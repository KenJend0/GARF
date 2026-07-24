"""
assembly/models/hybrid_geometry_features.py
============================================
Lightweight geometric feature extractor for the hybrid segmentation experiment.

For each point in a fragment, computes explicit descriptors from its local
k-nearest-neighbor (k-NN) neighborhood using local PCA. These descriptors
complement the implicit PTv3 features with interpretable geometric signals.

Features computed (all per-point):
  1. normal_consistency  (1D): alignment between the input normal and the
                               PCA-estimated normal of the local patch.
                               High → smooth, planar region.
                               Low  → noisy or geometrically ambiguous area.

  2. curvature           (1D, optional): λ_min / Σλ, where λ_i are the PCA
                               eigenvalues of the local neighborhood.
                               ≈ 0   → flat (all variance in one direction).
                               ≈ 1/3 → isotropic (corner / sharp feature).
                               Fracture surfaces tend to have higher curvature.

  3. roughness           (1D, optional): mean absolute distance of k-NN points
                               to the local tangent plane (defined by the
                               PCA-estimated normal and centroid).
                               High → bumpy / textured surface.
                               Low  → smooth / planar region.

  4. normals             (3D, optional): pass-through of the input normals.
                               Included as a feature so the MLP fusion head
                               can learn orientation-dependent patterns.

Design choices:
  - All computations run inside torch.no_grad() — no gradient through geo features.
  - Forward is per-fragment (N, 3), called in a loop over the flat batch.
  - Uses torch.linalg.eigh for stable, fast symmetric eigendecomposition.
  - k=16 neighbors is a good default: fast and statistically robust.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Brute-force k nearest neighbor indices for a single fragment.

    Args:
        points : (N, 3) coordinates of one fragment
        k      : number of neighbors (self excluded)

    Returns:
        idx    : (N, k) integer tensor of neighbor indices
    """
    # Squared pairwise distances via expansion:  ||a - b||^2 = a^2 + b^2 - 2 a·b
    sq = (points ** 2).sum(dim=-1, keepdim=True)        # (N, 1)
    dist2 = sq + sq.T - 2.0 * (points @ points.T)       # (N, N)
    dist2 = dist2.clamp(min=0.0)
    dist2.fill_diagonal_(float("inf"))                   # exclude self
    _, idx = dist2.topk(k, dim=-1, largest=False)        # (N, k)
    return idx


def _knn_indices_batched_chunk(points: torch.Tensor, k: int) -> torch.Tensor:
    """Un-chunked (K, N, N) brute-force kNN — see _knn_indices_batched for the
    chunking wrapper that keeps this call's memory bounded."""
    sq = (points ** 2).sum(dim=-1, keepdim=True)              # (K, N, 1)
    dist2 = sq + sq.transpose(1, 2) - 2.0 * (points @ points.transpose(1, 2))  # (K, N, N)
    dist2 = dist2.clamp(min=0.0)
    diag = torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
    dist2 = dist2.masked_fill(diag.unsqueeze(0), float("inf"))  # exclude self, per fragment
    _, idx = dist2.topk(k, dim=-1, largest=False)               # (K, N, k)
    return idx


def _knn_indices_batched(
    points: torch.Tensor, k: int, max_chunk_bytes: int = 300 * 1024 * 1024,
) -> torch.Tensor:
    """
    Brute-force k nearest neighbor indices, batched over K fragments that all
    share the same point count N (the common case: sample_method=uniform).

    Args:
        points : (K, N, 3) coordinates, K independent fragments
        k      : number of neighbors (self excluded)

    Returns:
        idx    : (K, N, k) integer tensor of neighbor indices, local to each
                 fragment (never crosses fragment boundaries — each fragment's
                 (N, N) distance matrix is computed independently within the
                 batched dim, exactly like calling _knn_indices K times).

    Chunked over K to bound peak GPU memory: the dense (K, N, N) distance
    tensor this needs grows linearly with K and quadratically with N (with
    this codebase's default num_points_to_sample=5000, that's ~100MB per
    fragment) -- an un-chunked call on a validation/training batch with K in
    the hundreds (batch_size=32 x up to max_parts=20 fragments/object) can
    request tens of GB in one allocation, which overflows the lab's 7.6GB
    GPUs even though no single fragment is large. Chunking bounds each call
    to roughly max_chunk_bytes and processes chunks sequentially; the result
    is identical to the un-chunked call since each fragment's kNN is
    independent of every other fragment (same rationale as _batched_eigh's
    chunking above, for an unrelated cusolver batch-size limit).
    """
    K, N, _ = points.shape
    bytes_per_frag = N * N * 4
    chunk_size = max(1, min(K, max_chunk_bytes // bytes_per_frag))

    if chunk_size >= K:
        return _knn_indices_batched_chunk(points, k)

    idx_chunks = [
        _knn_indices_batched_chunk(points[start:start + chunk_size], k)
        for start in range(0, K, chunk_size)
    ]
    return torch.cat(idx_chunks, dim=0)


def _gather_neighbors(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbor coordinates, supporting both a single fragment and a
    batch of K fragments (each fragment's indices only ever reference points
    within that same fragment — no cross-fragment mixing).

    Args:
        points : (N, 3)        or  (K, N, 3)
        idx    : (N, k)        or  (K, N, k)

    Returns:
        neighbors : (N, k, 3)  or  (K, N, k, 3)
    """
    if points.dim() == 2:
        return points[idx]                                  # (N, k, 3)

    K, N, _ = points.shape
    k = idx.shape[-1]
    batch_idx = torch.arange(K, device=points.device).view(K, 1, 1).expand(-1, N, k)
    return points[batch_idx, idx]                            # (K, N, k, 3)


def _batched_eigh(cov: torch.Tensor, chunk_size: int = 8192):
    """
    torch.linalg.eigh over a large batch of small (3, 3) matrices, chunked.

    cusolver's batched eigh path (cusolverDnXsyevBatched) has an internal
    batch-size ceiling — flattening (K, N, 3, 3) with K fragments * N=5000
    points easily exceeds it (e.g. K=15 -> 75,000 matrices), raising
    CUSOLVER_STATUS_INVALID_VALUE even with no NaNs involved. Chunking the
    eigh call keeps each call under the limit; it does not change the result
    (each 3x3 matrix is decomposed independently regardless of chunk
    boundaries) and is still far fewer Python-level calls than the original
    K-fragment loop this replaces.
    """
    orig_shape = cov.shape                                    # (..., 3, 3)
    flat = cov.reshape(-1, orig_shape[-2], orig_shape[-1])
    total = flat.shape[0]

    if total <= chunk_size:
        eigenvalues, eigenvectors = torch.linalg.eigh(flat)
    else:
        eigvals_chunks, eigvecs_chunks = [], []
        for start in range(0, total, chunk_size):
            ev, evec = torch.linalg.eigh(flat[start:start + chunk_size])
            eigvals_chunks.append(ev)
            eigvecs_chunks.append(evec)
        eigenvalues = torch.cat(eigvals_chunks, dim=0)
        eigenvectors = torch.cat(eigvecs_chunks, dim=0)

    eigenvalues = eigenvalues.reshape(*orig_shape[:-1])       # (..., 3)
    eigenvectors = eigenvectors.reshape(*orig_shape)          # (..., 3, 3)
    return eigenvalues, eigenvectors


def _local_pca(points: torch.Tensor, idx: torch.Tensor):
    """
    Compute local PCA covariance eigendecomposition for each point's neighborhood.
    Works for a single fragment or a batch of K fragments sharing the same N
    (each fragment's PCA is computed independently — this only vectorizes the
    K sequential Python calls into one, it does not mix fragments together).

    Args:
        points : (N, 3)        or  (K, N, 3)
        idx    : (N, k)        or  (K, N, k)  neighbor indices

    Returns:
        eigenvalues  : (N, 3)  or (K, N, 3)   sorted descending (λ_0 ≥ λ_1 ≥ λ_2)
        eigenvectors : (N, 3, 3)  or (K, N, 3, 3)  columns are eigenvectors
                       [..., 2] = direction of smallest variance
                       = PCA-estimated surface normal
    """
    k = idx.shape[-1]
    neighbors = _gather_neighbors(points, idx)          # (..., N, k, 3)
    centroid = neighbors.mean(dim=-2, keepdim=True)     # (..., N, 1, 3)
    centered = neighbors - centroid                     # (..., N, k, 3)

    # Unnormalized covariance matrix (..., N, 3, 3)
    cov = centered.transpose(-2, -1) @ centered / max(k - 1, 1)

    # torch.linalg.eigh: eigenvalues ascending, eigenvectors as columns
    # eigh does not support float16 on CUDA — upcast temporarily
    orig_dtype = cov.dtype
    eigenvalues, eigenvectors = _batched_eigh(cov.float())       # (...,3), (...,3,3)
    eigenvalues = eigenvalues.to(orig_dtype)
    eigenvectors = eigenvectors.to(orig_dtype)

    # Flip to descending order (conventional: λ_0 = largest variance)
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Feature extractor module
# ---------------------------------------------------------------------------

class HybridGeometryFeatures(nn.Module):
    """
    Explicit geometric descriptors per point, computed from local k-NN PCA.

    Output dimension is determined by which features are enabled:
        out_dim = 1                      (normal_consistency, always on)
                + 3 * use_normals
                + 1 * use_curvature
                + 1 * use_roughness

    Args:
        k              : number of nearest neighbors for local PCA (default 16)
        use_normals    : include the 3D surface normal as a feature (default True)
        use_curvature  : include the curvature scalar λ_min/Σλ    (default True)
        use_roughness  : include the roughness scalar              (default True)
    """

    def __init__(
        self,
        k: int = 16,
        use_normals: bool = True,
        use_curvature: bool = True,
        use_roughness: bool = True,
        use_dist_to_centroid: bool = False,
    ):
        super().__init__()
        self.k = k
        self.use_normals = use_normals
        self.use_curvature = use_curvature
        self.use_roughness = use_roughness
        self.use_dist_to_centroid = use_dist_to_centroid

        # Pre-compute output dimension so the caller can size the fusion MLP.
        self.out_dim = 1                        # normal_consistency always included
        if use_normals:
            self.out_dim += 3
        if use_curvature:
            self.out_dim += 1
        if use_roughness:
            self.out_dim += 1
        if use_dist_to_centroid:
            self.out_dim += 1

    @torch.no_grad()
    def forward_single(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geometric features for ONE fragment (no batch dimension).

        Args:
            xyz     : (N, 3)  point coordinates
            normals : (N, 3)  unit surface normals (from the dataset)

        Returns:
            features : (N, out_dim)  float32 geometric descriptors
        """
        # kNN builds an (N, N) distance matrix — kept on the input device (GPU).
        # The matrix is transient (freed after each fragment in the caller's
        # loop), so it does not accumulate across the K fragments in a batch.
        N = xyz.shape[0]
        k_actual = min(self.k, N - 1)

        if k_actual < 2:
            return torch.zeros(N, self.out_dim, device=xyz.device, dtype=xyz.dtype)

        idx = _knn_indices(xyz, k_actual)                       # (N, k)
        eigenvalues, eigenvectors = _local_pca(xyz, idx)        # (N,3), (N,3,3)

        # ---- Feature 1: normal consistency --------------------------------
        # eigenvectors[:, :, 2]  → direction of SMALLEST variance = PCA normal
        pca_normal = eigenvectors[:, :, 2]                      # (N, 3)

        # Orient PCA normal to face same hemisphere as input normal
        dot = (pca_normal * normals).sum(dim=-1, keepdim=True)  # (N, 1)
        pca_normal = pca_normal * dot.sign()                    # flip if anti-parallel

        # Consistency in [0, 1]:  1 → perfectly aligned, 0 → orthogonal
        consistency = dot.abs()                                 # (N, 1)

        features = [consistency]

        # ---- Feature 2: pass-through normals (optional) -------------------
        if self.use_normals:
            # Normals encode local orientation; useful for the MLP to learn
            # orientation-dependent fracture patterns (e.g. steep dihedral angles).
            features.append(normals)                            # (N, 3)

        # ---- Feature 3: curvature approximation (optional) ----------------
        if self.use_curvature:
            # Surface variation = λ_min / (λ_0 + λ_1 + λ_2)
            # Intuition: flat surface → λ_min ≈ 0, so curvature ≈ 0.
            #            corner/edge → λ_min comparable to others → curvature > 0.
            # Fracture surfaces typically show higher curvature than intact surfaces.
            total_var = eigenvalues.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            curvature = eigenvalues[:, 2:3] / total_var         # (N, 1)
            features.append(curvature)

        # ---- Feature 4: roughness (optional) ------------------------------
        if self.use_roughness:
            # Distance of each neighbor to the local tangent plane.
            # Tangent plane: defined by neighborhood centroid + PCA normal.
            neighbors = xyz[idx]                                # (N, k, 3)
            centroid = neighbors.mean(dim=1, keepdim=True)      # (N, 1, 3)
            offset = neighbors - centroid                       # (N, k, 3)

            # Signed projection onto pca_normal → absolute = perpendicular distance
            n_exp = pca_normal.unsqueeze(1)                     # (N, 1, 3)
            perp_dist = (offset * n_exp).sum(dim=-1).abs()      # (N, k)
            roughness = perp_dist.mean(dim=-1, keepdim=True)    # (N, 1)
            features.append(roughness)

        # ---- Feature 5: distance to fragment centroid (optional) ---------
        if self.use_dist_to_centroid:
            centroid = xyz.mean(dim=0, keepdim=True)              # (1, 3)
            dists = (xyz - centroid).norm(dim=-1, keepdim=True)   # (N, 1)
            # Normalise by mean (not max) — robust to outlier points
            mean_d = dists.mean().clamp(min=1e-6)
            features.append(dists / mean_d)                       # (N, 1) ratio > 1 = periphery

        return torch.cat(features, dim=-1)                       # (N, out_dim)

    @torch.no_grad()
    def forward_fragment_list(
        self,
        xyz_list: list,
        normals_list: list,
    ) -> list:
        """
        Batched equivalent of calling forward_single(xyz_list[k], normals_list[k])
        for k in range(K) in a Python loop — same formulas, same per-fragment
        kNN/PCA (fragments never see each other's points), just computed as one
        vectorized (K, N, ...) call instead of K sequential (N, ...) calls.

        Falls back to the per-fragment loop when fragments don't share the same
        point count N (only the uniform-sampling path guarantees that).

        Args:
            xyz_list     : list of K tensors (N, 3)
            normals_list : list of K tensors (N, 3)

        Returns:
            list of K tensors (N, out_dim) — same order, same values (up to
            float rounding) as the per-fragment loop.
        """
        K = len(xyz_list)
        if K == 0:
            return []
        N0 = xyz_list[0].shape[0]
        if any(x.shape[0] != N0 for x in xyz_list):
            return [self.forward_single(xyz_list[i], normals_list[i]) for i in range(K)]

        xyz = torch.stack(xyz_list, dim=0)              # (K, N, 3)
        normals = torch.stack(normals_list, dim=0)      # (K, N, 3)

        N = xyz.shape[1]
        k_actual = min(self.k, N - 1)
        if k_actual < 2:
            out = torch.zeros(K, N, self.out_dim, device=xyz.device, dtype=xyz.dtype)
            return [out[i] for i in range(K)]

        idx = _knn_indices_batched(xyz, k_actual)                  # (K, N, k)
        eigenvalues, eigenvectors = _local_pca(xyz, idx)           # (K,N,3), (K,N,3,3)

        pca_normal = eigenvectors[..., 2]                          # (K, N, 3)
        dot = (pca_normal * normals).sum(dim=-1, keepdim=True)     # (K, N, 1)
        pca_normal = pca_normal * dot.sign()
        consistency = dot.abs()

        features = [consistency]
        if self.use_normals:
            features.append(normals)
        if self.use_curvature:
            total_var = eigenvalues.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            curvature = eigenvalues[..., 2:3] / total_var          # (K, N, 1)
            features.append(curvature)
        if self.use_roughness:
            neighbors = _gather_neighbors(xyz, idx)                # (K, N, k, 3)
            centroid = neighbors.mean(dim=-2, keepdim=True)        # (K, N, 1, 3)
            offset = neighbors - centroid                          # (K, N, k, 3)
            n_exp = pca_normal.unsqueeze(-2)                       # (K, N, 1, 3)
            perp_dist = (offset * n_exp).sum(dim=-1).abs()         # (K, N, k)
            roughness = perp_dist.mean(dim=-1, keepdim=True)       # (K, N, 1)
            features.append(roughness)
        if self.use_dist_to_centroid:
            centroid = xyz.mean(dim=1, keepdim=True)               # (K, 1, 3)
            dists = (xyz - centroid).norm(dim=-1, keepdim=True)    # (K, N, 1)
            mean_d = dists.mean(dim=1, keepdim=True).clamp(min=1e-6)
            features.append(dists / mean_d)

        out = torch.cat(features, dim=-1)                          # (K, N, out_dim)
        return [out[i] for i in range(K)]

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience batched wrapper.  Processes each item in the batch independently.

        Args:
            xyz     : (B, N, 3)  or  (N, 3)
            normals : (B, N, 3)  or  (N, 3)

        Returns:
            features : (B, N, out_dim)  or  (N, out_dim)
        """
        if xyz.dim() == 2:
            return self.forward_single(xyz, normals)

        return torch.stack(
            [self.forward_single(xyz[i], normals[i]) for i in range(xyz.shape[0])],
            dim=0,
        )
