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


def _local_pca(points: torch.Tensor, idx: torch.Tensor):
    """
    Compute local PCA covariance eigendecomposition for each point's neighborhood.

    Args:
        points : (N, 3)
        idx    : (N, k) neighbor indices

    Returns:
        eigenvalues  : (N, 3)  sorted descending (λ_0 ≥ λ_1 ≥ λ_2)
        eigenvectors : (N, 3, 3)  columns are eigenvectors
                       eigenvectors[:, :, 2] = direction of smallest variance
                       = PCA-estimated surface normal
    """
    N, k = idx.shape
    neighbors = points[idx]                             # (N, k, 3)
    centroid = neighbors.mean(dim=1, keepdim=True)      # (N, 1, 3)
    centered = neighbors - centroid                     # (N, k, 3)

    # Unnormalized covariance matrix (N, 3, 3)
    cov = centered.transpose(1, 2) @ centered / max(k - 1, 1)

    # torch.linalg.eigh: eigenvalues ascending, eigenvectors as columns
    # eigh does not support float16 on CUDA — upcast temporarily
    orig_dtype = cov.dtype
    eigenvalues, eigenvectors = torch.linalg.eigh(cov.float())  # (N,3), (N,3,3)
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
    ):
        super().__init__()
        self.k = k
        self.use_normals = use_normals
        self.use_curvature = use_curvature
        self.use_roughness = use_roughness

        # Pre-compute output dimension so the caller can size the fusion MLP.
        self.out_dim = 1                        # normal_consistency always included
        if use_normals:
            self.out_dim += 3
        if use_curvature:
            self.out_dim += 1
        if use_roughness:
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
        # kNN builds an (N, N) distance matrix — run on CPU to avoid GPU OOM
        # for large fragments (N=5000 → 100 MB on GPU per fragment).
        out_device = xyz.device
        xyz     = xyz.cpu()
        normals = normals.cpu()

        N = xyz.shape[0]
        k_actual = min(self.k, N - 1)

        if k_actual < 2:
            return torch.zeros(N, self.out_dim, device=out_device, dtype=xyz.dtype)

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

        return torch.cat(features, dim=-1).to(out_device)        # (N, out_dim)

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
