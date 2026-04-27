"""
assembly/models/projection_3d_to_2d.py
========================================
Orthographic 3D → 2D projection module.

For each fragment point cloud (N, 3), produces V depth-map images (C, H, W)
and records which flat pixel(s) each 3D point maps to (for backprojection).

Views (XYZ convention):
  view 0 — front : u=X, v=Y, depth=Z
  view 1 — side  : u=Z, v=Y, depth=X
  view 2 — top   : u=X, v=Z, depth=Y

Image channels:
  ch 0 — depth                        : average depth in [0, 1]
  ch 1 — occupancy                    : 1.0 where any point landed, else 0.0
  ch 2 — nx  (if use_normals)         : average world-space normal x
  ch 3 — ny  (if use_normals)
  ch 4 — nz  (if use_normals)
  ch 5 — curvature   (if geo_features): λ_min / Σλ from local PCA
  ch 6 — roughness   (if geo_features): mean perpendicular distance to tangent plane
  ch 7 — consistency (if geo_features): input normal vs PCA normal alignment

Projection modes (use_bilinear flag):
  False — floor assignment.
    Each 3D point maps to its single nearest pixel (floor rounding).
    Stored as pix_corners[:, v, 0] with weight 1.0; corners 1-3 are zero.
    Fast and simple. Output shape is (N, V, 4) / (N, V, 4) same as bilinear
    so all downstream code is identical.

  True  — bilinear assignment.
    Each 3D point distributes to its 4 surrounding pixels with bilinear
    weights that sum to 1.0 per (point, view).
    Reduces quantization noise. Partial differentiability: gradient flows
    through the bilinear weights (which depend on continuous sub-pixel u/v
    coordinates), but NOT through the floor operation on pixel indices.

Accumulation:
  Multiple points mapping to the same pixel are averaged (scatter_add_ /
  count), not overwritten. This is physically correct: the image at a pixel
  shows the average depth / orientation of all 3D points that project there.

Post-processing:
  Optional Gaussian splatting (splat_sigma > 0): a fixed depthwise Gaussian
  blur is applied after rasterization. This fills gaps between sparse projected
  points, increasing the effective signal area seen by the CNN. The pix_corners
  mapping is computed BEFORE splatting so backprojection still references the
  original (un-blurred) pixel.

No trainable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# (u_axis, v_axis, depth_axis) indices into XYZ for each view
_VIEW_AXES = [
    (0, 1, 2),   # front : u=X, v=Y, depth=Z
    (2, 1, 0),   # side  : u=Z, v=Y, depth=X
    (0, 2, 1),   # top   : u=X, v=Z, depth=Y
]


def _normalize_fragment(pts: torch.Tensor) -> torch.Tensor:
    """
    Center to centroid and scale to fit in [-0.95, 0.95]^3.

    The 0.95 margin prevents projected points from landing on the very
    border pixel, avoiding edge-wrap artifacts, and gives bilinear corners
    a safe 1-pixel margin before the image boundary.

    Args:
        pts : (N, 3)
    Returns:
        (N, 3)  values in [-0.95, 0.95]
    """
    centroid = pts.mean(dim=0, keepdim=True)
    pts = pts - centroid
    scale = pts.abs().max().clamp(min=1e-6)
    return pts / scale * 0.95


def _gaussian_splat(
    images: torch.Tensor,    # (V, C, H, W)
    kernel_size: int = 5,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Apply a fixed (non-trainable) Gaussian blur independently to every
    (H, W) slice via depthwise convolution.

    The blur fills gaps between sparse projected points, increasing the
    effective signal area seen by the CNN without changing pix_corners
    (backprojection still points to the original pre-blur pixel).

    Args:
        images      : (V, C, H, W)
        kernel_size : odd integer kernel size
        sigma       : Gaussian standard deviation in pixels
    Returns:
        blurred (V, C, H, W)
    """
    V, C, H, W = images.shape
    flat = images.view(V * C, 1, H, W)

    x = torch.arange(kernel_size, dtype=torch.float32, device=images.device)
    x = x - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    out = F.conv2d(flat, kernel, padding=kernel_size // 2)   # (V*C, 1, H, W)
    return out.view(V, C, H, W)


def project_fragment(
    pts: torch.Tensor,                    # (N, 3)
    normals: torch.Tensor = None,         # (N, 3) optional
    geo_features: torch.Tensor = None,    # (N, G) optional — e.g. curvature/roughness/consistency
    num_views: int = 3,
    resolution: int = 128,
    splat_sigma: float = 1.5,
    splat_kernel: int = 5,
    use_bilinear: bool = False,
) -> tuple:
    """
    Project one fragment to V orthographic depth-map images.

    Returns:
        images      : (V, C, H, W)
                      C = 2 (base) + 3 (normals) + G (geo_features)
        pix_corners : (N, V, 4) long        flat pixel indices for bilinear corners
        pix_weights : (N, V, 4) float       bilinear weights summing to 1 per (N, V)
        count_flat  : (V, H*W) float        accumulated bilinear weight per pixel per view
    """
    N = pts.shape[0]
    H = W = resolution
    V = num_views
    G = geo_features.shape[1] if geo_features is not None else 0
    C = 2 + (3 if normals is not None else 0) + G
    device = pts.device
    dtype = pts.dtype

    pts_n = _normalize_fragment(pts)   # (N, 3) in [-0.95, 0.95]

    images      = torch.zeros(V, C, H, W, device=device, dtype=dtype)
    pix_corners = torch.zeros(N, V, 4, device=device, dtype=torch.long)
    pix_weights = torch.zeros(N, V, 4, device=device, dtype=dtype)
    count_flat  = torch.zeros(V, H * W, device=device, dtype=dtype)

    for v in range(V):
        u_ax, v_ax, d_ax = _VIEW_AXES[v]

        # Continuous pixel coordinates in [0, W-1] and [0, H-1]
        u_f = (pts_n[:, u_ax] + 0.95) / 1.9 * (W - 1)   # (N,) float
        v_f = (pts_n[:, v_ax] + 0.95) / 1.9 * (H - 1)   # (N,) float

        if use_bilinear:
            # 4-corner bilinear assignment
            u0 = u_f.long().clamp(0, W - 2)
            u1 = u0 + 1
            v0 = v_f.long().clamp(0, H - 2)
            v1 = v0 + 1

            wu1 = (u_f - u0.to(dtype)).clamp(0.0, 1.0)
            wu0 = 1.0 - wu1
            wv1 = (v_f - v0.to(dtype)).clamp(0.0, 1.0)
            wv0 = 1.0 - wv1

            # corners: top-left, top-right, bottom-left, bottom-right
            corners = torch.stack([
                v0 * W + u0,
                v0 * W + u1,
                v1 * W + u0,
                v1 * W + u1,
            ], dim=1)   # (N, 4)
            weights = torch.stack([
                wv0 * wu0,
                wv0 * wu1,
                wv1 * wu0,
                wv1 * wu1,
            ], dim=1)   # (N, 4)
        else:
            # Floor assignment — degenerate bilinear: 1 corner, weight=1
            col = u_f.long().clamp(0, W - 1)
            row = v_f.long().clamp(0, H - 1)
            flat_idx = row * W + col           # (N,)
            corners  = torch.zeros(N, 4, device=device, dtype=torch.long)
            weights  = torch.zeros(N, 4, device=device, dtype=dtype)
            corners[:, 0] = flat_idx
            weights[:, 0] = 1.0

        # Flatten (N*4,) for scatter operations
        flat_c = corners.reshape(-1)   # (N*4,)
        w_flat = weights.reshape(-1)   # (N*4,)

        # Accumulated bilinear weight (= point count in floor mode)
        cnt = torch.zeros(H * W, device=device, dtype=dtype)
        cnt.scatter_add_(0, flat_c, w_flat)
        count_flat[v] = cnt

        # Depth channel: weighted-average depth
        depth_norm = (pts_n[:, d_ax] + 0.95) / 1.9                      # (N,)
        depth_rep  = depth_norm.unsqueeze(1).expand(-1, 4).reshape(-1)   # (N*4,)
        depth_acc  = torch.zeros(H * W, device=device, dtype=dtype)
        depth_acc.scatter_add_(0, flat_c, depth_rep * w_flat)
        images[v, 0] = (depth_acc / cnt.clamp(min=1e-6)).view(H, W)

        # Occupancy channel: 1 where at least one point projected
        images[v, 1] = (cnt > 0).to(dtype).view(H, W)

        # Normal channels (optional): weighted-average per-axis normal
        if normals is not None:
            for dim in range(3):
                n_rep = normals[:, dim].to(dtype).unsqueeze(1).expand(-1, 4).reshape(-1)
                n_acc = torch.zeros(H * W, device=device, dtype=dtype)
                n_acc.scatter_add_(0, flat_c, n_rep * w_flat)
                images[v, 2 + dim] = (n_acc / cnt.clamp(min=1e-6)).view(H, W)

        # Geometric feature channels (optional): weighted-average per feature
        if geo_features is not None:
            base_ch = 5 if normals is not None else 2
            for g in range(G):
                g_rep = geo_features[:, g].to(dtype).unsqueeze(1).expand(-1, 4).reshape(-1)
                g_acc = torch.zeros(H * W, device=device, dtype=dtype)
                g_acc.scatter_add_(0, flat_c, g_rep * w_flat)
                images[v, base_ch + g] = (g_acc / cnt.clamp(min=1e-6)).view(H, W)

        pix_corners[:, v, :] = corners
        pix_weights[:, v, :] = weights

    # Gaussian splatting — applied after rasterization, before CNN
    if splat_sigma > 0:
        images = _gaussian_splat(images, kernel_size=splat_kernel, sigma=splat_sigma)

    return images, pix_corners, pix_weights, count_flat


class Project3DTo2D(nn.Module):
    """
    Stateless orthographic projection module — no trainable parameters.

    Accepts a list of per-fragment point tensors (variable N per fragment)
    and returns stacked images, corner indices, bilinear weights, and
    per-pixel accumulation counts for every fragment in the batch.

    Args:
        num_views        : 1 (front), 2 (front+side), or 3 (front+side+top)
        resolution       : image H = W
        use_normals      : if True, include nx/ny/nz channels
        splat_sigma      : Gaussian blur sigma in pixels; 0 = disable splatting
        splat_kernel     : Gaussian blur kernel size (odd)
        use_bilinear     : if True, bilinear 4-corner projection; else floor assignment
        use_geo_features : if True, include 3 extra geometric channels:
                           curvature, roughness, normal_consistency (in that order)
    """

    def __init__(
        self,
        num_views: int = 3,
        resolution: int = 128,
        use_normals: bool = False,
        splat_sigma: float = 0.0,
        splat_kernel: int = 5,
        use_bilinear: bool = False,
        use_geo_features: bool = False,
    ):
        super().__init__()
        assert 1 <= num_views <= 3, "num_views must be 1, 2 or 3"
        self.num_views        = num_views
        self.resolution       = resolution
        self.use_normals      = use_normals
        self.splat_sigma      = splat_sigma
        self.splat_kernel     = splat_kernel
        self.use_bilinear     = use_bilinear
        self.use_geo_features = use_geo_features
        self.num_channels = (
            2
            + (3 if use_normals else 0)
            + (3 if use_geo_features else 0)   # curvature + roughness + consistency
        )

    def forward(
        self,
        frag_list: list,
        normal_list: list = None,
        geo_features_list: list = None,
    ) -> tuple:
        """
        Args:
            frag_list         : list of K tensors, each (N_i, 3)
            normal_list       : list of K tensors, each (N_i, 3) or None
            geo_features_list : list of K tensors, each (N_i, G) or None

        Returns:
            images             : (K, V, C, H, W) float
            pix_corners_list   : list of K tensors (N_i, V, 4) long
            pix_weights_list   : list of K tensors (N_i, V, 4) float
            count_list         : list of K tensors (V, H*W) float
        """
        imgs_all = []
        crn_all  = []
        wgt_all  = []
        cnt_all  = []

        for k, pts in enumerate(frag_list):
            normals_k     = normal_list[k]       if (normal_list       is not None) else None
            geo_k         = geo_features_list[k] if (geo_features_list is not None) else None
            imgs, corners, weights, cnt = project_fragment(
                pts,
                normals=normals_k,
                geo_features=geo_k,
                num_views=self.num_views,
                resolution=self.resolution,
                splat_sigma=self.splat_sigma,
                splat_kernel=self.splat_kernel,
                use_bilinear=self.use_bilinear,
            )
            imgs_all.append(imgs)
            crn_all.append(corners)
            wgt_all.append(weights)
            cnt_all.append(cnt)

        return torch.stack(imgs_all, dim=0), crn_all, wgt_all, cnt_all
