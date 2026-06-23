"""
assembly/models/projection_mapping_utils.py
============================================
Pure-function helpers for extracting per-fragment tensors from GARF batch
dictionaries and for backprojecting 2D pixel predictions back to 3D points.

The GARF batch layout (from BreakingBadUniform.transform):
  pointclouds         : (B, max_parts * num_pts, 3)   flat; valid frags first
  pointclouds_normals : (B, max_parts * num_pts, 3)
  points_per_part     : (B, max_parts)                num_pts for valid, 0 for padding
  fracture_surface_gt : (B, max_parts * num_pts)      binary labels, 0 for padding

All functions are stateless — no nn.Module, no trainable parameters.
"""

import torch


def extract_fragment_list(
    pointclouds: torch.Tensor,       # (B, N_total, 3)
    points_per_part: torch.Tensor,   # (B, max_parts)
) -> tuple:
    """
    Reshape flat batch point-cloud tensor into per-fragment lists.

    Fast path: uniform sampling, all valid fragments share the same num_pts
    (sample_method=uniform — every slot, valid or padding, reserves exactly
    num_points_to_sample points, so N_total = P * num_pts always holds).

    Slow path: variable per-fragment sizes (sample_method=weighted — total
    points per object is fixed but split unevenly across the real fragments
    by surface area, padding slots get exactly 0 points). N_total // P does
    NOT recover the real per-fragment size here, so each object is sliced
    using its own points_per_part offsets instead (mirrors how FracSeg's own
    forward computes points_offset = cumsum(points_per_part[valid_pcs])).

    Returns:
        frag_list : list of K tensors (num_pts, 3)   valid fragments only
        valid_pcs : (B, max_parts) bool              True for non-padding fragments
        K         : total valid fragments in the batch
    """
    P = points_per_part.shape[1]
    valid_pcs = (points_per_part != 0)   # (B, P)

    if pointclouds.dim() != 3:
        # Already (B, P, num_pts, 3) — accept this layout defensively
        valid_frags = pointclouds[valid_pcs]     # (K, num_pts, 3)
        K = valid_frags.shape[0]
        return [valid_frags[k] for k in range(K)], valid_pcs, K

    B, N_total, C = pointclouds.shape
    uniform = bool((points_per_part[valid_pcs] == N_total // P).all())

    if uniform:
        num_pts = N_total // P
        pcs = pointclouds.view(B, P, num_pts, C)     # (B, P, num_pts, 3)
        valid_frags = pcs[valid_pcs]                 # (K, num_pts, 3)
        K = valid_frags.shape[0]
        return [valid_frags[k] for k in range(K)], valid_pcs, K

    # Variable per-fragment sizes: slice each object with its real offsets.
    frag_list = []
    for b in range(B):
        offset = 0
        for p in range(P):
            size = int(points_per_part[b, p].item())
            if size > 0:
                frag_list.append(pointclouds[b, offset:offset + size])
                offset += size
    return frag_list, valid_pcs, len(frag_list)


def extract_normal_list(
    normals: torch.Tensor,           # (B, N_total, 3) or (B, P, num_pts, 3)
    valid_pcs: torch.Tensor,         # (B, max_parts) bool, output of extract_fragment_list
    points_per_part: torch.Tensor,   # (B, max_parts)
) -> list:
    """
    Extract per-fragment normal tensors for valid fragments only.

    Returns:
        list of K tensors (num_pts, 3)
    """
    P = points_per_part.shape[1]

    if normals.dim() == 3:
        B, N_total, C = normals.shape
        num_pts = N_total // P
        norms = normals.view(B, P, num_pts, C)
    else:
        norms = normals  # already (B, P, num_pts, 3)

    valid_norms = norms[valid_pcs]   # (K, num_pts, 3)
    K = valid_norms.shape[0]
    return [valid_norms[k] for k in range(K)]


def extract_gt_for_valid_frags(
    fracture_surface_gt: torch.Tensor,  # (B, max_parts * num_pts) or (B, P, num_pts)
    valid_pcs: torch.Tensor,            # (B, max_parts) bool
    points_per_part: torch.Tensor,      # (B, max_parts)
) -> torch.Tensor:
    """
    Extract ground-truth labels for valid fragments only.

    Returns:
        (N_sum_valid,) long — labels for all points in all valid fragments
    """
    P = points_per_part.shape[1]

    if fracture_surface_gt.dim() == 2:
        B, N_flat = fracture_surface_gt.shape
        num_pts = N_flat // P
        gt = fracture_surface_gt.view(B, P, num_pts)
    else:
        gt = fracture_surface_gt  # already (B, P, num_pts)

    return gt[valid_pcs].reshape(-1).long()   # (K * num_pts,)


def extract_batch_idx(valid_pcs: torch.Tensor) -> torch.Tensor:
    """
    Return the batch-item index for each of the K valid fragments.

    torch.where on a (B, P) bool tensor returns (row_indices, col_indices)
    in row-major order — the same iteration order as extract_fragment_list
    (which also iterates valid_pcs in row-major order via boolean indexing).

    Used by CNNFracSeg.forward to aggregate per-batch-item context.

    Returns:
        (K,) long tensor with values in [0, B)
    """
    return torch.where(valid_pcs)[0]


def sample_features_at_points(
    feature_map: torch.Tensor,               # (K, V, C_feat, H, W)
    pix_corners_list: list,                  # list of K tensors (N_i, V, 4) long
    pix_weights_list: list,                  # list of K tensors (N_i, V, 4) float
    view_attn_weights: torch.Tensor = None,  # (K, V) softmax scores, or None
) -> list:
    """
    Sample C_feat-dimensional feature vectors at each 3D point's projected 2D
    location, then fuse across V views.

    Identical bilinear gather + view fusion logic as backproject_pixel_to_points,
    but operates on feature vectors (C_feat dims) instead of scalar predictions.

    Used by PointHead (Phase 3): extracts per-point 2D context from the CNN
    feature map before passing to the point-wise MLP.

    Returns:
        list of K tensors (N_k, C_feat)
    """
    K, V, C_feat, H, W = feature_map.shape

    result = []
    for k in range(K):
        corners_k = pix_corners_list[k]                          # (N_k, V, 4) long
        weights_k = pix_weights_list[k]                          # (N_k, V, 4) float
        feat_flat = feature_map[k].reshape(V, C_feat, H * W)    # (V, C_feat, H*W)

        N_k = corners_k.shape[0]

        # (V, N_k*4) — same reshape as backproject_pixel_to_points
        corners_vn4 = corners_k.permute(1, 0, 2).reshape(V, N_k * 4)
        weights_vn4 = weights_k.permute(1, 0, 2).reshape(V, N_k * 4)

        # Gather features at each bilinear corner: (V, C_feat, N_k*4)
        idx = corners_vn4.unsqueeze(1).expand(-1, C_feat, -1)
        feat_corners = feat_flat.gather(2, idx)

        # Apply bilinear weights and sum over 4 corners → (V, C_feat, N_k)
        w = weights_vn4.unsqueeze(1).expand(-1, C_feat, -1)
        feat_per_view = (feat_corners * w).reshape(V, C_feat, N_k, 4).sum(dim=3)

        # Multi-view fusion → (C_feat, N_k)
        if view_attn_weights is not None:
            alpha = view_attn_weights[k].view(V, 1, 1)
            fused = (feat_per_view * alpha).sum(dim=0)
        else:
            fused = feat_per_view.mean(dim=0)

        result.append(fused.T)   # (N_k, C_feat)

    return result


def backproject_pixel_to_points(
    pixel_seg: torch.Tensor,             # (K, V, H, W)   sigmoid predictions [0,1]
    pix_corners_list: list,              # list of K tensors (N_i, V, 4) long
    pix_weights_list: list,              # list of K tensors (N_i, V, 4) float
    view_attn_weights: torch.Tensor = None,  # (K, V) softmax scores over views, or None
) -> torch.Tensor:
    """
    Bilinear-interpolated, view-fused backprojection of pixel predictions to
    per-3D-point predictions.

    For each fragment k, view v, and point n:
      step 1 — bilinear gather:
        pred[k, v, n] = sum_{c=0}^{3}
            pixel_seg[k, v, corners[k][n, v, c]] * weights[k][n, v, c]
        (In floor mode: corners[1-3] are zero with weight zero — sum reduces
         to a single pixel lookup, identical to NearestNeighbour gather.)

    For each fragment k and point n:
      step 2 — multi-view fusion:
        if view_attn_weights provided (learned attention):
          out[k, n] = sum_v  alpha[k, v] * pred[k, v, n]
        else (uniform mean):
          out[k, n] = mean_v  pred[k, v, n]

    Fully differentiable:
      → through pixel_seg (CNN weights)
      → through pix_weights (bilinear weights ← sub-pixel 3D coordinates)
      → through view_attn_weights (ViewAttention MLP)

    Returns:
        (N_sum_valid,) float32
    """
    K, V, H, W = pixel_seg.shape
    all_preds = []

    for k in range(K):
        corners_k = pix_corners_list[k]        # (N_k, V, 4) long
        weights_k = pix_weights_list[k]        # (N_k, V, 4) float
        seg_flat  = pixel_seg[k].reshape(V, H * W)  # (V, H*W)

        N_k = corners_k.shape[0]

        # Reshape for batched gather: (V, N_k * 4)
        corners_vn4 = corners_k.permute(1, 0, 2).reshape(V, N_k * 4)
        weights_vn4 = weights_k.permute(1, 0, 2).reshape(V, N_k * 4)

        # Gather prediction at each bilinear corner: (V, N_k*4)
        pred_vn4 = seg_flat.gather(1, corners_vn4)

        # Weighted sum over 4 corners → (V, N_k)
        pred_per_view = (pred_vn4 * weights_vn4).reshape(V, N_k, 4).sum(dim=2)

        # Multi-view fusion → (N_k,)
        if view_attn_weights is not None:
            alpha      = view_attn_weights[k].unsqueeze(1)       # (V, 1)
            point_pred = (pred_per_view * alpha).sum(dim=0)      # (N_k,)
        else:
            point_pred = pred_per_view.mean(dim=0)               # (N_k,)

        all_preds.append(point_pred)

    return torch.cat(all_preds, dim=0)   # (N_sum_valid,)
