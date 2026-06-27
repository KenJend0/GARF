"""
assembly/models/pair_matching/soft_kabsch_matcher.py
======================================================
Phase 3A learned matcher (see PLAN_REASSEMBLY_MODULE.md, "Phase 3" section).

Minimal architecture, decided 2026-06-27 after the Phase 3A data-plumbing validation
(`scripts/phase3a_pair_dataset_check.py`):

    shared pointwise MLP encoder -> descriptor correlation -> row-softmax + learned
    dustbin column -> (V1 only) weighted Kabsch on the soft assignment

Deliberately NOT used at this stage (kept minimal, cf. Phase 2's lesson that stacking
unvalidated complexity made failures hard to diagnose): no Transformer/cross-attention,
no Sinkhorn (fracture surfaces are partial/many-to-one, not a clean balanced bipartite
match), no negative-pair sampling (Phase 3A = positive pairs only, registration question).

Two usage modes, both built from the same model:
  V0 -- correspondence-only: train with `soft_correspondence_loss` alone. Question:
        does the network learn to predict the target correspondence matrix better than
        chance? `weighted_kabsch` can still be called for MONITORING (Pose@30 etc.)
        without including it in the backward pass.
  V1 -- add `geodesic_rotation_loss` + translation L2 on the weighted-Kabsch pose,
        after a warmup of V0-only epochs (an untrained correspondence matrix produces
        an unstable/meaningless gradient through Kabsch's SVD).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointEncoder(nn.Module):
    """Shared pointwise MLP, applied independently to every point (no cross-point
    interaction here -- that happens in the descriptor correlation step). Default
    8 -> 64 -> 128 -> 128, matching the agreed minimal V0 spec (xyz + normals + cnn_score
    + dist_to_centroid = 8 input dims)."""

    def __init__(self, in_dim: int = 8, hidden=(64, 128), out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers = []
        for k in range(len(dims) - 1):
            layers.append(nn.Linear(dims[k], dims[k + 1]))
            is_last = k == len(dims) - 2
            if not is_last:
                layers.append(nn.LayerNorm(dims[k + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., out_dim]. nn.Linear broadcasts over leading dims,
        so this works directly on [B, N, in_dim] without reshaping."""
        return self.net(x)


class SoftCorrespondenceMatcher(nn.Module):
    """Descriptor correlation + row-softmax with a learned dustbin column.

    A point of i with no genuine contact in j (cf. Phase 2's multi-neighbor confound,
    ~65% of rows in the validated Phase 3A data) should match the dustbin, not be forced
    onto a wrong j point -- hence a dustbin column instead of a plain softmax over j only.
    The dustbin logit is a single learned scalar bias (broadcast to every row/pair),
    not a full Sinkhorn/optimal-transport formulation -- deliberately minimal for V0.

    Assumes desc_i/desc_j are L2-normalized (cosine similarity in [-1,1] -- true when
    PairMatcherModel.normalize_desc=True, the default). A LEARNED `logit_scale`
    (CLIP-style) replaces a fixed `/sqrt(D)` scaling: the latter was found empirically
    to collapse training to "always predict dustbin" -- with unit-norm descriptors,
    dividing by sqrt(D)=sqrt(128)~11.3 squashes match logits into [-0.09, 0.09] while
    dustbin_bias is a free, unconstrained scalar, so the optimizer could cut the loss
    cheaply by growing dustbin_bias alone instead of learning to separate descriptors.
    Both initialized so match logits and the dustbin logit start on a comparable scale
    (~10), removing that structural advantage.
    """

    def __init__(self, desc_dim: int, init_logit_scale: float = 10.0, init_dustbin_bias: float = 0.0):
        super().__init__()
        self.desc_dim = desc_dim
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))
        self.dustbin_bias = nn.Parameter(torch.full((1,), float(init_dustbin_bias)))

    def forward(self, desc_i: torch.Tensor, desc_j: torch.Tensor, valid_j: torch.Tensor):
        """desc_i, desc_j: [B, N, D]. valid_j: [B, N] bool (padded j columns are masked
        out with -inf before softmax, so they can structurally never receive any weight
        -- consistent with the `valid_target_col_rate=1.0` invariant already enforced in
        the dataset-check script). Returns (logits [B,N,N+1], P [B,N,N+1])."""
        B, N, D = desc_i.shape
        cos_sim = torch.einsum("bnd,bmd->bnm", desc_i, desc_j)
        scale = self.logit_scale.exp().clamp(max=50.0)
        S = scale * cos_sim
        S = S.masked_fill(~valid_j.unsqueeze(1), float("-inf"))
        dustbin_col = self.dustbin_bias.view(1, 1, 1).expand(B, N, 1)
        logits = torch.cat([S, dustbin_col], dim=-1)
        P = torch.softmax(logits, dim=-1)
        return logits, P


class PairMatcherModel(nn.Module):
    def __init__(self, in_dim: int = 8, hidden=(64, 128), desc_dim: int = 128,
                 dropout: float = 0.1, normalize_desc: bool = True,
                 init_logit_scale: float = 10.0, init_dustbin_bias: float = 0.0):
        super().__init__()
        self.encoder = PointEncoder(in_dim, hidden, desc_dim, dropout)
        self.matcher = SoftCorrespondenceMatcher(desc_dim, init_logit_scale, init_dustbin_bias)
        self.normalize_desc = normalize_desc

    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor, valid_j: torch.Tensor):
        desc_i = self.encoder(feat_i)
        desc_j = self.encoder(feat_j)
        if self.normalize_desc:
            # L2-normalize before the dot product so descriptor magnitude can't trivially
            # sharpen/flatten the softmax temperature -- the network must encode
            # similarity in direction, not norm.
            desc_i = F.normalize(desc_i, dim=-1)
            desc_j = F.normalize(desc_j, dim=-1)
        return self.matcher(desc_i, desc_j, valid_j)


def soft_correspondence_loss(
    P: torch.Tensor, target: torch.Tensor, valid_i: torch.Tensor,
    contact_row_weight: float = 2.0, dustbin_row_weight: float = 1.0, eps: float = 1e-8,
) -> torch.Tensor:
    """Soft cross-entropy between predicted P and target_ij, restricted to valid_i rows.
    Rows are reweighted by whether their target is a genuine contact (dustbin column == 0)
    or pure dustbin (== 1) -- without this, a model could trivially minimize the loss by
    always predicting dustbin given contact_row_rate~35% / dustbin_row_rate~65% (validated
    on the Phase 3A data check). contact_row_weight=2.0 / dustbin_row_weight=1.0 are the
    agreed starting values, not tuned."""
    log_p = torch.log(P.clamp(min=eps))
    ce = -(target * log_p).sum(dim=-1)  # [B, N]
    is_dustbin_row = target[..., -1] > 0.5
    weight = torch.where(
        is_dustbin_row,
        torch.full_like(ce, dustbin_row_weight),
        torch.full_like(ce, contact_row_weight),
    )
    weight = weight * valid_i.float()
    denom = weight.sum().clamp(min=eps)
    return (ce * weight).sum() / denom


def matching_entropy(P: torch.Tensor, valid_i: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean row entropy of P over valid_i rows (nats). Diagnostic only -- logged, not
    necessarily penalized at V0 (cf. plan: log first, add an entropy penalty only if the
    matrix turns out to stay too diffuse)."""
    ent = -(P.clamp(min=eps) * torch.log(P.clamp(min=eps))).sum(dim=-1)
    return (ent * valid_i.float()).sum() / valid_i.float().sum().clamp(min=eps)


def weighted_kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Differentiable weighted Kabsch: find R, t minimizing sum_n w_n ||R @ P_n + t - Q_n||^2.
    Batched over the leading dim. P, Q: [B, N, 3]; w: [B, N] (non-negative, NOT required to
    sum to 1 -- normalized internally). Same R @ P + t ~= Q convention as
    `scripts/phase2_geometric_baseline.py`'s `kabsch()`, so error metrics are comparable
    across Phase 2 and Phase 3.

    Returns (R [B,3,3], t [B,3], weight_sum [B]) -- weight_sum lets the caller mask out
    pairs whose total non-dustbin weight is too small to trust the resulting pose (e.g.
    almost everything routed to dustbin)."""
    w = w.clamp(min=0)
    weight_sum = w.sum(dim=1)
    w_norm = (w / weight_sum.clamp(min=eps).unsqueeze(1)).unsqueeze(-1)  # [B,N,1]
    p_mean = (w_norm * P).sum(dim=1, keepdim=True)
    q_mean = (w_norm * Q).sum(dim=1, keepdim=True)
    Pc = P - p_mean
    Qc = Q - q_mean
    H = torch.einsum("bn,bni,bnj->bij", w_norm.squeeze(-1), Pc, Qc)

    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
    ones = torch.ones_like(det)
    D = torch.diag_embed(torch.stack([ones, ones, torch.sign(det)], dim=-1))
    R = torch.matmul(torch.matmul(V, D), U.transpose(-1, -2))
    t = q_mean.squeeze(1) - torch.einsum("bij,bj->bi", R, p_mean.squeeze(1))
    return R, t, weight_sum


def geodesic_rotation_error(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic rotation error in radians, batched. [B,3,3] x [B,3,3] -> [B]."""
    rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_angle)
