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

Phase 4 (2026-06-27/28) added on top of this minimal baseline after closing Phase 3A
(CNN features carry real but weak signal, MLP+cosine matcher can't exploit it -- see
plan): 4A replaced the global dustbin_bias scalar with a per-point dustbin_head
(below), fixed through two rounds (class-imbalance reweighting, then bounding
dustbin_logit to the match-logit scale) but did NOT beat the simple global-bias
baseline on top8_gap -- confirming the bottleneck is the encoder's lack of
cross-fragment interaction, not the dustbin formulation. 4B (`CrossAttnPairMatcherModel`,
below) addresses that directly: self-attention within each fragment + cross-attention
between i and j BEFORE the same SoftCorrespondenceMatcher head (kept unchanged,
already validated) -- isolates the effect of interaction from everything else.
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
    """Descriptor correlation + row-softmax with a learned, PER-POINT dustbin column.

    A point of i with no genuine contact in j (cf. Phase 2's multi-neighbor confound,
    ~65% of rows in the validated Phase 3A data) should match the dustbin, not be forced
    onto a wrong j point -- hence a dustbin column instead of a plain softmax over j only.

    Phase 4A change: the dustbin logit was a single GLOBAL learned scalar bias
    (Phase 3A V0/C1/C2) -- it could only express "rows go to dustbin X% of the time on
    average", not "THIS point of i has no counterpart in j" (which is exactly the
    multi-neighbor confound this column exists to handle). Replaced with a small
    per-point head `dustbin_head(desc_i)`, trained with an auxiliary `L_contact` BCE
    loss (cf. soft_correspondence_loss + this module's docstring in the training
    script) that explicitly separates two questions: (1) does this row have a real
    contact at all, (2) if so, which column. The head's last layer is zero-init with
    its bias set to `init_dustbin_bias`, so at initialization every point starts with
    the SAME dustbin logit (matching the old global-bias behavior) -- it only
    differentiates per point once gradient flows through L_contact/L_corr.

    Assumes desc_i/desc_j are L2-normalized (cosine similarity in [-1,1] -- true when
    PairMatcherModel.normalize_desc=True, the default). A LEARNED `logit_scale`
    (CLIP-style) replaces a fixed `/sqrt(D)` scaling: the latter was found empirically
    to collapse training to "always predict dustbin" -- with unit-norm descriptors,
    dividing by sqrt(D)=sqrt(128)~11.3 squashes match logits into [-0.09, 0.09] while
    the old dustbin bias was a free, unconstrained scalar, so the optimizer could cut
    the loss cheaply by growing it alone instead of learning to separate descriptors.
    Both initialized so match logits and the dustbin logit start on a comparable scale
    (~10), removing that structural advantage -- still relevant with a per-point head.

    Second collapse found with the per-point dustbin_head (Phase 4A, after fixing
    L_contact's class imbalance): dustbin_logit is BCE-trained (L_contact) with no
    constraint tying its magnitude to the match-logit scale -- BCE's incentive to push
    confidently-correct (majority dustbin) rows to extreme logit values is unrelated to
    what the softmax in L_corr actually needs (a per-row comparison against THAT row's
    match logits, which stay roughly within [-scale, scale]). Observed: dustbin_logit
    mean drifted to 3.7-6+ while match logits stayed ~0.3-2.6, so dustbin started
    winning the softmax even on rows L_contact "correctly" ranked as lower-dustbin-than-
    average. Fixed by bounding dustbin_logit to (-scale, scale) via `scale * tanh(...)`,
    using the SAME learned scale as the match logits -- structurally removes the
    runaway degree of freedom instead of hoping the two losses stay balanced.
    """

    def __init__(self, desc_dim: int, init_logit_scale: float = 10.0, init_dustbin_bias: float = 0.0):
        super().__init__()
        self.desc_dim = desc_dim
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))
        self.dustbin_head = nn.Sequential(
            nn.Linear(desc_dim, desc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(desc_dim // 2, 1),
        )
        nn.init.zeros_(self.dustbin_head[-1].weight)
        # tanh(bias) * init_logit_scale == init_dustbin_bias at step 0 (matches the old
        # global-bias behavior exactly, since the head's weight is zero-init above).
        tanh_target = max(min(init_dustbin_bias / init_logit_scale, 0.999), -0.999)
        nn.init.constant_(self.dustbin_head[-1].bias, math.atanh(tanh_target))

    def forward(self, desc_i: torch.Tensor, desc_j: torch.Tensor, valid_j: torch.Tensor):
        """desc_i, desc_j: [B, N, D]. valid_j: [B, N] bool (padded j columns are masked
        out with -inf before softmax, so they can structurally never receive any weight
        -- consistent with the `valid_target_col_rate=1.0` invariant already enforced in
        the dataset-check script). Returns (logits [B,N,N+1], P [B,N,N+1], dustbin_logit
        [B,N] -- the raw per-point dustbin logit, exposed separately for L_contact)."""
        B, N, D = desc_i.shape
        cos_sim = torch.einsum("bnd,bmd->bnm", desc_i, desc_j)
        scale = self.logit_scale.exp().clamp(max=50.0)
        S = scale * cos_sim
        S = S.masked_fill(~valid_j.unsqueeze(1), float("-inf"))
        dustbin_raw = self.dustbin_head(desc_i).squeeze(-1)  # [B, N]
        dustbin_logit = scale * torch.tanh(dustbin_raw)  # bounded to (-scale, scale)
        logits = torch.cat([S, dustbin_logit.unsqueeze(-1)], dim=-1)
        P = torch.softmax(logits, dim=-1)
        return logits, P, dustbin_logit


class PairMatcherModel(nn.Module):
    def __init__(self, in_dim: int = 8, hidden=(64, 128), desc_dim: int = 128,
                 dropout: float = 0.1, normalize_desc: bool = True,
                 init_logit_scale: float = 10.0, init_dustbin_bias: float = 0.0):
        super().__init__()
        self.encoder = PointEncoder(in_dim, hidden, desc_dim, dropout)
        self.matcher = SoftCorrespondenceMatcher(desc_dim, init_logit_scale, init_dustbin_bias)
        self.normalize_desc = normalize_desc

    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor,
                valid_i: torch.Tensor, valid_j: torch.Tensor):
        """valid_i is accepted but unused -- this encoder processes i and j independently
        (no interaction, cf. Phase 4B's CrossAttnPairMatcherModel below, which DOES need
        it for self-attention masking). Kept in the signature so callers can use the same
        call site for both model classes."""
        desc_i = self.encoder(feat_i)
        desc_j = self.encoder(feat_j)
        if self.normalize_desc:
            # L2-normalize before the dot product so descriptor magnitude can't trivially
            # sharpen/flatten the softmax temperature -- the network must encode
            # similarity in direction, not norm.
            desc_i = F.normalize(desc_i, dim=-1)
            desc_j = F.normalize(desc_j, dim=-1)
        return self.matcher(desc_i, desc_j, valid_j)


class CrossAttentionBlock(nn.Module):
    """One block of: self-attention within each fragment, then cross-attention between
    fragments, then a per-point FFN -- standard transformer-encoder pattern, applied
    symmetrically to i and j (i attends to j, j attends to i, same block instance reused
    so weights are shared -- consistent with the rest of this module's "weight sharing
    between fragments" convention, e.g. PointEncoder is also a single shared MLP)."""

    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim),
        )

    def _apply(self, x: torch.Tensor, other: torch.Tensor, kpm_self: torch.Tensor, kpm_other: torch.Tensor):
        a, _ = self.self_attn(x, x, x, key_padding_mask=kpm_self, need_weights=False)
        x = self.norm_self(x + a)
        c, _ = self.cross_attn(x, other, other, key_padding_mask=kpm_other, need_weights=False)
        x = self.norm_cross(x + c)
        x = self.norm_ff(x + self.ff(x))
        return x

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, valid_i: torch.Tensor, valid_j: torch.Tensor):
        """x_i, x_j: [B, N, D]. valid_i/valid_j: [B, N] bool. nn.MultiheadAttention's
        key_padding_mask convention is True=IGNORE, the opposite of valid_i/valid_j
        (True=keep) -- inverted here, once, rather than at every call site."""
        kpm_i, kpm_j = ~valid_i, ~valid_j
        # Both fragments updated from the SAME pre-block (x_i, x_j) -- not sequentially
        # (which would make j's cross-attention see an already-updated i within the same
        # block, breaking the symmetry between "i attends to j" and "j attends to i").
        new_i = self._apply(x_i, x_j, kpm_i, kpm_j)
        new_j = self._apply(x_j, x_i, kpm_j, kpm_i)
        return new_i, new_j


class CrossAttnEncoder(nn.Module):
    """Phase 4B encoder: shared input projection -> L stacked CrossAttentionBlocks. Each
    fragment's points first attend to each other (self-attention), then to the OTHER
    fragment's points (cross-attention) -- the structural fix Phase 4A's per-point
    dustbin head couldn't provide: i and j are no longer encoded independently before
    the cosine-similarity head sees them."""

    def __init__(self, in_dim: int, desc_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, desc_dim), nn.LayerNorm(desc_dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(desc_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor,
                valid_i: torch.Tensor, valid_j: torch.Tensor):
        x_i, x_j = self.input_proj(feat_i), self.input_proj(feat_j)
        for block in self.blocks:
            x_i, x_j = block(x_i, x_j, valid_i, valid_j)
        return x_i, x_j


class CrossAttnPairMatcherModel(nn.Module):
    """Phase 4B model: CrossAttnEncoder -> the SAME SoftCorrespondenceMatcher used by
    PairMatcherModel (unchanged, already validated in Phase 3A/4A) -- isolates the effect
    of adding cross-fragment interaction from everything else (dustbin head, logit_scale
    bounding, loss functions all stay identical)."""

    def __init__(self, in_dim: int = 65, desc_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1, normalize_desc: bool = True,
                 init_logit_scale: float = 10.0, init_dustbin_bias: float = 0.0):
        super().__init__()
        self.encoder = CrossAttnEncoder(in_dim, desc_dim, num_layers, num_heads, dropout)
        self.matcher = SoftCorrespondenceMatcher(desc_dim, init_logit_scale, init_dustbin_bias)
        self.normalize_desc = normalize_desc

    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor,
                valid_i: torch.Tensor, valid_j: torch.Tensor):
        desc_i, desc_j = self.encoder(feat_i, feat_j, valid_i, valid_j)
        if self.normalize_desc:
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


def contact_loss(
    dustbin_logit: torch.Tensor, target: torch.Tensor, valid_i: torch.Tensor,
    contact_row_weight: float = 2.0, dustbin_row_weight: float = 1.0,
) -> torch.Tensor:
    """Phase 4A auxiliary loss: BCE between the per-point dustbin head's prediction and
    whether the row actually has a real contact (target's dustbin column == 0), restricted
    to valid_i rows. Separates two questions the single soft_correspondence_loss conflates:
    (1) does this row of i have ANY counterpart in j (this loss), (2) if so, which column
    (soft_correspondence_loss). `-dustbin_logit` is used as the "has contact" logit (high
    dustbin_logit = confidently dustbin = low contact probability).

    Reweighted by the SAME contact_row_weight/dustbin_row_weight as soft_correspondence_loss
    -- a first run without this reweighting collapsed to "always predict dustbin" again
    (dustbin_minus_max_match went from -1.39 to +2.44 over 15 epochs, contact_pred_rate
    85.8%->6.7%): with ~65% dustbin rows and a plain unweighted BCE, the optimizer could
    cheaply push dustbin_logit toward +inf for every row (correct on the 65% majority,
    wrong on the 35% minority) -- the exact same imbalance-driven shortcut as the original
    global dustbin_bias collapse, just relocated into this new auxiliary loss."""
    has_contact = (target[..., -1] <= 0.5).float()  # [B, N]
    bce = F.binary_cross_entropy_with_logits(-dustbin_logit, has_contact, reduction="none")
    row_weight = torch.where(
        has_contact.bool(),
        torch.full_like(bce, contact_row_weight),
        torch.full_like(bce, dustbin_row_weight),
    )
    weight = row_weight * valid_i.float()
    return (bce * weight).sum() / weight.sum().clamp(min=1e-8)


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
