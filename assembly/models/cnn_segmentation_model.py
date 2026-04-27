"""
assembly/models/cnn_segmentation_model.py
==========================================
CNN-based fracture segmentation model — experimental alternative to GARF PTv3.

Pipeline
--------
  batch["pointclouds"]  (B, N_total, 3)
    → extract_fragment_list            → K per-fragment point tensors
    → Project3DTo2D                    → (K, V, C, H, W) images + corner/weight maps
    → SimpleCNNBackbone.encode         → (K*V, btn_ch, H', W') bottleneck
    → [optional] global context inject → cross-fragment object descriptor fused in
    → [optional] ViewAttention         → (K, V) learned view importance weights
    → [optional] feature-level fusion  → fuse bottleneck over V, broadcast back
    → SimpleCNNBackbone.decode         → (K*V, 1, H, W) logits → sigmoid
    → backproject_pixel_to_points      → (N_sum_valid,) per-point predictions
    → dice_loss + acc/recall/precision/F1

Ablation flags:
  use_normals        → 5-ch vs 2-ch image input
  splat_sigma > 0    → Gaussian splatting after projection
  use_bilinear       → bilinear vs floor projection
  use_global_context → cross-fragment object-level context injection
  use_view_attn      → learned view attention weights
  feature_fusion     → "none" | "mean" | "max" | "concat"
                       moves view fusion from prediction level (Step 6/7)
                       to bottleneck level so all decoders share a common
                       cross-view representation before predicting

The model has an identical LightningModule API to FracSeg and HybridFracSeg:
  same criteria(), same step method signatures, same log keys.
  This makes metric tables directly comparable across all three models.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics

from assembly.models.pretraining.loss import dice_loss
from assembly.models.projection_3d_to_2d import Project3DTo2D
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures
from assembly.models.projection_mapping_utils import (
    extract_fragment_list,
    extract_normal_list,
    extract_gt_for_valid_frags,
    extract_batch_idx,
    backproject_pixel_to_points,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU helper block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ViewAttention(nn.Module):
    """
    Learns to assign importance weights to each view per fragment.

    Architecture: Linear → ReLU → Linear → softmax over V.
    Operates independently per (fragment, view); no spatial pooling needed
    because this runs on the global-average-pooled bottleneck.

    Training signal: end-to-end with segmentation loss. The module learns
    that certain views provide more reliable predictions per fragment geometry
    (e.g. top view for flat fragments, side view for tall ones).

    Args:
        in_ch     : bottleneck channel dimension (SimpleCNNBackbone.btn_ch)
        hidden_ch : hidden dimension of the attention MLP
    """

    def __init__(self, in_ch: int, hidden_ch: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 1),
        )

    def forward(self, gap_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gap_kv : (K, V, in_ch)  — global-average-pooled bottleneck per fragment·view
        Returns:
            (K, V) softmax attention weights summing to 1 over views
        """
        scores = self.mlp(gap_kv).squeeze(-1)    # (K, V)
        return torch.softmax(scores, dim=1)       # (K, V)


class SimpleCNNBackbone(nn.Module):
    """
    Minimal encoder–decoder CNN for per-pixel binary segmentation.

    Architecture (default: base_ch=16, num_blocks=3):

      Encoder
        Block 0 : Conv(in_ch→16) BN ReLU → Conv(16→16)   BN ReLU → MaxPool/2
        Block 1 : Conv(16→32)    BN ReLU → Conv(32→32)   BN ReLU → MaxPool/2
        Block 2 : Conv(32→64)    BN ReLU → Conv(64→64)   BN ReLU  (no pool — bottleneck)

      Context injection (optional between encode and decode):
        concat(bottleneck [btn_ch], global_ctx [btn_ch]) → 1×1 Conv → (btn_ch)

      Decoder
        Bilinear upsample × 4  → back to input H×W
        Conv(btn_ch→2*base_ch) BN ReLU
        Conv(2*base_ch→1, 1×1) → logits (sigmoid applied in CNNFracSeg)

    Same weights are used for every (fragment, view) call — implicit
    invariance to view permutation and lower parameter count.

    Args:
        in_ch      : input image channels (2 without normals, 5 with normals)
        base_ch    : base channel multiplier (default 16 → 32 → 64)
        num_blocks : encoder depth (2 or 3)
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 16, num_blocks: int = 3):
        super().__init__()

        channels = [in_ch] + [base_ch * (2 ** i) for i in range(num_blocks)]
        self.btn_ch = channels[-1]          # bottleneck channel dim (public attribute)
        self._pool_steps = num_blocks - 1   # last encoder block has no pool

        encoder_blocks = []
        for i in range(num_blocks):
            encoder_blocks.append(nn.Sequential(
                _ConvBnRelu(channels[i], channels[i + 1]),
                _ConvBnRelu(channels[i + 1], channels[i + 1]),
            ))
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # Context injection: concat(bottleneck, global_ctx) → bottleneck via 1×1 conv
        self.context_proj = nn.Conv2d(
            2 * self.btn_ch, self.btn_ch, kernel_size=1, bias=False
        )

        self.decoder_conv = nn.Sequential(
            _ConvBnRelu(self.btn_ch, base_ch * 2),
            nn.Conv2d(base_ch * 2, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, in_ch, H, W) → (B, btn_ch, H', W')
        H' = H / 2^(num_blocks-1) e.g. 128 → 32 for num_blocks=3
        """
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < self._pool_steps:
                x = F.max_pool2d(x, kernel_size=2)
        return x

    def inject_context(
        self,
        bottleneck: torch.Tensor,        # (B, btn_ch, H', W')
        context_spatial: torch.Tensor,   # (B, btn_ch, H', W')
    ) -> torch.Tensor:
        """
        Fuse global object context into per-fragment bottleneck.
        Concatenates along channel dim then projects back to btn_ch via 1×1 conv.
        (B, 2*btn_ch, H', W') → (B, btn_ch, H', W')
        """
        return self.context_proj(torch.cat([bottleneck, context_spatial], dim=1))

    def decode(self, x: torch.Tensor, output_hw: tuple) -> torch.Tensor:
        """
        (B, btn_ch, H', W') → (B, 1, H, W) logits
        Bilinear upsample avoids checkerboard artifacts from ConvTranspose2d.
        """
        H, W = output_hw
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward without context (used when use_global_context=False)."""
        H, W = x.shape[-2], x.shape[-1]
        return self.decode(self.encode(x), (H, W))


# ---------------------------------------------------------------------------
# U-Net backbone (skip connections encoder → decoder)
# ---------------------------------------------------------------------------

class UNetBackbone(nn.Module):
    """
    U-Net style encoder-decoder with skip connections.

    Architecture (default: base_ch=16, depth=4):
      Encoder:
        Level 0 : Conv(in_ch →  16) ×2  → MaxPool/2   [skip_0: (B, 16,  H,   W  )]
        Level 1 : Conv(16    →  32) ×2  → MaxPool/2   [skip_1: (B, 32,  H/2, W/2)]
        Level 2 : Conv(32    →  64) ×2  → MaxPool/2   [skip_2: (B, 64,  H/4, W/4)]
        Bottleneck: Conv(64  → 128) ×2               [(B, 128, H/8, W/8)]

      Decoder (each level: upsample × concat(skip) × Conv ×2):
        Level 2 : up → cat(skip_2) → Conv(128+64→64)  ×2
        Level 1 : up → cat(skip_1) → Conv(64+32 →32)  ×2
        Level 0 : up → cat(skip_0) → Conv(32+16 →16)  ×2
        Head    : Conv(16→1, 1×1)  → logits

    btn_ch exposed for compatibility with ViewAttention and context injection.
    Context injection is applied at the bottleneck before decoding.
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Build encoder channel sizes: [in_ch, base_ch, base_ch*2, ..., base_ch*2^(depth-1)]
        enc_chs = [in_ch] + [base_ch * (2 ** i) for i in range(depth)]
        self.btn_ch = enc_chs[-1]

        # Encoder blocks
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                _ConvBnRelu(enc_chs[i], enc_chs[i + 1]),
                _ConvBnRelu(enc_chs[i + 1], enc_chs[i + 1]),
            )
            for i in range(depth)
        ])

        # Context injection at bottleneck
        self.context_proj = nn.Conv2d(
            2 * self.btn_ch, self.btn_ch, kernel_size=1, bias=False
        )

        # Decoder blocks — input is concat(upsampled, skip)
        dec_chs = list(reversed(enc_chs[1:]))   # [btn_ch, btn_ch/2, ..., base_ch]
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                _ConvBnRelu(dec_chs[i] + dec_chs[i + 1], dec_chs[i + 1]),
                _ConvBnRelu(dec_chs[i + 1], dec_chs[i + 1]),
            )
            for i in range(depth - 1)
        ])

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def encode(self, x: torch.Tensor):
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            if i < self.depth - 1:
                skips.append(x)
                x = F.max_pool2d(x, kernel_size=2)
        return x, skips   # bottleneck, list of skip tensors

    def inject_context(self, bottleneck, context_spatial):
        return self.context_proj(torch.cat([bottleneck, context_spatial], dim=1))

    def decode(self, bottleneck: torch.Tensor, skips: list, output_hw: tuple):
        x = bottleneck
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips, (H, W))


# ---------------------------------------------------------------------------
# Main LightningModule
# ---------------------------------------------------------------------------

class CNNFracSeg(pl.LightningModule):
    """
    CNN fracture segmentation model.

    Drop-in LightningModule replacement for FracSeg and HybridFracSeg.
    Identical batch interface, identical criteria() and step method signatures,
    identical metric log keys — results are directly comparable.

    All six ablation flags default to OFF so that the model in its
    default configuration equals the Step-1 baseline. Each step config
    turns exactly one flag on, giving controlled one-variable-at-a-time
    ablations.

    Args:
        optimizer          : partial optimizer constructor
        lr_scheduler       : partial lr scheduler (optional)
        num_views          : number of orthographic views (1, 2 or 3)
        resolution         : image H = W (64 or 128)
        base_ch            : CNN base channel multiplier
        num_blocks         : CNN encoder depth
        use_normals        : add nx/ny/nz image channels (5-ch vs 2-ch)
        splat_sigma        : Gaussian splatting sigma in pixels; 0 = disabled
        splat_kernel       : Gaussian kernel size
        use_bilinear       : bilinear vs floor projection
        use_global_context : enable cross-fragment global reasoning
        use_view_attn      : enable learned view attention
        view_attn_hidden   : hidden dim of ViewAttention MLP
        backbone           : "simple" (default) or "unet" (skip connections)
        unet_depth         : encoder depth for UNetBackbone (default 4)
        feature_fusion     : "none" | "mean" | "max" | "concat"
                             Fuses bottleneck features across views BEFORE decoding
                             so all V decoders share a common cross-view representation.
                             "none"   — no feature fusion (Steps 1-7 behaviour)
                             "mean"   — mean over views (weighted by view_attn if enabled)
                             "max"    — element-wise max over views
                             "concat" — concat views then project back via 1×1 conv
        use_geo_features   : add 3 geometric feature channels to the projected image:
                             curvature (λ_min/Σλ), roughness, normal_consistency.
                             Requires use_normals=True (normals needed for kNN PCA).
        geo_k              : k-NN neighborhood size for geometric feature computation
    """

    def __init__(
        self,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        num_views: int = 3,
        resolution: int = 128,
        base_ch: int = 16,
        num_blocks: int = 3,
        use_normals: bool = False,
        splat_sigma: float = 0.0,
        splat_kernel: int = 5,
        use_bilinear: bool = False,
        use_global_context: bool = False,
        use_view_attn: bool = False,
        view_attn_hidden: int = 16,
        backbone: str = "simple",
        unet_depth: int = 4,
        feature_fusion: str = "none",
        use_geo_features: bool = False,
        geo_k: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler"])

        self.num_views          = num_views
        self.resolution         = resolution
        self.use_normals        = use_normals
        self.use_global_context = use_global_context
        self.use_view_attn      = use_view_attn
        self.feature_fusion     = feature_fusion
        self.use_geo_features   = use_geo_features
        self._optimizer         = optimizer
        self._lr_scheduler      = lr_scheduler

        self.projector = Project3DTo2D(
            num_views=num_views,
            resolution=resolution,
            use_normals=use_normals,
            splat_sigma=splat_sigma,
            splat_kernel=splat_kernel,
            use_bilinear=use_bilinear,
            use_geo_features=use_geo_features,
        )

        if use_geo_features:
            # No normals in geo extractor — already projected as image channels
            self.geo_extractor = HybridGeometryFeatures(
                k=geo_k,
                use_normals=False,
                use_curvature=True,
                use_roughness=True,
            )

        if backbone == "unet":
            self.cnn = UNetBackbone(
                in_ch=self.projector.num_channels,
                base_ch=base_ch,
                depth=unet_depth,
            )
        else:
            self.cnn = SimpleCNNBackbone(
                in_ch=self.projector.num_channels,
                base_ch=base_ch,
                num_blocks=num_blocks,
            )

        if use_view_attn:
            self.view_attn = ViewAttention(
                in_ch=self.cnn.btn_ch,
                hidden_ch=view_attn_hidden,
            )

        if feature_fusion == "concat":
            # Project V concatenated bottleneck channels back to btn_ch
            self.fusion_proj = nn.Conv2d(
                num_views * self.cnn.btn_ch, self.cnn.btn_ch, kernel_size=1, bias=False
            )

    # ------------------------------------------------------------------
    # Shared criteria — identical to FracSeg
    # ------------------------------------------------------------------

    def criteria(self, input_dict, output_dict):
        """
        Dice loss + accuracy / recall / precision / F1.
        Identical API to FracSeg.criteria() for direct metric comparison.
        """
        pred    = output_dict["coarse_seg_pred"]         # (N_sum_valid,)
        pred_b  = output_dict["coarse_seg_pred_binary"]  # (N_sum_valid,) bool
        gt      = output_dict["coarse_seg_gt"].float()   # (N_sum_valid,)
        gt_long = output_dict["coarse_seg_gt"]           # (N_sum_valid,) long

        loss = dice_loss(pred, gt)

        acc       = torchmetrics.functional.accuracy(pred_b, gt_long, task="binary")
        recall    = torchmetrics.functional.recall(pred_b, gt_long, task="binary")
        precision = torchmetrics.functional.precision(pred_b, gt_long, task="binary")
        f1        = torchmetrics.functional.f1_score(pred_b, gt_long, task="binary")

        return loss, {
            "coarse_seg_loss":      loss,
            "coarse_seg_acc":       acc,
            "coarse_seg_recall":    recall,
            "coarse_seg_precision": precision,
            "coarse_seg_f1":        f1,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        """
        9-step forward pass.

        Batch keys consumed:
          pointclouds          : (B, N_total, 3)
          pointclouds_normals  : (B, N_total, 3)  — read only when use_normals=True
          points_per_part      : (B, max_parts)
          fracture_surface_gt  : (B, N_total)      — used for GT extraction

        Returns dict with:
          coarse_seg_pred        : (N_sum_valid,) float  probabilities
          coarse_seg_pred_binary : (N_sum_valid,) bool   > 0.5 threshold
          coarse_seg_gt          : (N_sum_valid,) long   ground-truth labels
        """
        out = {}
        pointclouds     = batch["pointclouds"]       # (B, N_total, 3)
        points_per_part = batch["points_per_part"]   # (B, max_parts)
        B = pointclouds.shape[0]

        # 1. Extract valid per-fragment point lists
        frag_list, valid_pcs, K = extract_fragment_list(pointclouds, points_per_part)

        if K == 0:
            dummy = torch.zeros(0, device=pointclouds.device)
            out["coarse_seg_pred"]        = dummy
            out["coarse_seg_pred_binary"] = dummy.bool()
            out["coarse_seg_gt"]          = dummy.long()
            return out

        # 2. Normal list (if available and requested)
        normal_list = None
        if self.use_normals and "pointclouds_normals" in batch:
            normal_list = extract_normal_list(
                batch["pointclouds_normals"], valid_pcs, points_per_part
            )

        # 2.5 Geometric features per fragment (Step 9)
        # Computed from 3D points + normals via local k-NN PCA, then projected
        # to 2D as extra image channels. No gradients — pure geometric descriptors.
        geo_features_list = None
        if self.use_geo_features and normal_list is not None:
            geo_features_list = [
                self.geo_extractor.forward_single(frag_list[k], normal_list[k])
                for k in range(K)
            ]

        # 3. Orthographic projection → (K, V, C, H, W) + corner/weight maps
        images, pix_corners_list, pix_weights_list, _ = self.projector(
            frag_list, normal_list, geo_features_list
        )

        # 4. CNN encode — all K*V views in one batched call (weight sharing)
        V = self.num_views
        H = W = self.resolution
        C = self.projector.num_channels

        images_flat = images.view(K * V, C, H, W)
        enc_out = self.cnn.encode(images_flat)
        if isinstance(enc_out, tuple):
            bottleneck, _skips = enc_out   # UNet returns (bottleneck, skips)
        else:
            bottleneck, _skips = enc_out, None   # SimpleCNN returns bottleneck only
        btn_ch = bottleneck.shape[1]
        Hp, Wp = bottleneck.shape[2], bottleneck.shape[3]

        # 5. Global object context injection
        if self.use_global_context:
            # a. Global average pool per (fragment, view): (K*V, btn_ch)
            gap = bottleneck.mean(dim=[-2, -1])

            # b. Per-fragment mean over views: (K, btn_ch)
            frag_feat = gap.view(K, V, btn_ch).mean(dim=1)

            # c. Mean-aggregate per batch item using scatter_add_: (B, btn_ch)
            batch_idx   = extract_batch_idx(valid_pcs)                    # (K,) long
            global_feat = torch.zeros(B, btn_ch, device=bottleneck.device)
            count_b     = torch.zeros(B, 1, device=bottleneck.device)
            global_feat.scatter_add_(
                0,
                batch_idx.unsqueeze(1).expand(-1, btn_ch),
                frag_feat,
            )
            count_b.scatter_add_(
                0,
                batch_idx.unsqueeze(1),
                torch.ones(K, 1, device=bottleneck.device),
            )
            global_feat = global_feat / count_b.clamp(min=1.0)   # (B, btn_ch)

            # d. Broadcast global descriptor back to each (fragment, view):
            #    (K, btn_ch) → (K*V, btn_ch, H', W')
            global_per_frag = global_feat[batch_idx]              # (K, btn_ch)
            global_per_kv   = (
                global_per_frag.unsqueeze(1)                      # (K, 1, btn_ch)
                .expand(-1, V, -1)                                # (K, V, btn_ch)
                .reshape(K * V, btn_ch)                           # (K*V, btn_ch)
            )
            ctx_spatial = (
                global_per_kv
                .unsqueeze(-1).unsqueeze(-1)                      # (K*V, btn_ch, 1, 1)
                .expand(-1, -1, Hp, Wp)                           # (K*V, btn_ch, H', W')
            )

            # e. Inject via concat + 1×1 conv
            bottleneck = self.cnn.inject_context(bottleneck, ctx_spatial)

        # 6. Learned view attention scores (computed on post-context bottleneck)
        view_weights = None
        if self.use_view_attn:
            gap_ctx      = bottleneck.mean(dim=[-2, -1]).view(K, V, btn_ch)  # (K, V, btn_ch)
            view_weights = self.view_attn(gap_ctx)                            # (K, V) softmax

        # 6.5 Feature-level view fusion (Step 8)
        # Fuse bottleneck across V views → broadcast back so every decoder
        # shares the same cross-view representation before decoding.
        # Skip connections (UNet) remain view-specific — local spatial features
        # stay per-view while global semantics are shared.
        if self.feature_fusion != "none":
            btn_kv = bottleneck.view(K, V, btn_ch, Hp, Wp)  # (K, V, btn_ch, H', W')

            if self.feature_fusion == "mean":
                if view_weights is not None:
                    # Use view attention as weighted mean at feature level
                    w = view_weights.view(K, V, 1, 1, 1)
                    fused = (btn_kv * w).sum(dim=1)           # (K, btn_ch, H', W')
                else:
                    fused = btn_kv.mean(dim=1)
            elif self.feature_fusion == "max":
                fused = btn_kv.max(dim=1).values              # (K, btn_ch, H', W')
            else:  # "concat"
                fused = self.fusion_proj(
                    btn_kv.reshape(K, V * btn_ch, Hp, Wp)
                )                                             # (K, btn_ch, H', W')

            # Broadcast fused representation to all V decoder paths
            bottleneck = (
                fused.unsqueeze(1)                            # (K, 1, btn_ch, H', W')
                .expand(-1, V, -1, -1, -1)                   # (K, V, btn_ch, H', W')
                .reshape(K * V, btn_ch, Hp, Wp)              # (K*V, btn_ch, H', W')
            )
            # Fusion already captured view weighting — disable prediction-level weighting
            view_weights = None

        # 7. Decode → per-pixel logits → sigmoid → (K*V, 1, H, W)
        if _skips is not None:
            logits_flat = self.cnn.decode(bottleneck, _skips, (H, W))  # UNet
        else:
            logits_flat = self.cnn.decode(bottleneck, (H, W))          # SimpleCNN
        seg_flat     = torch.sigmoid(logits_flat).squeeze(1)       # (K*V, H, W)
        seg_per_frag = seg_flat.view(K, V, H, W)                   # (K, V, H, W)

        # 8. Bilinear + view-attention backprojection → (N_sum_valid,)
        # view_weights is None when feature_fusion != "none" (fusion already done)
        coarse_seg_pred = backproject_pixel_to_points(
            seg_per_frag, pix_corners_list, pix_weights_list, view_weights
        )

        out["coarse_seg_pred"]        = coarse_seg_pred
        out["coarse_seg_pred_binary"] = (coarse_seg_pred > 0.5)

        # 9. Extract GT for valid fragments only
        if "fracture_surface_gt" in batch:
            out["coarse_seg_gt"] = extract_gt_for_valid_frags(
                batch["fracture_surface_gt"], valid_pcs, points_per_part
            )

        return out

    # ------------------------------------------------------------------
    # Lightning steps — identical pattern to FracSeg
    # ------------------------------------------------------------------

    def training_step(self, batch):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log("train/loss", loss, on_step=True, on_epoch=False,
                 sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss

    def validation_step(self, batch):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log("val/loss", loss, on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, metrics = self.criteria(batch, out)
        self.log("test/loss", loss, on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters())
        if self._lr_scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self._lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
