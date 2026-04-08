"""
assembly/models/hybrid_segmentation_model.py
==============================================
Hybrid fracture segmentation model — experimental variant for ablation studies.

Design overview
---------------
Baseline FracSeg pipeline:
    [xyz | normals]  (6D)  →  PTv3  →  feat (64D)  →  MLP(64→16→1)  →  sigmoid

Hybrid extension (this file):
    [xyz | normals]  (6D)  →  PTv3  →  feat (64D)  ──────────────────────────┐
                                                                              ↓
    [xyz, normals]         →  GeoExtractor  →  geo_feat (D_geo)  →  cat([feat, geo_feat])
                                                                              ↓
                                                              FusionMLP(64+D_geo → 32 → 1)
                                                                              ↓
                                                                          sigmoid

Key design decisions:
  - PTv3 receives IDENTICAL input as the baseline (6D: xyz + normals).
    This makes the encoder weights directly transferable from a baseline checkpoint.
  - Geometric features are computed AFTER encoding (post-encoder fusion).
    No changes to the PTv3 architecture at all.
  - Geometric feature computation runs without gradient (torch.no_grad inside
    HybridGeometryFeatures), so training is as fast as baseline + small MLP overhead.
  - The model follows the SAME batch interface as FracSeg, so the baseline dataloader
    and experiment configs work without modification.
  - criteria(), training_step(), validation_step() mirror FracSeg exactly so metrics
    are directly comparable.

Ablation switches (via config):
  use_normals    → include 3D normals in geo_feat
  use_curvature  → include 1D curvature scalar
  use_roughness  → include 1D roughness scalar
  geo_k          → neighborhood size for local PCA
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics
import torch_scatter

from assembly.models.pretraining.loss import dice_loss
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures


class HybridFracSeg(pl.LightningModule):
    """
    Hybrid fracture segmentation model.

    Drop-in replacement for FracSeg with an explicit geometric feature branch
    fused after the PTv3 encoder.

    Args:
        pc_feat_dim      : PTv3 output feature dimension (must match encoder config)
        encoder          : PointTransformerV3 instance (same as baseline)
        optimizer        : partial optimizer constructor
        lr_scheduler     : partial lr scheduler constructor (optional)
        seg_warmup_epochs: kept for API compatibility with FracSeg (unused here)
        grid_size        : voxel resolution for PTv3 (same as baseline)
        geo_k            : k-NN neighborhood size for geometric feature extraction
        use_normals      : include surface normals in geometric features
        use_curvature    : include curvature scalar in geometric features
        use_roughness    : include roughness scalar in geometric features
        fusion_hidden_dim: hidden dimension of the fusion MLP head
    """

    def __init__(
        self,
        pc_feat_dim: int,
        encoder: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        seg_warmup_epochs: int = 10,
        grid_size: float = 0.02,
        # ---- Geometric feature settings ----
        geo_k: int = 16,
        use_normals: bool = True,
        use_curvature: bool = True,
        use_roughness: bool = True,
        # ---- Fusion MLP settings ----
        fusion_hidden_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.pc_feat_dim = pc_feat_dim
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.seg_warmup_epochs = seg_warmup_epochs
        self.grid_size = grid_size

        # ---- Geometric feature branch (no trainable parameters) ----
        self.geo_extractor = HybridGeometryFeatures(
            k=geo_k,
            use_normals=use_normals,
            use_curvature=use_curvature,
            use_roughness=use_roughness,
        )
        geo_dim = self.geo_extractor.out_dim  # depends on which features are enabled

        # ---- Fusion head: BatchNorm → Linear → ReLU → Linear → Sigmoid ----
        # Input: concat(ptv3_feat [pc_feat_dim], geo_feat [geo_dim])
        # BatchNorm operates jointly over both feature sets before the MLP.
        fused_dim = pc_feat_dim + geo_dim
        self.seg_head = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )
        # Sigmoid applied separately (same as baseline) to keep logits accessible.

    # ------------------------------------------------------------------
    # Geometric feature computation (mirrors PTv3 offset logic)
    # ------------------------------------------------------------------

    def _compute_geo_features(
        self,
        part_pcds: torch.Tensor,
        part_normals: torch.Tensor,
        points_per_part: torch.Tensor,
        valid_pcs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geometric features for each valid fragment.

        The flat tensor ordering MUST match the ordering that PTv3 produces,
        so we use the same valid-part offset logic as FracSeg.forward().

        Args:
            part_pcds     : (B*N, 3)  flat coordinates (same as fed to PTv3)
            part_normals  : (B*N, 3)  flat normals
            points_per_part: (B, P)   points per fragment (0 = padding)
            valid_pcs     : (B, P)    boolean mask of non-padding fragments

        Returns:
            geo_feats : (N_sum_valid, geo_dim)  in the same order as PTv3 features
        """
        # Cumulative offsets for VALID parts only — mirrors PTv3's offset tensor.
        valid_counts = points_per_part[valid_pcs]              # (n_valid,)
        offsets = torch.cumsum(valid_counts, dim=0).tolist()   # list of int

        geo_feats_list = []
        prev = 0
        for end in offsets:
            end = int(end)
            pts = part_pcds[prev:end]       # (n_pts_fragment, 3)
            nrm = part_normals[prev:end]    # (n_pts_fragment, 3)
            # forward_single works on a single fragment — k-NN stays within fragment
            geo = self.geo_extractor.forward_single(pts, nrm)  # (n_pts, geo_dim)
            geo_feats_list.append(geo)
            prev = end

        return torch.cat(geo_feats_list, dim=0)     # (N_sum_valid, geo_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        Forward pass — SAME batch interface as FracSeg.forward().

        Batch keys used:
            pointclouds         : (B, N, 3)
            pointclouds_normals : (B, N, 3)
            points_per_part     : (B, P)
            fracture_surface_gt : (B, N)   — used in criteria(), not here
            graph               : (B, P, P)

        Returns dict with:
            coarse_seg_pred         : (N_sum_valid,)  fracture probability ∈ [0,1]
            coarse_seg_pred_binary  : (N_sum_valid,)  binarized at 0.5
            coarse_seg              : (N_sum_valid,)  GT during train, pred during val/test
            fracture_surface_points_per_part : (B, P)
            coarse_seg_gt           : (B*N,)  flat GT labels (if available)
            valid_graph, graph_gt   : connectivity info (unchanged)
            point, super_point      : raw PTv3 output (for downstream use)
        """
        out_dict = {}

        pointclouds: torch.Tensor = batch["pointclouds"]
        normals: torch.Tensor = batch["pointclouds_normals"]
        points_per_part: torch.Tensor = batch["points_per_part"]   # (B, P)

        valid_pcs = points_per_part != 0    # (B, P)
        B, P = points_per_part.shape

        with torch.no_grad():
            valid_graph = valid_pcs.unsqueeze(2) & valid_pcs.unsqueeze(1)
            valid_graph = valid_graph & ~torch.eye(P, device=valid_pcs.device).bool()
            out_dict["valid_graph"] = valid_graph
            out_dict["graph_gt"] = batch["graph"]

        # ------------------------------------------------------------------
        # Build flat (N_sum_valid, 3) tensors that PTv3 expects.
        #
        # Uniform sampler  → pointclouds: (B, P, N_per, 3)
        #   padding fragments have points_per_part == 0 but still occupy space
        #   in the tensor → must filter them out before passing to PTv3.
        #
        # Weighted sampler → pointclouds: (B, N_total, 3)
        #   N_total = sum of valid fragment counts; no padding points in tensor.
        # ------------------------------------------------------------------
        if pointclouds.dim() == 4:
            # Uniform: (B, P, N_per, 3) — filter out padding fragments
            _B, _P, N_per, C = pointclouds.shape
            valid_flat = valid_pcs.reshape(-1)              # (B*P,)
            part_pcds = (
                pointclouds.reshape(_B * _P, N_per, C)[valid_flat]
                .reshape(-1, C)
            )                                               # (N_sum_valid, 3)
            part_normals = (
                normals.reshape(_B * _P, N_per, C)[valid_flat]
                .reshape(-1, C)
            )
            # GT labels: (B, P, N_per) → filter valid → (N_sum_valid,)
            if "fracture_surface_gt" in batch:
                gt = batch["fracture_surface_gt"]           # (B, P, N_per)
                out_dict["coarse_seg_gt"] = (
                    gt.reshape(_B * _P, N_per)[valid_flat].reshape(-1)
                )
        else:
            # Weighted: (B, N_total, 3) — already flat, no padding points
            C = pointclouds.shape[-1]
            part_pcds = pointclouds.reshape(-1, C)          # (B*N_total, 3)
            part_normals = normals.reshape(-1, C)
            if "fracture_surface_gt" in batch:
                out_dict["coarse_seg_gt"] = batch["fracture_surface_gt"].reshape(-1)

        # Cumulative offset for PTv3 (valid parts only, same as FracSeg)
        points_offset = torch.cumsum(points_per_part[valid_pcs], dim=-1)

        # ---- Geometric features (no gradient) — computed BEFORE PTv3 ----
        # Computing here avoids overlap with PTv3 activations kept for backprop.
        geo_feat = self._compute_geo_features(
            part_pcds, part_normals, points_per_part, valid_pcs
        )                                   # (N_sum_valid, geo_dim)

        # ---- PTv3 encoder (identical call to FracSeg) ----
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            super_point, point = self.encoder(
                {
                    "coord": part_pcds,
                    "offset": points_offset,
                    "feat": torch.cat([part_pcds, part_normals], dim=-1),  # 6D
                    "grid_size": torch.tensor(self.grid_size).to(part_pcds.device),
                }
            )
            ptv3_feat = point["feat"]       # (N_sum_valid, pc_feat_dim)

        out_dict["point"] = point
        out_dict["point"]["normal"] = part_normals
        out_dict["super_point"] = super_point

        geo_feat = geo_feat.to(dtype=ptv3_feat.dtype)

        # ---- Fusion: concatenate and predict ----
        fused = torch.cat([ptv3_feat, geo_feat], dim=-1).float()  # (N_sum_valid, fused_dim)
        logits = self.seg_head(fused).squeeze(-1)                  # (N_sum_valid,)
        coarse_seg_pred = torch.sigmoid(logits)
        coarse_seg_pred_binary = coarse_seg_pred > 0.5

        out_dict["coarse_seg_pred"] = coarse_seg_pred
        out_dict["coarse_seg_pred_binary"] = coarse_seg_pred_binary

        # coarse_seg: GT during training (teacher forcing), pred during val/test
        with torch.no_grad():
            if self.training and out_dict.get("coarse_seg_gt") is not None:
                out_dict["coarse_seg"] = out_dict["coarse_seg_gt"]
            else:
                out_dict["coarse_seg"] = coarse_seg_pred_binary

        return out_dict

    # ------------------------------------------------------------------
    # Loss & metrics  (identical to FracSeg for direct comparability)
    # ------------------------------------------------------------------

    def criteria(self, input_dict: dict, output_dict: dict):
        """
        Dice loss + classification metrics.
        Identical to FracSeg.criteria() for direct numeric comparability.
        """
        coarse_seg_loss = dice_loss(
            output_dict["coarse_seg_pred"],
            output_dict["coarse_seg_gt"].float(),
        )

        coarse_seg_acc = torchmetrics.functional.accuracy(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_recall = torchmetrics.functional.recall(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_precision = torchmetrics.functional.precision(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_f1 = torchmetrics.functional.f1_score(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )

        return coarse_seg_loss, {
            "coarse_seg_loss": coarse_seg_loss,
            "coarse_seg_acc": coarse_seg_acc,
            "coarse_seg_recall": coarse_seg_recall,
            "coarse_seg_precision": coarse_seg_precision,
            "coarse_seg_f1": coarse_seg_f1,
        }

    # ------------------------------------------------------------------
    # Lightning steps (identical to FracSeg)
    # ------------------------------------------------------------------

    def training_step(self, batch):
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss

    def validation_step(self, batch):
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is None:
            return {"optimizer": optimizer}
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
