"""
assembly/models/pretraining/frac_seg.py
=========================================
PHASE 1 : Pré-entraînement — Segmentation des surfaces de fracture.

Objectif : pour chaque point d'un fragment, prédire s'il appartient à :
  - la surface de FRACTURE (label 1) : zone cassée, complémentaire avec un autre fragment
  - la surface ORIGINALE  (label 0) : surface intacte de l'objet avant cassure

Pourquoi pré-entraîner cette tâche séparément ?
  Le feature extractor (PointTransformerV3) appris ici sera gelé et réutilisé
  dans la phase 2 (flow matching). L'idée : les features qui distinguent fracture/original
  sont exactement celles dont le denoiser a besoin pour faire du matching.

Architecture :
  PointTransformerV3 (backbone)
    → features par point (N, pc_feat_dim)
    → BatchNorm
    → Linear(feat_dim → 16) → ReLU → Linear(16 → 1) → Sigmoid
    → score ∈ [0, 1] par point

Loss : Dice Loss (meilleure que BCE sur classes déséquilibrées :
       les surfaces de fracture sont souvent minoritaires)
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics
import torch_scatter

from .loss import dice_loss


class FracSeg(pl.LightningModule):
    """
    Modèle de segmentation binaire : fracture vs surface originale.

    Hérite de pl.LightningModule → gestion automatique de train/val/test
    par PyTorch Lightning.

    Args:
        pc_feat_dim    : dimension des features produites par l'encodeur (ex: 64)
        encoder        : backbone PointTransformerV3 (instancié depuis le config YAML)
        optimizer      : constructeur d'optimiseur (ex: partial(Adam, lr=1e-3))
        lr_scheduler   : scheduler de learning rate (optionnel)
        seg_warmup_epochs : nb d'epochs où on entraîne uniquement la segmentation
        grid_size      : taille de la grille pour le voxel downsampling dans PTv3
    """

    def __init__(
        self,
        pc_feat_dim: int,
        encoder: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        seg_warmup_epochs: int = 10,
        grid_size: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.pc_feat_dim = pc_feat_dim
        self.encoder = encoder          # PointTransformerV3
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.seg_warmup_epochs = seg_warmup_epochs
        self.grid_size = grid_size      # résolution spatiale du voxel grid dans PTv3

        # Normalisation des features avant la tête de classification
        self.batch_norm = nn.BatchNorm1d(self.pc_feat_dim)

        # Tête de segmentation : simple MLP 2 couches
        # Entrée : features par point (N_sum, pc_feat_dim)
        # Sortie : score scalaire par point (N_sum,) → Sigmoid → [0, 1]
        self.coarse_segmenter = nn.Sequential(
            nn.Linear(self.pc_feat_dim, 16, 1),  # (N_sum, feat_dim) → (N_sum, 16)
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, 1),                 # (N_sum, 16) → (N_sum, 1)
            nn.Flatten(0, 1),                    # (N_sum, 1) → (N_sum,) scalaire par point
        )

    def criteria(self, input_dict, output_dict):
        """
        Calcule la loss et les métriques de segmentation.

        Loss : Dice Loss
          = 1 - 2*|pred ∩ gt| / (|pred| + |gt|)
          Robuste aux classes déséquilibrées (fracture ≪ surface originale).

        Métriques calculées :
          - accuracy  : % de points correctement classifiés
          - recall    : % de vrais positifs (fractures détectées) → important pour ne pas rater
          - precision : % de prédictions positives correctes → évite les faux positifs
          - F1        : moyenne harmonique recall/precision
        """
        coarse_seg_loss = dice_loss(
            output_dict["coarse_seg_pred"],
            output_dict["coarse_seg_gt"].float(),
        )

        loss = coarse_seg_loss

        # Métriques binaires (seuil 0.5 : score > 0.5 → fracture)
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

        return loss, {
            "coarse_seg_loss": coarse_seg_loss,
            "coarse_seg_acc": coarse_seg_acc,
            "coarse_seg_recall": coarse_seg_recall,
            "coarse_seg_precision": coarse_seg_precision,
            "coarse_seg_f1": coarse_seg_f1,
        }

    def training_step(self, batch):
        """Étape d'entraînement : forward + loss, loggée à chaque step."""
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Étape de test : loggée à chaque step ET par epoch."""
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        """Étape de validation : loggée par epoch."""
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def forward(self, batch):
        """
        Forward pass de la segmentation.

        Entrée du batch :
          pointclouds        : (B, N, 3) — coordonnées xyz des points (B objets, N points max)
          pointclouds_normals: (B, N, 3) — normales de surface
          points_per_part    : (B, P)    — nb de points par fragment (0 si fragment absent = padding)

        Pipeline :
          1. Aplatit les B*P fragments en une liste plate de fragments valides
          2. Passe par PointTransformerV3 → features point (N_sum, feat_dim)
          3. Tête de classification → score par point
          4. En entraînement : utilise les labels GT pour la segmentation (teacher forcing)
             En inference : utilise les prédictions binaires

        Returns:
            dict avec coarse_seg_pred, coarse_seg_pred_binary, coarse_seg,
                      fracture_surface_points_per_part, etc.
        """
        out_dict = dict()

        pointclouds: torch.Tensor = batch["pointclouds"]          # (B, N, 3)
        normals: torch.Tensor = batch["pointclouds_normals"]       # (B, N, 3)
        points_per_part: torch.Tensor = batch["points_per_part"]   # (B, P)

        # Masque des fragments valides (non-padding)
        valid_pcs = points_per_part != 0   # (B, P) booléen

        B, N, C = pointclouds.shape   # B=batch, N=nb points total, C=3 (xyz)
        _, P = points_per_part.shape  # P=nb max de fragments

        # Offset cumulatif : indique où commence/termine chaque fragment dans la liste plate
        # Utilisé par PointTransformerV3 pour savoir quels points appartiennent ensemble
        points_per_part_offset = torch.cumsum(points_per_part, dim=-1)  # (B, P)

        with torch.no_grad():
            # Construit le graphe de validité : valid_graph[b,i,j] = True si les fragments i et j
            # sont tous les deux valides dans l'objet b (utilisé pour le matching plus tard)
            valid_graph = valid_pcs.unsqueeze(2) & valid_pcs.unsqueeze(1)  # (B, P, P)
            # Exclut la diagonale (un fragment ne se "matche" pas avec lui-même)
            valid_graph = (
                valid_graph
                & ~torch.eye(valid_graph.shape[1], device=valid_graph.device).bool()
            )
            out_dict["valid_graph"] = valid_graph
            out_dict["graph_gt"] = batch["graph"]  # vérité terrain du graphe de connectivité

        # Aplatit les points : (B, N, 3) → (B*N, 3) pour PTv3 qui attend une liste plate
        part_pcds = pointclouds.view(-1, C)     # (B*N, 3)
        part_normals = normals.view(-1, C)      # (B*N, 3)

        # Offset pour les fragments valides uniquement (PTv3 attend les fragments valides)
        points_offset = torch.cumsum(points_per_part[valid_pcs], dim=-1)  # (X,) où X = nb fragments valides

        # --- Encodeur PointTransformerV3 ---
        # Utilise float16 (mixed precision) pour économiser de la mémoire GPU
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # PTv3 prend un dict avec :
            #   coord     : coordonnées xyz des points [N_total, 3]
            #   offset    : offsets cumulatifs des fragments valides [X]
            #   feat      : features initiales = [xyz + normales] [N_total, 6]
            #   grid_size : résolution pour le voxel downsampling
            super_point, point = self.encoder(
                {
                    "coord": part_pcds,
                    "offset": points_offset,
                    "feat": torch.cat([part_pcds, part_normals], dim=-1),  # concat xyz + normales
                    "grid_size": torch.tensor(self.grid_size).to(part_pcds.device),
                }
            )
            # super_point : features des super-points (après downsampling)
            # point       : features de TOUS les points (après upsampling/interpolation)
            points_features = point["feat"]  # (N_sum_valides, pc_feat_dim)

            points_features = self.batch_norm(points_features)  # stabilise l'entraînement
            out_dict["point"] = point
            out_dict["point"]["normal"] = part_normals
            out_dict["super_point"] = super_point
            assert points_features.isnan().sum() == 0, "points_features has nan"

        # --- Tête de segmentation ---
        # Prédit un score de fracture pour chaque point
        coarse_seg_pred = self.coarse_segmenter(points_features)  # (N_sum, 1) → (N_sum,)
        coarse_seg_pred = torch.sigmoid(coarse_seg_pred)          # → [0, 1]
        coarse_seg_pred_binary = coarse_seg_pred > 0.5            # binarisation au seuil 0.5

        out_dict["coarse_seg_pred"] = coarse_seg_pred
        out_dict["coarse_seg_pred_binary"] = coarse_seg_pred_binary

        with torch.no_grad():
            # Récupère les labels GT si disponibles
            if "fracture_surface_gt" in batch:
                out_dict["coarse_seg_gt"] = batch["fracture_surface_gt"].view(-1)  # (B*N,)

            # TEACHER FORCING : pendant l'entraînement, on utilise les labels GT
            # comme segmentation (pas les prédictions) pour stabiliser l'apprentissage
            # En inference : on utilise les prédictions du modèle
            if self.training and out_dict["coarse_seg_gt"] is not None:
                out_dict["coarse_seg"] = out_dict["coarse_seg_gt"]    # GT pendant train
            else:
                out_dict["coarse_seg"] = out_dict["coarse_seg_pred_binary"]  # pred pendant val/test

            # Compte le nombre de points de fracture par fragment
            # segment_csr fait une somme groupée : pour chaque fragment, somme de ses labels
            fracture_surface_points_per_part = torch_scatter.segment_csr(
                src=out_dict["coarse_seg"].view(B, N).float(),
                indptr=F.pad(points_per_part_offset, (1, 0)),  # ajoute 0 au début comme pointeur de départ
                reduce="sum",
            ).view(B, P)  # → (B, P) : nb de points fracture par fragment
            out_dict["fracture_surface_points_per_part"] = fracture_surface_points_per_part

        return out_dict

    def configure_optimizers(self):
        """Configure l'optimiseur et optionnellement un scheduler de LR."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
