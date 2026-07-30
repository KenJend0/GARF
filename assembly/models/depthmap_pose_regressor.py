"""
assembly/models/depthmap_pose_regressor.py
=============================================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — modèle appris qui remplace
`match_depthmaps()` (scripts/phase5a_depthmap_matching.py) : au lieu d'une
recherche exhaustive (rotation discrète × 2 flips + score relief/overlap),
régresse directement `(theta, shift_y, shift_x)` à partir des deux depth
maps `(dmap_i, valid_i)`/`(dmap_j, valid_j)`.

Deux hypothèses testées en parallèle (pas une, cf. PLAN_REASSEMBLY_MODULE.md
Phase 8, section "ambiguïté de réflexion (u,v)") : le plan (u,v) de `j` tel
quel, et son miroir (`dmap_j`/`valid_j` retournés selon l'axe des rangées,
équivalent à négater l'axe `v_j`). Mesuré sur 1251 paires GT réelles
(`scripts/phase8_reflection_prevalence_check.py`) : 52.2% des paires ont
besoin du miroir -- pas un cas rare, un deuxième bit d'ambiguïté
indépendant de celui déjà géré par `match_depthmaps` (`best_flip`, qui ne
concerne que le signe de la profondeur, jamais le plan). Chaque hypothèse
produit sa propre pose candidate + une confiance ; on garde celle dont la
confiance est la plus haute -- même principe que `best_flip`, appliqué à
une ambiguïté différente.

`sin`/`cos` pour l'angle (pas `theta` brut) : évite la discontinuité de
l'enroulement d'angle à 360°/0°, décodage par `atan2(sin, cos)`.
"""

import torch
import torch.nn as nn


def _normalize_depth(dmap: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Normalise chaque depth map PAR ÉCHANTILLON (pas globalement) en
    divisant par l'écart-type des profondeurs valides -- correctif
    2026-07-30, suite à un premier entraînement infructueux
    (angle_err/mirror_acc au niveau du hasard sur validation). Les
    profondeurs sont en unités PHYSIQUES absolues (mètres), et l'échelle
    varie énormément d'un fragment à l'autre (taille réelle différente) --
    sans cette normalisation, le réseau voit des plages de valeurs
    incohérentes d'un exemple à l'autre pour un même type de tâche
    géométrique. `dmap`/`valid` : (B, R, R)."""
    n_valid = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    mean = (dmap * valid).sum(dim=(1, 2), keepdim=True) / n_valid
    var = ((dmap - mean) ** 2 * valid).sum(dim=(1, 2), keepdim=True) / n_valid
    std = var.clamp_min(1e-8).sqrt()
    return (dmap - mean) / std * valid


class _SiameseEncoder(nn.Module):
    """Encodeur CNN partagé (poids communs pour i, j-normal, j-miroir) --
    entrée (2, R, R) = depth + validity, sortie un vecteur (feat_dim,)."""

    def __init__(self, in_ch: int = 2, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.GroupNorm(8, feat_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)   # (B, feat_dim)


class DepthmapPoseRegressor(nn.Module):
    """Params par défaut : ~90K (feat_dim=64, hidden=128/64) -- cohérent
    avec la philosophie "petit modèle" du CNN de segmentation (544K)."""

    def __init__(self, feat_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = _SiameseEncoder(in_ch=2, feat_dim=feat_dim)
        head_in = 4 * feat_dim   # concat(f_i, f_j, f_i - f_j, f_i * f_j)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 5),   # sin, cos, shift_y, shift_x, confidence_logit
        )

    def _fuse_and_predict(self, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([f_i, f_j, f_i - f_j, f_i * f_j], dim=1)
        return self.head(fused)   # (B, 5)

    def forward(self, dmap_i: torch.Tensor, valid_i: torch.Tensor,
                dmap_j: torch.Tensor, valid_j: torch.Tensor):
        """`dmap_*`/`valid_*` : (B, R, R) float -- `valid_i`/`valid_j` en
        {0,1} (ou probabilités). Retourne un dict avec, pour chaque
        hypothèse ("normal"/"mirror") : `sin`, `cos`, `shift_y`, `shift_x`,
        `confidence_logit` -- (B,) chacun -- plus `mirror_prob` (B,)
        (probabilité de l'hypothèse miroir, softmax des deux confidences)
        pour le décodage/la perte.

        Le miroir = retourner `dmap_j`/`valid_j` selon l'axe des RANGÉES
        (`dim=-2`) -- équivalent à négater `v_j` avant rasterisation (cf.
        docstring module et PLAN_REASSEMBLY_MODULE.md, Phase 8)."""
        dmap_i = _normalize_depth(dmap_i, valid_i)
        dmap_j = _normalize_depth(dmap_j, valid_j)

        x_i = torch.stack([dmap_i, valid_i], dim=1)          # (B, 2, R, R)
        x_j_normal = torch.stack([dmap_j, valid_j], dim=1)
        x_j_mirror = torch.flip(x_j_normal, dims=[-2])        # flip des rangées = -v_j

        f_i = self.encoder(x_i)
        f_j_normal = self.encoder(x_j_normal)
        f_j_mirror = self.encoder(x_j_mirror)

        out_normal = self._fuse_and_predict(f_i, f_j_normal)
        out_mirror = self._fuse_and_predict(f_i, f_j_mirror)

        conf = torch.stack([out_normal[:, 4], out_mirror[:, 4]], dim=1)   # (B, 2)
        mirror_prob = torch.softmax(conf, dim=1)[:, 1]                   # (B,)

        def _unpack(out):
            norm = torch.linalg.vector_norm(out[:, :2], dim=1, keepdim=True).clamp_min(1e-8)
            sin_cos = out[:, :2] / norm
            return {
                "sin": sin_cos[:, 0], "cos": sin_cos[:, 1],
                "shift_y": out[:, 2], "shift_x": out[:, 3],
                "confidence_logit": out[:, 4],
            }

        return {
            "normal": _unpack(out_normal),
            "mirror": _unpack(out_mirror),
            "mirror_prob": mirror_prob,
        }

    @staticmethod
    def decode(pred: dict):
        """Choisit l'hypothèse la plus confiante et retourne
        `(theta_deg, shift_y, shift_x, mirror_bool)` par élément du batch
        (numpy-friendly, pour l'intégration avec
        `build_correspondences_nn`/`kabsch` -- cf.
        scripts/phase8_pipeline_learned_check.py)."""
        mirror = pred["mirror_prob"] > 0.5
        chosen = {
            k: torch.where(mirror, pred["mirror"][k], pred["normal"][k])
            for k in ("sin", "cos", "shift_y", "shift_x")
        }
        theta_deg = torch.rad2deg(torch.atan2(chosen["sin"], chosen["cos"])) % 360.0
        return theta_deg, chosen["shift_y"], chosen["shift_x"], mirror


def pose_regressor_loss(pred: dict, theta_gt_deg: torch.Tensor, shift_y_gt: torch.Tensor,
                         shift_x_gt: torch.Tensor, mirror_gt: torch.Tensor,
                         w_angle: float = 1.0, w_shift: float = 0.01, w_conf: float = 1.0):
    """`theta_gt_deg`/`shift_*_gt`/`mirror_gt` : (B,) -- labels de
    `fit_theta_shift_mirror_from_gt` (scripts/phase8_depthmap_regressor_dataset.py).
    La régression angle/shift n'est appliquée QU'À l'hypothèse correcte
    (`mirror_gt`) -- l'autre hypothèse n'a pas de cible de régression
    valide (aucune rotation pure ne peut la représenter par construction,
    cf. PLAN_REASSEMBLY_MODULE.md), seule sa confiance est entraînée
    (doit être basse)."""
    theta_gt_rad = torch.deg2rad(theta_gt_deg)
    sin_gt, cos_gt = torch.sin(theta_gt_rad), torch.cos(theta_gt_rad)

    def _angle_shift_loss(branch):
        angle_loss = 1.0 - (branch["sin"] * sin_gt + branch["cos"] * cos_gt)
        shift_loss = (branch["shift_y"] - shift_y_gt) ** 2 + (branch["shift_x"] - shift_x_gt) ** 2
        return angle_loss + w_shift * shift_loss

    loss_normal = _angle_shift_loss(pred["normal"])
    loss_mirror = _angle_shift_loss(pred["mirror"])
    regression_loss = torch.where(mirror_gt, loss_mirror, loss_normal).mean()

    conf = torch.stack([pred["normal"]["confidence_logit"], pred["mirror"]["confidence_logit"]], dim=1)
    conf_loss = nn.functional.cross_entropy(conf, mirror_gt.long())

    total = w_angle * regression_loss + w_conf * conf_loss
    return total, {"regression_loss": regression_loss.detach(), "conf_loss": conf_loss.detach()}
