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

MISE À JOUR (2026-07-31, cadrage Phase 8 "corrélation croisée explicite") --
le résultat thresh0.3 (2026-07-30) montrait un sur-apprentissage sévère que
ni le volume de données, ni un split train/val cohérent, ni l'augmentation
par rotation n'ont résolu. Hypothèse retenue : l'encodeur siamois pool
chaque depth map en un vecteur global (`AdaptiveAvgPool2d(1)`) AVANT toute
comparaison i/j -- toute l'information spatiale (où se trouve quoi) est
détruite avant même la fusion, forçant le réseau à réapprendre depuis zéro
un signal de corrélation que la recherche FFT hand-crafted (`match_depthmaps`,
Phase 5A) calcule déjà explicitement et de façon validée. Plutôt qu'une
refonte complète (volume de corrélation spatiale ou Fourier-Mellin
différentiable, plus ambitieux et plus risqué), option choisie : injecter le
profil de corrélation par angle (`angle_correlation_profile`,
`scripts/phase5a_depthmap_matching.py`) comme feature auxiliaire dans la
fusion, pour CHAQUE hypothèse (normal/miroir) séparément -- le réseau peut
alors apprendre à pondérer/affiner ce signal plutôt qu'à le redécouvrir.

MISE À JOUR (2026-07-31, "features 3D") -- `phase8_eligibility_diagnostic.py`
a isolé le vrai goulot du modèle avec profil : le miroir mal classé (~27%
des paires) donne une erreur d'angle quasi aléatoire (88.6° médian) même
quand `n_corr` reste élevé -- l'éligibilité seule masquait ce problème (des
paires à miroir faux passent quand même le seuil de correspondances par
hasard, sur des nuages denses). Hypothèse : sur une fracture quasi plane
(planéité médiane 0.043), le relief seul porte peu de signal pour trancher
la réflexion, alors que le SENS des normales dans le plan (u,v) change
directement sous une réflexion (une réflexion inverse `v`, donc `n_v`),
contrairement au relief (`depth`, invariant à une réflexion in-plane pure).
Ajout de 3 canaux de normale par carte (`n_u, n_v, n_n`, moyennés par case
comme `depth` -- `rasterize_normal_channels()`,
`scripts/phase5a_depthmap_matching.py`), `in_ch` 2→5.

**Convention miroir pour les normales (vérifiée empiriquement, corrélation
exacte 1.0 vs -1.0 sans la correction, cf. plan)** : le flip spatial des
rangées (`torch.flip(dims=[-2])`, déjà appliqué à tous les canaux) suffit
pour `n_u`/`n_n` (repositionnement pur), mais **`n_v` doit en plus être
négatée en VALEUR** -- une réflexion `v → -v` inverse aussi la composante
`v` de tout vecteur exprimé dans ce repère, pas seulement sa position."""

import torch
import torch.nn as nn


def _normalize_profile(profile: torch.Tensor) -> torch.Tensor:
    """Normalise le profil de corrélation PAR ÉCHANTILLON (même logique que
    `_normalize_depth`) -- l'échelle absolue du score (`relief * overlap_frac`)
    varie avec la taille/densité du fragment, seule la FORME du profil
    (où sont les pics) porte le signal de rotation. `profile` : (B, n_angles)."""
    mean = profile.mean(dim=1, keepdim=True)
    std = profile.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (profile - mean) / std


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
    entrée (5, R, R) = depth + validity + normale (n_u, n_v, n_n), sortie
    un vecteur (feat_dim,)."""

    def __init__(self, in_ch: int = 5, feat_dim: int = 64):
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
    """Params par défaut : ~106K (feat_dim=64, hidden=128/64, n_angles=36,
    profile_feat_dim=16, in_ch=5 avec les canaux de normale) -- cohérent
    avec la philosophie "petit modèle" du CNN de segmentation (544K)."""

    def __init__(self, feat_dim: int = 64, hidden_dim: int = 128,
                 n_angles: int = 36, profile_feat_dim: int = 16):
        super().__init__()
        self.encoder = _SiameseEncoder(in_ch=5, feat_dim=feat_dim)
        self.n_angles = n_angles
        # Encode le profil de corrélation FFT (angle_correlation_profile,
        # phase5a_depthmap_matching.py) -- le mécanisme de "corrélation
        # croisée explicite" (2026-07-31, cf. docstring module).
        self.profile_encoder = nn.Sequential(
            nn.Linear(n_angles, 32), nn.ReLU(inplace=True),
            nn.Linear(32, profile_feat_dim), nn.ReLU(inplace=True),
        )
        head_in = 4 * feat_dim + profile_feat_dim   # concat(f_i, f_j, f_i - f_j, f_i * f_j, profile_emb)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 5),   # sin, cos, shift_y, shift_x, confidence_logit
        )

    def _fuse_and_predict(self, f_i: torch.Tensor, f_j: torch.Tensor,
                           profile_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([f_i, f_j, f_i - f_j, f_i * f_j, profile_emb], dim=1)
        return self.head(fused)   # (B, 5)

    def forward(self, dmap_i: torch.Tensor, valid_i: torch.Tensor,
                dmap_j: torch.Tensor, valid_j: torch.Tensor,
                nmap_i: torch.Tensor, nmap_j: torch.Tensor,
                profile_normal: torch.Tensor, profile_mirror: torch.Tensor):
        """`dmap_*`/`valid_*` : (B, R, R) float -- `valid_i`/`valid_j` en
        {0,1} (ou probabilités). `nmap_*` : (B, 3, R, R) float -- canaux de
        normale (n_u, n_v, n_n), sortie de `rasterize_normal_channels()`
        (`scripts/phase5a_depthmap_matching.py`), REPÈRE NORMAL (pas
        mirroré -- le mirroring de `j` est géré en interne ci-dessous, comme
        pour `dmap_j`/`valid_j`). `profile_normal`/`profile_mirror` :
        (B, n_angles) float -- sortie de `angle_correlation_profile()`
        (`scripts/phase5a_depthmap_matching.py`), calculée une fois pour
        `dmap_j` tel quel et une fois pour `dmap_j` mirroré (même convention
        que le flip ci-dessous) -- PAS recalculée ici (coûteux, FFT
        numpy/scipy, pas de version torch différentiable). Retourne un dict
        avec, pour chaque hypothèse ("normal"/"mirror") : `sin`, `cos`,
        `shift_y`, `shift_x`, `confidence_logit` -- (B,) chacun -- plus
        `mirror_prob` (B,) (probabilité de l'hypothèse miroir, softmax des
        deux confidences) pour le décodage/la perte.

        Le miroir = retourner `dmap_j`/`valid_j`/`nmap_j` selon l'axe des
        RANGÉES (`dim=-2`) -- équivalent à négater `v_j` avant rasterisation
        (cf. docstring module) -- PLUS, pour `nmap_j` uniquement, négater la
        VALEUR du canal `n_v` (indice 1 de `nmap_j`) : une réflexion `v→-v`
        inverse la composante `v` de tout vecteur exprimé dans ce repère,
        pas seulement sa position (vérifié empiriquement, cf. docstring
        module -- sans cette négation la corrélation avec le vrai miroir est
        exactement -1.0 au lieu de +1.0)."""
        dmap_i = _normalize_depth(dmap_i, valid_i)
        dmap_j = _normalize_depth(dmap_j, valid_j)

        x_i = torch.cat([dmap_i.unsqueeze(1), valid_i.unsqueeze(1), nmap_i], dim=1)   # (B, 5, R, R)
        x_j_normal = torch.cat([dmap_j.unsqueeze(1), valid_j.unsqueeze(1), nmap_j], dim=1)
        x_j_mirror = torch.flip(x_j_normal, dims=[-2]).clone()   # flip des rangées = -v_j
        x_j_mirror[:, 3, :, :] = -x_j_mirror[:, 3, :, :]         # n_v (canal 3 = 1er+2 = depth,valid,n_u,n_v,n_n)

        f_i = self.encoder(x_i)
        f_j_normal = self.encoder(x_j_normal)
        f_j_mirror = self.encoder(x_j_mirror)

        prof_n = self.profile_encoder(_normalize_profile(profile_normal))
        prof_m = self.profile_encoder(_normalize_profile(profile_mirror))

        out_normal = self._fuse_and_predict(f_i, f_j_normal, prof_n)
        out_mirror = self._fuse_and_predict(f_i, f_j_mirror, prof_m)

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
