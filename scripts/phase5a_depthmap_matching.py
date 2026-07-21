"""
scripts/phase5a_depthmap_matching.py
======================================
Phase 5A du plan de réassemblage (voir PLAN_REASSEMBLY_MODULE.md, section "Phase 5A").

Teste si les faces de fracture détectées par le CNN peuvent être représentées en
depth maps 2D locales et matchées par complémentarité de relief ("bosse contre creux").

Restreint aux objets à exactement 2 fragments (pré-check 5A.0 : 48% du val,
médiane planéité=0.043 — voir plan). Supprime le confond multi-voisins de la
Phase 2D : avec 2 fragments, chaque point fracture n'a qu'un seul voisin possible.

Pipeline par paire (i=fragment0, j=fragment1) :
  1. Masque fracture (GT / CNN thresh0.3 / random contrôle)
  2. Filtre : >= MIN_FRAC_POINTS points ET planéité PCA <= MAX_PLANARITY
  3. PCA sur les points fracture → repère local (u, v, n) + centroïde
  4. Rasterization en depth map 2D (grille RESOLUTION×RESOLUTION, taille physique commune)
  5. Sweep rotation [0°, 360°, pas 360/N_ANGLES] × FFT translation (complémentarité)
     Score (--score_mode, défaut "joint") = relief_score * overlap_frac — voir
     match_depthmaps() pour le détail. "relief" reproduit la formule d'origine
     (bug identifié le 2026-07-16 : garde-fou overlap>0.5 PIXEL, quasi inexistant,
     laisse le score exploser à faible recouvrement) ; "overlap_only" teste
     isolément si la forme du contour de la fracture suffit sans aucun relief.
     Testé aussi avec flip de la normale de j (ambiguïté de signe PCA)
  6. Reconstruction 3D des correspondances depuis le meilleur alignement 2D
  7. Kabsch sur les correspondances → R_est, t_est
  8. Évaluation : RotErr, TransErr, Pose@30°/0.1, Pose@15°/0.05

Trois conditions (même logique que toutes les phases précédentes) :
  - gt       : masque fracture_surface_gt (oracle mask, isole le matcher)
  - thresh0.3: masque CNN prédit (condition réelle)
  - random   : budget identique à gt, points tirés aléatoirement sur tout le fragment

Métriques comparées aux bornes connues :
  - Phase 2 global (RANSAC + descripteurs faits main) : Pose@30 ≈ 1.3-3.2%
  - Phase 2 gt_edge oracle                           : Pose@30 ≈ 9.6%

Usage (sur le serveur) :
    # Test rapide (quelques dizaines d'objets)
    python scripts/phase5a_depthmap_matching.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 50 \\
        --summary_json /tmp/student7/phase5a_quick.json

    # Run complet
    python scripts/phase5a_depthmap_matching.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val \\
        --summary_json /tmp/student7/phase5a_val.json

    # Comparer les 3 formules de score (2026-07-20, suite à la remise en cause de la
    # formule d'origine) : relancer avec --score_mode relief / overlap_only / joint
    # sur le MEME quick run pour comparer directement.
    # --batch_size 1 : le forward CNN (features géométriques, KNN O(N^2) par fragment)
    # peut OOM sur les GPU 8 Go du labo si un batch contient des fragments à beaucoup
    # de points -- observé le 2026-07-20 avec batch_size=4 par défaut (jusqu'à 8
    # fragments traités d'un coup). batch_size=1 limite à 2 fragments par forward.
    for MODE in relief overlap_only joint; do
        CUDA_VISIBLE_DEVICES=1 python scripts/phase5a_depthmap_matching.py \\
            --ckpt output/cnn_step15_final_model/last.ckpt \\
            --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
            --experiment cnn_step15_final_model \\
            --categories everyday --split val --max_batches 50 --batch_size 1 \\
            --score_mode $MODE \\
            --summary_json /tmp/student7/phase5a_${MODE}.json
    done
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate as ndimage_rotate, binary_dilation, gaussian_filter
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate
from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list
from torch.utils.data import DataLoader


# ── Hyperparamètres (overridables via CLI) ────────────────────────────────────
DEFAULT_RESOLUTION     = 64      # taille de la depth map (RESOLUTION × RESOLUTION px)
DEFAULT_N_ANGLES       = 36      # sweep rotation : 360/36 = 10° par pas
MIN_FRAC_POINTS        = 50      # moins de N pts fracture → skip (PCA bruitée)
DEFAULT_MAX_PLANARITY  = 0.15    # planéité PCA > seuil → face trop courbe → skip
                                  # (overridable via --max_planarity ; 0.333 = isotrope,
                                  # borne théorique haute — cf. discussion 2026-07-20 :
                                  # augmenter le seuil teste les faces jusqu'ici exclues,
                                  # au lieu de seulement stratifier ce qui est déjà gardé)
MIN_OVERLAP_PIXELS     = 20      # correspondances 3D < N → skip Kabsch
POSE_SUCCESS_THRESH    = [(30.0, 0.1), (15.0, 0.05)]

# Seuil de masse (après flou gaussien du canal "count") au-delà duquel une case
# est considérée "valid" en mode splat gaussien (piste 1, tentative 2 : après le
# rejet de la dilatation binaire seule le 2026-07-20, cf. plan — la dilatation
# gonfle le recouvrement de `random` presque autant que `gt`). Contrairement à
# la dilatation, le splat gaussien pondère par la position réelle des points
# plutôt que d'étendre aveuglément un masque déjà là. Valeur empirique, pas
# encore calée finement — à ajuster si besoin après le premier sweep.
GAUSSIAN_VALID_THRESH  = 0.2

# Tranches de planéité pour la stratification post-hoc (bornes alignées sur les
# percentiles du pré-check Phase 5A.0 : p25=0.016, p50=0.043, p75=0.098, p90=0.150 ;
# étendues jusqu'à 0.333 = isotrope, indépendamment du seuil --max_planarity utilisé,
# pour rester comparables d'un run à l'autre même si le seuil change). Sert à
# vérifier si "moins plat" corrèle vraiment avec un meilleur matching, plutôt que de
# le supposer en excluant simplement les faces les plus courbées.
PLANARITY_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.334]
PLANARITY_LABELS = ["<0.02", "0.02-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20+"]

# Tranches de nombre de points de fracture (moyenne des 2 fragments) pour la
# stratification post-hoc (bornes alignées sur les percentiles du pré-check
# Phase 5A.0 : p25=56, p50=340, p75=1805, p90=3052). Teste l'hypothèse de
# sparsité derrière le plafond OracleOvlp=0.758 (2026-07-20) : si le plafond
# est nettement plus bas sur les paires éparses que sur les paires denses,
# la sparsité d'échantillonnage explique une vraie part du plafond, et une
# densification (splat gaussien) a de bonnes chances d'aider spécifiquement
# ces cas. Si le plafond est stable quel que soit le nombre de points, le
# problème est ailleurs (asymétrie géométrique entre les deux masques).
N_FRAC_PTS_BINS = [0, 100, 300, 1000, 3000, 10**9]
N_FRAC_PTS_LABELS = ["<100", "100-300", "300-1000", "1000-3000", "3000+"]

# Tranches de MIN(n_i, n_j) — le côté le plus pauvre de la paire, pas la
# moyenne. Ajouté le 2026-07-20 après inspection visuelle de l'objet#21
# (script phase5a_visualize_pair.py) : un fragment "grand" (le reste de
# l'objet) échantillonne très peu de points sur sa fracture (crack = petite
# fraction de sa surface totale) tandis qu'un fragment "petit" (le bout
# cassé) en échantillonne beaucoup (crack = quasi toute sa surface), même
# budget de points total par fragment (5000 chacun sur l'exemple observé).
# La MOYENNE cache ce déséquilibre (131 et 2832 pts → moyenne ~1481, tranche
# "dense" de N_FRAC_PTS_BINS, alors que la paire échoue à cause du côté à
# 131 pts). Le MIN teste directement l'hypothèse : c'est le maillon faible
# qui limite le matching, pas la moyenne des deux côtés.
N_FRAC_PTS_MIN_BINS = [50, 100, 200, 500, 1000, 10**9]
N_FRAC_PTS_MIN_LABELS = ["50-100", "100-200", "200-500", "500-1000", "1000+"]


# ── Géométrie ─────────────────────────────────────────────────────────────────

def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    return R_scipy.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t tels que R @ p + t ≈ q pour chaque paire (p, q) de P, Q."""
    pm, qm = P.mean(0), Q.mean(0)
    Pc, Qc = P - pm, Q - qm
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rmat = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rmat, qm - Rmat @ pm


def rot_err_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos_a = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def trans_err(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est - t_gt))


# ── Depth map ─────────────────────────────────────────────────────────────────

def compute_pca_frame(pts: np.ndarray):
    """PCA des points fracture → (centroid, u, v, n, planarity).
    n = axe de plus petite variance (normale à la face quasi-plane).
    u = axe de plus grande variance (direction principale in-plane).
    planarity = lambda_min / sum(lambdas) ; proche de 0 → quasi-plan."""
    centroid = pts.mean(0)
    centered = pts - centroid
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # trié croissant
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    planarity = float(eigenvalues[0] / total) if total > 1e-12 else 1.0
    n = eigenvectors[:, 0]   # axe de normale (plus petite variance)
    v = eigenvectors[:, 1]   # axe in-plane secondaire
    u = eigenvectors[:, 2]   # axe in-plane principal
    return centroid, u, v, n, planarity


def points_to_valid_mask(u_coords, v_coords, u_min, v_min, resolution, pixel_size,
                          dilate_px: int = 0, gaussian_sigma_px: float = 0.0):
    """Points (u,v) déjà projetés → masque `valid` (case couverte). Partagée
    entre `rasterize()` et `oracle_overlap_frac()` pour traiter dilatation et
    splat gaussien de façon strictement cohérente aux deux endroits.

    `dilate_px` : dilatation binaire (piste 1, tentative 1 — REJETÉE le
    2026-07-20 : gonfle `random` presque autant que `gt`, `Pose@30` n'en
    profite pas). Conservée pour comparaison, pas recommandée par défaut.

    `gaussian_sigma_px` : splat gaussien (piste 1, tentative 2) — au lieu
    d'étendre aveuglément le masque, floute le canal "nombre de points par
    case" avec un noyau gaussien, pondérant chaque case voisine par la
    distance réelle aux points plutôt que de l'inclure en tout-ou-rien.
    Une case devient `valid` si la masse gaussienne accumulée dépasse
    `GAUSSIAN_VALID_THRESH`.
    """
    u_pix = np.clip(((u_coords - u_min) / pixel_size).astype(int), 0, resolution - 1)
    v_pix = np.clip(((v_coords - v_min) / pixel_size).astype(int), 0, resolution - 1)

    count = np.zeros((resolution, resolution), dtype=np.float64)
    np.add.at(count, (v_pix, u_pix), 1.0)

    if gaussian_sigma_px > 0:
        count_smooth = gaussian_filter(count, sigma=gaussian_sigma_px, mode="constant")
        valid = count_smooth > GAUSSIAN_VALID_THRESH
    else:
        valid = count > 0

    if dilate_px > 0:
        struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        valid = binary_dilation(valid, structure=struct)

    return valid, u_pix, v_pix


def rasterize(pts: np.ndarray, centroid, u, v, n, resolution: int, pixel_size: float,
              dilate_px: int = 0, gaussian_sigma_px: float = 0.0):
    """Projette les points fracture en depth map 2D.
    depth = composante selon n depuis le centroïde.
    Retourne (dmap [R,R], valid [R,R bool], u_min, v_min).

    `dilate_px` (piste 1 Phase 7, 2026-07-20, tentative 1 — REJETÉE) : dilate
    le masque `valid` de `dilate_px` pixels après rasterisation. Fait
    remonter `OracleOvlp` (0.757→0.992 à d=1 sur GT) mais `random` en
    profite presque autant (0.430→0.830) et `Pose@30` n'en profite PAS
    (18.28% vs 20.87% sans dilatation, mesuré le 2026-07-20) — un pansement
    sur le bruit d'échantillonnage, pas une vraie correction de
    correspondance. Conservée pour comparaison, désactivée par défaut.

    `gaussian_sigma_px` (piste 1, tentative 2, 2026-07-20) : au lieu
    d'étendre aveuglément le masque, floute `dmap` (canal profondeur, somme
    pondérée) ET le canal "nombre de points" avec un noyau gaussien de même
    sigma (flou correct d'une moyenne pondérée : flouter numérateur et
    dénominateur séparément, puis diviser — pas flouter la moyenne déjà
    calculée). Utilise la position réelle des points plutôt qu'un
    remplissage tout-ou-rien.
    """
    pts_c = pts - centroid
    u_coords = pts_c @ u
    v_coords = pts_c @ v
    depths   = pts_c @ n

    u_min = u_coords.min() - 0.5 * pixel_size
    v_min = v_coords.min() - 0.5 * pixel_size

    u_pix = np.clip(((u_coords - u_min) / pixel_size).astype(int), 0, resolution - 1)
    v_pix = np.clip(((v_coords - v_min) / pixel_size).astype(int), 0, resolution - 1)

    dmap_sum = np.zeros((resolution, resolution), dtype=np.float64)
    count    = np.zeros((resolution, resolution), dtype=np.float64)
    np.add.at(dmap_sum, (v_pix, u_pix), depths)
    np.add.at(count,    (v_pix, u_pix), 1.0)

    if gaussian_sigma_px > 0:
        dmap_sum = gaussian_filter(dmap_sum, sigma=gaussian_sigma_px, mode="constant")
        count    = gaussian_filter(count,    sigma=gaussian_sigma_px, mode="constant")
        valid = count > GAUSSIAN_VALID_THRESH
    else:
        valid = count > 0

    dmap = np.zeros((resolution, resolution), dtype=np.float64)
    dmap[valid] = dmap_sum[valid] / count[valid]

    if dilate_px > 0:
        struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        valid = binary_dilation(valid, structure=struct)

    return dmap, valid, u_min, v_min


# ── Matching ──────────────────────────────────────────────────────────────────

def match_depthmaps(dmap_i, valid_i, dmap_j, valid_j, n_angles: int, score_mode: str = "joint"):
    """Sweep rotation + FFT translation, score = complémentarité + qualité de recouvrement.

    Pour chaque rotation θ de dmap_j (et chaque flip de normale), on calcule pour
    TOUS les décalages simultanément (FFT) :
      - CC[sy,sx]      : corrélation croisée des profondeurs (complémentarité de relief)
      - overlap[sy,sx] : nombre de pixels valides communs à ce décalage précis
      - overlap_frac[sy,sx] = overlap / min(n_valid_i, n_valid_j_rot) — fraction du
        contour de la face qui coïncide à ce décalage (0=aucun recouvrement, 1=un
        contour totalement inclus dans l'autre). C'est le signal de FORME du contour
        de la zone de fracture, indépendant du relief.

    score_mode :
      - "relief"       : formule originale, -CC/overlap avec un garde-fou quasi inexistant
                         (overlap > 0.5 PIXEL) — reproduit le bug identifié le 2026-07-16 :
                         à faible recouvrement, diviser par un dénominateur minuscule peut
                         gonfler artificiellement le score. Conservé pour comparaison.
      - "overlap_only" : ignore complètement le relief, score = overlap_frac seul — teste
                         isolément l'hypothèse "le contour de la fracture suffit déjà à
                         fixer la rotation, même sans bosses/creux" (pertinent surtout à
                         2 fragments, cf. discussion du 2026-07-20).
      - "joint" (défaut, LE FIX) : score = relief_score * overlap_frac. Un bon score
                         de relief à un recouvrement quasi nul est automatiquement ramené
                         vers 0 (au lieu d'exploser par division), et un bon recouvrement
                         sans complémentarité de relief ne suffit pas non plus à gagner.
                         Aucun seuil arbitraire à caler : la pénalité est continue.

    Retourne (best_score, best_theta_deg, best_shift_yx_pixels, best_flip, best_overlap_frac).
    best_shift est unwrappé (peut être négatif).
    """
    R_sz = dmap_i.shape[0]
    a = dmap_i * valid_i.astype(np.float64)
    c = valid_i.astype(np.float64)
    A_fft = np.fft.rfft2(a)
    C_fft = np.fft.rfft2(c)
    n_valid_i = float(valid_i.sum())

    best_score = -np.inf
    best_theta = 0.0
    best_shift = (0, 0)
    best_flip  = False
    best_overlap_frac = 0.0

    for flip in (False, True):
        dmap_j_use = -dmap_j if flip else dmap_j

        for k in range(n_angles):
            theta = k * 360.0 / n_angles

            dmap_j_rot  = ndimage_rotate(dmap_j_use, theta, reshape=False, order=1, cval=0.0)
            valid_j_rot = ndimage_rotate(
                valid_j.astype(np.float64), theta, reshape=False, order=1, cval=0.0) > 0.5

            b = dmap_j_rot * valid_j_rot.astype(np.float64)
            d = valid_j_rot.astype(np.float64)
            B_fft = np.fft.rfft2(b)
            D_fft = np.fft.rfft2(d)

            # CC[sy,sx] = sum_{r,c} a[r,c] * b[r+sy, c+sx]  (circulaire)
            # Complémentarité : depth_i ≈ -depth_j_rot → CC doit être le plus NÉGATIF.
            CC      = np.real(np.fft.irfft2(A_fft * np.conj(B_fft), s=(R_sz, R_sz)))
            overlap = np.real(np.fft.irfft2(C_fft * np.conj(D_fft), s=(R_sz, R_sz)))

            n_valid_j_rot = float(valid_j_rot.sum())
            denom = max(min(n_valid_i, n_valid_j_rot), 1.0)
            overlap_frac = overlap / denom   # fraction du contour qui coïncide, dans [0,1]

            with np.errstate(divide='ignore', invalid='ignore'):
                if score_mode == "relief":
                    score_map = np.where(overlap > 0.5, -CC / overlap, -np.inf)
                elif score_mode == "overlap_only":
                    score_map = overlap_frac
                else:  # "joint" — le fix : pénalise en continu le faible recouvrement
                    relief_score = np.where(overlap > 0.5, -CC / overlap, 0.0)
                    score_map = relief_score * overlap_frac

            idx = np.argmax(score_map)
            sy, sx = np.unravel_index(idx, score_map.shape)
            score = float(score_map[sy, sx])

            if score > best_score:
                best_score = score
                best_theta = theta
                sy_u = sy if sy < R_sz // 2 else sy - R_sz
                sx_u = sx if sx < R_sz // 2 else sx - R_sz
                best_shift = (sy_u, sx_u)
                best_flip  = flip
                best_overlap_frac = float(overlap_frac[sy, sx])

    return best_score, best_theta, best_shift, best_flip, best_overlap_frac


def build_correspondences(
    dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
    dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
    pixel_size, best_theta, best_shift, best_flip,
):
    """Construit des paires de points 3D à partir de l'alignement depth-map.

    Pixel (r, c) de i → pixel (r+sy, c+sx) dans j_rotated → pixel dans j_original
    par rotation inverse -theta autour du centre de l'image.
    Retourne (pts_i_3d, pts_j_3d) ou (None, None) si overlap insuffisant.
    """
    R_sz = dmap_i.shape[0]
    sy, sx = best_shift
    theta_rad = best_theta * np.pi / 180.0
    cos_neg = np.cos(-theta_rad)
    sin_neg = np.sin(-theta_rad)
    half = R_sz / 2.0

    rows_i, cols_i = np.where(valid_i)
    if len(rows_i) == 0:
        return None, None

    # Position dans j_rotated (shift uniquement, pas encore la rotation inverse)
    rows_jr = rows_i + sy
    cols_jr = cols_i + sx

    # Rotation inverse autour du centre pour retrouver les coords dans j_original
    cx = cols_jr - half
    cy = rows_jr - half
    cols_j_orig = half + cos_neg * cx - sin_neg * cy
    rows_j_orig = half + sin_neg * cx + cos_neg * cy

    cj_idx = np.round(cols_j_orig).astype(int)
    rj_idx = np.round(rows_j_orig).astype(int)

    in_bounds = (cj_idx >= 0) & (cj_idx < R_sz) & (rj_idx >= 0) & (rj_idx < R_sz)
    ri = rows_i[in_bounds]; ci_arr = cols_i[in_bounds]
    rj = rj_idx[in_bounds]; cj_arr = cj_idx[in_bounds]

    has_data = valid_j[rj, cj_arr]
    ri = ri[has_data]; ci_arr = ci_arr[has_data]
    rj = rj[has_data]; cj_arr = cj_arr[has_data]

    if len(ri) < MIN_OVERLAP_PIXELS:
        return None, None

    # Coordonnées 3D de i
    uc_i = u_min_i + (ci_arr + 0.5) * pixel_size
    vc_i = v_min_i + (ri    + 0.5) * pixel_size
    di   = dmap_i[ri, ci_arr]
    pts_i = c_i + uc_i[:,None]*u_i + vc_i[:,None]*v_i + di[:,None]*n_i

    # Coordonnées 3D de j
    uc_j = u_min_j + (cj_arr + 0.5) * pixel_size
    vc_j = v_min_j + (rj     + 0.5) * pixel_size
    dj   = dmap_j[rj, cj_arr]
    if best_flip:
        dj = -dj
    pts_j = c_j + uc_j[:,None]*u_j + vc_j[:,None]*v_j + dj[:,None]*n_j

    return pts_i, pts_j


# ── Boucle principale ─────────────────────────────────────────────────────────

def planarity_stratification(planarity, rot_err, pose30, pose15):
    """Découpe les paires par tranche de planéité (moyenne des 2 fragments) et
    calcule RotErr moyen / Pose@30 / Pose@15 / N par tranche.

    Répond à la question : parmi les paires gardées (planéité <= MAX_PLANARITY),
    est-ce que "moins plat" corrèle vraiment avec un meilleur matching ? Si oui,
    ça confirme la thèse "trop plat = pas de signal". Si non (plat ou pas, même
    résultat), la thèse doit être nuancée — le vrai facteur limitant serait
    ailleurs (ex. le nombre de points, le bruit du masque, autre chose).
    Retourne une liste de dicts, un par tranche (vide si aucune paire dedans).
    """
    planarity = np.asarray(planarity)
    rot_err   = np.asarray(rot_err)
    pose30    = np.asarray(pose30)
    pose15    = np.asarray(pose15)
    idx = np.digitize(planarity, PLANARITY_BINS[1:-1])   # 0..len(labels)-1

    rows = []
    for b, label in enumerate(PLANARITY_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({
            "bin": label,
            "n": n,
            "rot_err_mean": float(rot_err[mask].mean()),
            "pose_30deg_0.1": 100.0 * float(pose30[mask].mean()),
            "pose_15deg_0.05": 100.0 * float(pose15[mask].mean()),
        })
    return rows


def frac_pts_stratification(n_frac_pts, oracle_ovlp, rot_err, pose30, pose15):
    """Découpe les paires par tranche de nombre de points de fracture (moyenne
    des 2 fragments) et calcule OracleOvlp / RotErr / Pose@30 / Pose@15 par tranche.

    Teste l'hypothèse de sparsité derrière le plafond OracleOvlp=0.758
    (2026-07-20, cf. PLAN_REASSEMBLY_MODULE.md) : si OracleOvlp est nettement
    plus bas sur les paires éparses (peu de points) et proche de 1.0 sur les
    paires denses, la sparsité d'échantillonnage explique le plafond — une
    densification (splat gaussien) devrait aider spécifiquement les cas
    épars. Si OracleOvlp est stable quel que soit le nombre de points, la
    sparsité n'est pas la vraie cause, il faut chercher ailleurs.
    Retourne une liste de dicts, un par tranche (vide si aucune paire dedans).
    """
    n_frac_pts  = np.asarray(n_frac_pts)
    oracle_ovlp = np.asarray(oracle_ovlp)
    rot_err     = np.asarray(rot_err)
    pose30      = np.asarray(pose30)
    pose15      = np.asarray(pose15)
    idx = np.digitize(n_frac_pts, N_FRAC_PTS_BINS[1:-1])   # 0..len(labels)-1

    rows = []
    for b, label in enumerate(N_FRAC_PTS_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({
            "bin": label,
            "n": n,
            "oracle_overlap_frac_mean": float(oracle_ovlp[mask].mean()),
            "rot_err_mean": float(rot_err[mask].mean()),
            "pose_30deg_0.1": 100.0 * float(pose30[mask].mean()),
            "pose_15deg_0.05": 100.0 * float(pose15[mask].mean()),
        })
    return rows


def frac_pts_min_stratification(n_frac_pts_min, oracle_ovlp, rot_err, pose30, pose15):
    """Comme `frac_pts_stratification`, mais sur MIN(n_i, n_j) au lieu de la
    moyenne des deux fragments.

    Ajouté le 2026-07-20 après inspection visuelle de l'objet#21
    (`phase5a_visualize_pair.py`) : un fragment "grand" (le reste de l'objet)
    échantillonne très peu de points sur sa fracture (crack = petite fraction
    de sa surface totale) tandis qu'un fragment "petit" (le bout cassé) en
    échantillonne beaucoup — même budget total de points par fragment. La
    MOYENNE cache ce déséquilibre (ex. 131 et 2832 pts → moyenne ~1481,
    tranche "dense", alors que la paire échoue à cause du côté à 131 pts).
    Teste directement l'hypothèse du maillon faible : c'est le côté le plus
    pauvre qui limite le matching, pas la moyenne des deux côtés.
    Retourne une liste de dicts, un par tranche (vide si aucune paire dedans).
    """
    n_frac_pts_min = np.asarray(n_frac_pts_min)
    oracle_ovlp    = np.asarray(oracle_ovlp)
    rot_err        = np.asarray(rot_err)
    pose30         = np.asarray(pose30)
    pose15         = np.asarray(pose15)
    idx = np.digitize(n_frac_pts_min, N_FRAC_PTS_MIN_BINS[1:-1])   # 0..len(labels)-1

    rows = []
    for b, label in enumerate(N_FRAC_PTS_MIN_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({
            "bin": label,
            "n": n,
            "oracle_overlap_frac_mean": float(oracle_ovlp[mask].mean()),
            "rot_err_mean": float(rot_err[mask].mean()),
            "pose_30deg_0.1": 100.0 * float(pose30[mask].mean()),
            "pose_15deg_0.05": 100.0 * float(pose15[mask].mean()),
        })
    return rows


def oracle_overlap_frac(frac_i, frac_j, c_j, u_j, v_j, u_min_j, v_min_j,
                         resolution, pixel_size, R_ij_gt, t_ij_gt,
                         dilate_px=0, gaussian_sigma_px=0.0):
    """Recouvrement des footprints à la VRAIE pose GT — aucune recherche, aucun score.

    Transforme les points fracture de i dans le repère de j via la vraie pose
    (R_ij_gt, t_ij_gt), les projette sur le plan (u_j, v_j) de j, et mesure la
    fraction de cases qui coïncident avec le footprint réel de j (reconstruit
    à partir de `frac_j`, PAS d'un `valid_j` déjà calculé ailleurs — pour que
    i et j soient traités par le MÊME `points_to_valid_mask()`, avec la même
    dilatation/splat, une comparaison symétrique et cohérente).

    Plafond théorique de `overlap_frac` (cf. discussion 2026-07-20) : les deux
    faces d'une même fracture partagent le même contour par construction, donc
    à la vraie pose l'overlap DEVRAIT tendre vers 1. Si le meilleur overlap
    trouvé par la recherche (`match_depthmaps`) est loin de ce plafond, le
    problème est la recherche (pose/score). Si le plafond lui-même est déjà
    loin de 1, le problème est en amont : la correspondance entre les DEUX
    masques de fracture (bruit d'échantillonnage, seuil CNN/GT), pas la
    recherche de pose.

    `dilate_px` (piste 1, tentative 1 — REJETÉE le 2026-07-20) et
    `gaussian_sigma_px` (piste 1, tentative 2) : voir `points_to_valid_mask()`.
    """
    pts_i_in_j = (R_ij_gt @ frac_i.T).T + t_ij_gt
    pts_c_i = pts_i_in_j - c_j
    u_coords_i = pts_c_i @ u_j
    v_coords_i = pts_c_i @ v_j

    pts_c_j = frac_j - c_j
    u_coords_j = pts_c_j @ u_j
    v_coords_j = pts_c_j @ v_j

    pred_valid, _, _ = points_to_valid_mask(
        u_coords_i, v_coords_i, u_min_j, v_min_j, resolution, pixel_size,
        dilate_px=dilate_px, gaussian_sigma_px=gaussian_sigma_px)
    valid_j, _, _ = points_to_valid_mask(
        u_coords_j, v_coords_j, u_min_j, v_min_j, resolution, pixel_size,
        dilate_px=dilate_px, gaussian_sigma_px=gaussian_sigma_px)

    overlap = int(np.logical_and(pred_valid, valid_j).sum())
    denom = max(min(int(pred_valid.sum()), int(valid_j.sum())), 1)
    return float(overlap / denom)


def run_match_at_resolution(frac_i, c_i, u_i, v_i, n_i,
                             frac_j, c_j, u_j, v_j, n_j,
                             span_i, span_j, resolution, args, R_ij_gt, t_ij_gt):
    """Exécute le pipeline complet (rasterize -> match_depthmaps -> Kabsch) à UNE
    résolution donnée. Factorisé pour être appelé à la fois pour le run principal
    (args.resolution) et pour le sweep densité x résolution (piste 3 Phase 7,
    2026-07-21) -- CONTRAIREMENT à oracle_overlap_frac (diagnostic pur sur le
    recouvrement de footprint), mesure la métrique qui compte réellement :
    Pose@30/Pose@15 sur le pipeline réel. Nécessaire car le sweep purement
    OracleOvlp (2026-07-21) s'est révélé biaisé de la même façon que la
    dilatation/le splat gaussien avant lui : un bon recouvrement mécanique ne
    garantit pas une bonne pose, cf. plan Phase 7. Retourne un dict avec soit
    {"skip": ...} soit les métriques de pose.
    """
    pixel_size = max(span_i, span_j) * 1.1 / resolution
    dmap_i, valid_i, u_min_i, v_min_i = rasterize(
        frac_i, c_i, u_i, v_i, n_i, resolution, pixel_size,
        dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(
        frac_j, c_j, u_j, v_j, n_j, resolution, pixel_size,
        dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)

    n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
    if n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS:
        return {"skip": "sparse_dmap", "n_pix": (n_pix_i, n_pix_j)}

    best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
        dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)

    pts_i3, pts_j3 = build_correspondences(
        dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
        dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
        pixel_size, best_theta, best_shift, best_flip,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        return {"skip": "no_overlap_3d"}

    R_est, t_est = kabsch(pts_i3, pts_j3)
    re = rot_err_deg(R_est, R_ij_gt)
    te = trans_err(t_est, t_ij_gt)
    return {
        "rot_err":      float(re),
        "trans_err":    float(te),
        "pose_success": {f"{int(r)}deg_{t}": bool(re < r and te < t)
                          for r, t in POSE_SUCCESS_THRESH},
        "best_overlap_frac": float(best_overlap_frac),
        "n_corr":       len(pts_i3),
        "n_dmap_pix":   (n_pix_i, n_pix_j),
    }


def density_pose_stratification(n_frac_pts_min, rot_err, pose30, pose15):
    """Comme `frac_pts_min_stratification`, mais sans `OracleOvlp` -- utilisée
    pour le sweep densité x résolution (piste 3 Phase 7, 2026-07-21) : mesure
    Pose@30/Pose@15 RÉELS (pipeline complet, pas juste le recouvrement de
    footprint) par tranche de densité `min(n_i,n_j)`, séparément pour CHAQUE
    résolution testée -- répond directement à la critique du 2026-07-21 ("on ne
    sait pas sur quel type de fragments telle résolution fonctionne, et
    l'overlap seul ne suffit pas"). Retourne une liste de dicts (vide si tranche
    non peuplée).
    """
    n_frac_pts_min = np.asarray(n_frac_pts_min)
    rot_err = np.asarray(rot_err)
    pose30  = np.asarray(pose30)
    pose15  = np.asarray(pose15)
    idx = np.digitize(n_frac_pts_min, N_FRAC_PTS_MIN_BINS[1:-1])

    rows = []
    for b, label in enumerate(N_FRAC_PTS_MIN_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({
            "bin": label,
            "n": n,
            "rot_err_mean": float(rot_err[mask].mean()),
            "pose_30deg_0.1": 100.0 * float(pose30[mask].mean()),
            "pose_15deg_0.05": 100.0 * float(pose15[mask].mean()),
        })
    return rows


def attrition_by_density_stratification(nmin_ok, nmin_skip_sparse, nmin_skip_nooverlap):
    """Répartit succès et les DEUX raisons d'échec possibles dans
    `run_match_at_resolution` (`sparse_dmap` : pas assez de cases occupées
    après rasterisation ; `no_overlap_3d` : alignement trouvé mais < 3
    correspondances 3D reconstruites après rotation/arrondi pixel) par tranche
    de densité `min(n_i,n_j)`, pour UNE résolution donnée.

    Ajouté le 2026-07-21 suite à la demande explicite de l'utilisateur
    d'investiguer directement POURQUOI une résolution plus fine perd des
    paires (le sweep précédent montrait juste que ça arrivait, sans dire
    laquelle des deux raisons domine ni si c'est uniforme ou concentré sur
    les paires éparses). Retourne une liste de dicts, un par tranche.
    """
    nmin_ok = np.asarray(nmin_ok)
    nmin_sp = np.asarray(nmin_skip_sparse)
    nmin_no = np.asarray(nmin_skip_nooverlap)
    idx_ok = np.digitize(nmin_ok, N_FRAC_PTS_MIN_BINS[1:-1]) if len(nmin_ok) else np.array([], dtype=int)
    idx_sp = np.digitize(nmin_sp, N_FRAC_PTS_MIN_BINS[1:-1]) if len(nmin_sp) else np.array([], dtype=int)
    idx_no = np.digitize(nmin_no, N_FRAC_PTS_MIN_BINS[1:-1]) if len(nmin_no) else np.array([], dtype=int)

    rows = []
    for b, label in enumerate(N_FRAC_PTS_MIN_LABELS):
        n_ok = int((idx_ok == b).sum())
        n_sp = int((idx_sp == b).sum())
        n_no = int((idx_no == b).sum())
        n_tot = n_ok + n_sp + n_no
        rows.append({
            "bin": label,
            "n_ok": n_ok,
            "n_skip_sparse_dmap": n_sp,
            "n_skip_no_overlap_3d": n_no,
            "n_total": n_tot,
            "skip_rate": 100.0 * (n_sp + n_no) / n_tot if n_tot else 0.0,
        })
    return rows


def process_pair(raw_i, raw_j, gt_i, gt_j, score_i, score_j, R_ij_gt, t_ij_gt, args, rng):
    """Traite une paire dirigée i→j. Retourne un dict de résultats par stratégie."""
    results = {}
    n_gt_i = int((gt_i == 1).sum())
    n_gt_j = int((gt_j == 1).sum())
    n_rand = max(n_gt_i, n_gt_j, MIN_FRAC_POINTS)

    masks = {
        "gt":       (raw_i[gt_i == 1],                            raw_j[gt_j == 1]),
        "thresh0.3":(raw_i[score_i > 0.3],                        raw_j[score_j > 0.3]),
        "random":   (raw_i[rng.choice(len(raw_i), min(n_rand, len(raw_i)), replace=False)],
                     raw_j[rng.choice(len(raw_j), min(n_rand, len(raw_j)), replace=False)]),
    }

    for strat, (frac_i, frac_j) in masks.items():
        if len(frac_i) < MIN_FRAC_POINTS or len(frac_j) < MIN_FRAC_POINTS:
            results[strat] = {"skip": "too_few_points",
                              "n_pts": (len(frac_i), len(frac_j))}
            continue

        c_i, u_i, v_i, n_i, plan_i = compute_pca_frame(frac_i)
        c_j, u_j, v_j, n_j, plan_j = compute_pca_frame(frac_j)

        if plan_i > args.max_planarity or plan_j > args.max_planarity:
            results[strat] = {"skip": "too_curved",
                              "planarity": (float(plan_i), float(plan_j))}
            continue

        # Taille physique commune : chaque pixel couvre la même surface pour les deux maps
        ci = frac_i - c_i
        cj = frac_j - c_j
        span_i = max(
            float((ci @ u_i).max() - (ci @ u_i).min()),
            float((ci @ v_i).max() - (ci @ v_i).min()),
            1e-8,
        )
        span_j = max(
            float((cj @ u_j).max() - (cj @ u_j).min()),
            float((cj @ v_j).max() - (cj @ v_j).min()),
            1e-8,
        )
        pixel_size = max(span_i, span_j) * 1.1 / args.resolution

        # Sweep densité x résolution (piste 3 Phase 7, 2026-07-21) : contrairement
        # au sweep OracleOvlp seul (biaisé -- moyenne globale, overlap pas pose),
        # exécute le pipeline COMPLET à chaque résolution candidate et garde
        # min(n_i, n_j) pour stratifier après-coup par densité (voir
        # `density_pose_stratification`). Coût : len(pose_resolution_sweep) runs
        # complets en plus, restreint par défaut à la stratégie `gt` (oracle,
        # même discipline "diagnostiquer sur l'oracle d'abord" que tout le
        # projet) via --pose_resolution_sweep_strategies.
        density_sweep_result = None
        if args.pose_resolution_sweep and strat in args.pose_resolution_sweep_strategies:
            n_frac_min_pair = min(len(frac_i), len(frac_j))
            density_sweep_result = {}
            for r in args.pose_resolution_sweep:
                res_r = run_match_at_resolution(
                    frac_i, c_i, u_i, v_i, n_i, frac_j, c_j, u_j, v_j, n_j,
                    span_i, span_j, r, args, R_ij_gt, t_ij_gt)
                res_r["n_frac_pts_min"] = n_frac_min_pair
                density_sweep_result[str(r)] = res_r

        dmap_i, valid_i, u_min_i, v_min_i = rasterize(
            frac_i, c_i, u_i, v_i, n_i, args.resolution, pixel_size,
            dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)
        dmap_j, valid_j, u_min_j, v_min_j = rasterize(
            frac_j, c_j, u_j, v_j, n_j, args.resolution, pixel_size,
            dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)

        n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
        if n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS:
            results[strat] = {"skip": "sparse_dmap", "n_pix": (n_pix_i, n_pix_j),
                              "density_resolution_sweep": density_sweep_result}
            continue

        oracle_ovlp = oracle_overlap_frac(
            frac_i, frac_j, c_j, u_j, v_j, u_min_j, v_min_j,
            args.resolution, pixel_size, R_ij_gt, t_ij_gt,
        )
        oracle_ovlp_sweep = {}
        for d in args.dilate_sweep:
            oracle_ovlp_sweep[f"d{d}"] = oracle_overlap_frac(
                frac_i, frac_j, c_j, u_j, v_j, u_min_j, v_min_j,
                args.resolution, pixel_size, R_ij_gt, t_ij_gt, dilate_px=d,
            )
        for g in args.gaussian_sweep:
            oracle_ovlp_sweep[f"g{g}"] = oracle_overlap_frac(
                frac_i, frac_j, c_j, u_j, v_j, u_min_j, v_min_j,
                args.resolution, pixel_size, R_ij_gt, t_ij_gt, gaussian_sigma_px=g,
            )
        # Sweep de résolution (piste 3 Phase 7, 2026-07-21) : diagnostic PUR, comme
        # dilate_sweep/gaussian_sweep ci-dessus -- ne touche pas au pipeline réel de
        # recherche, seulement à oracle_overlap_frac. Pour chaque résolution candidate,
        # pixel_size ET u_min_j/v_min_j sont recalculés (u_min/v_min dépendent de
        # pixel_size -- cf. rasterize()) pour garder la même étendue physique, seule
        # la finesse de la grille change. Teste si le plafond 0.758 (résolution 64
        # fixe) remonte à résolution plus fine ou plus grossière, et surtout si
        # l'écart gt/random se maintient (sinon même défaut que la dilatation : un
        # gain mécanique qui profite à tout le monde, pas un vrai signal).
        cj_coords_u = cj @ u_j
        cj_coords_v = cj @ v_j
        for r in args.resolution_sweep:
            pixel_size_r = max(span_i, span_j) * 1.1 / r
            u_min_j_r = float(cj_coords_u.min()) - 0.5 * pixel_size_r
            v_min_j_r = float(cj_coords_v.min()) - 0.5 * pixel_size_r
            oracle_ovlp_sweep[f"res{r}"] = oracle_overlap_frac(
                frac_i, frac_j, c_j, u_j, v_j, u_min_j_r, v_min_j_r,
                r, pixel_size_r, R_ij_gt, t_ij_gt,
            )

        best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
            dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)

        pts_i3, pts_j3 = build_correspondences(
            dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
            dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
            pixel_size, best_theta, best_shift, best_flip,
        )

        if pts_i3 is None or len(pts_i3) < 3:
            results[strat] = {"skip": "no_overlap_3d",
                              "density_resolution_sweep": density_sweep_result}
            continue

        R_est, t_est = kabsch(pts_i3, pts_j3)
        re = rot_err_deg(R_est, R_ij_gt)
        te = trans_err(t_est, t_ij_gt)
        results[strat] = {
            "rot_err":      float(re),
            "trans_err":    float(te),
            "pose_success": {f"{int(r)}deg_{t}": bool(re < r and te < t)
                             for r, t in POSE_SUCCESS_THRESH},
            "best_score":   float(best_score),
            "best_theta":   float(best_theta),
            "best_flip":    bool(best_flip),
            "overlap_frac": float(best_overlap_frac),
            "oracle_overlap_frac": float(oracle_ovlp),
            "oracle_overlap_frac_sweep": {k: float(v) for k, v in oracle_ovlp_sweep.items()},
            "n_corr":       len(pts_i3),
            "planarity":    (float(plan_i), float(plan_j)),
            "n_frac_pts":   (len(frac_i), len(frac_j)),
            "n_dmap_pix":   (n_pix_i, n_pix_j),
            "density_resolution_sweep": density_sweep_result,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="0 = tout le split ; >0 pour un test rapide")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resolution",  type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--n_angles",    type=int, default=DEFAULT_N_ANGLES)
    parser.add_argument("--score_mode",  default="joint",
                        choices=["relief", "overlap_only", "joint"],
                        help="relief = formule d'origine (buggée, garde-fou overlap>0.5px) ; "
                             "overlap_only = contour seul, sans relief (teste l'hypothèse "
                             "contour-suffit) ; joint = fix, relief_score * overlap_frac.")
    parser.add_argument("--max_planarity", type=float, default=DEFAULT_MAX_PLANARITY,
                        help="Seuil de planéité au-delà duquel une face est jugée "
                             "trop courbe et la paire skippée (défaut 0.15). Augmenter "
                             "(ex. 0.333 = isotrope, borne théorique) pour tester si les "
                             "faces jusqu'ici exclues matchent mieux ou moins bien — "
                             "cf. incohérence relevée le 2026-07-20 entre 'trop plat' et "
                             "'on exclut les plus courbées sans jamais les tester'.")
    parser.add_argument("--dilate_px", type=int, default=0,
                        help="Dilate le masque `valid` de N pixels dans le pipeline réel "
                             "(recherche + score), pas seulement le diagnostic oracle. "
                             "Défaut 0 = comportement inchangé. À tester avec 1 d'abord "
                             "(cf. sweep du 2026-07-20 : d=1 fait remonter OracleOvlp gt "
                             "de 0.757 à 0.992, mais random en profite presque autant — "
                             "probablement un pansement, pas une vraie correction).")
    parser.add_argument("--dilate_sweep", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Rayons de dilatation (en pixels) testés pour "
                             "oracle_overlap_frac (piste 1 de la roadmap Phase 7, "
                             "2026-07-20) — mesure si le plafond théorique (~0.758 "
                             "sur GT à dilate=0) remonte vers 1.0 en dilatant "
                             "légèrement les footprints, ce qui indiquerait que "
                             "l'écart est surtout du bruit de discrétisation/"
                             "échantillonnage plutôt qu'une vraie asymétrie des masques.")
    parser.add_argument("--gaussian_sigma_px", type=float, default=0.0,
                        help="Applique un splat gaussien (sigma en pixels) dans le "
                             "pipeline réel (recherche + score), pas seulement le "
                             "diagnostic oracle. Défaut 0.0 = comportement inchangé. "
                             "Piste 1, tentative 2 (2026-07-20), après le rejet de la "
                             "dilatation binaire seule (--dilate_px) qui gonflait "
                             "random presque autant que gt.")
    parser.add_argument("--gaussian_sweep", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Sigmas (en pixels) testés pour oracle_overlap_frac en "
                             "mode splat gaussien (piste 1, tentative 2, 2026-07-20) — "
                             "à comparer à --dilate_sweep : vérifier que l'écart gt vs "
                             "random se maintient (contrairement à la dilatation, où "
                             "il s'effondrait de 0.327 à 0.027 entre d=0 et d=4).")
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[24, 32, 48, 64, 96, 128],
                        help="Résolutions (RxR px) testées pour oracle_overlap_frac "
                             "(piste 3 Phase 7, 2026-07-21) -- diagnostic pur, ne "
                             "touche pas au pipeline réel de recherche. Mesure si le "
                             "plafond 0.758 (résolution 64 fixe) remonte à une autre "
                             "résolution, et si l'écart gt/random se maintient (sinon "
                             "même défaut que la dilatation/le splat, déjà rejetés).")
    parser.add_argument("--pose_resolution_sweep", type=int, nargs="+", default=[],
                        help="Résolutions RxR pour lesquelles le pipeline COMPLET "
                             "(rasterize->match->Kabsch) est exécuté, en plus du run "
                             "principal à --resolution (piste 3 Phase 7, 2026-07-21, "
                             "réponse à la critique du sweep OracleOvlp seul : bon "
                             "recouvrement != bonne pose, même défaut que dilatation/"
                             "splat). Vide par défaut (désactivé, coût nul). Chaque "
                             "résolution ajoutée coûte un run complet de plus par "
                             "paire pour les stratégies listées dans "
                             "--pose_resolution_sweep_strategies -- rester raisonnable "
                             "(4-6 valeurs) sur un run pleine échelle.")
    parser.add_argument("--pose_resolution_sweep_strategies", nargs="+", default=["gt"],
                        choices=["gt", "thresh0.3", "random"],
                        help="Stratégies sur lesquelles exécuter "
                             "--pose_resolution_sweep. Défaut : gt seul (oracle, "
                             "même discipline que le reste du projet -- diagnostiquer "
                             "sur l'oracle avant de dépenser le budget de calcul sur "
                             "thresh0.3/random).")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)
    print(f"Device: {device} | resolution={args.resolution} | n_angles={args.n_angles}")

    # ── Data ──────────────────────────────────────────────────────────────────
    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.train_dataset

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    # ── Modèle (CNN figé, utilisé pour gt_flat + score_flat) ──────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # ── Résultats ─────────────────────────────────────────────────────────────
    accum = {s: defaultdict(list) for s in ("gt", "thresh0.3", "random")}
    n_pairs_total = 0
    n_2frag_seen  = 0
    t0 = time.time()

    print(f"\nPhase 5A — depth-map matching sur {args.categories}/{args.split} "
          f"(objets à 2 fragments uniquement)...\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 20 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx} | "
                      f"paires traitées={n_pairs_total} | {elapsed:.0f}s écoulées")

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()

            valid_pcs_np = valid_pcs.cpu().numpy()
            B, P = valid_pcs_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_pcs_np[b, p]]

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            quats_np  = batch["quaternions"].numpy()
            trans_np  = batch["translations"].numpy()

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            gt_per_k    = [gt_flat  [offsets[k]:offsets[k+1]] for k in range(K)]
            score_per_k = [pred_flat[offsets[k]:offsets[k+1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()          for k in range(K)]

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                if len(ks_ps) != 2:
                    continue   # on ne traite que les objets à 2 fragments

                n_2frag_seen += 1
                (k0, p0), (k1, p1) = ks_ps

                raw0 = pc_per_k[k0] * scale_np[b, p0]
                raw1 = pc_per_k[k1] * scale_np[b, p1]
                gt0  = gt_per_k[k0];    gt1  = gt_per_k[k1]
                sc0  = score_per_k[k0]; sc1  = score_per_k[k1]

                R0 = quat_wxyz_to_rotmat(quats_np[b, p0])
                R1 = quat_wxyz_to_rotmat(quats_np[b, p1])
                R_ij = R1.T @ R0
                t_ij = R1.T @ (trans_np[b, p0] - trans_np[b, p1])

                pair_res = process_pair(
                    raw0, raw1, gt0, gt1, sc0, sc1, R_ij, t_ij, args, rng)

                n_pairs_total += 1
                for strat, res in pair_res.items():
                    if "skip" in res:
                        accum[strat]["n_skip"].append(1)
                        accum[strat][f"skip_{res['skip']}"].append(1)
                    else:
                        accum[strat]["rot_err"].append(res["rot_err"])
                        accum[strat]["trans_err"].append(res["trans_err"])
                        for k_ps, v_ps in res["pose_success"].items():
                            accum[strat][f"pose_{k_ps}"].append(float(v_ps))
                        accum[strat]["n_corr"].append(res["n_corr"])
                        accum[strat]["best_score"].append(res["best_score"])
                        accum[strat]["overlap_frac"].append(res["overlap_frac"])
                        accum[strat]["oracle_overlap_frac"].append(res["oracle_overlap_frac"])
                        for key, v in res["oracle_overlap_frac_sweep"].items():
                            accum[strat][f"oracle_ovlp_{key}"].append(v)
                        accum[strat]["planarity"].append(float(np.mean(res["planarity"])))
                        accum[strat]["n_frac_pts"].append(float(np.mean(res["n_frac_pts"])))
                        accum[strat]["n_frac_pts_min"].append(float(min(res["n_frac_pts"])))

                    # Sweep densité x résolution (piste 3 Phase 7, 2026-07-21) : hors du
                    # if/else ci-dessus -- présent que le run principal (à args.resolution)
                    # ait réussi ou non, puisque d'autres résolutions peuvent réussir même
                    # quand args.resolution échoue (sparse_dmap/no_overlap_3d).
                    dsweep = res.get("density_resolution_sweep")
                    if dsweep:
                        for r_key, r_res in dsweep.items():
                            if "skip" in r_res:
                                # Attrition (piste 3 Phase 7, 2026-07-21) : POURQUOI une
                                # résolution plus fine perd des paires -- garder la raison
                                # (sparse_dmap = pas assez de cases occupées après
                                # rasterisation ; no_overlap_3d = alignement trouvé mais
                                # < 3 correspondances 3D reconstruites après rotation/
                                # arrondi pixel) ET la densité, pour voir si l'attrition
                                # est uniforme ou concentrée sur les paires éparses.
                                reason = r_res["skip"]
                                accum[strat][f"dsweep_res{r_key}_skip_{reason}"].append(1)
                                accum[strat][f"dsweep_res{r_key}_skip_{reason}_nmin"].append(
                                    r_res["n_frac_pts_min"])
                                continue
                            accum[strat][f"dsweep_res{r_key}_rot_err"].append(r_res["rot_err"])
                            accum[strat][f"dsweep_res{r_key}_n_frac_pts_min"].append(
                                r_res["n_frac_pts_min"])
                            for k_ps, v_ps in r_res["pose_success"].items():
                                accum[strat][f"dsweep_res{r_key}_pose_{k_ps}"].append(float(v_ps))

    elapsed = time.time() - t0
    print(f"\nFini : {n_pairs_total} paires traitées ({n_2frag_seen} objets 2-frags vus)"
          f" en {elapsed:.0f}s ({elapsed/max(n_pairs_total,1):.2f}s/paire)\n")

    # ── Résumé tabulaire ──────────────────────────────────────────────────────
    header = (f"{'Strategy':<12} {'N':>6} {'Skip':>6} {'RotErr°':>8} "
              f"{'TransErr':>9} {'Pose@30/0.1':>11} {'Pose@15/0.05':>12} {'N_corr':>7} {'OvlpFrac':>8} {'OracleOvlp':>10}")
    print(header)
    print("-" * len(header))

    summary = {"config": vars(args), "n_pairs": n_pairs_total,
               "n_2frag_seen": n_2frag_seen, "elapsed_s": elapsed}

    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        n_ok   = len(acc.get("rot_err", []))
        n_skip = sum(acc.get("n_skip", []))

        if n_ok == 0:
            print(f"{strat:<12} {'—':>6} {n_skip:>6}")
            summary[strat] = {"n_ok": 0, "n_skip": n_skip}
            continue

        re_mean  = float(np.mean(acc["rot_err"]))
        te_mean  = float(np.mean(acc["trans_err"]))
        p30      = 100.0 * float(np.mean(acc.get("pose_30deg_0.1", [0])))
        p15      = 100.0 * float(np.mean(acc.get("pose_15deg_0.05", [0])))
        nc_mean  = float(np.mean(acc["n_corr"]))
        ov_mean  = float(np.mean(acc.get("overlap_frac", [0])))
        oo_mean  = float(np.mean(acc.get("oracle_overlap_frac", [0])))

        print(f"{strat:<12} {n_ok:>6} {n_skip:>6} {re_mean:>8.2f} "
              f"{te_mean:>9.4f} {p30:>10.2f}% {p15:>11.2f}% {nc_mean:>7.1f} {ov_mean:>8.3f} {oo_mean:>10.3f}")

        summary[strat] = {
            "n_ok": n_ok, "n_skip": n_skip,
            "rot_err_mean": re_mean,
            "rot_err_median": float(np.median(acc["rot_err"])),
            "trans_err_mean": te_mean,
            "pose_30deg_0.1":  p30,
            "pose_15deg_0.05": p15,
            "n_corr_mean": nc_mean,
            "overlap_frac_mean": ov_mean,
            "oracle_overlap_frac_mean": oo_mean,
        }

    # ── Stratification par planéité : "moins plat" corrèle-t-il avec un
    # meilleur matching PARMI les paires gardées, ou pas ? ─────────────────────
    print(f"\nSTRATIFICATION PAR PLANÉITÉ (moyenne des 2 fragments, tranches "
          f"alignées sur les percentiles Phase 5A.0)")
    strat_header = f"{'Strategy':<12} {'Bin':<12} {'N':>5} {'RotErr°':>8} {'Pose@30':>9} {'Pose@15':>9}"
    print(strat_header)
    print("-" * len(strat_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        rows = planarity_stratification(
            acc["planarity"], acc["rot_err"],
            acc.get("pose_30deg_0.1", [0] * len(acc["rot_err"])),
            acc.get("pose_15deg_0.05", [0] * len(acc["rot_err"])),
        )
        for row in rows:
            if row["n"] == 0:
                print(f"{strat:<12} {row['bin']:<12} {'—':>5}")
            else:
                print(f"{strat:<12} {row['bin']:<12} {row['n']:>5} "
                      f"{row['rot_err_mean']:>8.2f} {row['pose_30deg_0.1']:>8.2f}% "
                      f"{row['pose_15deg_0.05']:>8.2f}%")
        summary[strat]["planarity_strata"] = rows
    print("(Lecture : si Pose@30 monte et RotErr baisse en allant vers les tranches\n"
          " les moins plates, la thèse 'trop plat = pas de signal' est confirmée dans\n"
          " le détail. Si c'est plat ou pas pareil, le facteur limitant est ailleurs.)")

    # ── Stratification par nombre de points : la sparsité explique-t-elle le
    # plafond OracleOvlp=0.758 (2026-07-20) ? ───────────────────────────────────
    print(f"\nSTRATIFICATION PAR NOMBRE DE POINTS (moyenne des 2 fragments, tranches "
          f"alignées sur les percentiles Phase 5A.0)")
    npts_header = (f"{'Strategy':<12} {'Bin':<12} {'N':>5} {'OracleOvlp':>10} "
                   f"{'RotErr°':>8} {'Pose@30':>9} {'Pose@15':>9}")
    print(npts_header)
    print("-" * len(npts_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        rows_n = frac_pts_stratification(
            acc["n_frac_pts"], acc.get("oracle_overlap_frac", [0] * len(acc["rot_err"])),
            acc["rot_err"],
            acc.get("pose_30deg_0.1", [0] * len(acc["rot_err"])),
            acc.get("pose_15deg_0.05", [0] * len(acc["rot_err"])),
        )
        for row in rows_n:
            if row["n"] == 0:
                print(f"{strat:<12} {row['bin']:<12} {'—':>5}")
            else:
                print(f"{strat:<12} {row['bin']:<12} {row['n']:>5} "
                      f"{row['oracle_overlap_frac_mean']:>10.3f} {row['rot_err_mean']:>8.2f} "
                      f"{row['pose_30deg_0.1']:>8.2f}% {row['pose_15deg_0.05']:>8.2f}%")
        summary[strat]["n_frac_pts_strata"] = rows_n
    print("(Lecture : si OracleOvlp monte nettement des tranches éparses vers les\n"
          " tranches denses, la sparsité d'échantillonnage explique le plafond -- une\n"
          " densification (splat gaussien) devrait aider spécifiquement les cas épars.\n"
          " Si OracleOvlp est stable quel que soit le nombre de points, la sparsité\n"
          " n'est pas la vraie cause, il faut chercher ailleurs.)")

    # ── Stratification par MIN(n_i, n_j) : le maillon faible, pas la moyenne
    # (2026-07-20, suite à l'inspection visuelle de l'objet#21) ────────────────
    print(f"\nSTRATIFICATION PAR MIN(N_PTS_I, N_PTS_J) (le côté le plus pauvre "
          f"de la paire, pas la moyenne)")
    nmin_header = (f"{'Strategy':<12} {'Bin':<12} {'N':>5} {'OracleOvlp':>10} "
                   f"{'RotErr°':>8} {'Pose@30':>9} {'Pose@15':>9}")
    print(nmin_header)
    print("-" * len(nmin_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        rows_nmin = frac_pts_min_stratification(
            acc["n_frac_pts_min"], acc.get("oracle_overlap_frac", [0] * len(acc["rot_err"])),
            acc["rot_err"],
            acc.get("pose_30deg_0.1", [0] * len(acc["rot_err"])),
            acc.get("pose_15deg_0.05", [0] * len(acc["rot_err"])),
        )
        for row in rows_nmin:
            if row["n"] == 0:
                print(f"{strat:<12} {row['bin']:<12} {'—':>5}")
            else:
                print(f"{strat:<12} {row['bin']:<12} {row['n']:>5} "
                      f"{row['oracle_overlap_frac_mean']:>10.3f} {row['rot_err_mean']:>8.2f} "
                      f"{row['pose_30deg_0.1']:>8.2f}% {row['pose_15deg_0.05']:>8.2f}%")
        summary[strat]["n_frac_pts_min_strata"] = rows_nmin
    print("(Lecture : si Pose@30 monte nettement quand le côté le plus PAUVRE a plus\n"
          " de points -- même si l'autre côté est déjà dense -- ça confirme l'hypothèse\n"
          " du maillon faible (objet#21) : le déséquilibre entre les deux côtés est le\n"
          " vrai problème, pas la densité moyenne. Comparer à la table précédente\n"
          " (moyenne) : si celle-ci était plate mais celle-ci est nette, la moyenne\n"
          " cachait bien le signal.)")

    print(f"\nRéférence Phase 2 global : Pose@30 ≈ 1.3-3.2%  |  gt_edge oracle : Pose@30 ≈ 9.6%")
    print(f"score_mode={args.score_mode}  "
          f"(relief=formule d'origine buggée, overlap_only=contour seul, joint=fix)")
    print("(OracleOvlp = recouvrement des footprints à la VRAIE pose GT, sans recherche —\n"
          " plafond théorique de OvlpFrac. Si OracleOvlp << 1, le facteur limitant est la\n"
          " correspondance entre masques de fracture, pas la recherche de pose. Si OvlpFrac\n"
          " (trouvé par la recherche) est nettement < OracleOvlp, c'est la recherche qui rate\n"
          " un alignement pourtant disponible dans les données.)")

    # ── Sweep de dilatation : le plafond remonte-t-il en lissant les masques ? ──
    print(f"\nSWEEP DILATATION (OracleOvlp moyen par rayon de dilatation, piste 1 Phase 7)")
    dsweep_header = f"{'Strategy':<12}" + "".join(f"{'d='+str(d):>8}" for d in args.dilate_sweep)
    print(dsweep_header)
    print("-" * len(dsweep_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        row_vals = {}
        row = f"{strat:<12}"
        for d in args.dilate_sweep:
            vals = acc.get(f"oracle_ovlp_d{d}", [])
            m = float(np.mean(vals)) if vals else 0.0
            row_vals[str(d)] = m
            row += f"{m:>8.3f}"
        print(row)
        summary[strat]["oracle_ovlp_dilate_sweep"] = row_vals
    print("(Lecture : si le plafond remonte nettement vers 1.0 dès d=1-2px, l'écart est\n"
          " surtout du bruit de discrétisation/échantillonnage -- une densification légère\n"
          " suffirait (piste 1). S'il stagne, c'est une asymétrie plus profonde entre les\n"
          " deux masques, pas juste un problème de résolution de grille.)")

    # ── Sweep splat gaussien : piste 1, tentative 2, après le rejet de la
    # dilatation binaire seule (2026-07-20) ────────────────────────────────────
    print(f"\nSWEEP SPLAT GAUSSIEN (OracleOvlp moyen par sigma, piste 1 tentative 2)")
    gsweep_header = f"{'Strategy':<12}" + "".join(f"{'σ='+str(g):>8}" for g in args.gaussian_sweep)
    print(gsweep_header)
    print("-" * len(gsweep_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        row_vals = {}
        row = f"{strat:<12}"
        for g in args.gaussian_sweep:
            vals = acc.get(f"oracle_ovlp_g{g}", [])
            m = float(np.mean(vals)) if vals else 0.0
            row_vals[str(g)] = m
            row += f"{m:>8.3f}"
        print(row)
        summary[strat]["oracle_ovlp_gaussian_sweep"] = row_vals
    print("(Lecture : contrairement à la dilatation binaire, le splat gaussien pondère\n"
          " par la distance réelle aux points -- garde-fou à vérifier : l'écart gt vs\n"
          " random doit se maintenir ou grandir avec sigma, pas s'effondrer comme avec\n"
          " la dilatation. S'il s'effondre pareil, le splat a le même défaut.)")

    # ── Sweep de résolution : piste 3 Phase 7 (2026-07-21), diagnostic pur avant
    # toute modification du pipeline réel ──────────────────────────────────────
    print(f"\nSWEEP RÉSOLUTION (OracleOvlp moyen par résolution RxR, piste 3 Phase 7)")
    rsweep_header = f"{'Strategy':<12}" + "".join(f"{'R='+str(r):>8}" for r in args.resolution_sweep)
    print(rsweep_header)
    print("-" * len(rsweep_header))
    for strat in ("gt", "thresh0.3", "random"):
        acc = accum[strat]
        if not acc.get("rot_err"):
            continue
        row_vals = {}
        row = f"{strat:<12}"
        for r in args.resolution_sweep:
            vals = acc.get(f"oracle_ovlp_res{r}", [])
            m = float(np.mean(vals)) if vals else 0.0
            row_vals[str(r)] = m
            row += f"{m:>8.3f}"
        print(row)
        summary[strat]["oracle_ovlp_resolution_sweep"] = row_vals
    print("(Lecture : si OracleOvlp gt monte nettement en changeant R SANS que random\n"
          " suive au même rythme, la résolution actuelle (64, fixe pour tous) est mal\n"
          " calibrée et une résolution adaptative (liée à la densité de points) a de\n"
          " bonnes chances d'aider réellement. Si gt et random bougent pareil (comme la\n"
          " dilatation/le splat), c'est un gain mécanique, pas un vrai signal -- rejeter.)")

    # ── Sweep densité x résolution : Pose@30 RÉEL (pipeline complet), stratifié
    # par min(n_i, n_j) -- réponse à la critique du 2026-07-21 sur le sweep
    # OracleOvlp seul (biaisé, moyenne globale, overlap != pose) ────────────────
    if args.pose_resolution_sweep:
        print(f"\nSWEEP DENSITÉ x RÉSOLUTION (Pose@30/Pose@15 RÉELS, pipeline complet, "
              f"stratifié par min(n_i,n_j) -- piste 3 Phase 7)")
        for strat in args.pose_resolution_sweep_strategies:
            acc = accum[strat]
            for r in args.pose_resolution_sweep:
                re_key   = f"dsweep_res{r}_rot_err"
                nmin_key = f"dsweep_res{r}_n_frac_pts_min"
                p30_key  = f"dsweep_res{r}_pose_30deg_0.1"
                p15_key  = f"dsweep_res{r}_pose_15deg_0.05"
                if not acc.get(re_key):
                    print(f"\n{strat} @ R={r} : aucune paire (tout skip)")
                    continue
                rows_r = density_pose_stratification(
                    acc[nmin_key], acc[re_key], acc[p30_key], acc[p15_key])
                n_total = len(acc[re_key])
                p30_overall = 100.0 * float(np.mean(acc[p30_key]))
                print(f"\n{strat} @ R={r} (N={n_total}, Pose@30 global={p30_overall:.2f}%)")
                dr_header = f"  {'Bin':<12} {'N':>5} {'RotErr°':>8} {'Pose@30':>9} {'Pose@15':>9}"
                print(dr_header)
                for row in rows_r:
                    if row["n"] == 0:
                        print(f"  {row['bin']:<12} {'—':>5}")
                    else:
                        print(f"  {row['bin']:<12} {row['n']:>5} "
                              f"{row['rot_err_mean']:>8.2f} {row['pose_30deg_0.1']:>8.2f}% "
                              f"{row['pose_15deg_0.05']:>8.2f}%")
                summary.setdefault(strat, {}).setdefault(
                    "density_resolution_sweep", {})[str(r)] = {
                        "n": n_total, "pose_30deg_0.1_overall": p30_overall,
                        "strata": rows_r,
                    }
        print("\n(Lecture : pour chaque tranche de densité (côté le plus PAUVRE de la\n"
              " paire), quelle résolution donne le meilleur Pose@30 réel -- pas juste le\n"
              " meilleur OracleOvlp ? Si une résolution plus fine aide les tranches denses\n"
              " et une résolution plus grossière aide les tranches éparses, ça confirme\n"
              " l'hypothèse d'une résolution ADAPTATIVE liée à la densité. Si aucune\n"
              " résolution ne change Pose@30 dans aucune tranche, la résolution n'est pas\n"
              " le facteur limitant -- chercher ailleurs.)")

        # ── Attrition par résolution x densité : POURQUOI une résolution plus fine
        # perd des paires (2026-07-21, suite à la demande explicite de l'utilisateur
        # d'investiguer directement le mécanisme, pas juste le compromis observé) ──
        print(f"\nATTRITION PAR RÉSOLUTION x DENSITÉ (répartition succès / sparse_dmap / "
              f"no_overlap_3d par tranche min(n_i,n_j), pour chaque R -- piste 3 Phase 7)")
        for strat in args.pose_resolution_sweep_strategies:
            acc = accum[strat]
            for r in args.pose_resolution_sweep:
                nmin_ok = acc.get(f"dsweep_res{r}_n_frac_pts_min", [])
                nmin_sp = acc.get(f"dsweep_res{r}_skip_sparse_dmap_nmin", [])
                nmin_no = acc.get(f"dsweep_res{r}_skip_no_overlap_3d_nmin", [])
                if not (nmin_ok or nmin_sp or nmin_no):
                    continue
                rows_a = attrition_by_density_stratification(nmin_ok, nmin_sp, nmin_no)
                n_ok_tot  = len(nmin_ok)
                n_sp_tot  = len(nmin_sp)
                n_no_tot  = len(nmin_no)
                n_all_tot = n_ok_tot + n_sp_tot + n_no_tot
                print(f"\n{strat} @ R={r} (N_total={n_all_tot} : "
                      f"ok={n_ok_tot}, sparse_dmap={n_sp_tot}, no_overlap_3d={n_no_tot})")
                at_header = (f"  {'Bin':<12} {'N_ok':>6} {'Sparse':>7} "
                             f"{'NoOvlp3D':>9} {'Total':>7} {'SkipRate':>9}")
                print(at_header)
                for row in rows_a:
                    if row["n_total"] == 0:
                        print(f"  {row['bin']:<12} {'—':>6}")
                    else:
                        print(f"  {row['bin']:<12} {row['n_ok']:>6} "
                              f"{row['n_skip_sparse_dmap']:>7} {row['n_skip_no_overlap_3d']:>9} "
                              f"{row['n_total']:>7} {row['skip_rate']:>8.1f}%")
        print("\n(Lecture : si `no_overlap_3d` grandit avec R alors que `sparse_dmap` reste\n"
              " stable, l'attrition vient de la tolérance de correspondance après rotation\n"
              " (la grille se resserre mais n_angles=36 -- pas de step angulaire fin -- reste\n"
              " fixe, l'erreur de quantification angulaire devient relativement plus grosse\n"
              " en pixels à résolution fine). Si c'est concentré sur les tranches éparses,\n"
              " le problème est spécifique à la densité, pas à la résolution en général.)")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
