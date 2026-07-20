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
from scipy.ndimage import rotate as ndimage_rotate
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
MAX_PLANARITY          = 0.15    # planéité PCA > seuil → face trop courbe → skip
MIN_OVERLAP_PIXELS     = 20      # correspondances 3D < N → skip Kabsch
POSE_SUCCESS_THRESH    = [(30.0, 0.1), (15.0, 0.05)]

# Tranches de planéité pour la stratification post-hoc (bornes alignées sur les
# percentiles du pré-check Phase 5A.0 : p25=0.016, p50=0.043, p75=0.098, p90=0.150).
# Sert à vérifier si "moins plat" corrèle vraiment avec un meilleur matching parmi
# les paires GARDÉES (pas juste à exclure les plus courbées via MAX_PLANARITY et
# conclure "trop plat" sans jamais avoir testé le sens inverse de la corrélation).
PLANARITY_BINS = [0.0, 0.02, 0.05, 0.10, MAX_PLANARITY + 1e-9]
PLANARITY_LABELS = ["<0.02", "0.02-0.05", "0.05-0.10", f"0.10-{MAX_PLANARITY}"]


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


def rasterize(pts: np.ndarray, centroid, u, v, n, resolution: int, pixel_size: float):
    """Projette les points fracture en depth map 2D.
    depth = composante selon n depuis le centroïde.
    Retourne (dmap [R,R], valid [R,R bool], u_min, v_min)."""
    pts_c = pts - centroid
    u_coords = pts_c @ u
    v_coords = pts_c @ v
    depths   = pts_c @ n

    u_min = u_coords.min() - 0.5 * pixel_size
    v_min = v_coords.min() - 0.5 * pixel_size

    u_pix = np.clip(((u_coords - u_min) / pixel_size).astype(int), 0, resolution - 1)
    v_pix = np.clip(((v_coords - v_min) / pixel_size).astype(int), 0, resolution - 1)

    dmap  = np.zeros((resolution, resolution), dtype=np.float64)
    count = np.zeros((resolution, resolution), dtype=np.float64)
    np.add.at(dmap,  (v_pix, u_pix), depths)
    np.add.at(count, (v_pix, u_pix), 1.0)

    valid = count > 0
    dmap[valid] /= count[valid]
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

        if plan_i > MAX_PLANARITY or plan_j > MAX_PLANARITY:
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

        dmap_i, valid_i, u_min_i, v_min_i = rasterize(
            frac_i, c_i, u_i, v_i, n_i, args.resolution, pixel_size)
        dmap_j, valid_j, u_min_j, v_min_j = rasterize(
            frac_j, c_j, u_j, v_j, n_j, args.resolution, pixel_size)

        n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
        if n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS:
            results[strat] = {"skip": "sparse_dmap", "n_pix": (n_pix_i, n_pix_j)}
            continue

        best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
            dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)

        pts_i3, pts_j3 = build_correspondences(
            dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
            dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
            pixel_size, best_theta, best_shift, best_flip,
        )

        if pts_i3 is None or len(pts_i3) < 3:
            results[strat] = {"skip": "no_overlap_3d"}
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
            "n_corr":       len(pts_i3),
            "planarity":    (float(plan_i), float(plan_j)),
            "n_frac_pts":   (len(frac_i), len(frac_j)),
            "n_dmap_pix":   (n_pix_i, n_pix_j),
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
                        accum[strat]["planarity"].append(float(np.mean(res["planarity"])))

    elapsed = time.time() - t0
    print(f"\nFini : {n_pairs_total} paires traitées ({n_2frag_seen} objets 2-frags vus)"
          f" en {elapsed:.0f}s ({elapsed/max(n_pairs_total,1):.2f}s/paire)\n")

    # ── Résumé tabulaire ──────────────────────────────────────────────────────
    header = (f"{'Strategy':<12} {'N':>6} {'Skip':>6} {'RotErr°':>8} "
              f"{'TransErr':>9} {'Pose@30/0.1':>11} {'Pose@15/0.05':>12} {'N_corr':>7} {'OvlpFrac':>8}")
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

        print(f"{strat:<12} {n_ok:>6} {n_skip:>6} {re_mean:>8.2f} "
              f"{te_mean:>9.4f} {p30:>10.2f}% {p15:>11.2f}% {nc_mean:>7.1f} {ov_mean:>8.3f}")

        summary[strat] = {
            "n_ok": n_ok, "n_skip": n_skip,
            "rot_err_mean": re_mean,
            "rot_err_median": float(np.median(acc["rot_err"])),
            "trans_err_mean": te_mean,
            "pose_30deg_0.1":  p30,
            "pose_15deg_0.05": p15,
            "n_corr_mean": nc_mean,
            "overlap_frac_mean": ov_mean,
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

    print(f"\nRéférence Phase 2 global : Pose@30 ≈ 1.3-3.2%  |  gt_edge oracle : Pose@30 ≈ 9.6%")
    print(f"score_mode={args.score_mode}  "
          f"(relief=formule d'origine buggée, overlap_only=contour seul, joint=fix)")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
