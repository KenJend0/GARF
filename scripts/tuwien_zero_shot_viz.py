"""
scripts/tuwien_zero_shot_viz.py
==================================
Test EXPLORATOIRE (2026-08-03, demande utilisateur "juste essayer, j'ai un
truc à montrer au tuteur") -- fait tourner le CNN Step 15 (`CNNFracSeg`) en
zero-shot sur un objet réel scanné (TU Wien 3D Puzzles, format PCD :
"x y z nx ny nz" par ligne, cf. https://www.geometrie.tuwien.ac.at/ig/3dpuzzles.html),
puis estime une pose relative par paire de fragments adjacents avec le
matcher hand-crafted (`match_depthmaps`), et exporte tout dans un JSON pour
une visualisation HTML (3 vues : fragments bruts, fragments + masque
fracture CNN, poses estimées par paire).

AUCUNE vérité terrain disponible sur ce dataset (ni pose exacte, ni
adjacence, ni masque fracture GT, ni label de correspondance) -- à
présenter comme une démo QUALITATIVE, pas une évaluation quantitative
(pas de Pose@30/succès strict possible ici, contrairement à Breaking Bad).
Les paires "adjacentes" sont elles-mêmes déduites par proximité (voir
`infer_adjacency`), pas une vérité terrain.

Pipeline par fragment :
  1. Charge le PCD (x,y,z,nx,ny,nz).
  2. Sous-échantillonne à `--n_points` (défaut 5000, même convention que
     l'entraînement CNN Step 15 -- `data.num_points_to_sample=5000`).
  3. Repère LOCAL pour le CNN : recentre (moyenne, `recenter_pc`) + normalise
     par `max(abs(.))` (même convention Breaking Bad -- le CNN n'a jamais vu
     d'échelle physique absolue). Le CNN est entraîné avec `random_rotate`
     (Step 14/15), donc en principe robuste à l'orientation du scan brut --
     PAS besoin de ré-appliquer une rotation aléatoire, l'orientation native
     du scan suffit.
  4. Repère GLOBAL pour la visualisation : normalise par l'échelle globale
     de l'objet entier (tous fragments confondus) -- garde l'alignement
     approximatif fourni par TU Wien (pas de repère par-fragment).

Pipeline par paire adjacente (déduite par proximité, cf. `infer_adjacency`) :
  masque fracture CNN (thresh0.3) -> `match_depthmaps` (recherche FFT,
  hand-crafted, PAS le modèle appris Phase 8 -- plus robuste hors
  distribution Breaking Bad) -> `build_correspondences_nn` -> `kabsch`.
  Pas de zoom (pas de maillage disponible, juste un nuage de points dense).

Usage (sur le serveur, checkpoint CNN requis) :
    CUDA_VISIBLE_DEVICES=1 python scripts/tuwien_zero_shot_viz.py \\
        --pcd_dir /tmp/student7/tuwien_brick/data/brick/pcd4web \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --n_fragments 6 \\
        --out /tmp/student7/tuwien_brick_viz.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase5a_depthmap_matching import (
    compute_pca_frame, rasterize, match_depthmaps, build_correspondences_nn, kabsch,
)

MAX_PARTS = 20   # padding Breaking Bad standard -- juste besoin de >= nb de fragments réels
RESOLUTION = 64
N_ANGLES = 36
CONTACT_EPS = 0.01   # tolérance de proximité (repère global normalisé) pour déduire l'adjacence
MIN_CONTACT_PTS = 50
FRAC_THRESHOLD = 0.3


def load_pcd(path):
    with open(path) as f:
        n = int(f.readline())
        data = np.loadtxt(f, max_rows=n)
    return data[:, :3].astype(np.float64), data[:, 3:6].astype(np.float64)


def subsample(pts, nrm, n_points, rng):
    if len(pts) <= n_points:
        idx = np.arange(len(pts))
    else:
        idx = rng.choice(len(pts), n_points, replace=False)
    return pts[idx], nrm[idx]


def infer_adjacency(global_pts_list, eps=CONTACT_EPS, min_pts=MIN_CONTACT_PTS):
    """Approxime l'adjacence par proximité sous l'alignement fourni par
    TU Wien (PAS une vérité terrain -- cf. docstring module). Retourne une
    liste de (i, j) avec i < j."""
    trees = [cKDTree(p) for p in global_pts_list]
    pairs = []
    for i in range(len(global_pts_list)):
        for j in range(i + 1, len(global_pts_list)):
            d_ij, _ = trees[j].query(global_pts_list[i])
            d_ji, _ = trees[i].query(global_pts_list[j])
            if int((d_ij < eps).sum()) >= min_pts and int((d_ji < eps).sum()) >= min_pts:
                pairs.append((i, j))
    return pairs


def run_cnn_zero_shot(model, frag_pts_local, frag_nrm_local, device):
    """`frag_pts_local`/`frag_nrm_local` : liste de (n_points, 3) déjà
    recentrés + normalisés PAR FRAGMENT (repère local CNN). Retourne une
    liste de (n_points,) probabilités fracture, même ordre que l'entrée."""
    K = len(frag_pts_local)
    n_points = frag_pts_local[0].shape[0]
    assert all(p.shape[0] == n_points for p in frag_pts_local), (
        "tous les fragments doivent avoir le même nombre de points ici "
        "(padding géré via points_per_part, pas via des tailles variables)"
    )

    pointclouds = np.concatenate(frag_pts_local, axis=0)[None]           # (1, K*n, 3)
    pointclouds_normals = np.concatenate(frag_nrm_local, axis=0)[None]   # (1, K*n, 3)
    points_per_part = np.zeros((1, MAX_PARTS), dtype=np.int64)
    points_per_part[0, :K] = n_points
    fracture_surface_gt = np.zeros((1, K * n_points), dtype=np.int64)   # dummy, jamais utilisé

    batch = {
        "pointclouds": torch.from_numpy(pointclouds).float().to(device),
        "pointclouds_normals": torch.from_numpy(pointclouds_normals).float().to(device),
        "points_per_part": torch.from_numpy(points_per_part).to(device),
        "fracture_surface_gt": torch.from_numpy(fracture_surface_gt).to(device),
    }
    with torch.no_grad():
        out = model(batch)
    scores = out["coarse_seg_pred"].float().cpu().numpy()   # (K*n,)
    return [scores[k * n_points:(k + 1) * n_points] for k in range(K)]


def match_pair(pts_i_global, pts_j_global, score_i, score_j, threshold=FRAC_THRESHOLD):
    """Hand-crafted (`match_depthmaps`), PAS le modèle appris Phase 8 --
    plus susceptible de généraliser hors distribution Breaking Bad. Aucune
    vérité terrain disponible : retourne juste (R_est, t_est, n_corr) ou
    None si le masque est trop épars ou le matching échoue."""
    frac_i = pts_i_global[score_i > threshold]
    frac_j = pts_j_global[score_j > threshold]
    if len(frac_i) < 50 or len(frac_j) < 50:
        return None

    c_i, u_i, v_i, n_i, _ = compute_pca_frame(frac_i)
    c_j, u_j, v_j, n_j, _ = compute_pca_frame(frac_j)
    ci, cj = frac_i - c_i, frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    pixel_size = max(span_i, span_j) * 1.1 / RESOLUTION

    dmap_i, valid_i, u_min_i, v_min_i = rasterize(frac_i, c_i, u_i, v_i, n_i, RESOLUTION, pixel_size)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(frac_j, c_j, u_j, v_j, n_j, RESOLUTION, pixel_size)
    if int(valid_i.sum()) < 20 or int(valid_j.sum()) < 20:
        return None

    best_score, best_theta, best_shift, best_flip, _ = match_depthmaps(
        dmap_i, valid_i, dmap_j, valid_j, N_ANGLES, score_mode="joint")

    pts_i3, pts_j3 = build_correspondences_nn(
        frac_i, c_i, u_i, v_i, frac_j, c_j, u_j, v_j,
        u_min_i, v_min_i, u_min_j, v_min_j,
        pixel_size, RESOLUTION, best_theta, best_shift, contact_eps=0.05,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        return None

    R_est, t_est = kabsch(pts_i3, pts_j3)
    return R_est, t_est, len(pts_i3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_dir", required=True, help="Dossier contenant les .pcd (x y z nx ny nz)")
    parser.add_argument("--pcd_pattern", default="brick_part{:02d}.pcd")
    parser.add_argument("--n_fragments", type=int, required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint CNN Step 15")
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    print(f"Chargement de {args.n_fragments} fragments depuis {args.pcd_dir}...")
    raw_frags = []
    for i in range(1, args.n_fragments + 1):
        path = Path(args.pcd_dir) / args.pcd_pattern.format(i)
        pts, nrm = load_pcd(path)
        pts, nrm = subsample(pts, nrm, args.n_points, rng)
        raw_frags.append((pts, nrm))
        print(f"  fragment {i}: {len(pts)} points (sous-échantillonné)")

    # Repère GLOBAL (visualisation + matching) : normalisation par l'échelle
    # de l'objet ENTIER -- garde l'alignement approximatif fourni par TU Wien.
    all_pts = np.concatenate([p for p, _ in raw_frags], axis=0)
    global_scale = float(np.max(np.abs(all_pts)))
    global_pts = [p / global_scale for p, _ in raw_frags]
    global_nrm = [n for _, n in raw_frags]   # les normales ne changent pas d'échelle

    print(f"Échelle globale = {global_scale:.3f}")
    print("Déduction de l'adjacence par proximité (PAS une vérité terrain)...")
    pairs = infer_adjacency(global_pts)
    print(f"  {len(pairs)} paires adjacentes déduites : {pairs}")

    # Repère LOCAL par fragment (CNN) : recentre (moyenne) + normalise par
    # max(abs(.)) -- même convention que Breaking Bad (recenter_pc + scale).
    local_pts, local_nrm = [], []
    for pts, nrm in raw_frags:
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        scale = np.max(np.abs(pts_c))
        local_pts.append((pts_c / scale).astype(np.float32))
        local_nrm.append(nrm.astype(np.float32))

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    from assembly.models.cnn_segmentation_model import CNNFracSeg
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    print("Inférence CNN zero-shot...")
    scores = run_cnn_zero_shot(model, local_pts, local_nrm, device)
    for i, s in enumerate(scores):
        print(f"  fragment {i+1}: {int((s > FRAC_THRESHOLD).sum())}/{len(s)} points "
              f"prédits fracture (thresh {FRAC_THRESHOLD})")

    print("Matching hand-crafted par paire adjacente...")
    pair_results = []
    for i, j in pairs:
        res = match_pair(global_pts[i], global_pts[j], scores[i], scores[j])
        if res is None:
            print(f"  paire {i+1}-{j+1} : échec (masque trop épars ou matching raté)")
            continue
        R_est, t_est, n_corr = res
        print(f"  paire {i+1}-{j+1} : OK, n_corr={n_corr}")
        pair_results.append({
            "i": i, "j": j, "n_corr": n_corr,
            "R_est": R_est.tolist(), "t_est": t_est.tolist(),
        })

    fragments_out = []
    for i in range(args.n_fragments):
        fragments_out.append({
            "points": global_pts[i].tolist(),
            "frac_score": scores[i].tolist(),
        })

    out = {
        "n_fragments": args.n_fragments,
        "global_scale": global_scale,
        "inferred_adjacency": pairs,
        "fragments": fragments_out,
        "pairs": pair_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nExporté : {args.out}")


if __name__ == "__main__":
    main()
