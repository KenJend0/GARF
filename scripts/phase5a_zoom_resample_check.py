"""
scripts/phase5a_zoom_resample_check.py
=========================================
Phase 5A / Phase 7 du plan de réassemblage — voir PLAN_REASSEMBLY_MODULE.md.

Diagnostic du "zoom + rééchantillonnage local" (2026-07-22, idée de
l'utilisateur, suite à la stratification fine par densité de
`phase5a_skip_audit.py`) : la stratification a montré que `no_correspondence`
est massivement dominé par la sparsité (n_frac_pts_min médiane=64 pour
no_correspondence vs 257-335 pour les paires qui produisent une pose, PAS un
problème de courbure). L'utilisateur a explicitement écarté d'augmenter
`num_points_to_sample` globalement (gaspille le budget sur tout le
fragment) — l'idée testée ici : au lieu d'élargir le budget global, zoomer
spatialement sur la zone de fracture déjà repérée (GT ici, CNN plus tard) et
y rééchantillonner densément, en PLUS de l'échantillonnage normal.

Mécanisme (confirmé faisable en inspectant `assembly/data/breaking_bad/`) :
- Chaque fragment est stocké comme un MAILLAGE complet (`data["meshes"]`,
  conservé en val/test, supprimé en train pour la mémoire — pas un problème
  ici, tout Phase 5A/7 tourne sur val).
- `pointclouds_gt`/`fracture_surface_gt` (dans le batch) sont dans le repère
  du maillage (AVANT recentrage/rotation/rescale) -- donc les points
  fracture déjà échantillonnés servent de "graine" pour localiser la zone
  sur le maillage. Face la plus proche de chaque point : `cKDTree` (scipy)
  sur `mesh.triangles_center` -- PAS `trimesh.proximity.closest_point`
  (dépend de `rtree`, absent de l'environnement serveur ; l'approximation
  "centroïde de face le plus proche" suffit ici, on veut juste délimiter un
  voisinage, pas une projection géométrique exacte) -- sans avoir besoin de
  l'indice de face d'origine (jeté après `sample_points()`, jamais stocké
  dans `data`).
- On étend le jeu de faces "zoom" par 1 anneau d'adjacence
  (`mesh.face_adjacency`), on construit un `face_weight` nul partout sauf sur
  ces faces (pondéré par leur aire), et on tire nouveaux points via
  `trimesh.sample.sample_surface(mesh, count=EXTRA, face_weight=...)` --
  RESTREINT à la zone, indépendamment du budget global du fragment.
- Les nouveaux points (repère maillage/GT) sont ramenés dans le repère
  "input" (celui de `raw_i` dans les autres scripts Phase 5A) via la formule
  de reconstruction de la Phase 0 : `q = (pts_gt - translation) @ R(quat)`
  (inverse de `pointclouds_gt ≈ R(quat) @ q + translation`).

Compare, pour les MÊMES paires, deux conditions avec la cascade de résolution
déjà validée (`run_match_at_resolution`, de `phase5a_depthmap_matching.py`) :
  - baseline : masque fracture GT tel quel (comportement actuel)
  - zoomed   : masque fracture GT + points supplémentaires zoomés sur la
               même zone (mesh, pas de fuite de label -- la zone est
               localisée à partir des points fracture déjà connus, pas
               d'information nouvelle "trichée")

Mise à jour 2026-07-22 : `--extra_budget_sweep` teste PLUSIEURS budgets dans
la MÊME passe du dataloader (au lieu de relancer tout le run pour chaque
valeur, ce qui revenait à tâtonner) -- le budget le plus grand est tiré UNE
FOIS par fragment, les budgets plus petits prennent un sous-ensemble de ce
même tirage (statistiquement valide, pas besoin de retirer).

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase5a_zoom_resample_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --num_points_to_sample 10000 --expand_rings 0 \\
        --extra_budget_sweep 200 500 800 1200 \\
        --resolution_sweep 128 96 64 48 32 24 20 16 12 \\
        --csv_out /tmp/student7/phase5a_zoom_sweep.csv \\
        --summary_json /tmp/student7/phase5a_zoom_sweep.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import (
    compute_pca_frame, compute_pca_frame_robust, quat_wxyz_to_rotmat, run_match_at_resolution,
)

ABS_MIN_POINTS = 5

# Tranches identiques à phase5a_skip_audit.py (density_outcome_stratification)
# pour comparer directement baseline vs zoomed sur les mêmes bins.
DENSITY_BINS = [0, 50, 75, 100, 150, 200, 300, 500, 1000, 10**9]
DENSITY_LABELS = ["<50", "50-75", "75-100", "100-150", "150-200",
                  "200-300", "300-500", "500-1000", "1000+"]


def expand_faces_by_adjacency(mesh, face_idx, n_rings=1):
    """Étend un ensemble de faces de `n_rings` anneaux via l'adjacence du
    maillage -- couvre un peu plus que les seules faces déjà touchées par
    l'échantillon épars actuel (qui peut rater des faces fracture voisines
    juste par manque de chance du tirage aléatoire)."""
    face_set = set(int(f) for f in face_idx)
    adjacency = mesh.face_adjacency
    for _ in range(n_rings):
        new_faces = set()
        for a, b in adjacency:
            a, b = int(a), int(b)
            if a in face_set and b not in face_set:
                new_faces.add(b)
            elif b in face_set and a not in face_set:
                new_faces.add(a)
        face_set |= new_faces
    return np.array(sorted(face_set))


def zoom_resample(mesh, seed_pts_gt_frame, extra_budget, expand_rings, rng):
    """Localise la zone de fracture sur le maillage à partir des points
    fracture DÉJÀ connus (`seed_pts_gt_frame`, repère maillage), l'étend d'un
    anneau d'adjacence, et tire `extra_budget` nouveaux points RESTREINTS à
    cette zone (pondérés par aire, comme le reste du pipeline).
    Retourne les nouveaux points en repère maillage/GT, ou None si la zone
    est vide (ne devrait pas arriver si seed_pts_gt_frame est non-vide)."""
    if len(seed_pts_gt_frame) == 0:
        return None
    # cKDTree sur les centroïdes de faces plutôt que trimesh.proximity.closest_point
    # (qui dépend de `rtree`, absent de l'environnement du serveur) -- approximation
    # "face la plus proche du point" au lieu du point exact sur le triangle, largement
    # suffisant pour localiser la zone (on veut juste "quelles faces sont dans le
    # voisinage", pas une projection géométrique exacte).
    tree = cKDTree(mesh.triangles_center)
    _, seed_face_idx = tree.query(seed_pts_gt_frame)
    zoom_face_idx = expand_faces_by_adjacency(mesh, np.unique(seed_face_idx), expand_rings)
    if len(zoom_face_idx) == 0:
        return None

    face_weight = np.zeros(len(mesh.faces), dtype=np.float64)
    face_weight[zoom_face_idx] = mesh.area_faces[zoom_face_idx]
    if face_weight.sum() <= 0:
        return None

    seed = int(rng.integers(0, 2**31 - 1))
    new_pts_gt, _ = trimesh.sample.sample_surface(
        mesh, count=extra_budget, face_weight=face_weight, seed=seed)
    return np.asarray(new_pts_gt)


def zoom_resample_with_normals(mesh, seed_pts_gt_frame, extra_budget, expand_rings, rng):
    """Identique à `zoom_resample()`, mais retourne AUSSI les normales des
    nouveaux points échantillonnés (Phase 8, features 3D, 2026-07-31) --
    `mesh.face_normals[face_idx]`, même convention que `sample_points()`
    (`assembly/data/breaking_bad/weighted.py`, `meshes[i].face_normals[pcd[1]]`).

    Duplique volontairement `zoom_resample()` (pas un refactor partagé avec
    un flag) -- `zoom_resample()` est appelée telle quelle par plusieurs
    scripts déjà validés (phase6b_pipeline_*, phase8_pipeline_learned_check.py,
    phase8_eligibility_diagnostic.py) avec un seul retour (`np.ndarray`) ;
    changer sa signature casserait tous ces appels. Retourne
    `(new_pts_gt, new_normals_gt)` ou `(None, None)` si la zone est vide."""
    if len(seed_pts_gt_frame) == 0:
        return None, None
    tree = cKDTree(mesh.triangles_center)
    _, seed_face_idx = tree.query(seed_pts_gt_frame)
    zoom_face_idx = expand_faces_by_adjacency(mesh, np.unique(seed_face_idx), expand_rings)
    if len(zoom_face_idx) == 0:
        return None, None

    face_weight = np.zeros(len(mesh.faces), dtype=np.float64)
    face_weight[zoom_face_idx] = mesh.area_faces[zoom_face_idx]
    if face_weight.sum() <= 0:
        return None, None

    seed = int(rng.integers(0, 2**31 - 1))
    new_pts_gt, new_face_idx = trimesh.sample.sample_surface(
        mesh, count=extra_budget, face_weight=face_weight, seed=seed)
    new_normals_gt = mesh.face_normals[new_face_idx]
    return np.asarray(new_pts_gt), np.asarray(new_normals_gt)


def normals_to_input_frame(normals_gt, quat_wxyz):
    """Comme `to_input_frame()`, mais pour des NORMALES (vecteurs directions,
    pas des positions) -- pas de soustraction de `translation` (une
    translation ne change pas une direction), seulement la rotation :
    `n_input = n_gt @ R(quat)` (même formule que `to_input_frame`, sans le
    terme `- translation`)."""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return normals_gt @ R


def to_input_frame(pts_gt, quat_wxyz, translation):
    """Repère maillage/GT -> repère 'input' (celui de raw_i dans les autres
    scripts Phase 5A). Inverse de la formule de reconstruction Phase 0 :
    pointclouds_gt ≈ R(quat) @ q + translation  =>  q = (pts_gt - t) @ R(quat)
    (transposée batch-wise : (R.T @ v.T).T = v @ R)."""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return (pts_gt - translation) @ R


def run_cascade(frac_i, frac_j, R_ij_gt, t_ij_gt, args):
    """Pipeline PCA -> cascade de résolution -> pose, pour UN masque fracture
    donné (baseline ou zoomed). Retourne un dict résultat, comme
    phase5a_skip_audit.process_pair_audit mais sans les champs would_skip_*
    (pas leur objet ici -- ce script compare deux CONDITIONS, pas des seuils)."""
    n_i, n_j = len(frac_i), len(frac_j)
    if n_i < ABS_MIN_POINTS or n_j < ABS_MIN_POINTS:
        return {"reached_stage": "unusable_too_few", "n_frac_pts_min": min(n_i, n_j)}

    # `robust_pca` (2026-07-23) : PCA standard (défaut, comportement historique
    # inchangé) ou variante robuste (compute_pca_frame_robust -- rejet itératif
    # des points les plus loin du plan avant de fixer le repère). Motivé par
    # phase7_mask_confusion_viz.py : petits amas de faux positifs CNN isolés
    # de la vraie fracture, trop peu nombreux pour dégrader précision/rappel
    # mais suffisants pour biaiser le repère PCA. `getattr` rétrocompatible --
    # les scripts qui n'exposent pas cette option gardent le comportement
    # standard sans changement.
    if getattr(args, "robust_pca", False):
        c_i, u_i, v_i, n_i_ax, _ = compute_pca_frame_robust(
            frac_i, n_iters=getattr(args, "robust_pca_iters", 3),
            keep_frac=getattr(args, "robust_pca_keep_frac", 0.9))
        c_j, u_j, v_j, n_j_ax, _ = compute_pca_frame_robust(
            frac_j, n_iters=getattr(args, "robust_pca_iters", 3),
            keep_frac=getattr(args, "robust_pca_keep_frac", 0.9))
    else:
        c_i, u_i, v_i, n_i_ax, _ = compute_pca_frame(frac_i)
        c_j, u_j, v_j, n_j_ax, _ = compute_pca_frame(frac_j)

    ci, cj = frac_i - c_i, frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)

    resolutions_desc = sorted(args.resolution_sweep, reverse=True)
    result = {"reached_stage": "no_correspondence", "n_frac_pts_min": min(n_i, n_j),
              "resolution_used": None, "pose_30": False, "pose_15": False,
              "rot_err": None, "trans_err": None, "R_est": None, "t_est": None}
    for r in resolutions_desc:
        res_r = run_match_at_resolution(
            frac_i, c_i, u_i, v_i, n_i_ax, frac_j, c_j, u_j, v_j, n_j_ax,
            span_i, span_j, r, args.n_angles, args, R_ij_gt, t_ij_gt,
        )
        if "skip" not in res_r:
            result["reached_stage"] = "pose_computed"
            result["resolution_used"] = r
            result["rot_err"] = res_r["rot_err"]
            result["trans_err"] = res_r["trans_err"]
            result["pose_30"] = bool(res_r["pose_success"]["30deg_0.1"])
            result["pose_15"] = bool(res_r["pose_success"]["15deg_0.05"])
            # R_est/t_est (2026-07-22, Phase 6B) : la pose estimée par ce stade,
            # utilisable comme initialisation d'un raffinement point-à-point.
            result["R_est"] = res_r["R_est"]
            result["t_est"] = res_r["t_est"]
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"],
                        help="Nécessite les meshes (conservés en val/test, "
                             "supprimés en train pour la mémoire) -- pas de choix 'train'.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[128, 96, 64, 48, 32, 24, 20, 16, 12])
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--extra_budget", type=int, default=500,
                        help="Nombre de points supplémentaires tirés sur la zone zoomée, "
                             "PAR FRAGMENT, en plus de l'échantillonnage normal. Ignoré si "
                             "--extra_budget_sweep est donné.")
    parser.add_argument("--extra_budget_sweep", type=int, nargs="+", default=[],
                        help="Teste PLUSIEURS budgets dans la MÊME passe du dataloader "
                             "(2026-07-22, évite de relancer tout le run pour chaque valeur "
                             "-- même principe que --pose_resolution_sweep). Le budget le "
                             "plus grand est tiré UNE FOIS ; les budgets plus petits prennent "
                             "un sous-ensemble de ce même tirage (statistiquement valide -- "
                             "un sous-ensemble d'un tirage i.i.d. est un tirage i.i.d. valide "
                             "de cette taille, pas besoin de retirer). Remplace --extra_budget "
                             "si fourni.")
    parser.add_argument("--expand_rings", type=int, default=1,
                        help="Anneaux d'adjacence de faces pour étendre la zone zoom "
                             "au-delà des seules faces déjà touchées par l'échantillon épars.")
    parser.add_argument("--csv_out", default="", help="Dump complet, une ligne par paire.")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0

    budgets = sorted(args.extra_budget_sweep) if args.extra_budget_sweep else [args.extra_budget]
    max_budget = budgets[-1]

    rng = np.random.default_rng(args.seed)

    print("Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          "meshes conservés (split != train), AUCUN CNN chargé")
    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    cfg.data.num_points_to_sample = args.num_points_to_sample
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    t0 = time.time()
    print(f"\nDiagnostic zoom + rééchantillonnage local -- "
          f"{args.categories}/{args.split}, budgets={budgets}...\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx} | objets 2-frags vus={n_seen_2frag} | "
                      f"zoom échoué={n_zoom_failed} | {elapsed:.0f}s écoulées")

            points_per_part = batch["points_per_part"][0].numpy()
            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            pointclouds = batch["pointclouds"][0].numpy()
            pointclouds_gt = batch["pointclouds_gt"][0].numpy()
            fracture_gt = batch["fracture_surface_gt"][0].numpy()

            frag_pts    = extract_gt_variable(pointclouds, points_per_part)
            frag_pts_gt = extract_gt_variable(pointclouds_gt, points_per_part)
            frag_gt     = extract_gt_variable(fracture_gt, points_per_part)
            if len(frag_pts) != 2:
                continue

            p0, p1 = valid_slots[0], valid_slots[1]
            scale_np = batch["scale"][0].numpy()
            if scale_np.ndim == 1:
                scale_np = scale_np[:, None]
            quats_np = batch["quaternions"][0].numpy()
            trans_np = batch["translations"][0].numpy()
            meshes = batch["meshes"][0]   # liste de trimesh, indexée comme points_per_part

            raw_i = frag_pts[0] * scale_np[p0]
            raw_j = frag_pts[1] * scale_np[p1]
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij = R1.T @ R0
            t_ij = R1.T @ (trans_np[p0] - trans_np[p1])

            frac_i_base = raw_i[gt_i == 1]
            frac_j_base = raw_j[gt_j == 1]
            n_min_base = min(len(frac_i_base), len(frac_j_base))
            if n_min_base < ABS_MIN_POINTS:
                continue

            baseline_res = run_cascade(frac_i_base, frac_j_base, R_ij, t_ij, args)

            # ── Zoom : localise la zone fracture sur CHAQUE maillage à partir des
            # points fracture déjà connus (repère GT/maillage), tire UNE FOIS le
            # budget le plus grand du sweep, ramène en repère input, concatène.
            # Les budgets plus petits prennent un sous-ensemble de CE MÊME tirage
            # (cf. --extra_budget_sweep : statistiquement valide, évite de retirer
            # et de repasser tout le dataloader pour chaque valeur testée). ──────
            seed_i_gt = frag_pts_gt[0][gt_i == 1]
            seed_j_gt = frag_pts_gt[1][gt_j == 1]
            new_i_gt_max = zoom_resample(meshes[p0], seed_i_gt, max_budget,
                                          args.expand_rings, rng)
            new_j_gt_max = zoom_resample(meshes[p1], seed_j_gt, max_budget,
                                          args.expand_rings, rng)
            if new_i_gt_max is None or new_j_gt_max is None:
                n_zoom_failed += 1
                continue

            row = {
                "n_frac_pts_min_base": n_min_base,
                "base_stage": baseline_res["reached_stage"],
                "base_pose_30": baseline_res.get("pose_30", False),
                "zoom_by_budget": {},
            }
            for k in budgets:
                new_i_input = to_input_frame(new_i_gt_max[:k], quats_np[p0], trans_np[p0])
                new_j_input = to_input_frame(new_j_gt_max[:k], quats_np[p1], trans_np[p1])
                frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
                frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)
                zoomed_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij, t_ij, args)
                row["zoom_by_budget"][k] = {
                    "n_frac_pts_min": zoomed_res["n_frac_pts_min"],
                    "stage": zoomed_res["reached_stage"],
                    "pose_30": zoomed_res.get("pose_30", False),
                }
            rows.append(row)

    elapsed = time.time() - t0
    report_zoom_sweep(rows, budgets, args, n_seen_2frag, n_zoom_failed, elapsed)


def report_zoom_sweep(rows, budgets, args, n_seen_2frag, n_zoom_failed, elapsed):
    """Imprime toutes les tables de comparaison baseline vs zoomed (par budget)
    et sauvegarde CSV/JSON -- factorisé pour être réutilisé identiquement par
    le diagnostic GT (`main()` ci-dessus) et par la version `thresh0.3`/CNN
    (`phase5a_zoom_resample_thresh03_check.py`, 2026-07-22), qui produisent
    exactement la même structure de `rows` (n_frac_pts_min_base, base_stage,
    base_pose_30, zoom_by_budget[k] = {n_frac_pts_min, stage, pose_30})."""
    print(f"\nFini : {len(rows)} paires comparées ({n_seen_2frag} objets 2-frags vus, "
          f"{n_zoom_failed} zooms impossibles -- zone fracture introuvable sur le maillage) "
          f"en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    # ── Comparaison globale baseline vs zoomed, POUR CHAQUE budget du sweep ─────
    n = len(rows)
    base_no_corr = sum(1 for r in rows if r["base_stage"] == "no_correspondence")
    base_pose = sum(1 for r in rows if r["base_stage"] == "pose_computed")
    base_p30 = sum(1 for r in rows if r["base_pose_30"])
    print(f"BASELINE (N={n}) : no_correspondence={100*base_no_corr/n:.1f}%  "
          f"pose_computed={100*base_pose/n:.1f}%  Pose@30={100*base_p30/n:.1f}%")

    idx = np.digitize([r["n_frac_pts_min_base"] for r in rows], DENSITY_BINS[1:-1])
    summary_by_budget = {}

    for k in budgets:
        zoom_no_corr = sum(1 for r in rows if r["zoom_by_budget"][k]["stage"] == "no_correspondence")
        zoom_pose = sum(1 for r in rows if r["zoom_by_budget"][k]["stage"] == "pose_computed")
        zoom_p30 = sum(1 for r in rows if r["zoom_by_budget"][k]["pose_30"])

        print(f"\n=== extra_budget={k} ===")
        print("COMPARAISON GLOBALE (mêmes N paires, baseline vs zoomed) :")
        print(f"  {'':<20} {'baseline':>10} {'zoomed':>10}")
        print(f"  {'no_correspondence':<20} {100*base_no_corr/n:>9.1f}% {100*zoom_no_corr/n:>9.1f}%")
        print(f"  {'pose_computed':<20} {100*base_pose/n:>9.1f}% {100*zoom_pose/n:>9.1f}%")
        print(f"  {'Pose@30 (/ total)':<20} {100*base_p30/n:>9.1f}% {100*zoom_p30/n:>9.1f}%")

        # ── Par tranche de densité BASELINE (avant zoom) : où le zoom aide-t-il le
        # plus ? Compare directement à density_outcome_stratification de
        # phase5a_skip_audit.py (mêmes tranches). ───────────────────────────────
        print(f"  PAR TRANCHE DE DENSITÉ BASELINE (n_frac_pts_min AVANT zoom) :")
        header = (f"    {'Bin':<10} {'N':>5} {'NoCorr(base)':>13} {'NoCorr(zoom)':>13} "
                  f"{'Pose30(base)':>13} {'Pose30(zoom)':>13}")
        print(header)
        summary_bins = {}
        for b, label in enumerate(DENSITY_LABELS):
            bin_rows = [r for r, i in zip(rows, idx) if i == b]
            nb = len(bin_rows)
            if nb == 0:
                print(f"    {label:<10} {'—':>5}")
                continue
            b_nc = 100 * sum(1 for r in bin_rows if r["base_stage"] == "no_correspondence") / nb
            z_nc = 100 * sum(1 for r in bin_rows if r["zoom_by_budget"][k]["stage"] == "no_correspondence") / nb
            b_p3 = 100 * sum(1 for r in bin_rows if r["base_pose_30"]) / nb
            z_p3 = 100 * sum(1 for r in bin_rows if r["zoom_by_budget"][k]["pose_30"]) / nb
            print(f"    {label:<10} {nb:>5} {b_nc:>12.1f}% {z_nc:>12.1f}% "
                  f"{b_p3:>12.1f}% {z_p3:>12.1f}%")
            summary_bins[label] = {"n": nb, "no_corr_base": b_nc, "no_corr_zoom": z_nc,
                                    "pose30_base": b_p3, "pose30_zoom": z_p3}

        # ── Composition vs dégradation (2026-07-22) : la baisse de Pose@30 global
        # vient-elle juste du fait que PLUS de paires sont comptées (les nouvelles
        # récupérées sont intrinsèquement plus dures), ou le zoom abîme-t-il aussi
        # la précision des paires qui marchaient DÉJÀ en baseline ? ─────────────
        base_eligible = [r for r in rows if r["base_stage"] == "pose_computed"]
        newly_rescued = [r for r in rows if r["base_stage"] != "pose_computed"
                          and r["zoom_by_budget"][k]["stage"] == "pose_computed"]
        print(f"  COMPOSITION vs DÉGRADATION :")
        composition = {}
        if base_eligible:
            nbe = len(base_eligible)
            be_base_p30 = 100 * sum(1 for r in base_eligible if r["base_pose_30"]) / nbe
            be_zoom_p30 = 100 * sum(1 for r in base_eligible if r["zoom_by_budget"][k]["pose_30"]) / nbe
            print(f"    Paires DÉJÀ éligibles en baseline (N={nbe}) : "
                  f"Pose@30 base={be_base_p30:.1f}% -> zoomed={be_zoom_p30:.1f}%")
            composition = {"base_eligible_n": nbe, "base_eligible_pose30_base": be_base_p30,
                           "base_eligible_pose30_zoom": be_zoom_p30}
        if newly_rescued:
            nnr = len(newly_rescued)
            nr_p30 = 100 * sum(1 for r in newly_rescued if r["zoom_by_budget"][k]["pose_30"]) / nnr
            print(f"    Paires récupérées PAR le zoom (N={nnr}) : Pose@30 zoomed={nr_p30:.1f}%")
            composition["newly_rescued_n"] = nnr
            composition["newly_rescued_pose30_zoom"] = nr_p30

        summary_by_budget[str(k)] = {
            "no_correspondence_zoom": 100*zoom_no_corr/n,
            "pose_computed_zoom": 100*zoom_pose/n,
            "pose_30_zoom": 100*zoom_p30/n,
            "by_density_bin": summary_bins,
            "composition_vs_degradation": composition,
        }

    print("\n(Lecture : pour chaque budget, le zoom vaut la peine si NoCorr(zoom) << NoCorr(base)\n"
          " et Pose30(zoom) >= Pose30(base) SPÉCIFIQUEMENT sur les tranches basses (<150).\n"
          " Comparer les 'COMPOSITION vs DÉGRADATION' entre budgets : si la ligne 'déjà\n"
          " éligibles' se dégrade moins à un budget qu'à un autre, ce budget préserve mieux\n"
          " la précision sans sacrifier l'éligibilité -- chercher le meilleur compromis sur\n"
          " l'ensemble des budgets testés, pas juste le premier qui améliore l'éligibilité.)")

    if args.csv_out:
        import csv
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        _known_keys = {"n_frac_pts_min_base", "base_stage", "base_pose_30", "zoom_by_budget"}
        flat_rows = []
        for r in rows:
            flat = {"n_frac_pts_min_base": r["n_frac_pts_min_base"],
                    "base_stage": r["base_stage"], "base_pose_30": r["base_pose_30"]}
            # Passe automatiquement les champs additionnels que certains appelants
            # ajoutent (ex. n_seed_raw_i/j, n_seed_clustered_i/j dans
            # phase5a_zoom_resample_thresh03_check.py) -- pas de couplage explicite
            # nécessaire entre les deux scripts.
            for extra_key, extra_val in r.items():
                if extra_key not in _known_keys:
                    flat[extra_key] = extra_val
            for k in budgets:
                zb = r["zoom_by_budget"][k]
                flat[f"zoom_b{k}_n_frac_pts_min"] = zb["n_frac_pts_min"]
                flat[f"zoom_b{k}_stage"] = zb["stage"]
                flat[f"zoom_b{k}_pose_30"] = zb["pose_30"]
            flat_rows.append(flat)
        fieldnames = list(flat_rows[0].keys())
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"\nCSV complet sauvegardé ({len(flat_rows)} lignes) : {args.csv_out} "
              f"(permet de recalculer d'autres croisements sans relancer le run)")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_pairs": n, "n_zoom_failed": n_zoom_failed,
                "budgets": budgets,
                "baseline": {"no_correspondence": 100*base_no_corr/n,
                             "pose_computed": 100*base_pose/n, "pose_30": 100*base_p30/n},
                "by_budget": summary_by_budget,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
