"""
scripts/phase6b_pipeline_check.py
====================================
Pipeline complet en deux temps, sur les VRAIES poses (pas une perturbation
contrôlée) — voir PLAN_REASSEMBLY_MODULE.md, section Phase 7 / Phase 6A-6B.

Chaîne concrètement :
  Étage 1 : depth-map matching (cascade de résolution + zoom, `ring=0`,
            `phase5a_zoom_resample_check.py` -- validé le 2026-07-22, gain net
            sur l'éligibilité pour la population "facile" n_frac_pts_min>=50).
  Étage 2 : raffinement point-à-point (`trimmed_icp_normals`,
            `phase6a_convergence_basin_check.py` -- validé le 2026-07-22, gain
            x1.7 à x6.8 sur le bassin de convergence par rapport à l'ICP
            vanille, grâce à la pénalité d'orientation contre le glissement
            tangentiel sur les surfaces quasi-planes).

Question posée : quel est le taux de succès RÉEL du pipeline complet (étage 1
tel qu'il tourne vraiment, pas une perturbation propre et contrôlée) ? Le test
de bassin de convergence (Phase 6A) mesurait le raffinement en isolation ;
celui-ci mesure la chaîne bout-en-bout.

Protocole par paire (stratégie GT, oracle-first comme tout Phase 7) :
  1. Masque fracture GT tel quel (`frac_i_base`/`frac_j_base`) + zoom (mêmes
     graines/mécanisme que `phase5a_zoom_resample_check.py`, `ring=0`,
     `--extra_budget` fixe -- pas de sweep ici, on chaîne juste le pipeline).
  2. `run_cascade()` sur le masque zoomé -> pose ÉTAGE 1 (R_est, t_est) si
     `pose_computed`, sinon la paire est un échec total (pas de pose à
     raffiner).
  3. `trimmed_icp_normals()` initialisé par (R_est, t_est), sur le masque GT
     NON zoomé (les vrais points, pas les points de zoom synthétiques -- le
     zoom servait à l'éligibilité de l'étage 1, l'étage 2 a déjà assez de
     vrais points).
  4. Compare : précision étage-1-seul (`Pose@30`/`Pose@15`) vs précision
     étage-1+étage-2 (mêmes seuils, + seuil strict 5°/0.02 du bassin de
     convergence Phase 6A).

Réutilise `zoom_resample`/`to_input_frame`/`run_cascade` de
`phase5a_zoom_resample_check.py`, `trimmed_icp_normals` de
`phase6a_convergence_basin_check.py`, `extract_gt_variable` de
`phase5a_weighted_gt_check.py`, `quat_wxyz_to_rotmat`/`rot_err_deg`/
`trans_err` de `phase5a_depthmap_matching.py` -- aucune réimplémentation.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase6b_pipeline_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --num_points_to_sample 10000 --min_frac_pts 50 \\
        --expand_rings 0 --extra_budget 1200 \\
        --resolution_sweep 128 96 64 48 32 24 20 16 12 \\
        --csv_out /tmp/student7/phase6b_pipeline.csv \\
        --summary_json /tmp/student7/phase6b_pipeline.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat, rot_err_deg, trans_err
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame, run_cascade
from scripts.phase6a_convergence_basin_check import trimmed_icp_normals

ABS_MIN_POINTS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"],
                        help="Nécessite les meshes (conservés en val/test) -- pas 'train'.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--min_frac_pts", type=int, default=50,
                        help="N'inclut que n_frac_pts_min >= ce seuil -- population 'facile' "
                             "(2026-07-22, la tranche <50 est mise de côté, structurellement "
                             "dure même en GT).")
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[128, 96, 64, 48, 32, 24, 20, 16, 12])
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--extra_budget", type=int, default=1200,
                        help="Budget de zoom fixe pour l'étage 1 (validé 2026-07-22, "
                             "pas de plafond net trouvé jusqu'à 3000 en GT, 1200 = bon compromis).")
    parser.add_argument("--expand_rings", type=int, default=0,
                        help="Défaut 0 -- validé 2026-07-22, l'expansion à 1 anneau dégradait "
                             "la précision sans nécessité.")
    parser.add_argument("--max_icp_iters", type=int, default=50)
    parser.add_argument("--trim_ratio", type=float, default=0.7)
    parser.add_argument("--normal_dot_thresh", type=float, default=-0.3)
    parser.add_argument("--success_rot_thresh", type=float, default=5.0,
                        help="Seuil strict du bassin de convergence Phase 6A.")
    parser.add_argument("--success_trans_thresh", type=float, default=0.02)
    parser.add_argument("--correspondence_mode", default="grid", choices=["grid", "nn"],
                        help="'nn' = plus-proche-voisin continu (2026-07-23, "
                             "build_correspondences_nn) au lieu de la coïncidence de grille "
                             "après arrondi -- transmis tel quel à run_match_at_resolution via "
                             "l'objet args (aucun changement requis dans run_cascade).")
    parser.add_argument("--contact_eps", type=float, default=0.05,
                        help="Tolérance de proximité pour --correspondence_mode nn.")
    parser.add_argument("--robust_pca", action="store_true",
                        help="PCA robuste (compute_pca_frame_robust, 2026-07-23) au lieu du "
                             "repère PCA standard -- rejette itérativement les points les plus "
                             "loin du plan avant de fixer le repère, pour ne pas laisser un "
                             "petit amas isolé (faux positifs CNN) biaiser le centre/les axes.")
    parser.add_argument("--robust_pca_iters", type=int, default=3)
    parser.add_argument("--robust_pca_keep_frac", type=float, default=0.9)
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0

    rng = np.random.default_rng(args.seed)
    print("Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          "meshes conservés (split != train), AUCUN CNN chargé (stratégie GT, oracle-first)")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    cfg.data.num_points_to_sample = args.num_points_to_sample
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit" if args.split == "val" else "test")
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.test_dataset

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    rows = []
    n_seen_2frag = 0
    n_skipped_sparse = 0
    n_stage1_failed = 0
    n_zoom_failed = 0
    t0 = time.time()
    print(f"\nPipeline complet (étage 1 + étage 2) -- {args.categories}/{args.split}, "
          f"extra_budget={args.extra_budget}, min_frac_pts={args.min_frac_pts}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                      f"étage1 échoué={n_stage1_failed} | {elapsed:.0f}s écoulées")

            points_per_part = batch["points_per_part"][0].numpy()
            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            pointclouds = batch["pointclouds"][0].numpy()
            pointclouds_normals = batch["pointclouds_normals"][0].numpy()
            pointclouds_gt = batch["pointclouds_gt"][0].numpy()
            fracture_gt = batch["fracture_surface_gt"][0].numpy()

            frag_pts    = extract_gt_variable(pointclouds, points_per_part)
            frag_nrm    = extract_gt_variable(pointclouds_normals, points_per_part)
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
            meshes = batch["meshes"][0]

            raw_i = frag_pts[0] * scale_np[p0]
            raw_j = frag_pts[1] * scale_np[p1]
            nrm_i, nrm_j = frag_nrm[0], frag_nrm[1]
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            frac_i_base = raw_i[gt_i == 1]
            frac_j_base = raw_j[gt_j == 1]
            frac_i_nrm  = nrm_i[gt_i == 1]
            frac_j_nrm  = nrm_j[gt_j == 1]
            n_min_base = min(len(frac_i_base), len(frac_j_base))
            if n_min_base < max(ABS_MIN_POINTS, args.min_frac_pts):
                n_skipped_sparse += 1
                continue

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

            # ── Étage 1 : zoom (mêmes graines/mécanisme que phase5a_zoom_resample_check.py) ──
            seed_i_gt = frag_pts_gt[0][gt_i == 1]
            seed_j_gt = frag_pts_gt[1][gt_j == 1]
            new_i_gt = zoom_resample(meshes[p0], seed_i_gt, args.extra_budget,
                                      args.expand_rings, rng)
            new_j_gt = zoom_resample(meshes[p1], seed_j_gt, args.extra_budget,
                                      args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                n_zoom_failed += 1
                continue
            new_i_input = to_input_frame(new_i_gt, quats_np[p0], trans_np[p0])
            new_j_input = to_input_frame(new_j_gt, quats_np[p1], trans_np[p1])
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)

            stage1_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, args)
            if stage1_res["reached_stage"] != "pose_computed":
                n_stage1_failed += 1
                continue

            # ── Étage 2 : raffinement point-à-point, initialisé par la pose étage 1,
            # sur les VRAIS points GT (pas les points de zoom synthétiques). ──────────
            R_init, t_init = stage1_res["R_est"], stage1_res["t_est"]
            R_final, t_final, converged_early = trimmed_icp_normals(
                frac_i_base, frac_i_nrm, frac_j_base, frac_j_nrm, R_init, t_init,
                max_iters=args.max_icp_iters, trim_ratio=args.trim_ratio,
                normal_dot_thresh=args.normal_dot_thresh,
            )
            re_final = rot_err_deg(R_final, R_ij_gt)
            te_final = trans_err(t_final, t_ij_gt)

            rows.append({
                "n_frac_pts_min": n_min_base,
                "stage1_rot_err": stage1_res["rot_err"],
                "stage1_trans_err": stage1_res["trans_err"],
                "stage1_pose_30": stage1_res["pose_30"],
                "stage1_pose_15": stage1_res["pose_15"],
                "final_rot_err": re_final,
                "final_trans_err": te_final,
                "final_pose_30": bool(re_final < 30.0 and te_final < 0.1),
                "final_pose_15": bool(re_final < 15.0 and te_final < 0.05),
                "final_strict_success": bool(re_final < args.success_rot_thresh
                                              and te_final < args.success_trans_thresh),
            })

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires ayant atteint l'étage 2 sur {n_seen_2frag} objets "
          f"2-frags vus ({n_skipped_sparse} sous le seuil de densité, {n_zoom_failed} zooms "
          f"impossibles, {n_stage1_failed} échecs étage 1) en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    n = len(rows)
    n_total_attempted = n_seen_2frag - n_skipped_sparse - n_zoom_failed  # dénominateur étage 1
    s1_p30 = sum(1 for r in rows if r["stage1_pose_30"])
    s1_p15 = sum(1 for r in rows if r["stage1_pose_15"])
    f_p30  = sum(1 for r in rows if r["final_pose_30"])
    f_p15  = sum(1 for r in rows if r["final_pose_15"])
    f_strict = sum(1 for r in rows if r["final_strict_success"])

    print("COMPARAISON ÉTAGE 1 SEUL vs ÉTAGE 1 + ÉTAGE 2 (parmi les paires ayant atteint "
          "l'étage 2) :")
    print(f"  {'':<30} {'étage 1 seul':>14} {'+ étage 2':>12}")
    print(f"  {'Pose@30°/0.1':<30} {100*s1_p30/n:>13.1f}% {100*f_p30/n:>11.1f}%")
    print(f"  {'Pose@15°/0.05':<30} {100*s1_p15/n:>13.1f}% {100*f_p15/n:>11.1f}%")
    print(f"  {'Succès strict (5°/0.02)':<30} {'—':>14} {100*f_strict/n:>11.1f}%")
    print(f"\n  Rappel : {n}/{n_total_attempted} paires "
          f"({100*n/max(n_total_attempted,1):.1f}%) ont atteint l'étage 2 "
          f"(pose_computed à l'étage 1) -- le reste ({n_total_attempted-n} paires) "
          f"n'a jamais de pose à raffiner, quel que soit l'étage 2.")

    print("\n(Lecture : si Pose@30/Pose@15 montent nettement de 'étage 1 seul' à '+ étage 2', "
          "le raffinement rattrape une partie des poses imprécises de l'étage 1 -- le pipeline "
          "complet fait mieux que chaque étage seul. Le 'succès strict' donne le taux de "
          "reconvergence exacte (5°/0.02), la mesure la plus dure.)")

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV complet sauvegardé ({len(rows)} lignes) : {args.csv_out}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_pairs_stage2": n, "n_2frag_seen": n_seen_2frag,
                "n_skipped_sparse": n_skipped_sparse, "n_zoom_failed": n_zoom_failed,
                "n_stage1_failed": n_stage1_failed,
                "stage1_only": {"pose_30": 100*s1_p30/n, "pose_15": 100*s1_p15/n},
                "stage1_plus_stage2": {"pose_30": 100*f_p30/n, "pose_15": 100*f_p15/n,
                                       "strict_success": 100*f_strict/n},
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
