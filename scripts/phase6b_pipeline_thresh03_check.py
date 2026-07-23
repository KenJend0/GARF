"""
scripts/phase6b_pipeline_thresh03_check.py
=============================================
Version `thresh0.3` (CNN) du pipeline complet bout-en-bout — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7 / Plan étape par étape 2026-07-22.

Chaîne les deux étages validés séparément le 2026-07-22, EN CONDITION RÉELLE
(masque prédit par le CNN, pas l'oracle GT comme dans `phase6b_pipeline_check.py`) :
  Étage 1 : zoom (graines filtrées par clustering, `phase5a_zoom_resample_thresh03_check.py`)
            + cascade de résolution -> pose grossière (R_est, t_est).
  Étage 2 : `trimmed_icp_normals` initialisé par cette pose, sur les VRAIS
            points prédits par le CNN (masque `thresh0.3` brut, pas les
            points de zoom synthétiques -- même logique que la version GT).

Point d'arrêt convenu avec l'utilisateur (2026-07-22) : si le taux de succès
final est faible, ne PAS enchaîner sur autre chose (ex. ordre d'assemblage) --
rester ici, visualiser les cas de succès/échec, et améliorer AVANT de monter
d'un niveau.

Réutilise l'intégralité de la plomberie déjà validée : deux datasets en
lockstep + correction `init_rot` + filtrage par clustering
(`phase5a_zoom_resample_thresh03_check.py`), `zoom_resample`/`to_input_frame`/
`run_cascade` (`phase5a_zoom_resample_check.py`), `trimmed_icp_normals`
(`phase6a_convergence_basin_check.py`) -- aucune réimplémentation.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase6b_pipeline_thresh03_check.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --expand_rings 0 --extra_budget 1200 \\
        --resolution_sweep 128 96 64 48 32 24 20 16 12 \\
        --csv_out /tmp/student7/phase6b_thresh03.csv \\
        --summary_json /tmp/student7/phase6b_thresh03.json
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat, rot_err_deg, trans_err
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame, run_cascade
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase6a_convergence_basin_check import trimmed_icp_normals
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

ABS_MIN_POINTS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[128, 96, 64, 48, 32, 24, 20, 16, 12])
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--extra_budget", type=int, default=1200,
                        help="Budget de zoom fixe pour l'étage 1 (1200 = meilleur compromis "
                             "trouvé en GT le 2026-07-22).")
    parser.add_argument("--expand_rings", type=int, default=0,
                        help="Défaut 0 -- validé 2026-07-22, l'expansion à 1 anneau "
                             "dégradait la précision sans nécessité.")
    parser.add_argument("--max_icp_iters", type=int, default=50)
    parser.add_argument("--trim_ratio", type=float, default=0.7)
    parser.add_argument("--normal_dot_thresh", type=float, default=-0.3)
    parser.add_argument("--success_rot_thresh", type=float, default=5.0)
    parser.add_argument("--success_trans_thresh", type=float, default=0.02)
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── Deux datasets sur le même split, indexés en lockstep (cf.
    # phase5a_zoom_resample_thresh03_check.py pour le détail complet). ─────────
    print("Chargement du dataset CNN (sample_method=uniform, forcé par la config "
          "d'expérience)...")
    cnn_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cnn_cfg = load_config_and_model(cnn_fake_args)
    cnn_datamodule = instantiate(cnn_cfg.data)
    cnn_datamodule.setup("fit" if args.split == "val" else "test")
    cnn_dataset = cnn_datamodule.val_dataset if args.split == "val" else cnn_datamodule.test_dataset

    print("Chargement du dataset mesh (sample_method=weighted, forcé via model_type=garf) "
          "-- UNIQUEMENT pour data['meshes']...")
    mesh_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    mesh_cfg = load_config_and_model(mesh_fake_args)
    mesh_datamodule = instantiate(mesh_cfg.data)
    mesh_datamodule.setup("fit" if args.split == "val" else "test")
    mesh_dataset = mesh_datamodule.val_dataset if args.split == "val" else mesh_datamodule.test_dataset

    assert len(cnn_dataset) == len(mesh_dataset), (
        f"Tailles de dataset différentes ({len(cnn_dataset)} vs {len(mesh_dataset)})."
    )

    cnn_loader = DataLoader(
        cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=cnn_datamodule.dataset_cls.collate_fn,
    )

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    n_stage1_failed = 0
    t0 = time.time()
    print(f"\nPipeline complet (étage 1 + étage 2), thresh0.3/CNN -- "
          f"{args.categories}/{args.split}, extra_budget={args.extra_budget}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                      f"étage1 échoué={n_stage1_failed} | {elapsed:.0f}s écoulées")

            mesh_data = mesh_dataset[idx]
            assert batch["name"][0] == mesh_data["name"], (
                f"Désalignement lockstep à l'index {idx} : "
                f"cnn={batch['name'][0]!r} vs mesh={mesh_data['name']!r}"
            )

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K != 2:
                continue
            n_seen_2frag += 1

            gt_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_gt"], batch_gpu["points_per_part"]
            )
            nrm_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_normals"], batch_gpu["points_per_part"]
            )

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()     for k in range(K)]
            gt_per_k    = [gt_frag_list[k].cpu().numpy()  for k in range(K)]
            nrm_per_k   = [nrm_frag_list[k].cpu().numpy() for k in range(K)]

            valid_pcs_np = valid_pcs.cpu().numpy()
            p_slots = [p for p in range(valid_pcs_np.shape[1]) if valid_pcs_np[0, p]]
            if len(p_slots) != 2:
                continue
            p0, p1 = p_slots

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            quats_np = batch["quaternions"].numpy()
            trans_np = batch["translations"].numpy()

            raw0 = pc_per_k[0] * scale_np[0, p0]
            raw1 = pc_per_k[1] * scale_np[0, p1]
            sc0, sc1 = score_per_k[0], score_per_k[1]
            gtgt0, gtgt1 = gt_per_k[0], gt_per_k[1]
            nrm0, nrm1 = nrm_per_k[0], nrm_per_k[1]

            R0 = quat_wxyz_to_rotmat(quats_np[0, p0])
            R1 = quat_wxyz_to_rotmat(quats_np[0, p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[0, p0] - trans_np[0, p1])

            mask0 = sc0 > args.threshold
            mask1 = sc1 > args.threshold
            frac_i_base = raw0[mask0]
            frac_j_base = raw1[mask1]
            frac_i_nrm  = nrm0[mask0]
            frac_j_nrm  = nrm1[mask1]
            n_min_base = min(len(frac_i_base), len(frac_j_base))
            if n_min_base < max(ABS_MIN_POINTS, 50):
                continue   # tranche <50 mise de côté (2026-07-22), structurellement dure

            # ── Étage 1 : zoom avec graines filtrées par clustering + correction
            # init_rot (cf. phase5a_zoom_resample_thresh03_check.py pour le détail
            # complet du bug de repère et du fix). ─────────────────────────────
            R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
            cluster_mask0 = dominant_cluster_mask(raw0[mask0])
            cluster_mask1 = dominant_cluster_mask(raw1[mask1])
            seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init_rot.T
            seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init_rot.T
            meshes = mesh_data["meshes"]
            new_i_gt = zoom_resample(meshes[p0], seed_i_gt, args.extra_budget,
                                      args.expand_rings, rng)
            new_j_gt = zoom_resample(meshes[p1], seed_j_gt, args.extra_budget,
                                      args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                n_zoom_failed += 1
                continue

            new_i_rotated = new_i_gt @ R_init_rot
            new_j_rotated = new_j_gt @ R_init_rot
            new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
            new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)

            stage1_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, args)
            if stage1_res["reached_stage"] != "pose_computed":
                n_stage1_failed += 1
                continue

            # ── Étage 2 : raffinement point-à-point, initialisé par la pose étage 1,
            # sur les VRAIS points prédits par le CNN (pas les points de zoom). ────
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
          f"2-frags vus ({n_zoom_failed} zooms impossibles, {n_stage1_failed} échecs "
          f"étage 1) en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    n = len(rows)
    n_total_attempted = n_seen_2frag - n_zoom_failed
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
    print(f"\n  Rappel : {n}/{n_total_attempted} objets 2-frags vus "
          f"({100*n/max(n_total_attempted,1):.1f}%) atteignent l'étage 2.")

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
                "n_zoom_failed": n_zoom_failed, "n_stage1_failed": n_stage1_failed,
                "stage1_only": {"pose_30": 100*s1_p30/n, "pose_15": 100*s1_p15/n},
                "stage1_plus_stage2": {"pose_30": 100*f_p30/n, "pose_15": 100*f_p15/n,
                                       "strict_success": 100*f_strict/n},
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
