"""
scripts/phase9_lowdensity_zoom_check.py
========================================
Phase 9 (PLAN_REASSEMBLY_MODULE.md) -- teste si le zoom/rééchantillonnage
(`zoom_resample_with_normals`, Phase 5A) peut sauver les paires
actuellement écartées AVANT même d'être tentées, parce que le masque CNN
brut combiné (i+j) a moins de `ABS_MIN_POINTS=50` points
(`phase8_pipeline_learned_check.py`/`phase9_eligibility_funnel_check.py` :
33.5% de TOUTES les paires, le principal contributeur à l'inéligibilité,
loin devant `no_correspondence` à 6.9%).

Motivation (2026-08-03, question de l'utilisateur) : le zoom a été conçu
précisément pour densifier les zones de fracture pauvres -- mais dans le
pipeline actuel, le seuil de 50 points est appliqué sur le masque CNN BRUT,
AVANT le zoom, donc le zoom n'est JAMAIS tenté sur ces paires. Un audit
antérieur (`phase5a_skip_audit.py`, 2026-07-2x) avait déjà trouvé
"too_few_points (50 pts) : 227 rejetées -> 0% auraient réussi -> seuil
JUSTIFIÉ" -- mais CET audit précède l'introduction du zoom (il testait
juste "forcer la paire brute à passer", pas "forcer la paire jusqu'au
zoom puis voir si le zoom la sauve"). Question distincte, jamais testée
dans la configuration actuelle.

Ce script : pour les paires dont le masque brut a entre `--min_points_floor`
(défaut 5, même convention que `phase5a_skip_audit.py`) et 50 points
(actuellement écartées), tente quand même `dominant_cluster_mask` +
`zoom_resample_with_normals` + étage 1 appris + `build_correspondences_nn`
(résolution/contact_eps par défaut, mêmes réglages que le pipeline
principal) -- et mesure, PAS SEULEMENT l'éligibilité récupérée, mais le
taux de Pose@30 réel parmi les récupérées (même exigence que
`phase9_stage1_recovery_check.py` : une correspondance numérique n'est
utile que si la pose qui en sort est plausible). Résultat détaillé par
tranche de densité brute (5-20 / 20-35 / 35-49) pour voir s'il existe un
seuil plus bas mais encore raisonnable.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase9_lowdensity_zoom_check.py \\
        --regressor_ckpt output/phase8_depthmap_regressor_thresh03_mirrorhead/best.ckpt \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat
from scripts.phase5a_zoom_resample_check import (
    zoom_resample_with_normals, to_input_frame, normals_to_input_frame,
)
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase9_stage1_recovery_check import (
    load_regressor, run_stage1_at_resolution, _try_correspondences,
    BASE_RESOLUTION, BASE_CONTACT_EPS,
)

PIPELINE_MIN_POINTS = 50   # seuil actuel du pipeline (avant zoom)
DENSITY_BINS = [0, 20, 35, 50]   # tranches de densité brute à l'intérieur de la zone testée
BIN_LABELS = ["5-20", "20-35", "35-49"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor_ckpt", required=True)
    parser.add_argument("--ckpt",       required=True, help="Checkpoint CNN (segmentation)")
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split",      default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--min_points_floor", type=int, default=5,
                         help="Plancher numérique pur (même convention que "
                              "phase5a_skip_audit.py) -- en dessous, on ne tente même pas.")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    regressor = load_regressor(args.regressor_ckpt, device)

    from torch.utils.data import DataLoader
    from assembly.models.cnn_segmentation_model import CNNFracSeg
    from assembly.models.projection_mapping_utils import extract_fragment_list

    print("Chargement du dataset CNN (sample_method=uniform)...")
    cnn_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cnn_cfg = load_config_and_model(cnn_fake_args)
    cnn_datamodule = instantiate(cnn_cfg.data)
    cnn_datamodule.setup("fit" if args.split == "val" else "test")
    cnn_dataset = cnn_datamodule.val_dataset if args.split == "val" else cnn_datamodule.test_dataset

    print("Chargement du dataset mesh (sample_method=weighted) -- meshes uniquement...")
    mesh_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    mesh_cfg = load_config_and_model(mesh_fake_args)
    mesh_datamodule = instantiate(mesh_cfg.data)
    mesh_datamodule.setup("fit" if args.split == "val" else "test")
    mesh_dataset = mesh_datamodule.val_dataset if args.split == "val" else mesh_datamodule.test_dataset

    assert len(cnn_dataset) == len(mesh_dataset)
    cnn_loader = DataLoader(cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=cnn_datamodule.dataset_cls.collate_fn)

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    cnn_model.eval()
    cnn_model.to(device)

    n_targeted = 0          # paires dans la zone [min_points_floor, 50)
    n_below_floor = 0        # < min_points_floor, jamais tenté (trop dégénéré)
    n_zoom_failed = 0
    n_no_correspondence = 0
    n_pose_computed = 0
    n_pose30 = 0
    rows = []   # (n_min_raw, outcome, pose30) pour la ventilation par tranche
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | ciblées={n_targeted} | {time.time()-t0:.0f}s")

            mesh_data = mesh_dataset[idx]
            assert batch["name"][0] == mesh_data["name"]

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"])
            if K != 2:
                continue

            gt_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_gt"], batch_gpu["points_per_part"])
            nrm_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_normals"], batch_gpu["points_per_part"])

            out = cnn_model(batch_gpu)
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
            n_min_raw = min(int(mask0.sum()), int(mask1.sum()))

            # -- on cible EXACTEMENT la population actuellement écartée par
            # le pipeline principal (< PIPELINE_MIN_POINTS), mais au-dessus
            # du plancher numérique pur --
            if n_min_raw >= PIPELINE_MIN_POINTS:
                continue
            if n_min_raw < args.min_points_floor:
                n_below_floor += 1
                continue
            if mask0.sum() < 1 or mask1.sum() < 1:
                continue

            n_targeted += 1

            cluster_mask0 = dominant_cluster_mask(raw0[mask0])
            cluster_mask1 = dominant_cluster_mask(raw1[mask1])
            R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
            seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init_rot.T
            seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init_rot.T
            meshes = mesh_data["meshes"]
            new_i_gt, new_i_nrm_gt = zoom_resample_with_normals(
                meshes[p0], seed_i_gt, args.extra_budget, args.expand_rings, rng)
            new_j_gt, new_j_nrm_gt = zoom_resample_with_normals(
                meshes[p1], seed_j_gt, args.extra_budget, args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                n_zoom_failed += 1
                rows.append((n_min_raw, "zoom_failed", False))
                continue
            new_i_rotated = new_i_gt @ R_init_rot
            new_j_rotated = new_j_gt @ R_init_rot
            new_i_nrm_rotated = new_i_nrm_gt @ R_init_rot
            new_j_nrm_rotated = new_j_nrm_gt @ R_init_rot
            new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
            new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
            new_i_nrm_input = normals_to_input_frame(new_i_nrm_rotated, quats_np[0, p0])
            new_j_nrm_input = normals_to_input_frame(new_j_nrm_rotated, quats_np[0, p1])
            frac_i_base, frac_j_base = raw0[mask0], raw1[mask1]
            frac_i_nrm, frac_j_nrm = nrm0[mask0], nrm1[mask1]
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)
            nrm_i_zoom = np.concatenate([frac_i_nrm, new_i_nrm_input], axis=0)
            nrm_j_zoom = np.concatenate([frac_j_nrm, new_j_nrm_input], axis=0)

            if len(frac_i_zoom) < 3 or len(frac_j_zoom) < 3:
                n_zoom_failed += 1
                rows.append((n_min_raw, "zoom_failed", False))
                continue

            full_centroid_i = raw0.mean(axis=0)
            full_centroid_j = raw1.mean(axis=0)

            res = run_stage1_at_resolution(regressor, frac_i_zoom, frac_j_zoom,
                                            nrm_i_zoom, nrm_j_zoom, device,
                                            BASE_RESOLUTION, full_centroid_i, full_centroid_j)
            if res is None:
                n_zoom_failed += 1
                rows.append((n_min_raw, "zoom_failed", False))
                continue
            theta_b, shift_b, mirror_b, frame_b = res
            pose30_b = _try_correspondences(frac_i_zoom, frac_j_zoom, frame_b, BASE_RESOLUTION,
                                             theta_b, shift_b, mirror_b, BASE_CONTACT_EPS,
                                             R_ij_gt, t_ij_gt)
            if pose30_b is None:
                n_no_correspondence += 1
                rows.append((n_min_raw, "no_correspondence", False))
            else:
                n_pose_computed += 1
                n_pose30 += int(pose30_b)
                rows.append((n_min_raw, "pose_computed", bool(pose30_b)))

    elapsed = time.time() - t0
    print(f"\nFini : {n_targeted} paires ciblées (masque brut entre "
          f"{args.min_points_floor} et {PIPELINE_MIN_POINTS} points) en {elapsed:.0f}s "
          f"({n_below_floor} sous le plancher numérique, jamais tentées)\n")
    if n_targeted == 0:
        print("Aucune paire ciblée.")
        return

    print(f"Zoom impossible / étage 1 non tenté : {n_zoom_failed}/{n_targeted} "
          f"({100*n_zoom_failed/n_targeted:.1f}%)")
    print(f"no_correspondence après zoom          : {n_no_correspondence}/{n_targeted} "
          f"({100*n_no_correspondence/n_targeted:.1f}%)")
    print(f"Éligible (pose_computed) après zoom    : {n_pose_computed}/{n_targeted} "
          f"({100*n_pose_computed/n_targeted:.1f}%)")
    if n_pose_computed:
        print(f"  dont Pose@30 correct                 : {n_pose30}/{n_pose_computed} "
              f"({100*n_pose30/n_pose_computed:.1f}%)")
    print(f"\nGlobal (Pose@30 réel / paires ciblées) : {n_pose30}/{n_targeted} "
          f"({100*n_pose30/n_targeted:.1f}%)")

    print("\n=== Ventilation par tranche de densité brute (n_min_raw) ===")
    for lo, hi, label in zip(DENSITY_BINS[:-1], DENSITY_BINS[1:], BIN_LABELS):
        bin_rows = [r for r in rows if lo <= r[0] < hi]
        if not bin_rows:
            continue
        n_bin = len(bin_rows)
        n_elig_bin = sum(1 for r in bin_rows if r[1] == "pose_computed")
        n_p30_bin = sum(1 for r in bin_rows if r[2])
        print(f"  {label:8s} (n={n_bin:4d}) : éligible {100*n_elig_bin/n_bin:5.1f}% | "
              f"Pose@30 (parmi ciblées) {100*n_p30_bin/n_bin:5.1f}%")


if __name__ == "__main__":
    main()
