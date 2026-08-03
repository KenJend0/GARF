"""
scripts/phase9_eligibility_funnel_check.py
===========================================
Phase 9 (PLAN_REASSEMBLY_MODULE.md) -- vérification DIRECTE (pas déduite
par recoupement) de la répartition exacte des paires à 2 fragments entre
les différentes causes d'inéligibilité, avec les MÊMES conventions de
comptage que `phase8_pipeline_learned_check.py` (dénominateur = TOUTES les
paires à 2 fragments vues, `n_seen_2frag`, y compris celles rejetées avant
même de tenter l'étage 1).

Motivation (2026-08-03) : `phase9_stage1_recovery_check.py` mesurait une
éligibilité de base de 89.3%, très supérieure aux ~50% du pipeline
complet -- écart expliqué par recoupement (dénominateur différent : ce
script-là ne comptait que les paires ayant déjà passé le filtre de densité
`ABS_MIN_POINTS`), mais jamais vérifié directement. Ce script ajoute un
compteur explicite `n_density_failed` (masque CNN combiné i+j < 50 points,
AVANT tout zoom/rasterisation) pour trancher si c'est bien le principal
contributeur à l'inéligibilité globale, plutôt que `no_correspondence`
(sur lequel `phase9_stage1_recovery_check.py` a déjà montré que cascade de
résolution et `contact_eps` généreux ne récupèrent que des poses fausses,
0% Pose@30).

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase9_eligibility_funnel_check.py \\
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
    ABS_MIN_POINTS, BASE_RESOLUTION, BASE_CONTACT_EPS,
)


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

    n_seen_2frag = 0
    n_density_failed = 0
    n_zoom_failed = 0
    n_stage1_attempted = 0
    n_stage1_failed_too_few = 0     # < 3 points après zoom (rare, cf. run_stage1_at_resolution)
    n_no_correspondence = 0
    n_pose_computed = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | {time.time()-t0:.0f}s")

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

            # -- même compteur que phase8_pipeline_learned_check.py : incrémenté
            # ICI, avant le filtre de densité (n_seen_2frag = TOUTES les paires
            # à 2 fragments, dénominateur de l'éligibilité globale) --
            n_seen_2frag += 1

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
            if n_min_raw < ABS_MIN_POINTS:
                n_density_failed += 1
                continue

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
                n_stage1_failed_too_few += 1
                continue

            full_centroid_i = raw0.mean(axis=0)
            full_centroid_j = raw1.mean(axis=0)

            n_stage1_attempted += 1
            res = run_stage1_at_resolution(regressor, frac_i_zoom, frac_j_zoom,
                                            nrm_i_zoom, nrm_j_zoom, device,
                                            BASE_RESOLUTION, full_centroid_i, full_centroid_j)
            if res is None:
                n_stage1_failed_too_few += 1
                continue
            theta_b, shift_b, mirror_b, frame_b = res
            pose30_b = _try_correspondences(frac_i_zoom, frac_j_zoom, frame_b, BASE_RESOLUTION,
                                             theta_b, shift_b, mirror_b, BASE_CONTACT_EPS,
                                             R_ij_gt, t_ij_gt)
            if pose30_b is None:
                n_no_correspondence += 1
            else:
                n_pose_computed += 1

    elapsed = time.time() - t0
    print(f"\nFini : {n_seen_2frag} paires à 2 fragments vues en {elapsed:.0f}s\n")
    if n_seen_2frag == 0:
        print("Aucune paire vue.")
        return

    def pct(n):
        return 100 * n / n_seen_2frag

    print("RÉPARTITION DU FUNNEL D'ÉLIGIBILITÉ (dénominateur = toutes les paires "
          "2-fragments vues, même convention que phase8_pipeline_learned_check.py) :")
    print(f"  Masque CNN trop pauvre (< {ABS_MIN_POINTS} pts, AVANT tout zoom) : "
          f"{n_density_failed} ({pct(n_density_failed):.1f}%)")
    print(f"  Zoom impossible (mesh)                                          : "
          f"{n_zoom_failed} ({pct(n_zoom_failed):.1f}%)")
    print(f"  Étage 1 non tenté (< 3 points même après zoom, rare)            : "
          f"{n_stage1_failed_too_few} ({pct(n_stage1_failed_too_few):.1f}%)")
    print(f"  Étage 1 tenté, no_correspondence                                : "
          f"{n_no_correspondence} ({pct(n_no_correspondence):.1f}%)")
    print(f"  Éligible (pose_computed)                                        : "
          f"{n_pose_computed} ({pct(n_pose_computed):.1f}%)")
    print(f"\n  Étage 1 tenté au total : {n_stage1_attempted} ({pct(n_stage1_attempted):.1f}%)")
    if n_stage1_attempted:
        print(f"  Éligibilité PARMI les tentés : {100*n_pose_computed/n_stage1_attempted:.1f}%")


if __name__ == "__main__":
    main()
