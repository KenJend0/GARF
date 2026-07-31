"""
scripts/phase8_pipeline_learned_check.py
===========================================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — évaluation bout-en-bout du pipeline
avec le modèle appris (`assembly/models/depthmap_pose_regressor.py`) comme
étage 1, à la place de `match_depthmaps()` (recherche FFT). Même structure
que `phase6b_pipeline_check.py`/`phase6b_pipeline_thresh03_check.py` :
étage 1 (pose grossière) -> étage 2 (`trimmed_icp_normals`, raffinement).

Étage 1 appris (résolution FIXE, celle de l'entraînement -- PAS de cascade
multi-résolution comme le hand-crafted, qui n'a plus lieu d'être ici) :
  1. `canonical_pca_frame` + `rasterize` (mêmes fonctions que
     `phase8_build_regressor_dataset.py`, même résolution).
  2. Forward du modèle -> `(theta, shift_y, shift_x, mirror)` via
     `DepthmapPoseRegressor.decode`.
  3. `build_correspondences_nn` (avec `v_j` inversé si `mirror=True`) +
     `kabsch` -> pose grossière (R_est, t_est).

Réutilise SANS modification : zoom/rééchantillonnage + clustering + init_rot
(`phase5a_zoom_resample_check.py`/`phase5a_zoom_resample_thresh03_check.py`),
`trimmed_icp_normals` (`phase6a_convergence_basin_check.py`), métriques
(`rot_err_deg`/`trans_err`/Pose@30/15, `phase5a_depthmap_matching.py`).

Barre à dépasser (Step 15 + `nn`, meilleur résultat hand-crafted actuel) :
  GT       : éligibilité 99.1%, Pose@30 global 37.6%, succès strict 23.2%
  thresh0.3: éligibilité 62.6%, Pose@30 global 14.7%, succès strict 6.8%

Usage (sur le serveur, oracle-first D'ABORD) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_pipeline_learned_check.py \\
        --strategy gt --regressor_ckpt output/phase8_depthmap_regressor/best.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --summary_json /tmp/student7/phase8_pipeline_learned_gt.json

    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_pipeline_learned_check.py \\
        --strategy thresh03 --regressor_ckpt output/phase8_depthmap_regressor/best.ckpt \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --summary_json /tmp/student7/phase8_pipeline_learned_thresh03.json
"""

import argparse
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
from scripts.phase5a_depthmap_matching import (
    quat_wxyz_to_rotmat, rasterize, kabsch, rot_err_deg, trans_err, build_correspondences_nn,
    angle_correlation_profile, DEFAULT_N_ANGLES,
)
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase6a_convergence_basin_check import trimmed_icp_normals
from scripts.phase8_depthmap_regressor_dataset import canonical_pca_frame
from scripts.phase8_build_regressor_dataset import build_frame_and_rasterize, RESOLUTION
from assembly.models.depthmap_pose_regressor import DepthmapPoseRegressor

ABS_MIN_POINTS = 50


def load_regressor(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DepthmapPoseRegressor(feat_dim=ckpt["feat_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_learned_stage1(model, frac_i, frac_j, R_ij_gt, t_ij_gt, device):
    """Étage 1 appris -- retourne un dict analogue à `run_cascade()`
    (`phase5a_zoom_resample_check.py`), même clés (`reached_stage`,
    `R_est`/`t_est`, `pose_30`/`pose_15`, `rot_err`/`trans_err`)."""
    n_i, n_j = len(frac_i), len(frac_j)
    result = {"reached_stage": "unusable_too_few", "n_frac_pts_min": min(n_i, n_j),
              "pose_30": False, "pose_15": False, "rot_err": None, "trans_err": None,
              "R_est": None, "t_est": None}
    if n_i < 3 or n_j < 3:
        return result

    dmap_i, valid_i, dmap_j, valid_j, frame = build_frame_and_rasterize(frac_i, frac_j)

    # Profil de corrélation FFT (2026-07-31, "corrélation croisée explicite")
    # -- calculé à la volée ici, comme à l'entraînement (pas de version
    # torch différentiable, coût négligeable : 2 x 36 x 2-flips FFT 64x64).
    profile_normal = angle_correlation_profile(dmap_i, valid_i, dmap_j, valid_j, DEFAULT_N_ANGLES)
    profile_mirror = angle_correlation_profile(
        dmap_i, valid_i, np.flip(dmap_j, axis=0), np.flip(valid_j, axis=0), DEFAULT_N_ANGLES)

    t_dmap_i = torch.from_numpy(dmap_i[None]).float().to(device)
    t_valid_i = torch.from_numpy(valid_i[None]).float().to(device)
    t_dmap_j = torch.from_numpy(dmap_j[None]).float().to(device)
    t_valid_j = torch.from_numpy(valid_j[None]).float().to(device)
    t_profile_normal = torch.from_numpy(profile_normal[None].astype(np.float32)).to(device)
    t_profile_mirror = torch.from_numpy(profile_mirror[None].astype(np.float32)).to(device)

    pred = model(t_dmap_i, t_valid_i, t_dmap_j, t_valid_j, t_profile_normal, t_profile_mirror)
    theta_t, sy_t, sx_t, mirror_t = DepthmapPoseRegressor.decode(pred)
    theta = float(theta_t.item())
    shift = (float(sy_t.item()), float(sx_t.item()))
    mirror = bool(mirror_t.item())

    v_j_use = -frame["v_j"] if mirror else frame["v_j"]
    pts_i3, pts_j3 = build_correspondences_nn(
        frac_i, frame["c_i"], frame["u_i"], frame["v_i"],
        frac_j, frame["c_j"], frame["u_j"], v_j_use,
        frame["u_min_i"], frame["v_min_i"], frame["u_min_j"], frame["v_min_j"],
        frame["pixel_size"], RESOLUTION, theta, shift,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        result["reached_stage"] = "no_correspondence"
        return result

    R_est, t_est = kabsch(pts_i3, pts_j3)
    re = rot_err_deg(R_est, R_ij_gt)
    te = trans_err(t_est, t_ij_gt)
    result.update({
        "reached_stage": "pose_computed", "rot_err": re, "trans_err": te,
        "pose_30": bool(re < 30.0 and te < 0.1), "pose_15": bool(re < 15.0 and te < 0.05),
        "R_est": R_est, "t_est": t_est,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["gt", "thresh03"])
    parser.add_argument("--regressor_ckpt", required=True)
    parser.add_argument("--ckpt", default="", help="Checkpoint CNN, requis si --strategy thresh03")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--max_icp_iters", type=int, default=50)
    parser.add_argument("--trim_ratio", type=float, default=0.7)
    parser.add_argument("--normal_dot_thresh", type=float, default=-0.3)
    parser.add_argument("--success_rot_thresh", type=float, default=5.0)
    parser.add_argument("--success_trans_thresh", type=float, default=0.02)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.strategy == "thresh03" and not args.ckpt:
        parser.error("--ckpt (checkpoint CNN) est requis avec --strategy thresh03")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    regressor = load_regressor(args.regressor_ckpt, device)

    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    n_stage1_failed = 0
    t0 = time.time()

    def _handle_pair(frac_i_base, frac_j_base, frac_i_nrm, frac_j_nrm, frac_i_zoom, frac_j_zoom,
                      R_ij_gt, t_ij_gt, n_min_base):
        nonlocal n_stage1_failed
        stage1_res = run_learned_stage1(regressor, frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, device)
        if stage1_res["reached_stage"] != "pose_computed":
            n_stage1_failed += 1
            return
        R_init, t_init = stage1_res["R_est"], stage1_res["t_est"]
        R_final, t_final, _ = trimmed_icp_normals(
            frac_i_base, frac_i_nrm, frac_j_base, frac_j_nrm, R_init, t_init,
            max_iters=args.max_icp_iters, trim_ratio=args.trim_ratio,
            normal_dot_thresh=args.normal_dot_thresh,
        )
        re_final = rot_err_deg(R_final, R_ij_gt)
        te_final = trans_err(t_final, t_ij_gt)
        rows.append({
            "n_frac_pts_min": n_min_base,
            "stage1_pose_30": stage1_res["pose_30"], "stage1_pose_15": stage1_res["pose_15"],
            "final_pose_30": bool(re_final < 30.0 and te_final < 0.1),
            "final_pose_15": bool(re_final < 15.0 and te_final < 0.05),
            "final_strict_success": bool(re_final < args.success_rot_thresh
                                          and te_final < args.success_trans_thresh),
        })

    if args.strategy == "gt":
        print("Chargement du datamodule -- sample_method=weighted, AUCUN CNN chargé "
              "(stratégie GT, oracle-first)")
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
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=datamodule.dataset_cls.collate_fn)

        for idx, batch in enumerate(loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                      f"étage1 échoué={n_stage1_failed} | {time.time()-t0:.0f}s écoulées")

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
            if n_min_base < max(3, ABS_MIN_POINTS):
                continue

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

            seed_i_gt = frag_pts_gt[0][gt_i == 1]
            seed_j_gt = frag_pts_gt[1][gt_j == 1]
            new_i_gt = zoom_resample(meshes[p0], seed_i_gt, args.extra_budget, args.expand_rings, rng)
            new_j_gt = zoom_resample(meshes[p1], seed_j_gt, args.extra_budget, args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                n_zoom_failed += 1
                continue
            new_i_input = to_input_frame(new_i_gt, quats_np[p0], trans_np[p0])
            new_j_input = to_input_frame(new_j_gt, quats_np[p1], trans_np[p1])
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)

            _handle_pair(frac_i_base, frac_j_base, frac_i_nrm, frac_j_nrm,
                         frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, n_min_base)

    else:   # thresh03
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

        with torch.no_grad():
            for idx, batch in enumerate(cnn_loader):
                if args.max_batches > 0 and idx >= args.max_batches:
                    break
                if idx % 200 == 0:
                    print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                          f"étage1 échoué={n_stage1_failed} | {time.time()-t0:.0f}s écoulées")

                mesh_data = mesh_dataset[idx]
                assert batch["name"][0] == mesh_data["name"]

                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                frag_list, valid_pcs, K = extract_fragment_list(
                    batch_gpu["pointclouds"], batch_gpu["points_per_part"])
                if K != 2:
                    continue
                n_seen_2frag += 1

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
                if n_min_raw < ABS_MIN_POINTS:
                    continue

                cluster_mask0 = dominant_cluster_mask(raw0[mask0])
                cluster_mask1 = dominant_cluster_mask(raw1[mask1])
                R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
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
                frac_i_base, frac_j_base = raw0[mask0], raw1[mask1]
                frac_i_nrm, frac_j_nrm = nrm0[mask0], nrm1[mask1]
                frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
                frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)

                _handle_pair(frac_i_base, frac_j_base, frac_i_nrm, frac_j_nrm,
                             frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, n_min_raw)

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
    f_p30  = sum(1 for r in rows if r["final_pose_30"])
    f_p15  = sum(1 for r in rows if r["final_pose_15"])
    f_strict = sum(1 for r in rows if r["final_strict_success"])

    eligibility = n / max(n_total_attempted, 1)
    pose30_global = f_p30 / n_seen_2frag
    strict_global = f_strict / n_seen_2frag

    print(f"ÉTAGE 1 APPRIS + ÉTAGE 2 (ICP) -- stratégie {args.strategy} :")
    print(f"  Éligibilité étage 1     : {100*eligibility:.1f}% ({n}/{n_total_attempted})")
    print(f"  Pose@30 (parmi étage 2) : étage1 seul {100*s1_p30/n:.1f}% | + étage2 {100*f_p30/n:.1f}%")
    print(f"  Pose@30 GLOBAL (/{n_seen_2frag} objets vus) : {100*pose30_global:.1f}% ({f_p30} paires)")
    print(f"  Succès strict GLOBAL                     : {100*strict_global:.1f}% ({f_strict} paires)")

    barre = ("éligibilité 99.1%, Pose@30 global 37.6%, succès strict 23.2%" if args.strategy == "gt"
             else "éligibilité 62.6%, Pose@30 global 14.7%, succès strict 6.8%")
    print(f"\n(Barre à dépasser -- Step 15 + nn, hand-crafted : {barre})")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_pairs_stage2": n, "n_2frag_seen": n_seen_2frag,
                "n_zoom_failed": n_zoom_failed, "n_stage1_failed": n_stage1_failed,
                "eligibility": eligibility, "pose30_global": pose30_global,
                "strict_global": strict_global,
                "stage1_only_pose30": s1_p30 / n, "final_pose30_among_stage2": f_p30 / n,
                "final_pose15_among_stage2": f_p15 / n,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
