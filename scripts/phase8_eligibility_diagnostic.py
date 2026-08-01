"""
scripts/phase8_eligibility_diagnostic.py
===========================================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — diagnostique la perte d'éligibilité de
l'étage 1 appris (57.8% vs 62.6% hand-crafted, cf. `phase8_pipeline_learned_check.py
--strategy thresh03`, 2026-07-31). Même esprit que `phase5a_skip_audit.py` /
`phase7_isolated_fp_prevalence_check.py` : pousser CHAQUE paire jusqu'au bout
et comparer prédiction vs vérité, pas seulement compter les échecs.

Question posée : l'échec d'éligibilité vient-il (A) d'une mauvaise
classification du MIROIR (ambiguïté de réflexion (u,v), ~68% de précision
observée) ou (B) d'une imprécision de l'angle/décalage même quand le miroir
est correct ? Les deux appellent des fix très différents (A → feature qui
aide spécifiquement à discriminer l'orientation, ex. normales 3D ; B →
architecture/capacité du modèle en général).

Pour CHAQUE paire à 2 fragments (pas seulement celles qui atteignent
`pose_computed`), calcule le VRAI label `(theta_gt, shift_gt, mirror_gt)`
via `fit_theta_shift_mirror_from_gt` (dérivation exacte depuis la pose 3D,
PAS une recherche -- déjà validée par les auto-tests de
`phase8_depthmap_regressor_dataset.py`), et compare à la prédiction du
modèle. Reconstruit aussi `n_corr` (nombre de correspondances 3D trouvées
par `build_correspondences_nn` avec la pose PRÉDITE) pour relier la
précision de la prédiction à l'éligibilité réelle du pipeline.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_eligibility_diagnostic.py \\
        --regressor_ckpt output/phase8_depthmap_regressor_thresh03_corr/best.ckpt \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --summary_json /tmp/student7/phase8_eligibility_diagnostic.json
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
from scripts.phase5a_depthmap_matching import (
    quat_wxyz_to_rotmat, build_correspondences_nn, angle_correlation_profile, DEFAULT_N_ANGLES,
)
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase8_depthmap_regressor_dataset import fit_theta_shift_mirror_from_gt
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
def diagnose_pair(model, frac_i, frac_j, R_ij_gt, t_ij_gt, device):
    """Retourne un dict complet (pas juste succès/échec) -- comparaison
    prédiction vs label GT, pour CHAQUE paire, y compris celles qui
    échouent."""
    n_i, n_j = len(frac_i), len(frac_j)
    result = {"reached_stage": "unusable_too_few", "n_frac_pts_min": min(n_i, n_j)}
    if n_i < 3 or n_j < 3:
        return result

    dmap_i, valid_i, dmap_j, valid_j, frame = build_frame_and_rasterize(frac_i, frac_j)

    theta_gt, shift_gt, mirror_gt, label_residual = fit_theta_shift_mirror_from_gt(
        frac_i, frame["c_i"], frame["u_i"], frame["v_i"], frame["u_min_i"], frame["v_min_i"],
        frame["c_j"], frame["u_j"], frame["v_j"], frame["u_min_j"], frame["v_min_j"],
        frame["pixel_size"], RESOLUTION, R_ij_gt, t_ij_gt,
    )

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
    theta_pred = float(theta_t.item())
    shift_pred = (float(sy_t.item()), float(sx_t.item()))
    mirror_pred = bool(mirror_t.item())
    mirror_conf = float(pred["mirror_prob"].item())
    mirror_conf = mirror_conf if mirror_pred else 1.0 - mirror_conf   # confiance du CHOIX fait

    angle_err = min(abs(theta_pred - theta_gt), 360.0 - abs(theta_pred - theta_gt))
    shift_err = float(np.hypot(shift_pred[0] - shift_gt[0], shift_pred[1] - shift_gt[1]))

    v_j_use = -frame["v_j"] if mirror_pred else frame["v_j"]
    pts_i3, pts_j3 = build_correspondences_nn(
        frac_i, frame["c_i"], frame["u_i"], frame["v_i"],
        frac_j, frame["c_j"], frame["u_j"], v_j_use,
        frame["u_min_i"], frame["v_min_i"], frame["u_min_j"], frame["v_min_j"],
        frame["pixel_size"], RESOLUTION, theta_pred, shift_pred,
    )
    n_corr = 0 if pts_i3 is None else len(pts_i3)
    reached_stage = "pose_computed" if n_corr >= 3 else "no_correspondence"

    result.update({
        "reached_stage": reached_stage,
        "mirror_pred": mirror_pred, "mirror_gt": bool(mirror_gt),
        "mirror_correct": mirror_pred == bool(mirror_gt),
        "mirror_confidence": mirror_conf,
        "angle_err_deg": angle_err, "shift_err_px": shift_err,
        "n_corr": n_corr, "label_residual": float(label_residual),
    })
    return result


def _summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label:<28} N=0")
        return
    elig = sum(1 for r in rows if r["reached_stage"] == "pose_computed")
    angle_mean = float(np.mean([r["angle_err_deg"] for r in rows]))
    angle_med = float(np.median([r["angle_err_deg"] for r in rows]))
    n_corr_mean = float(np.mean([r["n_corr"] for r in rows]))
    print(f"  {label:<28} N={n:>5}  éligible={100*elig/n:>5.1f}%  "
          f"angle_err(moy/méd)={angle_mean:>6.1f}°/{angle_med:>5.1f}°  n_corr_moy={n_corr_mean:>6.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor_ckpt", required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint CNN Step 15")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--summary_json", default="")
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

    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                      f"paires diagnostiquées={len(rows)} | {time.time()-t0:.0f}s écoulées")

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

            out = cnn_model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()     for k in range(K)]
            gt_per_k    = [gt_frag_list[k].cpu().numpy()  for k in range(K)]

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
            frac_i_zoom = np.concatenate([raw0[mask0], new_i_input], axis=0)
            frac_j_zoom = np.concatenate([raw1[mask1], new_j_input], axis=0)

            diag = diagnose_pair(regressor, frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, device)
            rows.append(diag)

    elapsed = time.time() - t0
    usable = [r for r in rows if r["reached_stage"] != "unusable_too_few"]
    print(f"\nFini : {len(rows)} paires diagnostiquées sur {n_seen_2frag} objets 2-frags vus "
          f"({n_zoom_failed} zooms impossibles, {len(rows)-len(usable)} unusable_too_few) "
          f"en {elapsed:.0f}s\n")

    if not usable:
        print("Aucune paire exploitable.")
        return

    n_mirror_correct = sum(1 for r in usable if r["mirror_correct"])
    n_mirror_wrong = len(usable) - n_mirror_correct
    print(f"Précision miroir globale : {100*n_mirror_correct/len(usable):.1f}% "
          f"({n_mirror_correct}/{len(usable)})\n")

    print("ÉLIGIBILITÉ / PRÉCISION, STRATIFIÉ PAR MIROIR CORRECT vs INCORRECT :")
    _summarize([r for r in usable if r["mirror_correct"]], "Miroir CORRECT")
    _summarize([r for r in usable if not r["mirror_correct"]], "Miroir INCORRECT")
    print()
    _summarize(usable, "TOUTES paires")

    # Confiance du modèle sur son propre choix de miroir, stratifiée par
    # correction réelle -- si le modèle est déjà moins confiant sur les cas
    # où il se trompe, un simple seuil de confiance pourrait servir de filtre.
    conf_correct = np.mean([r["mirror_confidence"] for r in usable if r["mirror_correct"]])
    conf_wrong = np.mean([r["mirror_confidence"] for r in usable if not r["mirror_correct"]])
    print(f"\nConfiance moyenne du modèle sur son choix de miroir : "
          f"correct={conf_correct:.3f} | incorrect={conf_wrong:.3f}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_total": len(rows), "n_usable": len(usable),
                "mirror_accuracy": n_mirror_correct / len(usable),
                "mirror_confidence_correct": float(conf_correct),
                "mirror_confidence_wrong": float(conf_wrong),
                "rows": rows,
            }, f, indent=2)
        print(f"\nJSON résumé (+ toutes les lignes brutes) sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
