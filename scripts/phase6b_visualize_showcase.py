"""
scripts/phase6b_visualize_showcase.py
=========================================
Génère, pour la présentation Phase 8, une figure par catégorie de résultat
du pipeline HAND-CONÇU complet (zoom + cascade + `nn` + `trimmed_icp_normals`),
en condition réelle (masque CNN `thresh0.3`) : la MÊME chaîne que
`phase6b_pipeline_thresh03_check.py` (aucune réimplémentation de la logique
de matching/raffinement), avec `--correspondence_mode nn` (le réglage qui
produit la barre de référence 62.6% / 14.7% / 6.8% citée dans
PLAN_REASSEMBLY_MODULE.md).

Quatre catégories cherchées, une paire par catégorie :
  - not_eligible      : étage 1 échoue (pas assez de points, ou aucune
                        correspondance trouvée par la cascade).
  - fail              : étage 1 réussit mais la pose finale (après ICP)
                        rate Pose@30°/0.1.
  - pose30_not_strict : pose finale sous 30°/0.1 mais pas sous le seuil
                        strict (5°/0.02).
  - strict_success    : pose finale sous le seuil strict.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase6b_visualize_showcase.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --correspondence_mode nn \\
        --out_dir /tmp/student7/phase8_showcase_handcrafted
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat, rot_err_deg, trans_err
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame, run_cascade
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask, remove_tiny_clusters_mask
from scripts.phase6a_convergence_basin_check import trimmed_icp_normals
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

ABS_MIN_POINTS = 5
CATEGORIES = ["not_eligible", "fail", "pose30_not_strict", "strict_success"]
CATEGORY_TITLES = {
    "not_eligible": "Non éligible — étage 1 n'a pas trouvé de pose",
    "fail": "Échec — pose finale hors tolérance (30°/0.1)",
    "pose30_not_strict": "Succès large (Pose@30) mais pas strict (5°/0.02)",
    "strict_success": "Succès strict (5°/0.02)",
}


def classify(stage1_res, re_final, te_final, success_rot_thresh, success_trans_thresh):
    if stage1_res["reached_stage"] != "pose_computed":
        return "not_eligible"
    if not (re_final < 30.0 and te_final < 0.1):
        return "fail"
    if re_final < success_rot_thresh and te_final < success_trans_thresh:
        return "strict_success"
    return "pose30_not_strict"


def scatter3d(ax, pts, color, s=1, alpha=0.5, label=None):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=s, alpha=alpha, label=label)


def make_figure(category, obj_idx, raw_i, raw_j, frac_i_base, frac_j_base,
                 R_ij_gt, t_ij_gt, stage1_res, R_final, t_final, re_final, te_final,
                 success_rot_thresh, success_trans_thresh, out_path):
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    scatter3d(ax1, raw_i, "tab:blue")
    ax1.set_title(f"Fragment i — {len(raw_i)} pts")

    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    scatter3d(ax2, raw_j, "tab:orange")
    ax2.set_title(f"Fragment j — {len(raw_j)} pts")

    ax3 = fig.add_subplot(2, 4, 3, projection="3d")
    scatter3d(ax3, raw_i, "lightgray", s=1, alpha=0.3)
    scatter3d(ax3, frac_i_base, "red", s=4, alpha=0.9)
    ax3.set_title(f"Fracture i (thresh0.3, CNN) — {len(frac_i_base)} pts")

    ax4 = fig.add_subplot(2, 4, 4, projection="3d")
    scatter3d(ax4, raw_j, "lightgray", s=1, alpha=0.3)
    scatter3d(ax4, frac_j_base, "green", s=4, alpha=0.9)
    ax4.set_title(f"Fracture j (thresh0.3, CNN) — {len(frac_j_base)} pts")

    ax8 = fig.add_subplot(2, 4, 8, projection="3d")
    raw_j_gt = (R_ij_gt.T @ (raw_j - t_ij_gt).T).T
    frac_j_gt = (R_ij_gt.T @ (frac_j_base - t_ij_gt).T).T
    scatter3d(ax8, raw_i, "tab:blue", alpha=0.25, label="i")
    scatter3d(ax8, raw_j_gt, "tab:orange", alpha=0.25, label="j (vraie pose GT)")
    scatter3d(ax8, frac_i_base, "red", s=6, alpha=0.9)
    scatter3d(ax8, frac_j_gt, "limegreen", s=6, alpha=0.9)
    ax8.set_title("VRAIE POSE (GT) — référence")

    if stage1_res["reached_stage"] == "pose_computed":
        ax6 = fig.add_subplot(2, 4, 6, projection="3d")
        R1, t1 = stage1_res["R_est"], stage1_res["t_est"]
        raw_j_s1 = (R1.T @ (raw_j - t1).T).T
        scatter3d(ax6, raw_i, "tab:blue", alpha=0.25)
        scatter3d(ax6, raw_j_s1, "tab:orange", alpha=0.25)
        ax6.set_title(f"Étage 1 (pose grossière, R={stage1_res['resolution_used']})\n"
                      f"RotErr={stage1_res['rot_err']:.1f}° TransErr={stage1_res['trans_err']:.3f}")

        ax7 = fig.add_subplot(2, 4, 7, projection="3d")
        raw_j_final = (R_final.T @ (raw_j - t_final).T).T
        frac_j_final = (R_final.T @ (frac_j_base - t_final).T).T
        scatter3d(ax7, raw_i, "tab:blue", alpha=0.25)
        scatter3d(ax7, raw_j_final, "tab:orange", alpha=0.25)
        scatter3d(ax7, frac_i_base, "red", s=6, alpha=0.9)
        scatter3d(ax7, frac_j_final, "limegreen", s=6, alpha=0.9)
        ax7.set_title(f"Après ICP (pose finale)\nRotErr={re_final:.1f}° TransErr={te_final:.3f}")

    ax5 = fig.add_subplot(2, 4, 5)
    ax5.axis("off")
    lines = [
        f"objet #{obj_idx}",
        f"catégorie : {category}",
        "",
        f"n points fracture (min i/j) : {stage1_res['n_frac_pts_min']}",
        f"étage 1 : {stage1_res['reached_stage']}",
    ]
    if stage1_res["reached_stage"] == "pose_computed":
        lines += [
            f"résolution retenue : {stage1_res['resolution_used']}",
            f"étage 1 seul — RotErr={stage1_res['rot_err']:.1f}° TransErr={stage1_res['trans_err']:.3f}",
            f"pose finale — RotErr={re_final:.1f}° TransErr={te_final:.3f}",
            "",
            f"Pose@30°/0.1 : {'OUI' if re_final < 30.0 and te_final < 0.1 else 'non'}",
            f"succès strict ({success_rot_thresh}°/{success_trans_thresh}) : "
            f"{'OUI' if (re_final < success_rot_thresh and te_final < success_trans_thresh) else 'non'}",
        ]
    ax5.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace",
              transform=ax5.transAxes)

    fig.suptitle(f"Hand-conçu (Step 15 + nn) — {CATEGORY_TITLES[category]}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[128, 96, 64, 48, 32, 24, 20, 16, 12])
    parser.add_argument("--n_angles", type=int, default=36)
    parser.add_argument("--score_mode", default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--max_icp_iters", type=int, default=50)
    parser.add_argument("--trim_ratio", type=float, default=0.7)
    parser.add_argument("--normal_dot_thresh", type=float, default=-0.3)
    parser.add_argument("--success_rot_thresh", type=float, default=5.0)
    parser.add_argument("--success_trans_thresh", type=float, default=0.02)
    parser.add_argument("--correspondence_mode", default="nn", choices=["grid", "nn"])
    parser.add_argument("--contact_eps", type=float, default=0.05)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0
    args.robust_pca = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

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
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    found = {}
    n_seen_2frag = 0

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if len(found) == len(CATEGORIES):
                break
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | trouvés={list(found.keys())}")

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

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k = [frag_list[k].cpu().numpy() for k in range(K)]
            gt_per_k = [gt_frag_list[k].cpu().numpy() for k in range(K)]
            nrm_per_k = [nrm_frag_list[k].cpu().numpy() for k in range(K)]

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
            if n_min_raw < max(ABS_MIN_POINTS, 50):
                continue

            cluster_mask0 = dominant_cluster_mask(raw0[mask0])
            cluster_mask1 = dominant_cluster_mask(raw1[mask1])
            frac_i_base, frac_j_base = raw0[mask0], raw1[mask1]
            frac_i_nrm, frac_j_nrm = nrm0[mask0], nrm1[mask1]

            R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
            seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init_rot.T
            seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init_rot.T
            meshes = mesh_data["meshes"]
            new_i_gt = zoom_resample(meshes[p0], seed_i_gt, args.extra_budget, args.expand_rings, rng)
            new_j_gt = zoom_resample(meshes[p1], seed_j_gt, args.extra_budget, args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                continue

            new_i_rotated = new_i_gt @ R_init_rot
            new_j_rotated = new_j_gt @ R_init_rot
            new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
            new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)

            stage1_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, args)

            re_final, te_final, R_final, t_final = None, None, None, None
            if stage1_res["reached_stage"] == "pose_computed":
                R_init, t_init = stage1_res["R_est"], stage1_res["t_est"]
                R_final, t_final, _ = trimmed_icp_normals(
                    frac_i_base, frac_i_nrm, frac_j_base, frac_j_nrm,
                    R_init, t_init, max_iters=args.max_icp_iters, trim_ratio=args.trim_ratio,
                    normal_dot_thresh=args.normal_dot_thresh,
                )
                re_final = rot_err_deg(R_final, R_ij_gt)
                te_final = trans_err(t_final, t_ij_gt)

            category = classify(stage1_res, re_final if re_final is not None else 999.0,
                                 te_final if te_final is not None else 999.0,
                                 args.success_rot_thresh, args.success_trans_thresh)

            if category in found:
                continue
            found[category] = True
            out_path = out_dir / f"{category}.png"
            make_figure(category, n_seen_2frag, raw0, raw1, frac_i_base, frac_j_base,
                        R_ij_gt, t_ij_gt, stage1_res, R_final, t_final, re_final, te_final,
                        args.success_rot_thresh, args.success_trans_thresh, out_path)
            print(f"  [{len(found)}/{len(CATEGORIES)}] {category} -> {out_path.name} (objet #{n_seen_2frag})")

    missing = [c for c in CATEGORIES if c not in found]
    if missing:
        print(f"\nATTENTION : catégories jamais rencontrées dans les {n_seen_2frag} objets vus : {missing} "
              f"(augmenter --max_batches)")
    else:
        print(f"\nLes 4 catégories ont été trouvées, figures dans {out_dir}")


if __name__ == "__main__":
    main()
