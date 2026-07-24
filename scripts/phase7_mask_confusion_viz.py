"""
scripts/phase7_mask_confusion_viz.py
=======================================
Visualisation diagnostique — voir PLAN_REASSEMBLY_MODULE.md, section Phase 7,
point 3 du plan étape par étape (2026-07-22).

Question de l'utilisateur (2026-07-23) après le résultat thresh0.3+nn
(éligibilité 48.2%→62.6%, mais toujours sous le seuil des 20% de Pose@30
global) : "qu'est-ce qui manque concrètement au masque CNN ?" On sait DÉJÀ
(diagnostic précision/rappel, 2026-07-22 et 2026-07-23) que la précision et
le rappel du masque `thresh0.3` chutent sur les paires où l'étage 1 échoue
(79.6%/90.1% réussite vs 63.0%/84.1% échec) -- mais c'est une corrélation
globale, pas un diagnostic qualitatif. Ce script montre concrètement, sur
des paires individuelles, à QUOI ressemble l'erreur : le masque déborde-t-il
sur des zones plates non fracturées (faux positifs), rate-t-il des franges
fines de la fracture (faux négatifs), ou est-il simplement mal localisé ?

Protocole par paire : reproduit EXACTEMENT le critère de succès/échec étage 1
du pipeline réel (`phase6b_pipeline_thresh03_check.py`, 2026-07-23,
--correspondence_mode nn) -- deux datasets en lockstep, clustering pour les
graines de zoom, correction init_rot, run_cascade(nn) -- pour filtrer
--filter success/fail sur le MÊME critère que les chiffres déjà obtenus.
Pour chaque paire retenue, classe chaque point du masque BRUT (thresh0.3,
non zoomé) en TP (prédit ET GT), FP (prédit, pas GT), FN (pas prédit, GT) et
sauvegarde un nuage de points 3D coloré (vert=TP, rouge=FP, bleu=FN,
gris clair=TN) pour chaque fragment de la paire.

Réutilise `zoom_resample`/`to_input_frame`/`run_cascade`
(`phase5a_zoom_resample_check.py`), `dominant_cluster_mask`
(`phase5a_zoom_resample_thresh03_check.py`), `quat_wxyz_to_rotmat`
(`phase5a_depthmap_matching.py`) -- aucune réimplémentation.

Usage (sur le serveur, headless -- sauvegarde en PNG) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase7_mask_confusion_viz.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --expand_rings 0 --extra_budget 1200 \\
        --resolution_sweep 128 96 64 48 32 24 20 16 12 \\
        --correspondence_mode nn --contact_eps 0.05 \\
        --filter fail --n_vis 12 \\
        --out_dir /tmp/student7/phase7_mask_confusion_fail

    # Puis comparer avec les succès :
    --filter success --n_vis 12 --out_dir /tmp/student7/phase7_mask_confusion_success
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless server
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame, run_cascade
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

ABS_MIN_POINTS = 5


def confusion_labels(mask_pred, mask_gt):
    tp = mask_pred & mask_gt
    fp = mask_pred & ~mask_gt
    fn = ~mask_pred & mask_gt
    tn = ~mask_pred & ~mask_gt
    return tp, fp, fn, tn


def plot_confusion_pair(raw0, raw1, tp0, fp0, fn0, tn0, tp1, fp1, fn1, tn1,
                         meta, precision0, recall0, precision1, recall1, out_path):
    fig = plt.figure(figsize=(14, 7))

    def scatter_confusion(ax, raw, tp, fp, fn, tn, title):
        ax.scatter(raw[tn, 0], raw[tn, 1], raw[tn, 2], c="lightgray", s=1, alpha=0.25, label="TN")
        ax.scatter(raw[fn, 0], raw[fn, 1], raw[fn, 2], c="tab:blue", s=8, alpha=0.9, label="FN (raté)")
        ax.scatter(raw[fp, 0], raw[fp, 1], raw[fp, 2], c="tab:red", s=8, alpha=0.9, label="FP (faux+)")
        ax.scatter(raw[tp, 0], raw[tp, 1], raw[tp, 2], c="tab:green", s=8, alpha=0.9, label="TP")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)

    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    scatter_confusion(ax0, raw0, tp0, fp0, fn0, tn0,
                       f"Fragment i — precision={precision0:.1%} recall={recall0:.1%}")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    scatter_confusion(ax1, raw1, tp1, fp1, fn1, tn1,
                       f"Fragment j — precision={precision1:.1%} recall={recall1:.1%}")

    fig.suptitle(meta, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--correspondence_mode", default="nn", choices=["grid", "nn"])
    parser.add_argument("--contact_eps", type=float, default=0.05)
    parser.add_argument("--filter",      default="fail", choices=["any", "success", "fail"])
    parser.add_argument("--n_vis",       type=int, default=12)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0

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

    cnn_loader = DataLoader(
        cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=cnn_datamodule.dataset_cls.collate_fn,
    )

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    n_saved = 0
    n_seen_2frag = 0
    print(f"\nVisualisation confusion masque CNN vs GT -- filter={args.filter}, "
          f"correspondence_mode={args.correspondence_mode}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if n_saved >= args.n_vis:
                break
            if args.max_batches > 0 and idx >= args.max_batches:
                break

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

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()    for k in range(K)]
            gt_per_k    = [gt_frag_list[k].cpu().numpy() for k in range(K)]

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
            if n_min_raw < max(ABS_MIN_POINTS, 50):
                continue   # tranche <50 mise de côté (2026-07-22), structurellement dure

            fracture_gt_np = batch["fracture_surface_gt"].numpy()
            true0 = fracture_gt_np[0, p0] == 1
            true1 = fracture_gt_np[0, p1] == 1

            tp0, fp0, fn0, tn0 = confusion_labels(mask0, true0)
            tp1, fp1, fn1, tn1 = confusion_labels(mask1, true1)
            precision0 = int(tp0.sum()) / max(int(mask0.sum()), 1)
            precision1 = int(tp1.sum()) / max(int(mask1.sum()), 1)
            recall0 = int(tp0.sum()) / max(int(true0.sum()), 1)
            recall1 = int(tp1.sum()) / max(int(true1.sum()), 1)

            # ── Étage 1 réel (mêmes graines/clustering/correction init_rot que
            # phase6b_pipeline_thresh03_check.py) -- pour filtrer success/fail
            # sur le MÊME critère que les chiffres déjà obtenus le 2026-07-23. ──
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
                continue
            new_i_rotated = new_i_gt @ R_init_rot
            new_j_rotated = new_j_gt @ R_init_rot
            new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
            new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
            frac_i_zoom = np.concatenate([raw0[mask0], new_i_input], axis=0)
            frac_j_zoom = np.concatenate([raw1[mask1], new_j_input], axis=0)

            stage1_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij_gt, t_ij_gt, args)
            stage1_success = stage1_res["reached_stage"] == "pose_computed"

            if args.filter == "success" and not stage1_success:
                continue
            if args.filter == "fail" and stage1_success:
                continue

            out_path = out_dir / f"pair_{n_seen_2frag:04d}_{'success' if stage1_success else 'fail'}.png"
            meta = (f"objet#{n_seen_2frag} | étage1={'RÉUSSIT' if stage1_success else 'ÉCHOUE'} | "
                    f"n_min(brut)={n_min_raw}")
            plot_confusion_pair(raw0, raw1, tp0, fp0, fn0, tn0, tp1, fp1, fn1, tn1,
                                 meta, precision0, recall0, precision1, recall1, out_path)

            n_saved += 1
            print(f"  [{n_saved}/{args.n_vis}] {out_path.name} — "
                  f"prec(i/j)={precision0:.1%}/{precision1:.1%} "
                  f"rec(i/j)={recall0:.1%}/{recall1:.1%}")

    print(f"\n{n_saved} figures sauvegardées dans {out_dir}")


if __name__ == "__main__":
    main()
