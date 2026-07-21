"""
scripts/phase5a_visualize_pair.py
===================================
Visualisation diagnostique pour la Phase 5A (depth-map matching) — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7. Constat du 2026-07-20 : chaque
piste testée (dilatation, splat gaussien) améliore les métriques agrégées
sans jamais faire bouger Pose@30 — on ne voit plus ce qui cloche dans les
chiffres seuls. Ce script sert à regarder concrètement quelques paires.

Réutilise DIRECTEMENT les fonctions de scripts/phase5a_depthmap_matching.py
(aucune réimplémentation) — ce qui est affiché correspond exactement à ce que
mesure le pipeline réel, pas une approximation.

Pour chaque paire visualisée, génère une figure PNG à 8 panneaux :
  1-2. Fragment i / j — nuage de points complet (repère local, désassemblé)
  3-4. Fragment i / j — points de fracture surlignés (stratégie choisie)
  5-6. Depth map i / j — les cartes 2D réelles utilisées par la recherche
  7.   Overlay 3D : fragment i + fragment j transformé par la MEILLEURE pose
       trouvée par l'algorithme (R_est/t_est)
  8.   Overlay 3D : fragment i + fragment j transformé par la VRAIE pose GT
       (R_ij_gt/t_ij_gt) — comparer 7 et 8 montre l'écart visuellement, ce
       que RotErr/Pose@30 résument en un seul chiffre.

Dans les panneaux 7-8, les points de fracture sont surlignés en plus gros
et de couleur vive (rouge pour i, vert pour j) — pour juger si c'est bien
la ZONE DE FRACTURE qui s'aligne, pas juste la silhouette générale du
fragment.

Usage (sur le serveur, headless — sauvegarde en PNG, pas d'affichage) :
    python scripts/phase5a_visualize_pair.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val \\
        --strategy gt --score_mode joint \\
        --n_vis 6 --out_dir /tmp/student7/phase5a_viz

    # Ne visualiser que des échecs (Pose@30 faux) ou que des succès :
    --filter fail
    --filter success

    # Visualiser avec un lissage donné (comme testé le 2026-07-20) :
    --dilate_px 1
    --gaussian_sigma_px 0.5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless server
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate
from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list
from scripts.phase5a_depthmap_matching import (
    MIN_FRAC_POINTS, MIN_OVERLAP_PIXELS,
    compute_pca_frame, rasterize, match_depthmaps, build_correspondences,
    kabsch, rot_err_deg, trans_err, quat_wxyz_to_rotmat,
)


def make_masks(raw_i, raw_j, gt_i, gt_j, score_i, score_j, strategy, rng):
    if strategy == "gt":
        return raw_i[gt_i == 1], raw_j[gt_j == 1]
    if strategy == "thresh0.3":
        return raw_i[score_i > 0.3], raw_j[score_j > 0.3]
    # random : même budget que gt, cf. process_pair
    n_gt_i = int((gt_i == 1).sum())
    n_gt_j = int((gt_j == 1).sum())
    n_rand = max(n_gt_i, n_gt_j, MIN_FRAC_POINTS)
    return (raw_i[rng.choice(len(raw_i), min(n_rand, len(raw_i)), replace=False)],
            raw_j[rng.choice(len(raw_j), min(n_rand, len(raw_j)), replace=False)])


def visualize_pair(raw_i, raw_j, frac_i, frac_j, R_ij_gt, t_ij_gt, args, out_path, meta):
    """Calcule la pose (même logique que process_pair) et sauvegarde la figure."""
    c_i, u_i, v_i, n_i, plan_i = compute_pca_frame(frac_i)
    c_j, u_j, v_j, n_j, plan_j = compute_pca_frame(frac_j)

    if plan_i > args.max_planarity or plan_j > args.max_planarity:
        return None, {"skip": "too_curved"}

    ci = frac_i - c_i
    cj = frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    pixel_size = max(span_i, span_j) * 1.1 / args.resolution

    dmap_i, valid_i, u_min_i, v_min_i = rasterize(
        frac_i, c_i, u_i, v_i, n_i, args.resolution, pixel_size,
        dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(
        frac_j, c_j, u_j, v_j, n_j, args.resolution, pixel_size,
        dilate_px=args.dilate_px, gaussian_sigma_px=args.gaussian_sigma_px)

    n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
    if n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS:
        return None, {"skip": "sparse_dmap"}

    best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
        dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)

    pts_i3, pts_j3 = build_correspondences(
        dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
        dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
        pixel_size, best_theta, best_shift, best_flip,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        return None, {"skip": "no_overlap_3d"}

    R_est, t_est = kabsch(pts_i3, pts_j3)   # R_est @ i + t_est ≈ j
    re = rot_err_deg(R_est, R_ij_gt)
    te = trans_err(t_est, t_ij_gt)
    pose30 = bool(re < 30.0 and te < 0.1)

    # ── Overlays 3D : ramener j dans le repère de i (inverse de R_est/t_est) ──
    raw_j_found = (R_est.T @ (raw_j - t_est).T).T
    frac_j_found = (R_est.T @ (frac_j - t_est).T).T
    raw_j_gt = (R_ij_gt.T @ (raw_j - t_ij_gt).T).T
    frac_j_gt = (R_ij_gt.T @ (frac_j - t_ij_gt).T).T

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 11))

    def scatter3d(ax, pts, color, s=1, alpha=0.5, label=None):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=s, alpha=alpha, label=label)

    # 1-2 : nuages complets
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    scatter3d(ax1, raw_i, "tab:blue")
    ax1.set_title(f"Fragment i — {len(raw_i)} pts")

    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    scatter3d(ax2, raw_j, "tab:orange")
    ax2.set_title(f"Fragment j — {len(raw_j)} pts")

    # 3-4 : points de fracture surlignés
    ax3 = fig.add_subplot(2, 4, 3, projection="3d")
    scatter3d(ax3, raw_i, "lightgray", s=1, alpha=0.3)
    scatter3d(ax3, frac_i, "red", s=4, alpha=0.9)
    ax3.set_title(f"Fracture i ({args.strategy}) — {len(frac_i)} pts, planarity={plan_i:.3f}")

    ax4 = fig.add_subplot(2, 4, 4, projection="3d")
    scatter3d(ax4, raw_j, "lightgray", s=1, alpha=0.3)
    scatter3d(ax4, frac_j, "green", s=4, alpha=0.9)
    ax4.set_title(f"Fracture j ({args.strategy}) — {len(frac_j)} pts, planarity={plan_j:.3f}")

    # 5-6 : depth maps
    dmap_i_show = np.ma.masked_where(~valid_i, dmap_i)
    dmap_j_show = np.ma.masked_where(~valid_j, dmap_j)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("lightgray")

    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(dmap_i_show, cmap=cmap, origin="lower")
    ax5.set_title(f"Depth map i ({n_pix_i} px valides / {args.resolution**2})")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(2, 4, 6)
    im6 = ax6.imshow(dmap_j_show, cmap=cmap, origin="lower")
    ax6.set_title(f"Depth map j ({n_pix_j} px valides / {args.resolution**2})")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # 7 : pose trouvée
    ax7 = fig.add_subplot(2, 4, 7, projection="3d")
    scatter3d(ax7, raw_i, "tab:blue", alpha=0.25, label="i")
    scatter3d(ax7, raw_j_found, "tab:orange", alpha=0.25, label="j (pose trouvée)")
    scatter3d(ax7, frac_i, "red", s=6, alpha=0.9)
    scatter3d(ax7, frac_j_found, "limegreen", s=6, alpha=0.9)
    ax7.set_title(f"POSE TROUVÉE — RotErr={re:.1f}°  TransErr={te:.3f}  "
                  f"{'SUCCÈS' if pose30 else 'ÉCHEC'} @30°/0.1")

    # 8 : vraie pose GT
    ax8 = fig.add_subplot(2, 4, 8, projection="3d")
    scatter3d(ax8, raw_i, "tab:blue", alpha=0.25, label="i")
    scatter3d(ax8, raw_j_gt, "tab:orange", alpha=0.25, label="j (vraie pose GT)")
    scatter3d(ax8, frac_i, "red", s=6, alpha=0.9)
    scatter3d(ax8, frac_j_gt, "limegreen", s=6, alpha=0.9)
    ax8.set_title("VRAIE POSE (GT) — référence")

    fig.suptitle(
        f"{meta} | strategy={args.strategy} score_mode={args.score_mode} "
        f"dilate_px={args.dilate_px} gaussian_sigma_px={args.gaussian_sigma_px} | "
        f"OvlpFrac trouvé={best_overlap_frac:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return out_path, {
        "rot_err": re, "trans_err": te, "pose_30deg_0.1": pose30,
        "overlap_frac": best_overlap_frac, "n_frac_pts": (len(frac_i), len(frac_j)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resolution",  type=int, default=64)
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--max_planarity", type=float, default=0.15)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--dilate_px",   type=int, default=0)
    parser.add_argument("--gaussian_sigma_px", type=float, default=0.0)
    parser.add_argument("--strategy",    default="gt", choices=["gt", "thresh0.3", "random"])
    parser.add_argument("--filter",      default="any", choices=["any", "success", "fail"],
                        help="any = les N premières paires valides ; success/fail = ne "
                             "garder que les paires où Pose@30 est vrai/faux.")
    parser.add_argument("--n_vis",       type=int, default=6)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.train_dataset
    loader = DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    n_saved = 0
    n_seen_2frag = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if n_saved >= args.n_vis:
                break
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"])
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()

            valid_np = valid_pcs.cpu().numpy()
            B, P = valid_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_np[b, p]]
            if len(bp_pairs) != 2:
                continue
            n_seen_2frag += 1

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            quats_np = batch["quaternions"].numpy()
            trans_np = batch["translations"].numpy()

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            gt_per_k    = [gt_flat[offsets[k]:offsets[k+1]]   for k in range(K)]
            score_per_k = [pred_flat[offsets[k]:offsets[k+1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()          for k in range(K)]

            (k0, p0), (k1, p1) = bp_pairs
            raw_i = pc_per_k[k0] * scale_np[0, p0]
            raw_j = pc_per_k[k1] * scale_np[0, p1]
            gt_i, gt_j = gt_per_k[k0], gt_per_k[k1]
            sc_i, sc_j = score_per_k[k0], score_per_k[k1]

            R0 = quat_wxyz_to_rotmat(quats_np[0, p0])
            R1 = quat_wxyz_to_rotmat(quats_np[0, p1])
            R_ij = R1.T @ R0
            t_ij = R1.T @ (trans_np[0, p0] - trans_np[0, p1])

            frac_i, frac_j = make_masks(raw_i, raw_j, gt_i, gt_j, sc_i, sc_j, args.strategy, rng)
            if len(frac_i) < MIN_FRAC_POINTS or len(frac_j) < MIN_FRAC_POINTS:
                continue

            out_path = out_dir / f"pair_{n_seen_2frag:04d}.png"
            saved, info = visualize_pair(raw_i, raw_j, frac_i, frac_j, R_ij, t_ij,
                                         args, out_path, meta=f"objet#{n_seen_2frag}")
            if saved is None:
                continue   # skip (too_curved / sparse_dmap / no_overlap_3d)

            if args.filter == "success" and not info["pose_30deg_0.1"]:
                out_path.unlink(missing_ok=True)
                continue
            if args.filter == "fail" and info["pose_30deg_0.1"]:
                out_path.unlink(missing_ok=True)
                continue

            n_saved += 1
            print(f"  [{n_saved}/{args.n_vis}] {out_path.name} — "
                  f"RotErr={info['rot_err']:.1f}° TransErr={info['trans_err']:.3f} "
                  f"Pose@30={'OUI' if info['pose_30deg_0.1'] else 'non'} "
                  f"OvlpFrac={info['overlap_frac']:.3f} n_pts={info['n_frac_pts']}")

    print(f"\n{n_saved} figures sauvegardées dans {out_dir}")


if __name__ == "__main__":
    main()
