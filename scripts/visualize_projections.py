"""
scripts/visualize_projections.py
==================================
Visualise quelques fragments du dataset Breaking Bad via le vrai dataloader GARF :
  - colonne gauche  : nuage de points 3D coloré par label GT (rouge=fracture, bleu=intact)
  - colonnes droite : 3 projections orthographiques (front/side/top) — depth map

Usage (depuis le dossier code/) :
    python scripts/visualize_projections.py \
        --data_root /tmp/student7/garf/data/breaking_bad_vol.hdf5 \
        --n_objects 4 \
        --out_dir output/visualizations
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # pas de display sur serveur headless
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assembly.data.breaking_bad import BreakingBadDataModule
from assembly.models.projection_3d_to_2d import project_fragment
from assembly.models.projection_mapping_utils import extract_fragment_list


VIEW_NAMES = ["Front (XY)", "Side (ZY)", "Top (XZ)"]


def visualize_object(batch: dict, batch_idx: int, out_path: str, resolution: int = 128):
    """
    Visualise tous les fragments valides d'un objet dans le batch.
    batch_idx : index de l'objet dans le batch.
    """
    pc_4d = batch["pointclouds"][batch_idx]          # (P, num_pts, 3)
    ppp_1d = batch["points_per_part"][batch_idx]     # (P,)
    gt_flat = batch["fracture_surface_gt"][batch_idx]  # (P * num_pts,)

    P, num_pts = pc_4d.shape[0], pc_4d.shape[1]
    valid_mask = ppp_1d > 0                          # (P,) bool
    K = int(valid_mask.sum())
    if K == 0:
        print(f"  objet {batch_idx}: aucun fragment valide, ignoré.")
        return

    frag_list = [pc_4d[p] for p in range(P) if valid_mask[p]]  # list of (num_pts, 3)
    gt_2d     = gt_flat.view(P, num_pts)
    gt_frags  = gt_2d[valid_mask]                    # (K, num_pts)

    n_cols = 1 + 3   # 3D + 3 vues
    n_rows = min(K, 6)   # max 6 fragments par figure

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"Objet #{batch_idx} — {K} fragments valides", fontsize=13, y=1.01)

    for k in range(n_rows):
        pts_k = frag_list[k]               # (num_pts, 3)
        gt_k  = gt_frags[k].numpy()        # (num_pts,)

        images, _, _, _ = project_fragment(
            pts_k,
            normals=None,
            num_views=3,
            resolution=resolution,
            splat_sigma=0.0,
            use_bilinear=False,
        )
        # images : (3, 2, H, W)   ch0=depth, ch1=occupancy

        # ── 3D scatter ──────────────────────────────────────────────────
        ax3d = fig.add_subplot(n_rows, n_cols, k * n_cols + 1, projection="3d")
        pts_np = pts_k.numpy()
        colors = ["#e74c3c" if g == 1 else "#3498db" for g in gt_k]
        ax3d.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                     c=colors, s=1.5, alpha=0.6, linewidths=0)
        frac_ratio = int(gt_k.sum())
        ax3d.set_title(f"Frag {k+1}  —  {frac_ratio}/{len(gt_k)} pts fracture",
                       fontsize=8)
        ax3d.set_xlabel("X", fontsize=6); ax3d.set_ylabel("Y", fontsize=6)
        ax3d.set_zlabel("Z", fontsize=6)
        ax3d.tick_params(labelsize=5)

        # ── 3 vues 2D ───────────────────────────────────────────────────
        for v, vname in enumerate(VIEW_NAMES):
            depth_img = images[v, 0].numpy()   # (H, W)
            occ_img   = images[v, 1].numpy()   # (H, W)

            depth_masked = np.where(occ_img > 0, depth_img, np.nan)

            ax = fig.add_subplot(n_rows, n_cols, k * n_cols + 2 + v)
            im = ax.imshow(depth_masked, cmap="plasma", origin="upper",
                           vmin=0, vmax=1, interpolation="nearest")
            ax.contour(occ_img, levels=[0.5], colors="white",
                       linewidths=0.5, alpha=0.4)
            occupied = int((occ_img > 0).sum())
            ax.set_title(f"Frag {k+1} — {vname}\n{occupied} px occupés", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Légende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="Surface de fracture"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=8, label="Surface intacte"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--n_objects", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--out_dir", default="output/visualizations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Chargement du datamodule...")
    dm = BreakingBadDataModule(
        data_root=args.data_root,
        categories=["everyday"],
        min_parts=2,
        max_parts=8,          # limité pour des figures lisibles
        num_points_to_sample=1024,  # moins de points pour aller vite
        min_points_per_part=20,
        sample_method="uniform",
        batch_size=args.n_objects,
        num_workers=0,
        num_removal=0,
        num_redundancy=0,
        multi_ref=False,
    )
    dm.setup("fit")
    loader = dm.val_dataloader()

    batch = next(iter(loader))
    B = batch["pointclouds"].shape[0]
    print(f"Batch chargé : {B} objets, shape={batch['pointclouds'].shape}")

    for i in range(min(args.n_objects, B)):
        print(f"[{i+1}/{min(args.n_objects, B)}] Visualisation objet {i}...")
        out_path = os.path.join(args.out_dir, f"obj_{i:03d}.png")
        visualize_object(batch, i, out_path, resolution=args.resolution)

    print(f"\nTerminé. Figures dans {args.out_dir}/")


if __name__ == "__main__":
    main()
