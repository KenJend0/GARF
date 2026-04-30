"""
scripts/analyze_errors.py
==========================
Post-hoc error analysis for CNN fracture segmentation models.

Runs the model on the validation set and computes per-fragment metrics.
Generates:
  - Statistical tables: F1/Prec/Rec by fragment size, fracture ratio, n_parts
  - Error plots: FP/FN distributions
  - Qualitative visualizations: best and worst fragments (GT vs Pred vs Error)

Usage:
    python scripts/analyze_errors.py \
        --ckpt /tmp/student7/output/cnn_step9_geo_features/version_0/checkpoints/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step9_geo_features \
        --out_dir /tmp/student7/analysis/step9 \
        --split val \
        --n_vis 6 \
        --max_batches 200
"""

import argparse
import functools
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # headless server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.serialization
torch.serialization.add_safe_globals([functools.partial])

# Make sure codebase is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx], replace=True)

from assembly.models.projection_mapping_utils import extract_fragment_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def per_fragment_metrics(pred_flat, gt_flat, frag_sizes):
    """
    Split flat (N_sum,) tensors by fragment and compute per-fragment metrics.

    Returns list of dicts, one per fragment.
    """
    records = []
    offset = 0
    for sz in frag_sizes:
        p = pred_flat[offset: offset + sz]
        g = gt_flat[offset: offset + sz]
        offset += sz

        pred_b = (p > 0.5)
        tp = int((pred_b & (g == 1)).sum())
        fp = int((pred_b & (g == 0)).sum())
        fn = int((~pred_b & (g == 1)).sum())
        tn = int((~pred_b & (g == 0)).sum())

        n_frac = int((g == 1).sum())
        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = safe_div(2 * prec * rec, prec + rec)

        records.append({
            "n_pts":          sz,
            "n_fracture":     n_frac,
            "fracture_ratio": safe_div(n_frac, sz),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            # store tensors for visualization
            "_pred": p.cpu(),
            "_gt":   g.cpu(),
        })
    return records


def bin_stats(records, key, n_bins=4):
    """
    Bin fragments by `key` and compute mean F1/Prec/Rec per bin.
    Returns list of (bin_label, mean_f1, mean_prec, mean_rec, count).
    """
    vals = np.array([r[key] for r in records])
    if key == "n_parts":
        groups = defaultdict(list)
        for r in records:
            label = str(int(r[key])) if int(r[key]) < 6 else "6+"
            groups[label].append(r)
        bins = [str(u) for u in sorted(int(k) for k in groups if k != "6+")]
        if "6+" in groups:
            bins.append("6+")
        return [
            (
                b,
                np.mean([r["f1"] for r in groups[b]]),
                np.mean([r["precision"] for r in groups[b]]),
                np.mean([r["recall"] for r in groups[b]]),
                len(groups[b]),
            )
            for b in bins
        ]
    else:
        quantiles = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
        quantiles[-1] += 1e-6   # include max
        results = []
        for i in range(n_bins):
            lo, hi = quantiles[i], quantiles[i + 1]
            mask = (vals >= lo) & (vals < hi)
            subset = [r for r, m in zip(records, mask) if m]
            if not subset:
                continue
            label = f"[{lo:.2f}, {hi:.2f})"
            if key == "n_pts":
                label = f"[{int(lo)}, {int(hi)})"
            results.append((
                label,
                np.mean([r["f1"] for r in subset]),
                np.mean([r["precision"] for r in subset]),
                np.mean([r["recall"] for r in subset]),
                len(subset),
            ))
        return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bin_metrics(stats, title, xlabel, out_path):
    labels  = [s[0] for s in stats]
    f1s     = [s[1] for s in stats]
    precs   = [s[2] for s in stats]
    recs    = [s[3] for s in stats]
    counts  = [s[4] for s in stats]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    ax.bar(x - w, f1s,   width=w, label="F1",        color="#2196F3")
    ax.bar(x,     precs,  width=w, label="Precision", color="#FF9800")
    ax.bar(x + w, recs,   width=w, label="Recall",    color="#4CAF50")

    for i, (f, p, r, c) in enumerate(zip(f1s, precs, recs, counts)):
        ax.text(x[i] - w, f + 0.005, f"{f:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i],     p + 0.005, f"{p:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + w, r + 0.005, f"{r:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i], 0.01, f"n={c}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_fp_fn_dist(records, out_path):
    fp_rates = [r["fp"] / max(r["n_pts"], 1) for r in records]
    fn_rates = [r["fn"] / max(r["n_fracture"], 1) for r in records if r["n_fracture"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(fp_rates, bins=30, color="#FF5722", edgecolor="white", linewidth=0.5)
    axes[0].set_title("False Positive Rate per Fragment\n(FP / n_pts)")
    axes[0].set_xlabel("FP rate")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(fp_rates), color="black", linestyle="--",
                    label=f"mean={np.mean(fp_rates):.3f}")
    axes[0].legend()

    axes[1].hist(fn_rates, bins=30, color="#9C27B0", edgecolor="white", linewidth=0.5)
    axes[1].set_title("False Negative Rate per Fragment\n(FN / n_fracture)")
    axes[1].set_xlabel("FN rate")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(fn_rates), color="black", linestyle="--",
                    label=f"mean={np.mean(fn_rates):.3f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_qualitative(records, xyz_list, n_vis, out_dir, tag):
    """
    For the n_vis best and n_vis worst fragments (by F1), save 3-panel figures:
      [GT fracture] [Prediction] [Error: FP=red, FN=blue, TP=green]
    Uses XY projection for readability.
    """
    # Filter fragments with at least some fracture points
    filtered = [(i, r) for i, r in enumerate(records) if r["n_fracture"] > 5]
    if not filtered:
        return

    by_f1 = sorted(filtered, key=lambda x: x[1]["f1"])
    selected = by_f1[:n_vis] + by_f1[-n_vis:]   # worst + best

    for rank, (idx, rec) in enumerate(selected):
        quality = "worst" if rank < n_vis else "best"
        xyz = xyz_list[idx]
        pred_b = (rec["_pred"] > 0.5).numpy()
        gt_b   = rec["_gt"].numpy().astype(bool)

        # Color map: TP=green, FP=red, FN=blue, TN=lightgray
        colors = np.full((len(gt_b), 3), [0.85, 0.85, 0.85])   # TN gray
        colors[gt_b & pred_b]   = [0.2,  0.8,  0.2]            # TP green
        colors[~gt_b & pred_b]  = [0.9,  0.1,  0.1]            # FP red
        colors[gt_b  & ~pred_b] = [0.1,  0.3,  0.9]            # FN blue

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        views = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
        for ax, (ax1, ax2, vlabel) in zip(axes, views):
            # GT fracture
            pass

        # Single-panel with error map (XY projection)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            xyz[:, 0], xyz[:, 1],
            c=colors, s=2, linewidths=0,
        )
        patches = [
            mpatches.Patch(color=[0.2, 0.8, 0.2], label=f"TP={rec['tp']}"),
            mpatches.Patch(color=[0.9, 0.1, 0.1], label=f"FP={rec['fp']}"),
            mpatches.Patch(color=[0.1, 0.3, 0.9], label=f"FN={rec['fn']}"),
            mpatches.Patch(color=[0.85, 0.85, 0.85], label=f"TN={rec['tn']}"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        ax.set_title(
            f"{quality.upper()} #{rank % n_vis + 1}  |  "
            f"F1={rec['f1']:.3f}  Prec={rec['precision']:.3f}  Rec={rec['recall']:.3f}\n"
            f"n_pts={rec['n_pts']}  frac_ratio={rec['fracture_ratio']:.2f}"
        )
        ax.set_aspect("equal")
        ax.axis("off")

        fname = out_dir / f"{tag}_{quality}_{rank % n_vis + 1:02d}_f1{rec['f1']:.3f}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()

    print(f"  Saved {2 * n_vis} qualitative figures to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True,  help="Path to .ckpt checkpoint")
    p.add_argument("--data_root",   required=True,  help="Path to breaking_bad_vol.hdf5")
    p.add_argument("--experiment",  required=True,  help="Hydra experiment name (e.g. cnn_step9_geo_features)")
    p.add_argument("--out_dir",     default="/tmp/student7/analysis", help="Output directory for plots")
    p.add_argument("--split",       default="val",  choices=["val", "test"])
    p.add_argument("--n_vis",       type=int, default=6, help="Qualitative examples per category")
    p.add_argument("--max_batches", type=int, default=300, help="Max batches to process (0=all)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sweep_threshold", action="store_true",
                   help="Sweep decision threshold and print F1/Prec/Rec table")
    return p.parse_args()


def load_config_and_model(args):
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={args.experiment}",
                f"data.data_root={args.data_root}",
                f"data.batch_size={args.batch_size}",
                f"data.num_workers={args.num_workers}",
            ],
        )
    return cfg


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.ckpt}")

    # --- Load config + datamodule ---
    print("\nLoading Hydra config...")
    cfg = load_config_and_model(args)

    print("Instantiating datamodule...")
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    # --- Load model ---
    print("Loading model from checkpoint...")
    from assembly.models.cnn_segmentation_model import CNNFracSeg
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Inference loop ---
    all_records = []
    all_xyz     = []
    n_parts_per_frag = []

    print(f"\nRunning inference on {args.split} set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}...")

            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Extract fragment metadata BEFORE forward (we need sizes + n_parts)
            frag_list, valid_pcs, K = extract_fragment_list(
                batch["pointclouds"], batch["points_per_part"]
            )
            if K == 0:
                continue

            frag_sizes = [f.shape[0] for f in frag_list]

            # n_parts per fragment: count non-zero parts in the same batch item
            B, P = valid_pcs.shape
            batch_idx_per_frag = []
            for b in range(B):
                for p_idx in range(P):
                    if valid_pcs[b, p_idx]:
                        batch_idx_per_frag.append(b)
            n_parts_in_obj = [
                int(valid_pcs[b].sum().item())
                for b in batch_idx_per_frag
            ]

            # Forward
            out = model(batch)
            pred_flat = out["coarse_seg_pred"].float()
            gt_flat   = out["coarse_seg_gt"].long()

            # Split by fragment
            records = per_fragment_metrics(pred_flat, gt_flat, frag_sizes)

            for rec, xyz_k, n_parts in zip(records, frag_list, n_parts_in_obj):
                rec["n_parts"] = n_parts
                all_records.append(rec)
                all_xyz.append(xyz_k.cpu())
                n_parts_per_frag.append(n_parts)

    print(f"\nAnalyzed {len(all_records)} fragments total.")

    # Strip tensor fields before numpy processing
    records_clean = [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in all_records]

    # --- Global stats ---
    f1s   = [r["f1"]        for r in records_clean]
    precs = [r["precision"] for r in records_clean]
    recs  = [r["recall"]    for r in records_clean]

    print("\n" + "=" * 52)
    print("GLOBAL STATS (fragment-level)")
    print("=" * 52)
    print(f"  Mean F1:        {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
    print(f"  Mean Precision: {np.mean(precs):.4f}  ± {np.std(precs):.4f}")
    print(f"  Mean Recall:    {np.mean(recs):.4f}  ± {np.std(recs):.4f}")
    print(f"  Median F1:      {np.median(f1s):.4f}")
    print(f"  Fragments with F1 > 0.9: {sum(f>0.9 for f in f1s)}/{len(f1s)}")
    print(f"  Fragments with F1 < 0.5: {sum(f<0.5 for f in f1s)}/{len(f1s)}")

    total_fp = sum(r["fp"] for r in records_clean)
    total_fn = sum(r["fn"] for r in records_clean)
    total_tp = sum(r["tp"] for r in records_clean)
    print(f"\n  Global TP={total_tp:,}  FP={total_fp:,}  FN={total_fn:,}")
    print(f"  FP / (FP+TP) = {total_fp/(total_fp+total_tp+1e-8):.3f}  ← false discovery rate")
    print(f"  FN / (FN+TP) = {total_fn/(total_fn+total_tp+1e-8):.3f}  ← miss rate")

    # --- Analysis by fragment size ---
    print("\n[F1 by Fragment Size (n_pts)]")
    stats_size = bin_stats(all_records, "n_pts", n_bins=4)
    for label, f1, prec, rec, cnt in stats_size:
        print(f"  {label:<20}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (n={cnt})")
    plot_bin_metrics(stats_size, "F1 / Precision / Recall by Fragment Size",
                     "Fragment size (n_pts)", out_dir / "metrics_by_size.png")

    # --- Analysis by fracture ratio ---
    print("\n[F1 by Fracture Surface Ratio]")
    stats_frac = bin_stats(all_records, "fracture_ratio", n_bins=4)
    for label, f1, prec, rec, cnt in stats_frac:
        print(f"  {label:<20}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (n={cnt})")
    plot_bin_metrics(stats_frac, "F1 / Precision / Recall by Fracture Ratio",
                     "Fracture surface ratio", out_dir / "metrics_by_fracture_ratio.png")

    # --- Analysis by number of parts ---
    print("\n[F1 by Number of Object Parts]")
    stats_parts = bin_stats(all_records, "n_parts", n_bins=4)
    for label, f1, prec, rec, cnt in stats_parts:
        print(f"  {label:<20}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (n={cnt})")
    plot_bin_metrics(stats_parts, "F1 / Precision / Recall by Number of Parts",
                     "Number of parts in object", out_dir / "metrics_by_n_parts.png")

    # --- FP/FN distribution ---
    plot_fp_fn_dist(all_records, out_dir / "fp_fn_distribution.png")

    # --- Qualitative visualizations ---
    print(f"\nGenerating qualitative visualizations (n_vis={args.n_vis})...")
    plot_qualitative(all_records, all_xyz, args.n_vis, out_dir, tag="step9")

    # --- F1 scatter: fracture_ratio vs F1 (colored by n_parts) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    fr  = np.array([r["fracture_ratio"] for r in all_records])
    f1v = np.array([r["f1"]             for r in all_records])
    npt = np.array([r["n_parts"]         for r in all_records])
    sc  = ax.scatter(fr, f1v, c=npt, cmap="viridis", s=6, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="n_parts")
    ax.set_xlabel("Fracture surface ratio")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs Fracture Ratio (color = n_parts)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_vs_fracture_ratio_scatter.png", dpi=120)
    plt.close()
    print(f"  Saved: {out_dir}/f1_vs_fracture_ratio_scatter.png")

    # --- Threshold sweep (optional) ---
    if args.sweep_threshold:
        print("\n[Threshold Sweep — Global F1 / Precision / Recall]")
        print(f"  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'FDR':>7}")
        print("  " + "-" * 42)
        all_pred = torch.cat([r["_pred"] for r in all_records])
        all_gt   = torch.cat([r["_gt"].float()   for r in all_records])
        for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            pb = (all_pred > thresh)
            tp = int((pb & (all_gt == 1)).sum())
            fp = int((pb & (all_gt == 0)).sum())
            fn = int((~pb & (all_gt == 1)).sum())
            prec = safe_div(tp, tp + fp)
            rec  = safe_div(tp, tp + fn)
            f1   = safe_div(2 * prec * rec, prec + rec)
            fdr  = safe_div(fp, fp + tp)
            marker = " ←" if thresh == 0.50 else ""
            print(f"  {thresh:>7.2f}  {f1:>7.4f}  {prec:>7.4f}  {rec:>7.4f}  {fdr:>7.4f}{marker}")

        # Also sweep by fracture_ratio bin to find optimal threshold per group
        print("\n[Threshold Sweep — Low fracture ratio fragments only (<0.14)]")
        low_frac = [r for r in all_records if r["fracture_ratio"] < 0.14]
        if low_frac:
            lf_pred = torch.cat([r["_pred"] for r in low_frac])
            lf_gt   = torch.cat([r["_gt"].float()   for r in low_frac])
            print(f"  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
            print("  " + "-" * 33)
            for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                pb = (lf_pred > thresh)
                tp = int((pb & (lf_gt == 1)).sum())
                fp = int((pb & (lf_gt == 0)).sum())
                fn = int((~pb & (lf_gt == 1)).sum())
                prec = safe_div(tp, tp + fp)
                rec  = safe_div(tp, tp + fn)
                f1   = safe_div(2 * prec * rec, prec + rec)
                print(f"  {thresh:>7.2f}  {f1:>7.4f}  {prec:>7.4f}  {rec:>7.4f}")

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
