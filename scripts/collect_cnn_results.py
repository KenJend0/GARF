"""
scripts/collect_cnn_results.py
================================
Reads CSVLogger metrics.csv from each CNN ablation step and prints a
comparison table: Acc / F1 / Precision / Recall (val, best epoch).

Three blocks are reported per step (tutor requirement):
  1. Row at best val/coarse_seg_f1 epoch       — overall best epoch
  2. Best val/coarse_seg_precision epoch       — isolated precision maximum
  3. Best val/coarse_seg_recall epoch          — isolated recall maximum

Blocks 2 and 3 are critical for step 3 (splatting) where F1 may improve
but precision may drop (blur smears thin fracture surfaces).

Multi-seed support:
  Pass --seeds 1116 42 2024 to average metrics across seed runs.
  Expects output dirs named cnn_step{N}_{name}_seed{SEED} for non-default seeds.

Usage:
    # Single seed (default 1116):
    python scripts/collect_cnn_results.py

    # Multi-seed average:
    python scripts/collect_cnn_results.py --seeds 1116 42 2024

Output:
    stdout — formatted table ready to copy into a report
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Optional


STEPS = [
    ("cnn_step1_baseline",    "Step 1 - Baseline"),
    ("cnn_step2_normals",     "Step 2 + Normals"),
    ("cnn_step3_splatting",   "Step 3 + Splatting"),
    ("cnn_step4_bilinear",    "Step 4 + Bilinear"),
    ("cnn_step5_context",     "Step 5 + Context"),
    ("cnn_step6_attention",   "Step 6 + Attention"),
    ("cnn_step7_unet",        "Step 7 + U-Net"),
    ("cnn_step8a_feat_mean",  "Step 8a + FeatFuse(mean)"),
    ("cnn_step8b_feat_max",   "Step 8b + FeatFuse(max)"),
    ("cnn_step8c_feat_concat","Step 8c + FeatFuse(concat)"),
]

VAL_METRICS = [
    "val/coarse_seg_acc",
    "val/coarse_seg_f1",
    "val/coarse_seg_precision",
    "val/coarse_seg_recall",
]

SHORT_NAMES = {
    "val/coarse_seg_acc":       "Acc",
    "val/coarse_seg_f1":        "F1",
    "val/coarse_seg_precision": "Prec",
    "val/coarse_seg_recall":    "Rec",
}


def load_csv(csv_path: str) -> list:
    """Load all rows from a metrics.csv file. Returns list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def best_row_by(rows: list, metric: str) -> dict:
    """Return the row with the highest value for `metric`, ignoring NaN / missing."""
    best_val = -1.0
    best_row = {}
    for row in rows:
        try:
            v = float(row[metric])
        except (KeyError, ValueError):
            continue
        if v > best_val:
            best_val = v
            best_row = row
    return best_row


def find_csv_files(step_name: str, seed: int, output_dir: str = "output") -> list:
    """Find all metrics.csv files for a given (step, seed) combination."""
    if seed == 1116:
        dir_name = step_name
    else:
        dir_name = f"{step_name}_seed{seed}"
    pattern = os.path.join(output_dir, dir_name, "version_*", "metrics.csv")
    return sorted(glob.glob(pattern))


def safe_float(row: dict, key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return None


def avg_rows(rows_list: list, metric: str) -> Optional[float]:
    """Average a metric value across multiple best-row dicts. Returns None if all missing."""
    vals = []
    for row in rows_list:
        v = safe_float(row, metric)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def format_val(v) -> str:
    if v is None:
        return "  N/A  "
    return f"{v:.4f}"


def print_table(seeds: list, output_dir: str = "output"):
    col_w = 28
    metric_w = 8

    header = f"{'Step':<{col_w}}" + "".join(f"{SHORT_NAMES[m]:>{metric_w}}" for m in VAL_METRICS)
    sep    = "-" * len(header)

    def print_block(title: str):
        print(f"\n  [{title}]")
        print("  " + header)
        print("  " + sep)

    # Best F1
    print_block("Best val F1 epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = []
        for seed in seeds:
            csvs = find_csv_files(step_key, seed, output_dir)
            if not csvs:
                continue
            all_rows = []
            for p in csvs:
                all_rows.extend(load_csv(p))
            best = best_row_by(all_rows, "val/coarse_seg_f1")
            if best:
                rows_per_seed.append(best)

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data — run train_cnn_segmentation.py first)")
            continue

        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        n_seeds = len(rows_per_seed)
        suffix = f"  (n={n_seeds} seeds)" if n_seeds > 1 else ""
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals) + suffix)

    # Best Precision (isolated)
    print_block("Best val Precision epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = []
        for seed in seeds:
            csvs = find_csv_files(step_key, seed, output_dir)
            if not csvs:
                continue
            all_rows = []
            for p in csvs:
                all_rows.extend(load_csv(p))
            best = best_row_by(all_rows, "val/coarse_seg_precision")
            if best:
                rows_per_seed.append(best)

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data)")
            continue

        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals))

    # Best Recall (isolated)
    print_block("Best val Recall epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = []
        for seed in seeds:
            csvs = find_csv_files(step_key, seed, output_dir)
            if not csvs:
                continue
            all_rows = []
            for p in csvs:
                all_rows.extend(load_csv(p))
            best = best_row_by(all_rows, "val/coarse_seg_recall")
            if best:
                rows_per_seed.append(best)

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data)")
            continue

        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect CNN ablation results from CSVLogger output."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1116],
        metavar="SEED",
        help="Seed(s) to include. Default: 1116.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Root output directory containing cnn_step* folders. Default: ./output",
    )
    args = parser.parse_args()

    print(f"\nCNN Ablation Results — seeds: {args.seeds}")
    print(f"Reading from: {args.output_dir}/cnn_step*/version_*/metrics.csv\n")
    print_table(args.seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
