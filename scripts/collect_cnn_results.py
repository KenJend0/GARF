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
    ("cnn_step8c_feat_concat",          "Step 8c + FeatFuse(concat)"),
    ("cnn_step8d_simplecnn_feat_mean",  "Step 8d + SimpleCNN+Fuse(mean)"),
]

# Hard-coded results for steps whose CSV output is no longer available.
# Format: step_key → {"f1": {...}, "precision": {...}, "recall": {...}}
# Each inner dict has keys matching VAL_METRICS.
HARDCODED: dict[str, dict[str, dict]] = {
    "cnn_step1_baseline": {
        "f1":        {"val/coarse_seg_acc": 0.7897, "val/coarse_seg_f1": 0.6805, "val/coarse_seg_precision": 0.6029, "val/coarse_seg_recall": 0.8107},
        "precision": {"val/coarse_seg_acc": 0.7897, "val/coarse_seg_f1": 0.6805, "val/coarse_seg_precision": 0.6029, "val/coarse_seg_recall": 0.8107},
        "recall":    {"val/coarse_seg_acc": 0.5428, "val/coarse_seg_f1": 0.5207, "val/coarse_seg_precision": 0.3797, "val/coarse_seg_recall": 0.9165},
    },
    "cnn_step2_normals": {
        "f1":        {"val/coarse_seg_acc": 0.7988, "val/coarse_seg_f1": 0.7090, "val/coarse_seg_precision": 0.6139, "val/coarse_seg_recall": 0.8732},
        "precision": {"val/coarse_seg_acc": 0.7983, "val/coarse_seg_f1": 0.7000, "val/coarse_seg_precision": 0.6145, "val/coarse_seg_recall": 0.8484},
        "recall":    {"val/coarse_seg_acc": 0.7528, "val/coarse_seg_f1": 0.6774, "val/coarse_seg_precision": 0.5575, "val/coarse_seg_recall": 0.9096},
    },
    "cnn_step3_splatting": {
        "f1":        {"val/coarse_seg_acc": 0.8240, "val/coarse_seg_f1": 0.7193, "val/coarse_seg_precision": 0.6338, "val/coarse_seg_recall": 0.8715},
        "precision": {"val/coarse_seg_acc": 0.8089, "val/coarse_seg_f1": 0.6846, "val/coarse_seg_precision": 0.6340, "val/coarse_seg_recall": 0.8001},
        "recall":    {"val/coarse_seg_acc": 0.7954, "val/coarse_seg_f1": 0.7005, "val/coarse_seg_precision": 0.5892, "val/coarse_seg_recall": 0.9113},
    },
    "cnn_step4_bilinear": {
        "f1":        {"val/coarse_seg_acc": 0.8242, "val/coarse_seg_f1": 0.7200, "val/coarse_seg_precision": 0.6306, "val/coarse_seg_recall": 0.8732},
        "precision": {"val/coarse_seg_acc": 0.8300, "val/coarse_seg_f1": 0.7171, "val/coarse_seg_precision": 0.6438, "val/coarse_seg_recall": 0.8450},
        "recall":    {"val/coarse_seg_acc": 0.7956, "val/coarse_seg_f1": 0.6981, "val/coarse_seg_precision": 0.5841, "val/coarse_seg_recall": 0.9115},
    },
    "cnn_step5_context": {
        "f1":        {"val/coarse_seg_acc": 0.8447, "val/coarse_seg_f1": 0.7272, "val/coarse_seg_precision": 0.6742, "val/coarse_seg_recall": 0.8269},
        "precision": {"val/coarse_seg_acc": 0.8386, "val/coarse_seg_f1": 0.6907, "val/coarse_seg_precision": 0.6869, "val/coarse_seg_recall": 0.7399},
        "recall":    {"val/coarse_seg_acc": 0.8248, "val/coarse_seg_f1": 0.7262, "val/coarse_seg_precision": 0.6286, "val/coarse_seg_recall": 0.8882},
    },
}

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

    def get_best_rows(step_key, seeds, metric):
        """Return list of best-row dicts, falling back to HARDCODED if no CSV found."""
        rows_per_seed = []
        for seed in seeds:
            csvs = find_csv_files(step_key, seed, output_dir)
            if not csvs:
                continue
            all_rows = []
            for p in csvs:
                all_rows.extend(load_csv(p))
            best = best_row_by(all_rows, metric)
            if best:
                rows_per_seed.append(best)

        if not rows_per_seed and step_key in HARDCODED:
            block = {"f1": "val/coarse_seg_f1", "precision": "val/coarse_seg_precision", "recall": "val/coarse_seg_recall"}
            key = next(k for k, v in block.items() if v == metric)
            rows_per_seed = [HARDCODED[step_key][key]]

        return rows_per_seed

    # Best F1
    print_block("Best val F1 epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = get_best_rows(step_key, seeds, "val/coarse_seg_f1")

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data — run train_cnn_segmentation.py first)")
            continue

        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        n_seeds = len(rows_per_seed)
        suffix = f"  (n={n_seeds} seeds)" if n_seeds > 1 else ""
        hardcoded_marker = " *" if step_key in HARDCODED and not find_csv_files(step_key, seeds[0], output_dir) else ""
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals) + hardcoded_marker + suffix)

    # Best Precision (isolated)
    print_block("Best val Precision epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = get_best_rows(step_key, seeds, "val/coarse_seg_precision")

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data)")
            continue

        hardcoded_marker = " *" if step_key in HARDCODED and not find_csv_files(step_key, seeds[0], output_dir) else ""
        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals) + hardcoded_marker)

    # Best Recall (isolated)
    print_block("Best val Recall epoch — Acc / F1 / Prec / Rec")
    for step_key, step_label in STEPS:
        rows_per_seed = get_best_rows(step_key, seeds, "val/coarse_seg_recall")

        if not rows_per_seed:
            print(f"  {step_label:<{col_w}}  (no data)")
            continue

        hardcoded_marker = " *" if step_key in HARDCODED and not find_csv_files(step_key, seeds[0], output_dir) else ""
        vals = [format_val(avg_rows(rows_per_seed, m)) for m in VAL_METRICS]
        print(f"  {step_label:<{col_w}}" + "".join(f"{v:>{metric_w}}" for v in vals) + hardcoded_marker)

    print("  (* hard-coded from previous run)")

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
