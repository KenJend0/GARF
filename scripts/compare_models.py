"""
scripts/compare_models.py
==========================
Merges structured summary JSON files (produced by analyze_errors.py
--summary_json) from multiple model/category runs into a single
comparison table — apples-to-apples, since every summary is computed
by the exact same metric code regardless of model_type (cnn|garf).

Usage:
    # Run analyze_errors.py once per model/category with --summary_json:
    python scripts/analyze_errors.py --ckpt output/cnn_step15_final_model/last.ckpt \\
        --experiment cnn_step15_final_model --model_type cnn --categories everyday \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --out_dir /tmp/student7/analysis/cnn_everyday --geometric --sweep_threshold \\
        --label "CNN Step15 (everyday)" --summary_json /tmp/student7/summaries/cnn_everyday.json

    python scripts/analyze_errors.py --ckpt /storage/student7/teyssir/checkpoints/GARF_mini.ckpt \\
        --experiment cnn_step15_final_model --model_type garf --categories everyday \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --out_dir /tmp/student7/analysis/garf_everyday --geometric --sweep_threshold \\
        --label "GARF-mini (everyday)" --summary_json /tmp/student7/summaries/garf_everyday.json

    # Then merge into one table:
    python scripts/compare_models.py /tmp/student7/summaries/*.json \\
        --out_dir output/comparison
"""

import argparse
import glob
import json
from pathlib import Path


def load_summaries(patterns: list) -> list:
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    summaries = []
    for p in paths:
        with open(p) as fh:
            summaries.append(json.load(fh))
    return summaries


def fmt(v, pct=False, decimals=1):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{decimals}f}%"
    return f"{v:.4f}"


def format_table(summaries: list) -> str:
    cols = [
        ("label",            "Model",          False),
        ("categories",       "Categories",     False),
        ("n_fragments",      "N frag.",        False),
        ("params",           "Params",         False),
        ("mean_f1",          "Mean F1",        True),
        ("median_f1",        "Median F1",      True),
        ("mean_precision",   "Mean Prec",      True),
        ("mean_recall",      "Mean Rec",       True),
        ("best_f1_pooled",   "F1 pooled (best thr)", True),
        ("best_threshold",   "Best thr",       False),
        ("boundary_f1_mean", "Boundary F1",    True),
        ("hausdorff_mean",   "Hausdorff",      False),
        ("chamfer_mean",     "Chamfer",        False),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for s in summaries:
        row = []
        for key, _, pct in cols:
            v = s.get(key)
            if key == "params" and v is not None:
                row.append(f"{v:,}")
            elif key == "n_fragments" and v is not None:
                row.append(f"{v:,}")
            elif key in ("label", "categories") and v is not None:
                row.append(str(v))
            elif key == "best_threshold" and v is not None:
                row.append(f"{v:.2f}")
            elif pct:
                row.append(fmt(v, pct=True))
            else:
                row.append(fmt(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+", help="Summary JSON files or glob patterns")
    p.add_argument("--out_dir", default="output/comparison")
    args = p.parse_args()

    summaries = load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"No summary JSON files matched: {args.summaries}")

    table = format_table(summaries)
    print(table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison.md"
    with open(md_path, "w") as fh:
        fh.write("# Model comparison\n\n")
        fh.write(table + "\n")
    print(f"\nSaved Markdown table to: {md_path}")


if __name__ == "__main__":
    main()
