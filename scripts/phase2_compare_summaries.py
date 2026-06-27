"""
scripts/phase2_compare_summaries.py
=====================================
Compare several --summary_json outputs from phase2_geometric_baseline.py and/or
phase2d_interface_clustering_diagnostic.py in compact tables, instead of
grepping/catting full text logs.

Usage:
    python scripts/phase2_compare_summaries.py /tmp/student7/phase2_results/*.json
"""

import argparse
import json
from pathlib import Path


# Phase 2 (geometric baseline matching) summary schema: {"strategies": {name: {...}}}
PHASE2_COLUMNS = [
    ("corr_prec", "{:.2%}"),
    ("ransac_valid_rate", "{:.2%}"),
    ("pose_success_30_0.1", "{:.2%}"),
    ("rot_err_deg", "{:.1f}"),
    ("trans_err", "{:.3f}"),
    ("inlier_ratio", "{:.2%}"),
    ("normal_dot_mean_inliers", "{:+.3f}"),
    ("score_gap", "{:+.2f}"),
]
PHASE2_HEADERS = ["Label", "Strategy", "CorrPrec", "RansacValid", "Pose@30", "RotErr", "TransErr", "InlierRatio", "NormalDot", "ScoreGap"]

# Phase 2D (interface clustering diagnostic) summary schema: {"metrics": {...}}
PHASE2D_COLUMNS = [
    ("mean_clusters_per_fragment", "{:.2f}"),
    ("mean_cluster_degree_ratio", "{:.2f}"),
    ("frac_fragments_clusters_ge_degree", "{:.2%}"),
    ("noise_rate", "{:.2%}"),
    ("cluster_purity_weighted", "{:.2%}"),
    ("mixed_cluster_rate", "{:.2%}"),
    ("edge_coverage", "{:.2%}"),
    ("best_cluster_corrprec", "{:.2%}"),
]
PHASE2D_HEADERS = ["Label", "Clusters/Frag", "Cluster/Deg", "Frag>=Deg", "Noise", "Purity", "MixedRate", "EdgeCov", "BestCorrPrec"]


def fmt(value, spec):
    if value is None:
        return "n/a"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def print_table(headers, rows):
    if not rows:
        return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  " + "-" * (len(line) - 2))
    for row in rows:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="Paths to --summary_json outputs")
    args = parser.parse_args()

    phase2_rows, phase2d_rows = [], []
    for path in args.json_files:
        data = json.loads(Path(path).read_text())
        label = data.get("label", Path(path).stem)
        if "strategies" in data:
            for strategy, metrics in data["strategies"].items():
                row = [label, strategy] + [fmt(metrics.get(col), spec) for col, spec in PHASE2_COLUMNS]
                phase2_rows.append(row)
        elif "metrics" in data:
            metrics = data["metrics"]
            row = [label] + [fmt(metrics.get(col), spec) for col, spec in PHASE2D_COLUMNS]
            phase2d_rows.append(row)
        else:
            print(f"  [warn] unrecognized summary schema in {path}, skipping")

    if phase2_rows:
        print("PHASE 2 — GEOMETRIC BASELINE MATCHING runs")
        print_table(PHASE2_HEADERS, phase2_rows)
    if phase2d_rows:
        print("PHASE 2D — INTERFACE CLUSTERING DIAGNOSTIC runs")
        print_table(PHASE2D_HEADERS, phase2d_rows)


if __name__ == "__main__":
    main()
