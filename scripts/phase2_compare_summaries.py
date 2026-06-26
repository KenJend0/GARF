"""
scripts/phase2_compare_summaries.py
=====================================
Compare several --summary_json outputs from phase2_geometric_baseline.py in one
compact table, instead of grepping/catting full text logs.

Usage:
    python scripts/phase2_compare_summaries.py /tmp/student7/phase2_results/*.json
"""

import argparse
import json
from pathlib import Path


COLUMNS = [
    ("corr_prec", "{:.2%}"),
    ("ransac_valid_rate", "{:.2%}"),
    ("pose_success_30_0.1", "{:.2%}"),
    ("rot_err_deg", "{:.1f}"),
    ("trans_err", "{:.3f}"),
    ("inlier_ratio", "{:.2%}"),
    ("normal_dot_mean_inliers", "{:+.3f}"),
    ("score_gap", "{:+.2f}"),
]
HEADERS = ["Label", "Strategy", "CorrPrec", "RansacValid", "Pose@30", "RotErr", "TransErr", "InlierRatio", "NormalDot", "ScoreGap"]


def fmt(value, spec):
    if value is None:
        return "n/a"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="Paths to --summary_json outputs")
    args = parser.parse_args()

    rows = []
    for path in args.json_files:
        data = json.loads(Path(path).read_text())
        label = data.get("label", Path(path).stem)
        for strategy, metrics in data.get("strategies", {}).items():
            row = [label, strategy] + [fmt(metrics.get(col), spec) for col, spec in COLUMNS]
            rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(HEADERS)]
    line = "  " + "  ".join(h.ljust(w) for h, w in zip(HEADERS, widths))
    print(line)
    print("  " + "-" * (len(line) - 2))
    for row in rows:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
