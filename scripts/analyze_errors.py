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

    # + geometric metrics (Phase 1 — Boundary F1 / Hausdorff / Chamfer)
    python scripts/analyze_errors.py \
        --ckpt /storage/student7/teyssir/checkpoints/cnn_step11_last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step11 \
        --out_dir /tmp/student7/analysis/step11_geo \
        --geometric \
        --k_boundary 5
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

try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

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
# Geometric metrics (Phase 1)
# ---------------------------------------------------------------------------

def _boundary_mask(xyz: np.ndarray, gt_labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Boolean mask — True for points whose k-NN neighbourhood contains both classes."""
    if not _SKLEARN_OK or len(xyz) < k + 1:
        return np.zeros(len(xyz), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)
    _, idxs = nbrs.kneighbors(xyz)
    neighbor_labels = gt_labels[idxs[:, 1:]]          # (N, k)
    return (neighbor_labels != gt_labels[:, None]).any(axis=1)


def _boundary_f1(pred_prob: np.ndarray, gt: np.ndarray,
                 xyz: np.ndarray, k: int = 5):
    """F1 / Precision / Recall computed only on boundary points."""
    mask = _boundary_mask(xyz, gt, k=k)
    n_boundary = mask.sum()
    if n_boundary < 2:
        return float("nan"), float("nan"), float("nan"), int(n_boundary)
    p = (pred_prob[mask] > 0.5)
    g = gt[mask].astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    return f1, prec, rec, int(n_boundary)


def _hausdorff(pred_prob: np.ndarray, gt: np.ndarray, xyz: np.ndarray) -> float:
    """One-sided + two-sided Hausdorff distance (in point-cloud units) between
    predicted fracture set and GT fracture set."""
    if not _SCIPY_OK:
        return float("nan")
    pred_pts = xyz[pred_prob > 0.5]
    gt_pts   = xyz[gt.astype(bool)]
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")
    d_p2g = cKDTree(gt_pts).query(pred_pts)[0].max()
    d_g2p = cKDTree(pred_pts).query(gt_pts)[0].max()
    return float(max(d_p2g, d_g2p))


def _chamfer(pred_prob: np.ndarray, gt: np.ndarray, xyz: np.ndarray) -> float:
    """Symmetric Chamfer distance between predicted and GT fracture point sets."""
    if not _SCIPY_OK:
        return float("nan")
    pred_pts = xyz[pred_prob > 0.5]
    gt_pts   = xyz[gt.astype(bool)]
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")
    d_p2g = cKDTree(gt_pts).query(pred_pts)[0].mean()
    d_g2p = cKDTree(pred_pts).query(gt_pts)[0].mean()
    return float(d_p2g + d_g2p)


def compute_occupancy_correlation(
    records: list,
    count_list: list,   # list of K tensors (V, H*W) — from out["count_list"]
    pix_corners_list: list,  # list of K tensors (N_k, V, 4) long
) -> dict:
    """
    Analyse de corrélation entre l'occupancy pixel (nb points/pixel) et l'erreur.

    Pour chaque point, récupère l'occupancy de son pixel home (view 0, corner 0)
    et calcule :
      - Histogramme de distribution de l'occupancy
      - % de points dans des pixels à occupancy > 3 (zone de conflit)
      - Erreur moyenne par bin d'occupancy (fracture misclassifiée ?)

    Returns dict with summary stats, also attaches "occupancy" key to each record.
    """
    all_occ    = []
    all_errors = []   # 1 = mauvaise prédiction, 0 = bonne

    for rec, cnt_kv, corners_k in zip(records, count_list, pix_corners_list):
        # cnt_kv : (V, H*W), corners_k : (N_k, V, 4)
        cnt_v0   = cnt_kv[0]                            # (H*W,) — view 0
        home_px  = corners_k[:, 0, 0].cpu().long()      # (N_k,) — floor-rounded pixel, view 0
        occ      = cnt_v0[home_px].cpu().numpy()        # (N_k,) — occupancy per point

        pred = (rec["_pred"].numpy() > 0.5).astype(int)
        gt   = rec["_gt"].numpy().astype(int)
        err  = (pred != gt).astype(float)

        all_occ.append(occ)
        all_errors.append(err)
        rec["_occupancy"] = occ

    all_occ    = np.concatenate(all_occ)
    all_errors = np.concatenate(all_errors)

    threshold = 3
    high_occ_mask = all_occ > threshold
    pct_high_occ  = high_occ_mask.mean() * 100

    # Error rate by occupancy bin
    bins   = [1, 2, 3, 5, 10, 30, int(all_occ.max()) + 1]
    labels = ["1", "2", "3", "4-5", "6-10", "11+"]
    bin_stats = []
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (all_occ >= lo) & (all_occ < hi)
        if mask.sum() == 0:
            continue
        bin_stats.append({
            "bin": label,
            "n_pts": int(mask.sum()),
            "error_rate": float(all_errors[mask].mean()),
            "pct_of_total": float(mask.mean() * 100),
        })

    corr = float(np.corrcoef(all_occ, all_errors)[0, 1]) if len(all_occ) > 10 else float("nan")

    return {
        "pct_high_occ": pct_high_occ,
        "threshold": threshold,
        "corr_occ_error": corr,
        "bin_stats": bin_stats,
    }


def print_occupancy_summary(occ_stats: dict) -> None:
    print("\n" + "=" * 58)
    print("OCCUPANCY ↔ ERROR CORRELATION")
    print("=" * 58)
    print(f"  Points dans pixels à occupancy > {occ_stats['threshold']} : "
          f"{occ_stats['pct_high_occ']:.1f}%")
    print(f"  Pearson corr(occupancy, error) = {occ_stats['corr_occ_error']:.3f}")
    print(f"\n  {'Occ bin':<8}  {'n_pts':>8}  {'% total':>8}  {'error rate':>10}")
    print("  " + "-" * 42)
    for b in occ_stats["bin_stats"]:
        print(f"  {b['bin']:<8}  {b['n_pts']:>8,}  {b['pct_of_total']:>7.1f}%"
              f"  {b['error_rate']:>9.3f}")
    print()


def plot_occupancy_vs_error(records: list, occ_stats: dict, out_dir) -> None:
    """Bar chart: error rate per occupancy bin."""
    bins  = [b["bin"]        for b in occ_stats["bin_stats"]]
    errs  = [b["error_rate"] for b in occ_stats["bin_stats"]]
    ns    = [b["n_pts"]      for b in occ_stats["bin_stats"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bins, errs, color="#E64A19", edgecolor="white")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"n={n:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Points per pixel (occupancy)")
    ax.set_ylabel("Error rate (misclassified points)")
    ax.set_title(f"Error rate vs Pixel Occupancy  "
                 f"(corr={occ_stats['corr_occ_error']:.3f})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "occupancy_vs_error.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def compute_geometric_metrics(records: list, xyz_list: list, k_boundary: int = 5) -> None:
    """Compute and attach boundary_f1, boundary_prec, boundary_rec, n_boundary,
    hausdorff, chamfer to every record in-place."""
    if not (_SCIPY_OK or _SKLEARN_OK):
        print("  [warn] scipy/sklearn not available — skipping geometric metrics")
        return

    for rec, xyz in zip(records, xyz_list):
        pred = rec["_pred"].numpy() if hasattr(rec["_pred"], "numpy") else np.asarray(rec["_pred"])
        gt   = rec["_gt"].numpy()   if hasattr(rec["_gt"],   "numpy") else np.asarray(rec["_gt"])
        xyz_np = xyz.numpy() if hasattr(xyz, "numpy") else np.asarray(xyz)

        bf1, bprec, brec, nb = _boundary_f1(pred, gt, xyz_np, k=k_boundary)
        rec["boundary_f1"]   = bf1
        rec["boundary_prec"] = bprec
        rec["boundary_rec"]  = brec
        rec["n_boundary"]    = nb
        rec["hausdorff"]     = _hausdorff(pred, gt, xyz_np)
        rec["chamfer"]       = _chamfer(pred, gt, xyz_np)


def print_geometric_summary(records: list) -> None:
    bf1s = [r["boundary_f1"] for r in records if not np.isnan(r["boundary_f1"])]
    hds  = [r["hausdorff"]   for r in records if not np.isnan(r["hausdorff"])]
    cds  = [r["chamfer"]     for r in records if not np.isnan(r["chamfer"])]

    print("\n" + "=" * 62)
    print("GEOMETRIC METRICS (fragment-level, boundary k=5)")
    print("=" * 62)
    print(f"  {'Metric':<22}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  n")
    print("  " + "-" * 56)
    if bf1s:
        print(f"  {'Boundary F1':<22}  {np.mean(bf1s):>8.4f}  {np.median(bf1s):>8.4f}  {np.std(bf1s):>8.4f}  {len(bf1s)}")
    if hds:
        print(f"  {'Hausdorff distance':<22}  {np.mean(hds):>8.4f}  {np.median(hds):>8.4f}  {np.std(hds):>8.4f}  {len(hds)}")
    if cds:
        print(f"  {'Chamfer distance':<22}  {np.mean(cds):>8.4f}  {np.median(cds):>8.4f}  {np.std(cds):>8.4f}  {len(cds)}")

    # Quick correlation: high-F1 fragments — is boundary_f1 also high?
    paired = [(r["f1"], r["boundary_f1"]) for r in records
              if not np.isnan(r["boundary_f1"])]
    if len(paired) > 10:
        f1v  = np.array([x[0] for x in paired])
        bf1v = np.array([x[1] for x in paired])
        corr = np.corrcoef(f1v, bf1v)[0, 1]
        print(f"\n  Pearson corr(F1, Boundary F1) = {corr:.3f}")

    print()


def plot_geometric_distributions(records: list, out_dir: Path) -> None:
    """Three-panel figure: Boundary F1, Hausdorff, Chamfer distributions."""
    bf1s = [r["boundary_f1"] for r in records if not np.isnan(r["boundary_f1"])]
    hds  = [r["hausdorff"]   for r in records if not np.isnan(r["hausdorff"])]
    cds  = [r["chamfer"]     for r in records if not np.isnan(r["chamfer"])]

    if not bf1s:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(bf1s, bins=40, color="#00897B", edgecolor="white", linewidth=0.5)
    axes[0].axvline(np.mean(bf1s), color="red",    linestyle="--", label=f"mean={np.mean(bf1s):.3f}")
    axes[0].axvline(np.median(bf1s), color="orange", linestyle="--", label=f"median={np.median(bf1s):.3f}")
    axes[0].set_xlabel("Boundary F1")
    axes[0].set_ylabel("Number of fragments")
    axes[0].set_title("Boundary F1 Distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if hds:
        axes[1].hist(hds, bins=40, color="#E53935", edgecolor="white", linewidth=0.5)
        axes[1].axvline(np.mean(hds), color="black", linestyle="--", label=f"mean={np.mean(hds):.4f}")
        axes[1].set_xlabel("Hausdorff distance (point-cloud units)")
        axes[1].set_title("Hausdorff Distance Distribution")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    if cds:
        axes[2].hist(cds, bins=40, color="#8E24AA", edgecolor="white", linewidth=0.5)
        axes[2].axvline(np.mean(cds), color="black", linestyle="--", label=f"mean={np.mean(cds):.4f}")
        axes[2].set_xlabel("Chamfer distance (point-cloud units)")
        axes[2].set_title("Chamfer Distance Distribution")
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "geometric_metrics_distributions.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")

    # Scatter: F1 vs Boundary F1
    paired = [(r["f1"], r["boundary_f1"]) for r in records
              if not np.isnan(r["boundary_f1"])]
    if len(paired) > 5:
        fig, ax = plt.subplots(figsize=(6, 5))
        f1v, bf1v = zip(*paired)
        ax.scatter(f1v, bf1v, s=4, alpha=0.4, color="#00897B")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Global F1")
        ax.set_ylabel("Boundary F1")
        ax.set_title("Global F1 vs Boundary F1 (points below diagonal = boundary worse)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path2 = out_dir / "f1_vs_boundary_f1_scatter.png"
        plt.savefig(out_path2, dpi=120)
        plt.close()
        print(f"  Saved: {out_path2}")


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


def plot_f1_histogram(records, out_path, threshold=0.5):
    """Histogram of per-fragment F1 scores."""
    f1s = [r["f1"] for r in records]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(f1s, bins=40, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(f1s), color="red", linestyle="--", label=f"mean={np.mean(f1s):.3f}")
    ax.axvline(np.median(f1s), color="orange", linestyle="--", label=f"median={np.median(f1s):.3f}")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Number of fragments")
    ax.set_title(f"F1 Distribution per Fragment (threshold={threshold})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def bin_stats_complexity(records):
    """
    Bin fragments by object complexity into: 2 parts, 3-5 parts, 6+ parts.
    Returns list of (label, mean_f1, mean_prec, mean_rec, count).
    """
    groups = {"2": [], "3-5": [], "6+": []}
    for r in records:
        n = int(r["n_parts"])
        if n == 2:
            groups["2"].append(r)
        elif 3 <= n <= 5:
            groups["3-5"].append(r)
        else:
            groups["6+"].append(r)
    result = []
    for label in ["2", "3-5", "6+"]:
        subset = groups[label]
        if not subset:
            continue
        result.append((
            label,
            np.mean([r["f1"] for r in subset]),
            np.mean([r["precision"] for r in subset]),
            np.mean([r["recall"] for r in subset]),
            len(subset),
        ))
    return result


def analyze_isolated_fp(records, xyz_list, k_neighbors=10, min_cluster_size=5):
    """
    Count and characterize isolated FP clusters.
    A FP point is 'isolated' if its k nearest neighbors contain fewer than
    min_cluster_size FP points.
    Returns summary dict.
    """
    if not _SKLEARN_OK:
        print("  [warn] sklearn not available — skipping isolated FP analysis")
        return {}

    total_fp = 0
    isolated_fp = 0
    frag_isolated_counts = []

    for rec, xyz in zip(records, xyz_list):
        fp_mask = (rec["_pred"] > 0.5).numpy() & (rec["_gt"].numpy() == 0)
        n_fp = fp_mask.sum()
        total_fp += n_fp
        if n_fp < 2:
            frag_isolated_counts.append(0)
            continue

        fp_xyz = xyz[fp_mask].numpy() if hasattr(xyz, 'numpy') else xyz[fp_mask]
        if len(fp_xyz) < 2:
            frag_isolated_counts.append(0)
            continue

        k = min(k_neighbors, len(fp_xyz) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(fp_xyz)
        dists, idxs = nbrs.kneighbors(fp_xyz)
        # For each FP point, count how many of its k neighbors are also FP
        neighbors_fp = (idxs[:, 1:].shape[1])  # all neighbors are FP by construction
        # A point is "isolated" if it has fewer than min_cluster_size FP neighbors
        n_isolated = int((k < min_cluster_size) or (len(fp_xyz) < min_cluster_size))
        # Simpler: if the whole FP cluster is small (<= min_cluster_size pts), it's isolated
        is_isolated_cluster = len(fp_xyz) <= min_cluster_size
        n_isolated = int(is_isolated_cluster) * len(fp_xyz)
        isolated_fp += n_isolated
        frag_isolated_counts.append(n_isolated)

    return {
        "total_fp": total_fp,
        "isolated_fp": isolated_fp,
        "isolated_fp_rate": isolated_fp / max(total_fp, 1),
        "frags_with_isolated_fp": sum(1 for c in frag_isolated_counts if c > 0),
    }


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
    p.add_argument("--experiment",  required=True,  help="Hydra experiment name (e.g. cnn_step9_geo_features) — used for the datamodule config only")
    p.add_argument("--model_type",  default="cnn", choices=["cnn", "garf"],
                   help="'cnn' loads CNNFracSeg, 'garf' loads PTv3 FracSeg (extracted from a full GARF "
                        "checkpoint if needed, e.g. GARF_mini.ckpt)")
    p.add_argument("--out_dir",     default="/tmp/student7/analysis", help="Output directory for plots")
    p.add_argument("--split",       default="val",  choices=["val", "test"])
    p.add_argument("--categories",  default=None,   help="Comma-separated categories override, e.g. artifact")
    p.add_argument("--n_vis",       type=int, default=6, help="Qualitative examples per category")
    p.add_argument("--max_batches", type=int, default=300, help="Max batches to process (0=all)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sweep_threshold", action="store_true",
                   help="Sweep decision threshold and print F1/Prec/Rec table")
    p.add_argument("--geometric", action="store_true",
                   help="Compute geometric metrics: Boundary F1, Hausdorff, Chamfer distance")
    p.add_argument("--k_boundary", type=int, default=5,
                   help="k for kNN boundary detection (default 5)")
    return p.parse_args()


def load_frac_seg_from_garf(garf_ckpt: str, device: torch.device):
    """
    Load the PTv3 FracSeg model from a full GARF checkpoint (DenoiserFlowMatching,
    e.g. GARF_mini.ckpt) or a standalone FracSeg checkpoint.

    FracSeg is instantiated via Hydra (configs/model/frac_seg.yaml supplies
    pc_feat_dim/encoder/optimizer) rather than via FracSeg.load_from_checkpoint,
    because the extracted feature_extractor.* weights carry no hyperparameters
    for Lightning to reconstruct the constructor args from.
    """
    ckpt_data = torch.load(garf_ckpt, map_location="cpu", weights_only=False)
    keys = list(ckpt_data.get("state_dict", {}).keys())
    is_garf = any(k.startswith("feature_extractor.") for k in keys)

    if is_garf:
        state = {
            k.replace("feature_extractor.", ""): v
            for k, v in ckpt_data["state_dict"].items()
            if k.startswith("feature_extractor.")
        }
    else:
        state = ckpt_data.get("state_dict", ckpt_data)

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=frac_seg"])
    model = instantiate(cfg.model)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  FracSeg weights loaded from {garf_ckpt}")
    print(f"  {len(state) - len(unexpected)} tensors matched | "
          f"{len(missing)} missing | {len(unexpected)} unexpected")

    return model


def load_config_and_model(args):
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    overrides = [
        f"experiment={args.experiment}",
        f"data.data_root={args.data_root}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
    ]
    if args.categories is not None:
        cats = args.categories.split(",")
        cats_str = "[" + ",".join(cats) + "]"
        overrides.append(f"data.categories={cats_str}")
    if args.model_type == "garf":
        # FracSeg.forward expects the flat concatenated (B, N_total, 3) layout
        # produced by BreakingBadWeighted — the CNN's sample_method=uniform
        # gives an already per-part-split (B, P, N, 3) tensor, which crashes
        # FracSeg's `B, N, C = pointclouds.shape` unpacking.
        overrides.append("data.sample_method=weighted")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=overrides,
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
    if args.split == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    # --- Load model ---
    print("Loading model from checkpoint...")
    if args.model_type == "garf":
        model = load_frac_seg_from_garf(args.ckpt, device)
    else:
        from assembly.models.cnn_segmentation_model import CNNFracSeg
        model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Inference loop ---
    all_records       = []
    all_xyz           = []
    all_count_list    = []   # for occupancy correlation analysis
    all_corners_list  = []   # for occupancy correlation analysis
    n_parts_per_frag  = []

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

            # Collect count_list and corners for occupancy analysis
            count_list_batch   = out.get("count_list", [None] * len(records))
            corners_batch      = out.get("pix_corners_list", [None] * len(records))

            for rec, xyz_k, n_parts, cnt_k, crn_k in zip(
                records, frag_list, n_parts_in_obj,
                count_list_batch, corners_batch,
            ):
                rec["n_parts"] = n_parts
                all_records.append(rec)
                all_xyz.append(xyz_k.cpu())
                all_count_list.append(cnt_k.cpu() if cnt_k is not None else None)
                all_corners_list.append(crn_k.cpu() if crn_k is not None else None)
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

    # --- Analysis by complexity (2 parts / 3-5 / 6+) ---
    print("\n[F1 by Object Complexity (2 / 3-5 / 6+ parts)]")
    stats_complexity = bin_stats_complexity(all_records)
    for label, f1, prec, rec, cnt in stats_complexity:
        print(f"  {label:<8}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (n={cnt})")
    plot_bin_metrics(stats_complexity, "F1 / Precision / Recall by Object Complexity",
                     "Number of parts (complexity)", out_dir / "metrics_by_complexity.png")

    # --- F1 histogram ---
    plot_f1_histogram(all_records, out_dir / "f1_histogram.png")

    # --- FP/FN distribution ---
    plot_fp_fn_dist(all_records, out_dir / "fp_fn_distribution.png")

    # --- Occupancy ↔ Error correlation (Phase 1 — requires count_list from model) ---
    has_occ = all_count_list[0] is not None and all_corners_list[0] is not None
    if has_occ:
        print("\n[Occupancy ↔ Error Correlation]")
        occ_stats = compute_occupancy_correlation(all_records, all_count_list, all_corners_list)
        print_occupancy_summary(occ_stats)
        plot_occupancy_vs_error(all_records, occ_stats, out_dir)
    else:
        print("\n[Occupancy analysis skipped — count_list not in model output]")

    # --- Geometric metrics (Phase 1) ---
    if args.geometric:
        print(f"\n[Geometric Metrics — Boundary F1 / Hausdorff / Chamfer (k={args.k_boundary})]")
        if not _SKLEARN_OK:
            print("  [warn] sklearn not found — install it: pip install scikit-learn")
        if not _SCIPY_OK:
            print("  [warn] scipy not found — install it: pip install scipy")
        compute_geometric_metrics(all_records, all_xyz, k_boundary=args.k_boundary)
        print_geometric_summary(all_records)
        plot_geometric_distributions(all_records, out_dir)

    # --- Isolated FP analysis ---
    print("\n[Isolated FP Cluster Analysis]")
    iso = analyze_isolated_fp(all_records, all_xyz)
    if iso:
        print(f"  Total FP points:         {iso['total_fp']:,}")
        print(f"  Isolated FP points:      {iso['isolated_fp']:,}  ({iso['isolated_fp_rate']:.1%})")
        print(f"  Frags with isolated FP:  {iso['frags_with_isolated_fp']}")

    # --- Qualitative visualizations ---
    print(f"\nGenerating qualitative visualizations (n_vis={args.n_vis})...")
    plot_qualitative(all_records, all_xyz, args.n_vis, out_dir, tag=args.experiment)

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
        import json
        THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.81, 0.05)]
        print("\n[Threshold Sweep — Global F1 / Precision / Recall]")
        print(f"  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'FDR':>7}")
        print("  " + "-" * 42)
        all_pred = torch.cat([r["_pred"] for r in all_records])
        all_gt   = torch.cat([r["_gt"].float()   for r in all_records])

        best_thresh, best_f1 = 0.50, 0.0
        sweep_results = []
        for thresh in THRESHOLDS:
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
            sweep_results.append({"threshold": thresh, "f1": f1, "precision": prec, "recall": rec, "fdr": fdr})
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        print(f"\n  Best threshold on {args.split} set: {best_thresh:.2f}  (F1={best_f1:.4f})")
        threshold_file = out_dir / "best_threshold.json"
        with open(threshold_file, "w") as fh:
            json.dump({"split": args.split, "best_threshold": best_thresh, "best_f1": best_f1,
                       "sweep": sweep_results}, fh, indent=2)
        print(f"  Saved best threshold to: {threshold_file}")

        # Precision-Recall curve
        fig, ax = plt.subplots(figsize=(6, 5))
        precs_sw = [r["precision"] for r in sweep_results]
        recs_sw  = [r["recall"]    for r in sweep_results]
        f1s_sw   = [r["f1"]        for r in sweep_results]
        ax.plot(recs_sw, precs_sw, "b-o", markersize=5)
        for r in sweep_results:
            ax.annotate(f"{r['threshold']:.2f}", (r["recall"], r["precision"]),
                        fontsize=7, textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall curve ({args.split} set)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "threshold_pr_curve.png", dpi=120)
        plt.close()
        print(f"  Saved: {out_dir}/threshold_pr_curve.png")

        # Also sweep by fracture_ratio bin to find optimal threshold per group
        print("\n[Threshold Sweep — Low fracture ratio fragments only (<0.14)]")
        low_frac = [r for r in all_records if r["fracture_ratio"] < 0.14]
        if low_frac:
            lf_pred = torch.cat([r["_pred"] for r in low_frac])
            lf_gt   = torch.cat([r["_gt"].float()   for r in low_frac])
            print(f"  {'Thresh':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
            print("  " + "-" * 33)
            for thresh in THRESHOLDS:
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
