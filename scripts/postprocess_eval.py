"""
scripts/postprocess_eval.py
============================
Évalue l'impact du post-processing spatial sur les prédictions CNN.

Méthodes testées :
  1. Aucun post-processing (raw)
  2. kNN majority vote (chaque point adopte le vote majoritaire de ses k voisins)
  3. Connected components — supprime les petits clusters de points fracture (< min_cluster_size)

Compare F1 / Precision / Recall / FP rate avant et après post-processing.

Usage:
    python scripts/postprocess_eval.py \\
        --ckpt output/cnn_step9_geo_features/version_0/checkpoints/last.ckpt \\
        --data_root /path/to/breaking_bad_vol.hdf5 \\
        --experiment cnn_step9_geo_features \\
        --split val \\
        --threshold 0.65 \\
        --k 10 \\
        --min_cluster 20 \\
        --out_dir output/postprocess_eval
"""

import argparse
import sys
import functools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([functools.partial])
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx], replace=True)

from assembly.models.projection_mapping_utils import extract_fragment_list


# ---------------------------------------------------------------------------
# Post-processing methods
# ---------------------------------------------------------------------------

def knn_majority_vote(xyz: np.ndarray, pred_b: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Pour chaque point, remplace la prédiction par le vote majoritaire de ses k voisins.
    Réduit les prédictions isolées bruitées.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("sklearn required: pip install scikit-learn")

    if len(xyz) <= k:
        return pred_b.copy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xyz)
    _, idxs = nbrs.kneighbors(xyz)
    # idxs[i, 0] = i lui-même, idxs[i, 1:] = ses k voisins
    neighbor_votes = pred_b[idxs[:, 1:]]   # (N, k) bool
    majority = neighbor_votes.mean(axis=1) >= 0.5
    return majority


def remove_small_clusters(xyz: np.ndarray, pred_b: np.ndarray,
                           min_cluster_size: int = 20, k: int = 10) -> np.ndarray:
    """
    Supprime les clusters de points fracture de taille < min_cluster_size.
    Utilise des composantes connexes sur le graphe kNN.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        raise ImportError("sklearn + scipy required: pip install scikit-learn scipy")

    pred_out = pred_b.copy()
    fp_indices = np.where(pred_b)[0]
    if len(fp_indices) < 2:
        return pred_out

    # Construire le graphe sur les points fracture uniquement
    fp_xyz = xyz[fp_indices]
    k_eff = min(k, len(fp_xyz) - 1)
    if k_eff < 1:
        return pred_out

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(fp_xyz)
    _, idxs = nbrs.kneighbors(fp_xyz)

    # Matrice d'adjacence sparse (uniquement entre points fracture)
    n = len(fp_indices)
    rows = np.repeat(np.arange(n), k_eff)
    cols = idxs[:, 1:].flatten()
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))

    n_comp, labels = connected_components(adj, directed=False)

    # Compter la taille de chaque composante
    comp_sizes = np.bincount(labels, minlength=n_comp)

    # Supprimer les petites composantes
    small_comps = np.where(comp_sizes < min_cluster_size)[0]
    for comp in small_comps:
        local_indices = np.where(labels == comp)[0]
        global_indices = fp_indices[local_indices]
        pred_out[global_indices] = False

    return pred_out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_metrics(pred_b: np.ndarray, gt_b: np.ndarray) -> dict:
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())
    tn = int((~pred_b & ~gt_b).sum())
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    fdr  = safe_div(fp, fp + tp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "fdr": fdr}


def aggregate_metrics(records: list) -> dict:
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    fdr  = safe_div(fp, fp + tp)
    mean_f1 = np.mean([r["f1"] for r in records])
    return {"global_f1": f1, "global_prec": prec, "global_rec": rec,
            "fdr": fdr, "mean_frag_f1": mean_f1,
            "total_tp": tp, "total_fp": fp, "total_fn": fn}


def print_summary(label: str, agg: dict):
    print(f"  [{label:30s}]  "
          f"F1={agg['global_f1']:.4f}  "
          f"Prec={agg['global_prec']:.4f}  "
          f"Rec={agg['global_rec']:.4f}  "
          f"FDR={agg['fdr']:.4f}  "
          f"mean-frag-F1={agg['mean_frag_f1']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--data_root",    required=True)
    p.add_argument("--experiment",   required=True)
    p.add_argument("--split",        default="val", choices=["val", "test"])
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--k",            type=int, default=10, help="Neighbors for kNN vote")
    p.add_argument("--min_cluster",  type=int, default=20, help="Min cluster size to keep")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--max_batches",  type=int, default=300)
    p.add_argument("--out_dir",      default="output/postprocess_eval")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}  k={args.k}  min_cluster={args.min_cluster}")

    # --- Load config + datamodule ---
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

    datamodule = instantiate(cfg.data)
    if args.split == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    # --- Load model ---
    from assembly.models.cnn_segmentation_model import CNNFracSeg
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Inference + collect raw predictions ---
    raw_records    = []
    knn_records    = []
    cluster_records = []
    xyz_all        = []

    print(f"\nRunning inference on {args.split} set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}...")

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            frag_list, valid_pcs, K = extract_fragment_list(
                batch["pointclouds"], batch["points_per_part"]
            )
            if K == 0:
                continue

            frag_sizes = [f.shape[0] for f in frag_list]

            out = model(batch)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat   = out["coarse_seg_gt"].long().cpu().numpy().astype(bool)

            offset = 0
            for k_idx, (sz, xyz_k) in enumerate(zip(frag_sizes, frag_list)):
                p_raw = pred_flat[offset: offset + sz]
                g     = gt_flat[offset: offset + sz]
                offset += sz

                pred_b_raw = p_raw > args.threshold
                xyz_np     = xyz_k.cpu().numpy()

                # Raw
                raw_records.append(compute_metrics(pred_b_raw, g))

                # kNN majority vote
                pred_b_knn = knn_majority_vote(xyz_np, pred_b_raw, k=args.k)
                knn_records.append(compute_metrics(pred_b_knn, g))

                # Connected components — remove small clusters
                pred_b_cc = remove_small_clusters(xyz_np, pred_b_raw,
                                                  min_cluster_size=args.min_cluster,
                                                  k=args.k)
                cluster_records.append(compute_metrics(pred_b_cc, g))

                xyz_all.append(xyz_np)

    print(f"\nAnalyzed {len(raw_records)} fragments total.")
    print("\n" + "=" * 80)
    print("POST-PROCESSING COMPARISON")
    print("=" * 80)
    print_summary(f"Raw (threshold={args.threshold})", aggregate_metrics(raw_records))
    print_summary(f"kNN majority vote (k={args.k})", aggregate_metrics(knn_records))
    print_summary(f"Remove clusters (<{args.min_cluster} pts)", aggregate_metrics(cluster_records))
    print("=" * 80)

    # Save JSON summary
    import json
    summary = {
        "threshold": args.threshold,
        "k": args.k,
        "min_cluster": args.min_cluster,
        "n_fragments": len(raw_records),
        "raw":          aggregate_metrics(raw_records),
        "knn_vote":     aggregate_metrics(knn_records),
        "cluster_filter": aggregate_metrics(cluster_records),
    }
    out_file = out_dir / f"postprocess_summary_thr{args.threshold}.json"
    with open(out_file, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary to: {out_file}")

    # Delta table
    print("\nDelta vs Raw:")
    agg_raw = aggregate_metrics(raw_records)
    for label, agg in [
        (f"kNN vote (k={args.k})", aggregate_metrics(knn_records)),
        (f"Cluster filter (<{args.min_cluster})", aggregate_metrics(cluster_records)),
    ]:
        df1 = agg["global_f1"]   - agg_raw["global_f1"]
        dp  = agg["global_prec"] - agg_raw["global_prec"]
        dr  = agg["global_rec"]  - agg_raw["global_rec"]
        sign = lambda x: "+" if x >= 0 else ""
        print(f"  {label:35s}  ΔF1={sign(df1)}{df1:+.4f}  "
              f"ΔPrec={sign(dp)}{dp:+.4f}  ΔRec={sign(dr)}{dr:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
