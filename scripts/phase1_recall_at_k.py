"""
scripts/phase1_recall_at_k.py
==============================
Phase 1 du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Mesure si le filtre CNN Step 15 garde assez de vrais points de fracture pour
servir de prior à un matching pair-à-pair en aval. Pour chaque fragment, calcule
recall/precision/reduction_ratio du masque "points de fracture" sous plusieurs
stratégies de filtrage :
  - top-K  (les K points avec le score de fracture le plus haut)
  - threshold (score > seuil)
  - top-percent (les X% points les plus probables)

Résultats agrégés par split (everyday/artifact) x bucket de complexité
(2-5 / 6-10 / 11+ fragments), pour voir si le recall s'effondre sur les objets
complexes (là où le prior serait le plus utile).

Usage (sur le serveur) :
    python scripts/phase1_recall_at_k.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --split val --max_batches 300

    python scripts/phase1_recall_at_k.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories artifact --split val --max_batches 300
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list


TOPK_LIST = [256, 512, 1024]
THRESHOLD_LIST = [0.2, 0.3, 0.5]
TOPPERCENT_LIST = [5, 10, 20, 30]


def complexity_bucket(n_parts: int) -> str:
    if n_parts <= 5:
        return "2-5"
    if n_parts <= 10:
        return "6-10"
    return "11+"


def filter_metrics(scores: np.ndarray, gt: np.ndarray, n_keep: int):
    """Garde les n_keep points de score le plus haut. Retourne (recall, precision, n_pts_kept)."""
    n_pts = len(scores)
    n_keep = min(n_keep, n_pts)
    if n_keep == 0:
        return 0.0, 0.0, 0
    keep_idx = np.argpartition(-scores, n_keep - 1)[:n_keep]
    keep_mask = np.zeros(n_pts, dtype=bool)
    keep_mask[keep_idx] = True

    n_frac_total = int(gt.sum())
    tp = int((keep_mask & (gt == 1)).sum())
    fp = int((keep_mask & (gt == 0)).sum())
    recall = tp / n_frac_total if n_frac_total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return recall, precision, n_keep


def threshold_metrics(scores: np.ndarray, gt: np.ndarray, thresh: float):
    keep_mask = scores > thresh
    n_pts = len(scores)
    n_frac_total = int(gt.sum())
    tp = int((keep_mask & (gt == 1)).sum())
    fp = int((keep_mask & (gt == 0)).sum())
    n_keep = int(keep_mask.sum())
    recall = tp / n_frac_total if n_frac_total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return recall, precision, n_keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reuse the exact same Hydra/datamodule loading path as analyze_errors.py
    fake_args = argparse.Namespace(
        experiment=args.experiment,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        categories=args.categories,
        model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    if args.split == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # records[filter_name][bucket] -> list of (recall, precision, reduction_ratio)
    records = defaultdict(lambda: defaultdict(list))

    print(f"\nRunning inference on {args.categories}/{args.split}...")
    n_fragments = 0
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

            B, P = valid_pcs.shape
            n_parts_per_frag = []
            for b in range(B):
                n_parts_in_obj = int(valid_pcs[b].sum().item())
                for p_idx in range(P):
                    if valid_pcs[b, p_idx]:
                        n_parts_per_frag.append(n_parts_in_obj)

            out = model(batch)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat = out["coarse_seg_gt"].long().cpu().numpy()

            offset = 0
            for sz, n_parts in zip(frag_sizes, n_parts_per_frag):
                scores = pred_flat[offset: offset + sz]
                gt = gt_flat[offset: offset + sz]
                offset += sz
                n_fragments += 1
                bucket = complexity_bucket(n_parts)

                for k in TOPK_LIST:
                    r, p, n_keep = filter_metrics(scores, gt, k)
                    records[f"top{k}"][bucket].append((r, p, n_keep / sz))

                for t in THRESHOLD_LIST:
                    r, p, n_keep = threshold_metrics(scores, gt, t)
                    records[f"thresh{t}"][bucket].append((r, p, n_keep / sz))

                for pct in TOPPERCENT_LIST:
                    n_keep_target = max(1, int(sz * pct / 100))
                    r, p, n_keep = filter_metrics(scores, gt, n_keep_target)
                    records[f"top{pct}pct"][bucket].append((r, p, n_keep / sz))

    print(f"\nAnalyzed {n_fragments} fragments ({args.categories}/{args.split}).")

    print("\n" + "=" * 95)
    print(f"RECALL@K — {args.categories}/{args.split}")
    print("=" * 95)
    print(f"  {'Filtrage':<14} {'Bucket':<8} {'Recall':>8} {'Precision':>10} {'Reduction':>10} {'n_frags':>8}")
    print("  " + "-" * 88)
    for filt_name, buckets in records.items():
        for bucket in ["2-5", "6-10", "11+"]:
            if bucket not in buckets:
                continue
            vals = buckets[bucket]
            recalls = [v[0] for v in vals if not np.isnan(v[0])]
            precs = [v[1] for v in vals if not np.isnan(v[1])]
            reds = [v[2] for v in vals]
            if not recalls:
                continue
            print(
                f"  {filt_name:<14} {bucket:<8} {np.mean(recalls):>8.2%} "
                f"{np.mean(precs):>10.2%} {np.mean(reds):>10.2%} {len(vals):>8}"
            )


if __name__ == "__main__":
    main()
