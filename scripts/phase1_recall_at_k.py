"""
scripts/phase1_recall_at_k.py
==============================
Phase 1 du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Mesure si le filtre CNN Step 15 garde assez de vrais points de fracture pour
servir de prior à un matching pair-à-pair en aval, à deux niveaux :

  - fragment_fracture_recall@K : recall du filtre sur TOUS les points fracture
    GT d'un fragment (fracture_surface_gt est un label fragment-level — un
    fragment touchant plusieurs voisins a des points fracture vers chacun d'eux).
  - edge_contact_recall@K : recall du filtre restreint aux points fracture d'un
    fragment i qui sont géométriquement proches (NN < eps, après reconstruction
    en pose GT confirmée en Phase 0) d'un voisin spécifique j (graph[i,j]=True).
    C'est la métrique pertinente pour le matching pair-à-pair : on ne veut pas
    juste "des points cassés", on veut les points de contact du bon voisin.

Stratégies de filtrage : top-K, threshold, top-percent.
Agrégation : par split (everyday/artifact) x bucket de complexité (2-5/6-10/11+).

Usage (sur le serveur) :
    python scripts/phase1_recall_at_k.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --split val --max_batches 300
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list


TOPK_LIST = [256, 512, 1024]
THRESHOLD_LIST = [0.2, 0.3, 0.5]
TOPPERCENT_LIST = [5, 10, 20, 30]
EPS_LIST = [0.02, 0.05]


def complexity_bucket(n_parts: int) -> str:
    if n_parts <= 5:
        return "2-5"
    if n_parts <= 10:
        return "6-10"
    return "11+"


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()


def filter_mask_topk(scores: np.ndarray, n_keep: int) -> np.ndarray:
    n_pts = len(scores)
    n_keep = min(n_keep, n_pts)
    mask = np.zeros(n_pts, dtype=bool)
    if n_keep == 0:
        return mask
    keep_idx = np.argpartition(-scores, n_keep - 1)[:n_keep]
    mask[keep_idx] = True
    return mask


def filter_mask_threshold(scores: np.ndarray, thresh: float) -> np.ndarray:
    return scores > thresh


def recall_precision(mask: np.ndarray, gt: np.ndarray):
    n_frac = int(gt.sum())
    tp = int((mask & (gt == 1)).sum())
    fp = int((mask & (gt == 0)).sum())
    recall = tp / n_frac if n_frac > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return recall, precision


def build_filters(scores: np.ndarray):
    """Returns dict filter_name -> boolean mask, for all configured filters."""
    n_pts = len(scores)
    masks = {}
    for k in TOPK_LIST:
        masks[f"top{k}"] = filter_mask_topk(scores, k)
    for t in THRESHOLD_LIST:
        masks[f"thresh{t}"] = filter_mask_threshold(scores, t)
    for pct in TOPPERCENT_LIST:
        n_keep = max(1, int(n_pts * pct / 100))
        masks[f"top{pct}pct"] = filter_mask_topk(scores, n_keep)
    return masks


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

    # records[filter_name][bucket]["fragment"|f"edge_eps{eps}"] -> list of recall values
    # records[filter_name][bucket]["precision"] -> list of precision values
    # records[filter_name][bucket]["reduction"] -> list of reduction ratios
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print(f"\nRunning inference on {args.categories}/{args.split}...")
    n_fragments = 0
    n_edges = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}...")

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat = out["coarse_seg_gt"].long().cpu().numpy()

            # bp_pairs[k] = (b, p) — matches the row-major flatten order of
            # tensor[valid_pcs] used by extract_fragment_list (uniform path).
            valid_pcs_np = valid_pcs.cpu().numpy()
            B, P = valid_pcs_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_pcs_np[b, p]]

            quats_np = batch["quaternions"].numpy()        # (B, P, 4)
            trans_np = batch["translations"].numpy()       # (B, P, 3)
            scale_np = batch["scale"].numpy()               # (B, P, 1) or (B, P)
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            graph_np = batch["graph"].numpy()                # (B, P, P) bool

            # Per-k arrays
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            scores_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            global_per_k = []
            n_parts_per_k = []
            for k, (b, p) in enumerate(bp_pairs):
                rot_mat = quat_wxyz_to_rotmat(quats_np[b, p])
                pc_global = (rot_mat @ (pc_local_per_k[k] * scale_np[b, p]).T).T + trans_np[b, p]
                global_per_k.append(pc_global)
                n_parts_per_k.append(int(valid_pcs_np[b].sum()))

            # Build a KD-tree per fragment once, reused across its edges.
            trees = [cKDTree(g) for g in global_per_k]

            # Group k-indices by object b
            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                n_parts = n_parts_per_k[ks_ps[0][0]]
                bucket = complexity_bucket(n_parts)

                # Per-fragment metrics (fragment-level fracture recall)
                for k, p in ks_ps:
                    n_fragments += 1
                    scores = scores_per_k[k]
                    gt = gt_per_k[k]
                    masks = build_filters(scores)
                    for filt_name, mask in masks.items():
                        r, prec = recall_precision(mask, gt)
                        records[filt_name][bucket]["fragment"].append(r)
                        records[filt_name][bucket]["precision"].append(prec)
                        records[filt_name][bucket]["kept_ratio"].append(mask.mean())

                # Per-edge contact metrics
                for idx_i in range(len(ks_ps)):
                    for idx_j in range(idx_i + 1, len(ks_ps)):
                        k_i, p_i = ks_ps[idx_i]
                        k_j, p_j = ks_ps[idx_j]
                        if not graph_np[b, p_i, p_j]:
                            continue
                        n_edges += 1

                        gt_i = gt_per_k[k_i].astype(bool)
                        gt_j = gt_per_k[k_j].astype(bool)
                        frac_idx_i = np.where(gt_i)[0]
                        frac_idx_j = np.where(gt_j)[0]

                        masks_i = build_filters(scores_per_k[k_i])
                        masks_j = build_filters(scores_per_k[k_j])

                        for eps in EPS_LIST:
                            # contact_i_to_j: fracture points of i within eps of fragment j
                            if len(frac_idx_i) > 0:
                                d_i, _ = trees[k_j].query(global_per_k[k_i][frac_idx_i])
                                contact_i = frac_idx_i[d_i < eps]
                            else:
                                contact_i = np.array([], dtype=int)
                            if len(frac_idx_j) > 0:
                                d_j, _ = trees[k_i].query(global_per_k[k_j][frac_idx_j])
                                contact_j = frac_idx_j[d_j < eps]
                            else:
                                contact_j = np.array([], dtype=int)

                            for filt_name in masks_i:
                                if len(contact_i) > 0:
                                    er_i = masks_i[filt_name][contact_i].mean()
                                    records[filt_name][bucket][f"edge_eps{eps}"].append(er_i)
                                if len(contact_j) > 0:
                                    er_j = masks_j[filt_name][contact_j].mean()
                                    records[filt_name][bucket][f"edge_eps{eps}"].append(er_j)

    print(f"\nAnalyzed {n_fragments} fragments, {n_edges} edges ({args.categories}/{args.split}).")

    print("\n" + "=" * 115)
    print(f"RECALL@K — {args.categories}/{args.split}")
    print("=" * 115)
    header = (
        f"  {'Filtrage':<14} {'Bucket':<8} {'FragRecall':>11} "
        + " ".join(f"{'Edge@'+str(e):>11}" for e in EPS_LIST)
        + f" {'Precision':>10} {'KeptRatio':>10} {'n_frags':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for filt_name, buckets in records.items():
        for bucket in ["2-5", "6-10", "11+"]:
            if bucket not in buckets:
                continue
            d = buckets[bucket]
            frag_r = [v for v in d["fragment"] if not np.isnan(v)]
            prec = [v for v in d["precision"] if not np.isnan(v)]
            kept = d["kept_ratio"]
            if not frag_r:
                continue
            edge_strs = []
            for eps in EPS_LIST:
                vals = d.get(f"edge_eps{eps}", [])
                edge_strs.append(f"{np.mean(vals):>11.2%}" if vals else f"{'n/a':>11}")
            print(
                f"  {filt_name:<14} {bucket:<8} {np.mean(frag_r):>11.2%} "
                + " ".join(edge_strs)
                + f" {np.mean(prec):>10.2%} {np.mean(kept):>10.2%} {len(frag_r):>8}"
            )


if __name__ == "__main__":
    main()
