"""
scripts/phase2d_interface_clustering_diagnostic.py
=====================================================
Phase 2D du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Question testée : les points fracture d'un fragment se découpent-ils naturellement en
patches spatiaux qui correspondent chacun à un voisin, SANS connaître la pose GT
(contrairement à `gt_edge` en Phase 2C, qui nécessite cette connaissance pour restreindre
le masque) ? Si oui, un clustering spatial naïf pourrait approximer `gt_edge` en
pratique : matcher chaque cluster contre son voisin plutôt que tout le masque fracture
du fragment en bloc (qui mélange les interfaces de plusieurs voisins -- la cause
principale de l'effondrement observé sur les masques réels en Phase 2C : `CorrPrec`
22.79% sur `gt_edge` contre 2.6-3.2% sur `gt`/`thresh0.2-0.5`, malgré le même scoring
normal-aware).

PAS de RANSAC, PAS de descripteurs ici -- diagnostic pur :
  1. Clustering par connectivité (graphe de proximité radius + composantes connexes,
     scipy uniquement, pas de dépendance sklearn) sur les points fracture (masque GT ou
     CNN-seuil) d'un fragment, dans son propre repère local (la structure spatiale du
     clustering ne dépend pas de la pose -- rotation/translation-invariante).
  2. Étiquetage GT (diagnostic uniquement) : pour chaque point fracture, le voisin réel
     le plus proche en repère assemblé (formule Phase 0), si la distance est < eps_contact.
  3. Pureté de chaque cluster = proportion du label voisin majoritaire dans ce cluster.
  4. edge_coverage : pour chaque arête (i,j), existe-t-il un cluster de i majoritairement
     étiqueté j ?
  5. best_cluster_corrprec_upper_bound : pour chaque arête (i,j), pureté du meilleur
     cluster de i pour j -- borne supérieure de CorrPrec atteignable si on ne donnait au
     matcher QUE ce cluster (au lieu de tout le masque fracture).

Lecture :
  - purity > 70%, edge_coverage > 70%, best_cluster_corrprec >> CorrPrec(gt/thresh global)
    => interfaces spatialement séparables, un module cluster -> pairwise matching est
       une suite crédible.
  - purity faible / clusters mélangés / edge_coverage basse
    => la séparation spatiale naïve ne suffit pas (besoin d'un modèle appris
       pair-specific ou de features plus riches).

Usage (sur le serveur) :
    python scripts/phase2d_interface_clustering_diagnostic.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --split val --max_batches 60 \
        --mask_strategy gt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from scripts.analyze_errors import load_config_and_model
from scripts.phase2_geometric_baseline import build_mask, quat_wxyz_to_rotmat
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list


CONTACT_EPS = 0.05  # same tolerance as Phase 0/1's edge_contact_recall


def cluster_points(points: np.ndarray, eps: float, min_cluster_size: int) -> np.ndarray:
    """Connected-components clustering via a radius proximity graph. Returns a label per
    point: cluster id (>=0) or -1 for noise (isolated points or clusters smaller than
    min_cluster_size)."""
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)
    tree = cKDTree(points)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return -np.ones(n, dtype=int)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    small = np.where(sizes < min_cluster_size)[0]
    labels = labels.copy()
    labels[np.isin(labels, small)] = -1
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1116)
    parser.add_argument(
        "--mask_strategy", default="gt",
        help="Mask used to select fracture points before clustering: 'gt' or "
             "'thresh<T>' (e.g. thresh0.3). See build_mask() in phase2_geometric_baseline.py.",
    )
    parser.add_argument(
        "--cluster_eps", type=float, default=0.05,
        help="Radius for the proximity graph used in connected-components clustering.",
    )
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument(
        "--summary_json", default=None,
        help="Optional path to dump a compact JSON summary -- avoids grepping the full "
             "text log. Consumed by scripts/phase2_compare_summaries.py.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Display name for this run in the summary JSON (default: derived from "
             "mask_strategy/cluster_eps/min_cluster_size).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    if args.split == "val":
        datamodule.setup("fit")
        dataset = datamodule.val_dataset
    else:
        datamodule.setup("test")
        dataset = datamodule.test_dataset

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # Aggregated diagnostics
    n_clusters_per_frag = []
    degree_per_frag = []           # graph degree (n. real neighbors), paired 1:1 with n_clusters_per_frag
    purity_weighted = []          # (purity, cluster_size) pairs -> size-weighted mean
    mixed_cluster_flags = []      # purity < 0.5
    noise_rate_per_frag = []
    edge_covered = []             # bool per directed edge (i,j): does i have a cluster majority-labeled j?
    best_cluster_corrprec = []    # purity of the best cluster per directed edge

    print(f"\nRunning Phase 2D clustering diagnostic on {args.categories}/{args.split} "
          f"(mask_strategy={args.mask_strategy})...")
    n_fragments, n_edges = 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 10 == 0:
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

            valid_pcs_np = valid_pcs.cpu().numpy()
            B, P = valid_pcs_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_pcs_np[b, p]]

            quats_np = batch["quaternions"].numpy()
            trans_np = batch["translations"].numpy()
            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            graph_np = batch["graph"].numpy()

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            scores_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            raw_per_k, global_per_k = [], []
            for k, (b, p) in enumerate(bp_pairs):
                scale_k = scale_np[b, p]
                raw_k = pc_local_per_k[k] * scale_k
                raw_per_k.append(raw_k)
                R_k = quat_wxyz_to_rotmat(quats_np[b, p])
                global_per_k.append((R_k @ raw_k.T).T + trans_np[b, p])

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                for idx_i in range(len(ks_ps)):
                    k_i, p_i = ks_ps[idx_i]
                    mask_i = build_mask(args.mask_strategy, scores_per_k[k_i], gt_per_k[k_i], rng)
                    frac_idx_i = np.where(mask_i)[0]
                    if len(frac_idx_i) < args.min_cluster_size:
                        continue
                    n_fragments += 1

                    # Neighbor label per fracture point of i: nearest neighbor fragment j
                    # (among i's actual graph neighbors) within CONTACT_EPS.
                    neighbors_of_i = [
                        (idx_j, k_j, p_j) for idx_j in range(len(ks_ps))
                        for k_j, p_j in [ks_ps[idx_j]]
                        if idx_j != idx_i and graph_np[b, p_i, p_j]
                    ]

                    frac_global_i = global_per_k[k_i][frac_idx_i]
                    cluster_labels = cluster_points(frac_global_i, args.cluster_eps, args.min_cluster_size)
                    valid_clusters = sorted(set(cluster_labels.tolist()) - {-1})
                    n_clusters_per_frag.append(len(valid_clusters))
                    degree_per_frag.append(len(neighbors_of_i))
                    noise_rate_per_frag.append(float((cluster_labels == -1).mean()))

                    if not neighbors_of_i:
                        continue

                    best_dist = np.full(len(frac_idx_i), np.inf)
                    neighbor_label = np.full(len(frac_idx_i), -1, dtype=int)  # index into neighbors_of_i, or -1
                    for n_idx, (_, k_j, p_j) in enumerate(neighbors_of_i):
                        d, _ = cKDTree(global_per_k[k_j]).query(frac_global_i)
                        closer = d < best_dist
                        best_dist = np.where(closer, d, best_dist)
                        neighbor_label = np.where(closer, n_idx, neighbor_label)
                    neighbor_label[best_dist >= CONTACT_EPS] = -1  # "none" -- no real contact

                    # Per-cluster purity + edge coverage
                    edge_best = defaultdict(float)  # n_idx -> best purity seen for this neighbor
                    for c in valid_clusters:
                        c_mask = cluster_labels == c
                        c_size = int(c_mask.sum())
                        labels_in_c = neighbor_label[c_mask]
                        vals, counts = np.unique(labels_in_c, return_counts=True)
                        majority_idx = np.argmax(counts)
                        majority_label, majority_count = vals[majority_idx], counts[majority_idx]
                        purity = majority_count / c_size
                        purity_weighted.append((purity, c_size))
                        mixed_cluster_flags.append(purity < 0.5)
                        if majority_label != -1:
                            edge_best[majority_label] = max(edge_best[majority_label], purity)

                    for n_idx in range(len(neighbors_of_i)):
                        n_edges += 1
                        covered = n_idx in edge_best
                        edge_covered.append(covered)
                        best_cluster_corrprec.append(edge_best.get(n_idx, 0.0))

    print(f"\nAnalyzed {n_fragments} fragments, {n_edges} directed edges "
          f"({args.categories}/{args.split}, mask_strategy={args.mask_strategy}).")

    print("\n" + "=" * 80)
    print(f"PHASE 2D — INTERFACE CLUSTERING DIAGNOSTIC — {args.categories}/{args.split}")
    print(f"  (mask_strategy={args.mask_strategy}, cluster_eps={args.cluster_eps}, "
          f"min_cluster_size={args.min_cluster_size})")
    print("=" * 80)
    purities = np.array([p for p, _ in purity_weighted])
    sizes = np.array([s for _, s in purity_weighted])
    weighted_purity = float((purities * sizes).sum() / sizes.sum()) if len(sizes) else float("nan")

    # Granularity: n_clusters vs graph degree per fragment. "Mean clusters/fragment"
    # alone isn't interpretable on its own -- the actual goal is ~as many patches as
    # interfaces, not just "more clusters". Only defined where degree > 0.
    n_arr = np.array(n_clusters_per_frag)
    deg_arr = np.array(degree_per_frag)
    has_degree = deg_arr > 0
    ratio = n_arr[has_degree] / deg_arr[has_degree] if has_degree.any() else np.array([])
    mean_ratio = float(ratio.mean()) if len(ratio) else float("nan")
    frac_clusters_ge_degree = float((ratio >= 1.0).mean()) if len(ratio) else float("nan")

    print(f"  Mean clusters / fragment        : {np.mean(n_clusters_per_frag):.2f}")
    print(f"  Mean cluster/degree ratio        : {mean_ratio:.2f}")
    print(f"  Fragments with clusters>=degree  : {frac_clusters_ge_degree:.2%}")
    print(f"  Noise rate (points)              : {np.mean(noise_rate_per_frag):.2%}")
    print(f"  Cluster purity (size-weighted)   : {weighted_purity:.2%}")
    print(f"  Mixed cluster rate (<50% purity) : {np.mean(mixed_cluster_flags):.2%}")
    print(f"  Edge coverage                    : {np.mean(edge_covered):.2%}")
    print(f"  Best-cluster CorrPrec upper bound: {np.mean(best_cluster_corrprec):.2%}")
    print(f"  n_fragments={n_fragments}  n_directed_edges={n_edges}  n_clusters_total={len(purity_weighted)}")

    if args.summary_json:
        import json as _json

        label = args.label or f"{args.mask_strategy}_eps{args.cluster_eps}_min{args.min_cluster_size}"
        summary = {
            "label": label,
            "categories": args.categories,
            "split": args.split,
            "config": {
                "mask_strategy": args.mask_strategy,
                "cluster_eps": args.cluster_eps,
                "min_cluster_size": args.min_cluster_size,
            },
            "metrics": {
                "n_fragments": n_fragments,
                "n_directed_edges": n_edges,
                "mean_clusters_per_fragment": float(np.mean(n_clusters_per_frag)) if n_clusters_per_frag else None,
                "mean_cluster_degree_ratio": mean_ratio,
                "frac_fragments_clusters_ge_degree": frac_clusters_ge_degree,
                "noise_rate": float(np.mean(noise_rate_per_frag)) if noise_rate_per_frag else None,
                "cluster_purity_weighted": weighted_purity,
                "mixed_cluster_rate": float(np.mean(mixed_cluster_flags)) if mixed_cluster_flags else None,
                "edge_coverage": float(np.mean(edge_covered)) if edge_covered else None,
                "best_cluster_corrprec": float(np.mean(best_cluster_corrprec)) if best_cluster_corrprec else None,
            },
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            _json.dump(summary, fh, indent=2)
        print(f"\nSaved compact summary to: {summary_path}")


if __name__ == "__main__":
    main()
