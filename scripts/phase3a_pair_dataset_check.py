"""
scripts/phase3a_pair_dataset_check.py
======================================
Phase 3A du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md, section
"Phase 3 -- Matcher appris", cadrage du 2026-06-27).

Plomberie de données pour le futur module appris (encodeur + matching souple +
weighted Kabsch), PAS le modèle lui-même. Construit, pour chaque paire positive
(`graph[i,j]=True`), le format d'entrée exact que le modèle consommera, et vérifie
les conventions verrouillées avant d'écrire la moindre couche apprise :

  - Paires DIRECTED par défaut : (i,j) et (j,i) sont deux exemples distincts (la
    matrice de correspondance et la pose relative R_ij/t_ij sont orientées). `--undirected`
    bascule sur i<j uniquement.
  - Sampling à N points : PAS de répétition. Si le masque a >= N points, sous-échantillonne
    (random ou FPS, `--sample_mode`). Si 0 < masque < N, garde tout + padding (valid=False
    sur le padding). Si masque vide, fallback top-N par probabilité CNN (`fallback=True`
    loggué). Format `points_i/j [N,3]` + `valid_i/j [N]` -- les points paddés sont ignorés
    en loss/métriques, jamais dupliqués.
  - Label de correspondance `target_ij [N, N+1]` : colonnes 0..N-1 = points sélectionnés de
    j (alignés à `points_j`), colonne N = dustbin. Un point de i sans aucun vrai contact
    (distance < `--contact_eps` sous la pose GT) reçoit tout son poids sur le dustbin --
    pas forcé sur un faux match. `--label_mode hard` (1 colonne à 1, NN sous la pose GT) ou
    `soft` (pondération gaussienne sur tous les j proches, `--label_sigma`).

Réutilise directement les briques Phase 2 déjà validées : `extract_fragment_list`,
`CNNFracSeg`, `quat_wxyz_to_rotmat`/formule R_ij-t_ij (`phase2_geometric_baseline.py`),
`HybridGeometryFeatures`. La logique de distance pair-specific est la même que `gt_edge`
(Phase 2A) et `oracle_cluster_pair` (Phase 2E), réutilisée ici comme signal d'apprentissage
plutôt que comme métrique d'évaluation seule.

Usage (sur le serveur) :
    python scripts/phase3a_pair_dataset_check.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --split val \
        --mask_strategy thresh0.3 --num_points 512 \
        --contact_eps 0.05 --label_sigma 0.02 --label_mode soft --sample_mode random \
        --max_batches 10 --summary_json /tmp/student7/phase3a_pair_dataset_check.json
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
from scripts import phase2_geometric_baseline as p2b
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures
from assembly.models.projection_mapping_utils import extract_fragment_list


N_EXAMPLES_TO_PRINT = 3


def fps_sample(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point-sampling indices into `points` (n_pool, 3), random start.
    O(n_samples * n_pool) -- fine for n_pool up to a few thousand fracture points and
    n_samples up to 1024. Chosen over random sampling to preserve spatial diversity across
    the masked fracture surface (Phase 2's bottleneck was interface coverage, not just
    point count -- random sampling can over-represent a dense patch and miss a small,
    separate interface entirely)."""
    n_pool = points.shape[0]
    if n_samples >= n_pool:
        return np.arange(n_pool)
    selected = np.empty(n_samples, dtype=np.int64)
    start = int(rng.integers(n_pool))
    selected[0] = start
    dist = np.linalg.norm(points - points[start], axis=1)
    for k in range(1, n_samples):
        idx = int(np.argmax(dist))
        selected[k] = idx
        new_dist = np.linalg.norm(points - points[idx], axis=1)
        dist = np.minimum(dist, new_dist)
    return selected


def select_points(pool_idx: np.ndarray, pool_xyz: np.ndarray, N: int, mode: str, rng: np.random.Generator):
    """Select up to N points from a candidate pool (mask or fallback indices into a
    fragment's full point array). Returns (selected_idx [N] int, valid [N] bool).

    NO repetition: if the pool has fewer than N points, the remainder is padded with
    valid=False (ignored downstream), never filled by repeating real points -- repetition
    would create duplicate fake correspondences and inflate target density artificially.
    """
    n_pool = len(pool_idx)
    selected = np.zeros(N, dtype=pool_idx.dtype if n_pool else np.int64)
    valid = np.zeros(N, dtype=bool)
    if n_pool == 0:
        return selected, valid
    if n_pool >= N:
        if mode == "random":
            sub = rng.choice(n_pool, size=N, replace=False)
        elif mode == "fps":
            sub = fps_sample(pool_xyz, N, rng)
        else:
            raise ValueError(mode)
        selected[:] = pool_idx[sub]
        valid[:] = True
    else:
        selected[:n_pool] = pool_idx
        valid[:n_pool] = True
    return selected, valid


def build_pair_sample(
    raw_i, raw_j, normals_i, normals_j, scores_i, scores_j, desc_i, desc_j,
    mask_i_bool, mask_j_bool, R_ij_gt, t_ij_gt, args, rng,
):
    """Build the full Phase 3A input/label tensors for one directed pair i->j.
    Returns a dict (see module docstring for the exact field list) plus a `diag` dict of
    per-pair scalars consumed by the summary aggregation."""
    N = args.num_points
    diag = {}

    n_mask_i, n_mask_j = int(mask_i_bool.sum()), int(mask_j_bool.sum())
    diag["mask_points_i"], diag["mask_points_j"] = n_mask_i, n_mask_j
    # Informational, NOT a pipeline bug: a real fragment can have a tiny fracture mask
    # (e.g. a small chip touching several neighbors, Phase 2B's avail_rate confound) while
    # its neighbor has thousands of points -- this is expected data asymmetry, tracked as
    # a rate rather than flagged per-pair as a "sanity issue".
    diag["asymmetric_pair"] = n_mask_i < 0.5 * N or n_mask_j < 0.5 * N

    fallback_i = n_mask_i == 0
    fallback_j = n_mask_j == 0
    diag["fallback"] = fallback_i or fallback_j

    pool_idx_i = (
        np.where(mask_i_bool)[0] if not fallback_i
        else np.argsort(-scores_i)[: min(N, len(scores_i))]
    )
    pool_idx_j = (
        np.where(mask_j_bool)[0] if not fallback_j
        else np.argsort(-scores_j)[: min(N, len(scores_j))]
    )

    sel_i, valid_i = select_points(pool_idx_i, raw_i[pool_idx_i], N, args.sample_mode, rng)
    sel_j, valid_j = select_points(pool_idx_j, raw_j[pool_idx_j], N, args.sample_mode, rng)

    diag["padding_i"] = float((~valid_i).mean())
    diag["padding_j"] = float((~valid_j).mean())
    diag["valid_points_i"] = int(valid_i.sum())
    diag["valid_points_j"] = int(valid_j.sum())

    points_i = np.where(valid_i[:, None], raw_i[sel_i], 0.0)
    points_j = np.where(valid_j[:, None], raw_j[sel_j], 0.0)
    normals_out_i = np.where(valid_i[:, None], normals_i[sel_i], 0.0)
    normals_out_j = np.where(valid_j[:, None], normals_j[sel_j], 0.0)
    cnn_score_i = np.where(valid_i, scores_i[sel_i], 0.0)[:, None]
    cnn_score_j = np.where(valid_j, scores_j[sel_j], 0.0)[:, None]
    geom_i = np.where(valid_i[:, None], desc_i[sel_i], 0.0)
    geom_j = np.where(valid_j[:, None], desc_j[sel_j], 0.0)

    # Distance in j's local frame: transform i's selected points by the GT relative pose
    # (raw_i and raw_j live in independent per-fragment random-rotation frames, cf. Phase 0
    # -- only comparable after applying R_ij/t_ij, same as gt_edge/oracle_cluster_pair).
    pred_i_in_j = (R_ij_gt @ points_i.T).T + t_ij_gt
    d_mat = np.linalg.norm(pred_i_in_j[:, None, :] - points_j[None, :, :], axis=-1)  # [N,N]
    d_mat[~valid_i, :] = np.inf
    d_mat[:, ~valid_j] = np.inf

    target = np.zeros((N, N + 1), dtype=np.float32)
    contact_rows = 0
    matches_per_contact_row = []
    effective_matches_per_contact_row = []
    for a in range(N):
        if not valid_i[a]:
            continue  # padded source row: no target, excluded from loss via valid_i
        row = d_mat[a]
        close = row < args.contact_eps
        if not close.any():
            target[a, N] = 1.0  # dustbin: no real contact for this point
            continue
        contact_rows += 1
        if args.label_mode == "hard":
            b_star = int(np.argmin(row))
            target[a, b_star] = 1.0
            matches_per_contact_row.append(1)
            effective_matches_per_contact_row.append(1.0)
        elif args.label_mode == "soft":
            w = np.exp(-(row[close] ** 2) / (args.label_sigma ** 2))
            w = w / w.sum()
            target[a, np.where(close)[0]] = w
            # raw count of points within contact_eps -- on dense fracture surfaces this can
            # be large (tens of points) without meaning the label is actually diffuse: most
            # of those points can carry near-zero weight after the Gaussian normalization.
            matches_per_contact_row.append(int(close.sum()))
            # effective number of matches (inverse participation ratio, 1/sum(w^2)): 1.0 for
            # a one-hot-like label (sharp, weight concentrated on the true NN), tends toward
            # close.sum() for a near-uniform label (diffuse, weak positional signal). This is
            # the metric that actually tells us whether contact_eps/label_sigma produce a
            # usable supervision target, not just "how many points are nearby".
            effective_matches_per_contact_row.append(float(1.0 / np.sum(w ** 2)))
        else:
            raise ValueError(args.label_mode)

    n_valid_rows = int(valid_i.sum())
    diag["contact_row_rate"] = contact_rows / n_valid_rows if n_valid_rows else float("nan")
    diag["dustbin_row_rate"] = (
        (n_valid_rows - contact_rows) / n_valid_rows if n_valid_rows else float("nan")
    )
    diag["target_density"] = float((target[:, :N] > 0).sum()) / max(n_valid_rows, 1)
    diag["mean_matches_per_contact_row"] = (
        float(np.mean(matches_per_contact_row)) if matches_per_contact_row else float("nan")
    )
    diag["mean_effective_matches_per_contact_row"] = (
        float(np.mean(effective_matches_per_contact_row)) if effective_matches_per_contact_row else float("nan")
    )
    # valid_target_col_rate: fraction of non-dustbin target mass that lands on a valid
    # (non-padded) j column -- should always be 1.0 by construction (d_mat forced to inf
    # on padded columns), kept as an explicit sanity check rather than an assumption.
    mass_on_cols = target[:, :N].sum()
    mass_on_valid_cols = target[:, :N][:, valid_j].sum()
    diag["valid_target_col_rate"] = (
        float(mass_on_valid_cols / mass_on_cols) if mass_on_cols > 0 else float("nan")
    )

    sample = {
        "points_i": points_i, "points_j": points_j,
        "normals_i": normals_out_i, "normals_j": normals_out_j,
        "cnn_score_i": cnn_score_i, "cnn_score_j": cnn_score_j,
        "geom_feats_i": geom_i, "geom_feats_j": geom_j,
        "valid_i": valid_i, "valid_j": valid_j,
        "target_ij": target,
        "R_ij": R_ij_gt, "t_ij": t_ij_gt,
    }
    return sample, diag


def run_sanity_checks(sample, diag, pair_meta) -> list:
    """Hard failures / strong warnings, returns a list of message strings (empty if clean)."""
    issues = []
    target, valid_i, valid_j = sample["target_ij"], sample["valid_i"], sample["valid_j"]
    N = valid_i.shape[0]

    row_sums = target[valid_i].sum(axis=1)
    if row_sums.size and not np.allclose(row_sums, 1.0, atol=1e-4):
        bad = int((~np.isclose(row_sums, 1.0, atol=1e-4)).sum())
        issues.append(f"{pair_meta}: {bad}/{valid_i.sum()} valid rows have target sum != 1")

    if (target[~valid_i] != 0).any():
        issues.append(f"{pair_meta}: padded source rows (valid_i=False) carry nonzero target")

    mass_on_invalid_j = target[:, :N][:, ~valid_j].sum()
    if mass_on_invalid_j > 1e-6:
        issues.append(f"{pair_meta}: target puts {mass_on_invalid_j:.4f} mass on padded j columns")

    for key in ("points_i", "points_j", "normals_i", "normals_j", "cnn_score_i", "cnn_score_j", "geom_feats_i", "geom_feats_j"):
        if not np.isfinite(sample[key]).all():
            issues.append(f"{pair_meta}: NaN/Inf in {key}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1116)
    parser.add_argument(
        "--mask_strategy", default="thresh0.3",
        help="Fracture mask strategy, reuses p2b.build_mask: gt, thresh<T>, random, all.",
    )
    parser.add_argument("--num_points", type=int, default=512, help="N points per fragment side.")
    parser.add_argument("--contact_eps", type=float, default=0.05)
    parser.add_argument("--label_sigma", type=float, default=0.02)
    parser.add_argument("--label_mode", default="soft", choices=["hard", "soft"])
    parser.add_argument("--sample_mode", default="random", choices=["random", "fps"])
    parser.add_argument(
        "--undirected", action="store_true",
        help="Emit only i<j pairs instead of both (i,j) and (j,i) (default: directed).",
    )
    parser.add_argument("--summary_json", default=None)
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

    geo_extractor = HybridGeometryFeatures(
        k=16, use_normals=False, use_curvature=True,
        use_roughness=True, use_dist_to_centroid=True,
    )

    diag_acc = defaultdict(list)
    n_batches, n_objects, n_pairs = 0, 0, 0
    issues_all = []
    examples_printed = 0

    print(f"\nPhase 3A pair-dataset check on {args.categories}/{args.split} "
          f"(mask={args.mask_strategy}, N={args.num_points}, sample_mode={args.sample_mode}, "
          f"label_mode={args.label_mode}, directed={not args.undirected})...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            n_batches += 1

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
            normals_np = batch["pointclouds_normals"].numpy()
            names = batch["name"]

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            scores_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            normals_per_k, raw_per_k = [], []
            for k, (b, p) in enumerate(bp_pairs):
                scale_k = scale_np[b, p]
                raw_k = pc_local_per_k[k] * scale_k
                raw_per_k.append(raw_k)
                normals_per_k.append(
                    normals_np[b, p] if normals_np.ndim == 4
                    else normals_np[offsets[k]:offsets[k + 1]]
                )

            desc_per_k = []
            for k in range(K):
                xyz_t = torch.from_numpy(raw_per_k[k]).float().to(device)
                nrm_t = torch.from_numpy(normals_per_k[k]).float().to(device)
                desc_per_k.append(geo_extractor.forward_single(xyz_t, nrm_t).cpu().numpy())

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                n_objects += 1
                pos_pairs_this_object = 0
                for idx_i in range(len(ks_ps)):
                    j_range = (
                        range(idx_i + 1, len(ks_ps)) if args.undirected
                        else [j for j in range(len(ks_ps)) if j != idx_i]
                    )
                    for idx_j in j_range:
                        k_i, p_i = ks_ps[idx_i]
                        k_j, p_j = ks_ps[idx_j]
                        if not graph_np[b, p_i, p_j]:
                            continue
                        n_pairs += 1
                        pos_pairs_this_object += 1

                        R_i = p2b.quat_wxyz_to_rotmat(quats_np[b, p_i])
                        R_j = p2b.quat_wxyz_to_rotmat(quats_np[b, p_j])
                        R_j_inv = R_j.T
                        R_ij_gt = R_j_inv @ R_i
                        t_ij_gt = R_j_inv @ (trans_np[b, p_i] - trans_np[b, p_j])

                        mask_i_bool = p2b.build_mask(
                            args.mask_strategy, scores_per_k[k_i], gt_per_k[k_i], rng
                        )
                        mask_j_bool = p2b.build_mask(
                            args.mask_strategy, scores_per_k[k_j], gt_per_k[k_j], rng
                        )

                        sample, diag = build_pair_sample(
                            raw_per_k[k_i], raw_per_k[k_j],
                            normals_per_k[k_i], normals_per_k[k_j],
                            scores_per_k[k_i], scores_per_k[k_j],
                            desc_per_k[k_i], desc_per_k[k_j],
                            mask_i_bool, mask_j_bool, R_ij_gt, t_ij_gt, args, rng,
                        )

                        pair_meta = f"{names[b]} part_i={p_i} part_j={p_j}"
                        issues = run_sanity_checks(sample, diag, pair_meta)
                        issues_all.extend(issues)

                        for key, val in diag.items():
                            diag_acc[key].append(val)

                        if examples_printed < N_EXAMPLES_TO_PRINT:
                            print(
                                f"\n  Example {examples_printed + 1}: {pair_meta}\n"
                                f"    mask_count_i={diag['mask_points_i']} mask_count_j={diag['mask_points_j']} "
                                f"fallback={diag['fallback']}\n"
                                f"    valid_i={diag['valid_points_i']} valid_j={diag['valid_points_j']}\n"
                                f"    contact_row_rate={diag['contact_row_rate']:.2%} "
                                f"dustbin_row_rate={diag['dustbin_row_rate']:.2%} "
                                f"target_density={diag['target_density']:.4f}"
                            )
                            examples_printed += 1

                if pos_pairs_this_object == 0:
                    issues_all.append(f"batch {batch_idx} object {names[b]}: no positive pair")

            if batch_idx % 5 == 0:
                print(f"  batch {batch_idx}... ({n_pairs} pairs so far)")

    print("\n" + "=" * 100)
    print(f"PHASE 3A — PAIR DATASET CHECK SUMMARY — {args.categories}/{args.split}")
    print("=" * 100)
    mean_pos_per_obj = n_pairs / n_objects if n_objects else float("nan")
    print(f"  n_batches={n_batches}  n_objects={n_objects}  n_positive_pairs_directed={n_pairs}  "
          f"mean_pos_pairs_per_object={mean_pos_per_obj:.2f}")

    def pct(key, q):
        # nanpercentile: a handful of degenerate pairs (e.g. an empty fragment) can leave
        # contact_row_rate/target_density etc. as NaN -- exclude them from the aggregate
        # rather than let one NaN silently poison the whole summary.
        vals = diag_acc.get(key, [])
        return float(np.nanpercentile(vals, q)) if vals else float("nan")

    def mean(key):
        vals = diag_acc.get(key, [])
        return float(np.nanmean(vals)) if vals else float("nan")

    print(f"  mean_mask_points_i={mean('mask_points_i'):.1f}  mean_mask_points_j={mean('mask_points_j'):.1f}")
    for key in ("mask_points_i", "mask_points_j"):
        print(f"    {key}: p10={pct(key,10):.0f} p50={pct(key,50):.0f} p90={pct(key,90):.0f}")
    print(f"  padding_rate_i={mean('padding_i'):.2%}  padding_rate_j={mean('padding_j'):.2%}")
    print(f"  fallback_rate={mean('fallback'):.2%}")
    print(f"  asymmetric_pair_rate={mean('asymmetric_pair'):.2%}  "
          f"(informational: one side's mask < 50% of N -- real fragment-size asymmetry, not a bug)")
    print(f"  mean_valid_points_i={mean('valid_points_i'):.1f}  mean_valid_points_j={mean('valid_points_j'):.1f}")
    print(f"  contact_row_rate={mean('contact_row_rate'):.2%}  dustbin_row_rate={mean('dustbin_row_rate'):.2%}")
    print(f"  target_density={mean('target_density'):.4f}  mean_matches_per_contact_row={mean('mean_matches_per_contact_row'):.2f}")
    print(f"  mean_effective_matches_per_contact_row={mean('mean_effective_matches_per_contact_row'):.2f}  "
          f"(1.0 = sharp/one-hot-like label, -> raw count = uniform/diffuse label, no real positional signal)")
    print(f"  valid_target_col_rate={mean('valid_target_col_rate'):.4f}")

    print("\n" + "=" * 100)
    print(f"SANITY CHECK ISSUES: {len(issues_all)}")
    print("=" * 100)
    for issue in issues_all[:30]:
        print(f"  WARNING: {issue}")
    if len(issues_all) > 30:
        print(f"  ... ({len(issues_all) - 30} more, truncated)")

    if args.summary_json:
        import json as _json

        summary = {
            "categories": args.categories, "split": args.split,
            "config": {
                "mask_strategy": args.mask_strategy, "num_points": args.num_points,
                "contact_eps": args.contact_eps, "label_sigma": args.label_sigma,
                "label_mode": args.label_mode, "sample_mode": args.sample_mode,
                "directed": not args.undirected,
            },
            "n_batches": n_batches, "n_objects": n_objects,
            "n_positive_pairs_directed": n_pairs,
            "mean_pos_pairs_per_object": mean_pos_per_obj,
            "mean_mask_points_i": mean("mask_points_i"), "mean_mask_points_j": mean("mask_points_j"),
            "mask_points_i_p10_p50_p90": [pct("mask_points_i", q) for q in (10, 50, 90)],
            "mask_points_j_p10_p50_p90": [pct("mask_points_j", q) for q in (10, 50, 90)],
            "padding_rate_i": mean("padding_i"), "padding_rate_j": mean("padding_j"),
            "fallback_rate": mean("fallback"),
            "asymmetric_pair_rate": mean("asymmetric_pair"),
            "mean_valid_points_i": mean("valid_points_i"), "mean_valid_points_j": mean("valid_points_j"),
            "contact_row_rate": mean("contact_row_rate"), "dustbin_row_rate": mean("dustbin_row_rate"),
            "target_density": mean("target_density"),
            "mean_matches_per_contact_row": mean("mean_matches_per_contact_row"),
            "mean_effective_matches_per_contact_row": mean("mean_effective_matches_per_contact_row"),
            "valid_target_col_rate": mean("valid_target_col_rate"),
            "n_sanity_issues": len(issues_all),
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            _json.dump(summary, fh, indent=2)
        print(f"\nSaved compact summary to: {summary_path}")


if __name__ == "__main__":
    main()
