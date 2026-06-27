"""
scripts/phase2e_cluster_based_matching.py
=============================================
Phase 2E du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Conclusion Phase 2D : la segmentation fracture (CNN ou GT) mélange les points de TOUTES
les interfaces d'un fragment (`EdgeCoverage` global ≈ 43-44% via clustering spatial, mais
le masque non filtré donne `CorrPrec` ≈ 2.6-3.2% au lieu de ~37% atteignable par cluster).
Ce script teste si APPROXIMER la restriction pair-specific de `gt_edge` par du clustering
spatial NON SUPERVISÉ (pas de connaissance de la pose GT, déployable en pratique) améliore
le matching réel par rapport au masque global brut -- sur le masque CNN `thresh0.3`, pas
un oracle.

Pipeline, sur les paires positives uniquement (`graph[i,j]=True`) :
  1. Masque fracture CNN (thresh0.3) sur chaque fragment.
  2. Clustering par connectivité (mêmes paramètres que la Phase 2D, eps=0.02 = meilleur
     compromis trouvé) sur les points fracture de CHAQUE fragment séparément, dans son
     propre repère local -- aucune information de pose GT utilisée ici (le clustering est
     invariant à la rotation/translation, contrairement à l'étiquetage GT de la Phase 2D
     qui ne servait qu'au diagnostic).
  3. "global" (référence) : matching (1-NN descripteur + RANSAC normal_soft) sur tout le
     masque fracture de chaque fragment, comme en Phase 2C.
  4. "cluster_based" : matching sur CHAQUE paire de clusters (cluster de i x cluster de j),
     on garde l'hypothèse de pose la mieux notée (même critère de score que le RANSAC
     normal-aware) parmi toutes les paires de clusters testées. Si aucune paire de
     clusters ne produit de pose valide, l'arête est comptée en `no_cluster_match`.

Ne teste QUE les paires positives (Protocole A, comme tout Phase 2) -- pas encore de
discrimination positif/négatif.

Usage (sur le serveur) :
    python scripts/phase2e_cluster_based_matching.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --split val --max_batches 60
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
from scripts.phase2d_interface_clustering_diagnostic import cluster_points
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures
from assembly.models.projection_mapping_utils import extract_fragment_list


def match_subset(
    idx_i_sub, idx_j_sub, raw_i, raw_j, desc_i_full, desc_j_full,
    normal_i_full, normal_j_full, rng, args,
):
    """Run the descriptor + RANSAC normal-aware matching pipeline on a given subset of
    point indices for fragments i and j (either the full fracture mask, or a single
    cluster pair). Returns (R_est, t_est, n_inliers, P_cand, Q_cand, normal_P, normal_Q)
    or None if no valid pose was found."""
    if len(idx_i_sub) < p2b.MIN_INLIERS or len(idx_j_sub) < p2b.MIN_INLIERS:
        return None

    desc_i = desc_i_full[idx_i_sub]
    desc_j = desc_j_full[idx_j_sub]
    d_mat = np.linalg.norm(desc_i[:, None, :] - desc_j[None, :, :], axis=-1)
    i_idx_corr, j_idx_corr = p2b.build_correspondences(d_mat, args.corr_mode)

    P_cand = raw_i[idx_i_sub][i_idx_corr]
    Q_cand = raw_j[idx_j_sub][j_idx_corr]
    normal_P = normal_i_full[idx_i_sub][i_idx_corr]
    normal_Q = normal_j_full[idx_j_sub][j_idx_corr]

    pose = p2b.ransac_pose(
        P_cand, Q_cand, rng,
        sample_size=args.ransac_sample_size,
        min_dispersion=args.ransac_min_dispersion,
        score_mode=args.score_mode, score_lambda=args.score_lambda,
        tau=args.score_tau, min_inliers_for_score=args.min_inliers_for_score,
        normal_P_cand=normal_P, normal_Q_cand=normal_Q, normal_tau=args.normal_tau,
    )
    if pose is None:
        return None
    R_est, t_est, n_inliers = pose
    return R_est, t_est, n_inliers, P_cand, Q_cand, normal_P, normal_Q


def score_pose(R_est, t_est, P_cand, Q_cand, normal_P, normal_Q, args):
    """Re-score a candidate pose with the same criterion RANSAC optimizes, so hypotheses
    from different cluster pairs (different candidate sets) can be ranked against each
    other on a common scale."""
    resid = np.linalg.norm((R_est @ P_cand.T).T + t_est - Q_cand, axis=1)
    normal_dot = ((R_est @ normal_P.T).T * normal_Q).sum(axis=1)
    inlier_mask = (
        (resid < p2b.RANSAC_THRESH) & (normal_dot < args.normal_tau)
        if args.normal_tau is not None else resid < p2b.RANSAC_THRESH
    )
    score, _ = p2b._hypothesis_score(
        resid, inlier_mask, args.score_mode, args.score_lambda,
        tau=args.score_tau, min_inliers_for_score=args.min_inliers_for_score,
        normal_dot=normal_dot,
    )
    return score


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
    parser.add_argument("--mask_strategy", default="thresh0.3")
    parser.add_argument("--cluster_eps", type=float, default=0.02)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--corr_mode", default="1nn")
    parser.add_argument("--ransac_sample_size", type=int, default=6)
    parser.add_argument("--ransac_min_dispersion", type=float, default=0.1)
    parser.add_argument("--inlier_thresh", type=float, default=0.03)
    parser.add_argument("--score_mode", default="count_times_quality_and_normal")
    parser.add_argument("--score_lambda", type=float, default=50.0)
    parser.add_argument("--score_tau", type=float, default=None)
    parser.add_argument("--min_inliers_for_score", type=int, default=None)
    parser.add_argument("--normal_tau", type=float, default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    args.score_tau = args.score_tau if args.score_tau is not None else args.inlier_thresh
    args.min_inliers_for_score = (
        args.min_inliers_for_score if args.min_inliers_for_score is not None
        else max(6, args.ransac_sample_size)
    )
    p2b.RANSAC_THRESH = args.inlier_thresh

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
        k=16, use_normals=False, use_curvature=True, use_roughness=True, use_dist_to_centroid=True,
    )

    results = {"global": defaultdict(list), "cluster_based": defaultdict(list)}
    no_cluster_match = 0
    n_edges = 0

    print(f"\nRunning Phase 2E cluster-based matching on {args.categories}/{args.split}...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 10 == 0:
                print(f"  batch {batch_idx}...")

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            frag_list, valid_pcs, K = extract_fragment_list(batch_gpu["pointclouds"], batch_gpu["points_per_part"])
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

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            scores_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            raw_per_k, normals_per_k, desc_per_k = [], [], []
            for k, (b, p) in enumerate(bp_pairs):
                raw_k = pc_local_per_k[k] * scale_np[b, p]
                raw_per_k.append(raw_k)
                nrm_k = normals_np[b, p] if normals_np.ndim == 4 else normals_np[offsets[k]:offsets[k + 1]]
                normals_per_k.append(nrm_k)
                xyz_t = torch.from_numpy(raw_k).float().to(device)
                nrm_t = torch.from_numpy(nrm_k).float().to(device)
                desc_per_k.append(geo_extractor.forward_single(xyz_t, nrm_t).cpu().numpy())

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                for idx_i in range(len(ks_ps)):
                    for idx_j in range(idx_i + 1, len(ks_ps)):
                        k_i, p_i = ks_ps[idx_i]
                        k_j, p_j = ks_ps[idx_j]
                        if not graph_np[b, p_i, p_j]:
                            continue
                        n_edges += 1

                        R_i = p2b.quat_wxyz_to_rotmat(quats_np[b, p_i])
                        R_j = p2b.quat_wxyz_to_rotmat(quats_np[b, p_j])
                        R_ij_gt = R_j.T @ R_i
                        t_ij_gt = R_j.T @ (trans_np[b, p_i] - trans_np[b, p_j])

                        mask_i = p2b.build_mask(args.mask_strategy, scores_per_k[k_i], gt_per_k[k_i], rng)
                        mask_j = p2b.build_mask(args.mask_strategy, scores_per_k[k_j], gt_per_k[k_j], rng)
                        frac_idx_i = np.where(mask_i)[0]
                        frac_idx_j = np.where(mask_j)[0]

                        # --- "global": match on the WHOLE fracture mask, as in Phase 2C ---
                        result = match_subset(
                            frac_idx_i, frac_idx_j, raw_per_k[k_i], raw_per_k[k_j],
                            desc_per_k[k_i], desc_per_k[k_j],
                            normals_per_k[k_i], normals_per_k[k_j], rng, args,
                        )
                        if result is not None:
                            R_est, t_est, n_inliers, P_cand, Q_cand, nP, nQ = result
                            rot_err = p2b.rotation_error_deg(R_est, R_ij_gt)
                            trans_err = float(np.linalg.norm(t_est - t_ij_gt))
                            results["global"]["rot_err_deg"].append(rot_err)
                            results["global"]["trans_err"].append(trans_err)
                            results["global"]["pose_success_30_0.1"].append(rot_err < 30.0 and trans_err < 0.1)
                            results["global"]["ransac_valid"].append(True)
                        else:
                            results["global"]["ransac_valid"].append(False)

                        # --- "cluster_based": match on every (cluster_i, cluster_j) pair,
                        # keep the best-scoring hypothesis across all pairs tested ---
                        if len(frac_idx_i) < args.min_cluster_size or len(frac_idx_j) < args.min_cluster_size:
                            no_cluster_match += 1
                            continue
                        clusters_i = cluster_points(raw_per_k[k_i][frac_idx_i], args.cluster_eps, args.min_cluster_size)
                        clusters_j = cluster_points(raw_per_k[k_j][frac_idx_j], args.cluster_eps, args.min_cluster_size)
                        valid_i = sorted(set(clusters_i.tolist()) - {-1})
                        valid_j = sorted(set(clusters_j.tolist()) - {-1})

                        best = None  # (score, R_est, t_est)
                        for c_a in valid_i:
                            idx_i_sub = frac_idx_i[clusters_i == c_a]
                            for c_b in valid_j:
                                idx_j_sub = frac_idx_j[clusters_j == c_b]
                                res = match_subset(
                                    idx_i_sub, idx_j_sub, raw_per_k[k_i], raw_per_k[k_j],
                                    desc_per_k[k_i], desc_per_k[k_j],
                                    normals_per_k[k_i], normals_per_k[k_j], rng, args,
                                )
                                if res is None:
                                    continue
                                R_e, t_e, n_in, P_c, Q_c, nP_c, nQ_c = res
                                sc = score_pose(R_e, t_e, P_c, Q_c, nP_c, nQ_c, args)
                                if not np.isfinite(sc):
                                    continue
                                if best is None or sc > best[0]:
                                    best = (sc, R_e, t_e)

                        if best is None:
                            no_cluster_match += 1
                            results["cluster_based"]["ransac_valid"].append(False)
                            continue

                        _, R_est, t_est = best
                        rot_err = p2b.rotation_error_deg(R_est, R_ij_gt)
                        trans_err = float(np.linalg.norm(t_est - t_ij_gt))
                        results["cluster_based"]["rot_err_deg"].append(rot_err)
                        results["cluster_based"]["trans_err"].append(trans_err)
                        results["cluster_based"]["pose_success_30_0.1"].append(rot_err < 30.0 and trans_err < 0.1)
                        results["cluster_based"]["ransac_valid"].append(True)

    print(f"\nAnalyzed {n_edges} adjacent fragment pairs ({args.categories}/{args.split}).")

    print("\n" + "=" * 90)
    print(f"PHASE 2E — CLUSTER-BASED PAIRWISE MATCHING — {args.categories}/{args.split}")
    print(f"  (mask_strategy={args.mask_strategy}, cluster_eps={args.cluster_eps}, "
          f"score_mode={args.score_mode}, inlier_thresh={args.inlier_thresh})")
    print("=" * 90)
    print(f"  {'Variant':<14} {'RansacValid':>12} {'Pose@30d_0.1':>13} {'RotErr(deg)':>12} {'TransErr':>10} {'n':>6}")
    for variant in ["global", "cluster_based"]:
        d = results[variant]
        valid = d.get("ransac_valid", [])
        if not valid:
            continue
        rot_errs = d.get("rot_err_deg", [])
        trans_errs = d.get("trans_err", [])
        pose30 = d.get("pose_success_30_0.1", [])
        print(
            f"  {variant:<14} {np.mean(valid):>12.2%} "
            f"{np.mean(pose30) if pose30 else float('nan'):>13.2%} "
            f"{np.mean(rot_errs) if rot_errs else float('nan'):>12.2f} "
            f"{np.mean(trans_errs) if trans_errs else float('nan'):>10.4f} {len(valid):>6}"
        )
    print(f"\n  no_cluster_match rate (cluster_based): {no_cluster_match / max(n_edges, 1):.2%}  ({no_cluster_match}/{n_edges})")

    if args.summary_json:
        import json as _json
        summary = {
            "label": args.label or f"phase2e_{args.mask_strategy}",
            "categories": args.categories, "split": args.split,
            "config": vars(args),
            "metrics": {
                "n_edges": n_edges,
                "no_cluster_match_rate": no_cluster_match / max(n_edges, 1),
            },
        }
        for variant in ["global", "cluster_based"]:
            d = results[variant]
            valid = d.get("ransac_valid", [])
            summary["metrics"][f"{variant}_ransac_valid_rate"] = float(np.mean(valid)) if valid else None
            summary["metrics"][f"{variant}_pose_success_30_0.1"] = (
                float(np.mean(d["pose_success_30_0.1"])) if d.get("pose_success_30_0.1") else None
            )
            summary["metrics"][f"{variant}_rot_err_deg"] = (
                float(np.mean(d["rot_err_deg"])) if d.get("rot_err_deg") else None
            )
            summary["metrics"][f"{variant}_trans_err"] = (
                float(np.mean(d["trans_err"])) if d.get("trans_err") else None
            )
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            _json.dump(summary, fh, indent=2)
        print(f"\nSaved compact summary to: {summary_path}")


if __name__ == "__main__":
    main()
