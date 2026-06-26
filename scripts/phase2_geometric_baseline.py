"""
scripts/phase2_geometric_baseline.py
======================================
Phase 2 du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Pipeline minimal, sans réseau de matching appris :

    masque fracture (GT / GT restreint au contact pair-specific / CNN seuil / random / all)
    -> descripteurs géométriques rotation-invariants (HybridGeometryFeatures)
    -> correspondances candidates (1-NN en espace descripteur)
    -> RANSAC (échantillons de 3 correspondances + Kabsch)
    -> pose relative estimée (Kabsch pondéré sur les inliers)
    -> comparaison à la pose relative GT (dérivée des poses confirmées en Phase 0)

Comparaison obligatoire à 3 conditions (isole la responsabilité d'un échec) :
  - masque fracture GT       : borne haute, isole "le matching marche-t-il du tout ?"
  - masque CNN (seuil 0.2/0.3/0.5) : la condition réelle d'usage
  - random / all             : référence basse (le filtre fracture aide-t-il vraiment ?)

Travaille directement sur le repère d'entrée non-assemblé (pointclouds_model), pas sur
pointclouds_gt — c'est l'information réellement disponible à l'inférence. Le facteur
`scale` est ré-appliqué (recalculable depuis les points eux-mêmes, ce n'est pas une fuite
de label) pour revenir au repère "centré-roté" où la pose relative GT est définie
(cf. formule dans PLAN_REASSEMBLY_MODULE.md, Phase 0).

Trois métriques distinctes, à ne pas confondre (cf. plan) :
  - correspondence_precision (diagnostic, calculé indépendamment de RANSAC) : fraction des
    candidats 1-NN qui sont géométriquement corrects SOUS LA VRAIE POSE GT. Isole "le
    descripteur/la mise en correspondance produit-elle de vrais matches du tout ?" d'un
    éventuel échec de RANSAC/Kabsch en aval. Si ce chiffre est ~0%, RANSAC ne peut
    structurellement pas trouver la bonne pose, quel que soit le nombre d'itérations --
    ce n'est alors pas un problème de RANSAC mais de descripteur (ou de masque qui mélange
    plusieurs interfaces, cf. le même problème fragment-level identifié en Phase 1).
  - ransac_valid : RANSAC a trouvé >=3 inliers -- dit seulement qu'une pose a été produite,
    pas qu'elle est correcte (3 inliers peuvent satisfaire le seuil résiduel par hasard).
  - pose_success_(rot_thresh, trans_thresh) : la pose estimée est réellement proche de la
    GT (rotation_error < rot_thresh ET translation_error < trans_thresh).

Protocole A uniquement (registration sur paires positives, graph[i,j]=True) : mesure si,
pour deux fragments qui vont vraiment ensemble, la baseline retrouve leur pose relative.
Le protocole B (discrimination positif/négatif -- distinguer une vraie paire d'une fausse
via le score de matching) est une question différente, pas encore implémentée ici.

Usage (sur le serveur) :
    python scripts/phase2_geometric_baseline.py \
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
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures
from assembly.models.projection_mapping_utils import extract_fragment_list


ALL_MASK_STRATEGIES = ["gt", "gt_edge", "thresh0.2", "thresh0.3", "thresh0.5", "random", "all"]
RANSAC_ITERS = 500
RANSAC_THRESH = 0.05   # inlier distance, same units as Phase 0's eps
MIN_INLIERS = 3

# pose_success thresholds: (rotation error max in degrees, translation error max).
# A pair "succeeds" only if BOTH are met -- distinct from ransac_valid (>=3 inliers,
# which says RANSAC produced *a* pose, not that it is a *correct* one: 3 inliers can
# satisfy the residual threshold by geometric coincidence on a wrong pose).
POSE_SUCCESS_THRESHOLDS = [(15.0, 0.05), (30.0, 0.1)]
TOPK_DIAG_LIST = [5, 10, 20]


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation minimizing ||R @ P + t - Q||^2 over correspondences P->Q."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    Rmat = Vt.T @ D @ U.T
    t = q_mean - Rmat @ p_mean
    return Rmat, t


def ransac_pose(P_cand: np.ndarray, Q_cand: np.ndarray, rng: np.random.Generator):
    """RANSAC over candidate correspondences. Returns (R, t, n_inliers) or None."""
    n = len(P_cand)
    if n < MIN_INLIERS:
        return None

    best_inliers, best_mask = -1, None
    for _ in range(RANSAC_ITERS):
        sample = rng.choice(n, size=3, replace=False)
        try:
            Rmat, t = kabsch(P_cand[sample], Q_cand[sample])
        except np.linalg.LinAlgError:
            continue
        pred = (Rmat @ P_cand.T).T + t
        resid = np.linalg.norm(pred - Q_cand, axis=1)
        inlier_mask = resid < RANSAC_THRESH
        n_in = int(inlier_mask.sum())
        if n_in > best_inliers:
            best_inliers, best_mask = n_in, inlier_mask

    if best_inliers < MIN_INLIERS:
        return None

    Rmat, t = kabsch(P_cand[best_mask], Q_cand[best_mask])
    return Rmat, t, best_inliers


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos_angle = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def build_mask(strategy: str, scores: np.ndarray, gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_pts = len(scores)
    if strategy == "gt":
        return gt.astype(bool)
    if strategy.startswith("thresh"):
        t = float(strategy.replace("thresh", ""))
        return scores > t
    if strategy == "all":
        return np.ones(n_pts, dtype=bool)
    if strategy == "random":
        n_keep = max(int(gt.sum()), 10)
        n_keep = min(n_keep, n_pts)
        idx = rng.choice(n_pts, size=n_keep, replace=False)
        mask = np.zeros(n_pts, dtype=bool)
        mask[idx] = True
        return mask
    raise ValueError(strategy)


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
        "--strategies", default=None,
        help=f"Comma-separated subset of {ALL_MASK_STRATEGIES} (default: all). "
             "Each strategy costs roughly the same O(Ni*Nj) work -- restricting "
             "this is the main lever to cut runtime (a full run took 1h05 for "
             "60 batches / 634 edges / 6 strategies).",
    )
    args = parser.parse_args()
    mask_strategies = args.strategies.split(",") if args.strategies else ALL_MASK_STRATEGIES

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
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    geo_extractor = HybridGeometryFeatures(
        k=16, use_normals=False, use_curvature=True,
        use_roughness=True, use_dist_to_centroid=True,
    )

    # results[strategy] -> dict of lists
    results = defaultdict(lambda: defaultdict(list))

    print(f"\nRunning Phase 2 geometric baseline on {args.categories}/{args.split}...")
    n_edges = 0
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
            normals_np = batch["pointclouds_normals"].numpy()  # (B, P, N, 3) or (B, N_total, 3)

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            scores_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            normals_per_k = []
            raw_per_k = []
            for k, (b, p) in enumerate(bp_pairs):
                scale_k = scale_np[b, p]
                raw_k = pc_local_per_k[k] * scale_k          # un-normalize, no leak (scale = max|x|)
                raw_per_k.append(raw_k)
                if normals_np.ndim == 4:
                    normals_per_k.append(normals_np[b, p])
                else:
                    normals_per_k.append(normals_np[offsets[k]:offsets[k + 1]])

            # Geometric descriptors per fragment (rotation/translation-invariant scalars)
            desc_per_k = []
            for k in range(K):
                xyz_t = torch.from_numpy(raw_per_k[k]).float().to(device)
                nrm_t = torch.from_numpy(normals_per_k[k]).float().to(device)
                desc = geo_extractor.forward_single(xyz_t, nrm_t).cpu().numpy()
                desc_per_k.append(desc)

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

                        # GT relative pose: raw_j ≈ R_ij @ raw_i + t_ij
                        R_i = quat_wxyz_to_rotmat(quats_np[b, p_i])
                        R_j = quat_wxyz_to_rotmat(quats_np[b, p_j])
                        R_j_inv = R_j.T
                        R_ij_gt = R_j_inv @ R_i
                        t_ij_gt = R_j_inv @ (trans_np[b, p_i] - trans_np[b, p_j])

                        # Global/assembled-frame reconstruction (Phase 0 formula), needed
                        # for "gt_edge" -- raw_i and raw_j each live in their OWN independent
                        # local frame (random per-fragment rotation), so a Euclidean distance
                        # between them directly is meaningless. Must compare in the shared
                        # assembled frame, same as Phase 0/1.
                        global_i_full = (R_i @ raw_per_k[k_i].T).T + trans_np[b, p_i]
                        global_j_full = (R_j @ raw_per_k[k_j].T).T + trans_np[b, p_j]

                        for strategy in mask_strategies:
                            if strategy == "gt_edge":
                                # Oracle restricted to pair-specific contact points: GT
                                # fracture points of i (resp. j) whose nearest neighbor in
                                # the FULL point cloud of j (resp. i), in the shared assembled
                                # frame, is < eps. Isolates the multi-neighbor confound from
                                # "gt" (which keeps ALL of a fragment's fracture points, even
                                # those facing OTHER neighbors -- cf. avail_rate=38% diagnostic,
                                # same issue as Phase 1's fragment-level vs edge_contact_recall).
                                gt_i = gt_per_k[k_i].astype(bool)
                                gt_j = gt_per_k[k_j].astype(bool)
                                mask_i = np.zeros_like(gt_i)
                                mask_j = np.zeros_like(gt_j)
                                if gt_i.any():
                                    d_i_to_j = np.linalg.norm(
                                        global_i_full[gt_i][:, None, :]
                                        - global_j_full[None, :, :], axis=-1
                                    ).min(axis=1)
                                    mask_i[np.where(gt_i)[0]] = d_i_to_j < RANSAC_THRESH
                                if gt_j.any():
                                    d_j_to_i = np.linalg.norm(
                                        global_j_full[gt_j][:, None, :]
                                        - global_i_full[None, :, :], axis=-1
                                    ).min(axis=1)
                                    mask_j[np.where(gt_j)[0]] = d_j_to_i < RANSAC_THRESH
                            else:
                                mask_i = build_mask(strategy, scores_per_k[k_i], gt_per_k[k_i], rng)
                                mask_j = build_mask(strategy, scores_per_k[k_j], gt_per_k[k_j], rng)

                            idx_i_keep = np.where(mask_i)[0]
                            idx_j_keep = np.where(mask_j)[0]
                            if len(idx_i_keep) < MIN_INLIERS or len(idx_j_keep) < MIN_INLIERS:
                                results[strategy]["ransac_valid"].append(False)
                                for thresh in POSE_SUCCESS_THRESHOLDS:
                                    results[strategy][f"pose_success_{thresh}"].append(False)
                                continue

                            # 1-NN correspondence in descriptor space, i -> j
                            desc_i = desc_per_k[k_i][idx_i_keep]      # (Ni, D)
                            desc_j = desc_per_k[k_j][idx_j_keep]      # (Nj, D)
                            d_mat = np.linalg.norm(
                                desc_i[:, None, :] - desc_j[None, :, :], axis=-1
                            )
                            best_j = d_mat.argmin(axis=1)             # (Ni,)

                            P_cand = raw_per_k[k_i][idx_i_keep]
                            P_cand_all = raw_per_k[k_i][idx_i_keep]   # (Ni,3) -- all filtered i points
                            Q_full = raw_per_k[k_j][idx_j_keep]       # (Nj,3) -- all filtered j points
                            Q_cand = Q_full[best_j]

                            # Diagnostic 1: is the 1-NN candidate set itself usable at all?
                            # A candidate is "correct" if it's geometrically consistent with
                            # the TRUE pose (independent of whether RANSAC/Kabsch can recover
                            # that pose from the candidate set). Distinguishes "descriptor too
                            # weak" from "no true correspondence exists among candidates"
                            # (e.g. fragment touches >1 neighbor, mask mixes multiple interfaces).
                            pred_under_gt = (R_ij_gt @ P_cand.T).T + t_ij_gt
                            resid_under_gt = np.linalg.norm(pred_under_gt - Q_cand, axis=1)
                            corr_precision = float((resid_under_gt < RANSAC_THRESH).mean())
                            results[strategy]["correspondence_precision"].append(corr_precision)
                            results[strategy]["n_candidates_diag"].append(len(P_cand))

                            # Diagnostic 2: top-K correspondence recall. Among i-points that DO
                            # have at least one geometrically correct j-point available in the
                            # filtered set (avail_mask -- isolates the multi-neighbor confound:
                            # if False, no true match exists here regardless of descriptor
                            # quality), does the true match appear within the descriptor's
                            # top-K nearest neighbors? Separates "descriptor has weak signal,
                            # 1-NN is too strict" from "descriptor has no signal at all".
                            target_all = (R_ij_gt @ P_cand_all.T).T + t_ij_gt          # (Ni,3)
                            resid_mat = np.linalg.norm(
                                target_all[:, None, :] - Q_full[None, :, :], axis=-1
                            )                                                          # (Ni,Nj)
                            correct_mat = resid_mat < RANSAC_THRESH                     # (Ni,Nj)
                            avail_mask = correct_mat.any(axis=1)                        # (Ni,)
                            results[strategy]["avail_rate"].append(float(avail_mask.mean()))
                            if avail_mask.any():
                                rank = np.argsort(d_mat, axis=1)                        # (Ni,Nj)
                                for k_top in TOPK_DIAG_LIST:
                                    k_eff = min(k_top, rank.shape[1])
                                    topk_idx = rank[:, :k_eff]
                                    hit = np.take_along_axis(correct_mat, topk_idx, axis=1).any(axis=1)
                                    results[strategy][f"topk_recall_{k_top}"].append(
                                        float(hit[avail_mask].mean())
                                    )

                            pose = ransac_pose(P_cand, Q_cand, rng)
                            if pose is None:
                                results[strategy]["ransac_valid"].append(False)
                                for thresh in POSE_SUCCESS_THRESHOLDS:
                                    results[strategy][f"pose_success_{thresh}"].append(False)
                                continue

                            R_est, t_est, n_inliers = pose
                            rot_err = rotation_error_deg(R_est, R_ij_gt)
                            trans_err = float(np.linalg.norm(t_est - t_ij_gt))

                            # ransac_valid: RANSAC found >=3 inliers (produced *a* pose).
                            # pose_success: that pose is actually close to GT -- distinct,
                            # since 3 inliers can satisfy the residual threshold on a wrong pose.
                            results[strategy]["ransac_valid"].append(True)
                            results[strategy]["rot_err_deg"].append(rot_err)
                            results[strategy]["trans_err"].append(trans_err)
                            results[strategy]["n_candidates"].append(len(P_cand))
                            results[strategy]["inlier_ratio"].append(n_inliers / len(P_cand))
                            for thresh in POSE_SUCCESS_THRESHOLDS:
                                rot_thresh, trans_thresh = thresh
                                success = (rot_err < rot_thresh) and (trans_err < trans_thresh)
                                results[strategy][f"pose_success_{thresh}"].append(success)

    print(f"\nAnalyzed {n_edges} adjacent fragment pairs ({args.categories}/{args.split}).")

    print("\n" + "=" * 100)
    print(f"PHASE 2 — GEOMETRIC BASELINE MATCHING — {args.categories}/{args.split}")
    print("=" * 100)
    pose_success_cols = [f"Pose@{t[0]:g}d_{t[1]:g}" for t in POSE_SUCCESS_THRESHOLDS]
    header = (
        f"  {'Strategy':<12} {'CorrPrec':>10} {'RansacValid':>12} "
        + " ".join(f"{c:>14}" for c in pose_success_cols)
        + f" {'RotErr(deg)':>12} {'TransErr':>10} {'InlierRatio':>12} {'n_edges':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for strategy in mask_strategies:
        d = results[strategy]
        valid = d.get("ransac_valid", [])
        if not valid:
            continue
        ransac_valid_rate = np.mean(valid)
        corr_prec = d.get("correspondence_precision", [])
        rot_errs = d.get("rot_err_deg", [])
        trans_errs = d.get("trans_err", [])
        inlier_ratios = d.get("inlier_ratio", [])
        corr_str = f"{np.mean(corr_prec):>10.2%}" if corr_prec else f"{'n/a':>10}"
        rot_str = f"{np.mean(rot_errs):>12.2f}" if rot_errs else f"{'n/a':>12}"
        trans_str = f"{np.mean(trans_errs):>10.4f}" if trans_errs else f"{'n/a':>10}"
        ir_str = f"{np.mean(inlier_ratios):>12.2%}" if inlier_ratios else f"{'n/a':>12}"
        pose_strs = [
            f"{np.mean(d[f'pose_success_{t}']):>14.2%}" for t in POSE_SUCCESS_THRESHOLDS
        ]
        print(
            f"  {strategy:<12} {corr_str} {ransac_valid_rate:>12.2%} "
            + " ".join(pose_strs)
            + f" {rot_str} {trans_str} {ir_str} {len(valid):>8}"
        )

    # --- Top-K correspondence diagnostic ---
    print("\n" + "=" * 100)
    print(f"TOP-K CORRESPONDENCE DIAGNOSTIC — {args.categories}/{args.split}")
    print("  avail_rate: fraction of i-points with >=1 geometrically correct j-point in the")
    print("  filtered set (isolates the multi-neighbor confound). topk_recall_K (computed only")
    print("  over available points): does the true match appear in the descriptor's top-K?")
    print("=" * 100)
    topk_cols = [f"Top{k}" for k in TOPK_DIAG_LIST]
    header2 = (
        f"  {'Strategy':<12} {'AvailRate':>10} " + " ".join(f"{c:>10}" for c in topk_cols)
    )
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for strategy in mask_strategies:
        d = results[strategy]
        avail = d.get("avail_rate", [])
        if not avail:
            continue
        topk_strs = []
        for k in TOPK_DIAG_LIST:
            vals = d.get(f"topk_recall_{k}", [])
            topk_strs.append(f"{np.mean(vals):>10.2%}" if vals else f"{'n/a':>10}")
        print(f"  {strategy:<12} {np.mean(avail):>10.2%} " + " ".join(topk_strs))


if __name__ == "__main__":
    main()
