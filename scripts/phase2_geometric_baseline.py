"""
scripts/phase2_geometric_baseline.py
======================================
Phase 2 du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Pipeline minimal, sans réseau de matching appris :

    masque fracture (GT / CNN seuil / random / all)
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

Deux métriques de succès distinctes (à ne pas confondre, cf. plan) :
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


MASK_STRATEGIES = ["gt", "thresh0.2", "thresh0.3", "thresh0.5", "random", "all"]
RANSAC_ITERS = 500
RANSAC_THRESH = 0.05   # inlier distance, same units as Phase 0's eps
MIN_INLIERS = 3

# pose_success thresholds: (rotation error max in degrees, translation error max).
# A pair "succeeds" only if BOTH are met -- distinct from ransac_valid (>=3 inliers,
# which says RANSAC produced *a* pose, not that it is a *correct* one: 3 inliers can
# satisfy the residual threshold by geometric coincidence on a wrong pose).
POSE_SUCCESS_THRESHOLDS = [(15.0, 0.05), (30.0, 0.1)]


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

                        for strategy in MASK_STRATEGIES:
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
                            Q_cand = raw_per_k[k_j][idx_j_keep][best_j]

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
        f"  {'Strategy':<12} {'RansacValid':>12} "
        + " ".join(f"{c:>14}" for c in pose_success_cols)
        + f" {'RotErr(deg)':>12} {'TransErr':>10} {'InlierRatio':>12} {'n_edges':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for strategy in MASK_STRATEGIES:
        d = results[strategy]
        valid = d.get("ransac_valid", [])
        if not valid:
            continue
        ransac_valid_rate = np.mean(valid)
        rot_errs = d.get("rot_err_deg", [])
        trans_errs = d.get("trans_err", [])
        inlier_ratios = d.get("inlier_ratio", [])
        rot_str = f"{np.mean(rot_errs):>12.2f}" if rot_errs else f"{'n/a':>12}"
        trans_str = f"{np.mean(trans_errs):>10.4f}" if trans_errs else f"{'n/a':>10}"
        ir_str = f"{np.mean(inlier_ratios):>12.2%}" if inlier_ratios else f"{'n/a':>12}"
        pose_strs = [
            f"{np.mean(d[f'pose_success_{t}']):>14.2%}" for t in POSE_SUCCESS_THRESHOLDS
        ]
        print(
            f"  {strategy:<12} {ransac_valid_rate:>12.2%} "
            + " ".join(pose_strs)
            + f" {rot_str} {trans_str} {ir_str} {len(valid):>8}"
        )


if __name__ == "__main__":
    main()
