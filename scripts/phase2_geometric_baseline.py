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

Sanity check oracle (avant d'interpréter tout le reste) : Kabsch simple (sans RANSAC) sur
des correspondances NON pilotées par le descripteur -- le plus proche voisin de chaque
point sous la VRAIE pose GT. Teste si la convention de pose/scale/Kabsch elle-même peut
récupérer R_ij_gt/t_ij_gt, indépendamment du matching. Doit être quasi nul ; si non, le
bug est dans le pipeline de pose, pas dans le matcher.

`--corr_mode` filtre les correspondances 1-NN en espace descripteur : `1nn` (défaut),
`mutual` (mutual nearest neighbor), `ratio<R>` (test de Lowe), `mutual_ratio<R>`.

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

# Object name format: "<category>/<ObjectFamily>/<hash>/fractured_<n>" (e.g.
# "everyday/BeerBottle/6da7fa.../fractured_37"). Crude but cheap symmetry proxy --
# many everyday objects (bottles, bowls, jars...) are axisymmetric, so a fracture ring
# around the main axis may have a genuine rotational ambiguity that no local descriptor
# can resolve, independent of matching algorithm quality. This is a coarse heuristic
# (manual keyword list, not a real symmetry detector) meant to sanity-check that
# hypothesis cheaply by re-aggregating results already computed, not to be precise.
SYMMETRIC_LIKE_KEYWORDS = ["bottle", "bowl", "vase", "jar", "cup", "mug", "plate", "pot"]


def object_family_and_group(name: str):
    parts = name.split("/")
    family = parts[1] if len(parts) > 1 else "unknown"
    group = "symmetric-like" if any(kw in family.lower() for kw in SYMMETRIC_LIKE_KEYWORDS) else "irregular-like"
    return family, group


def record(results, results_by_group, results_by_family, strategy, group, family, key, value):
    results[strategy][key].append(value)
    results_by_group[(strategy, group)][key].append(value)
    results_by_family[(strategy, family)][key].append(value)


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


def _is_dispersed(points: np.ndarray, min_dispersion: float) -> bool:
    """True if the sampled points span at least min_dispersion (max pairwise distance).
    Rejects near-degenerate (clustered/coplanar-ish) samples, which on a locally flat
    fracture patch give an ill-conditioned in-plane rotation estimate (Kabsch SVD has
    little signal to constrain tangential rotation/translation when the sample barely
    spans the patch)."""
    if min_dispersion <= 0:
        return True
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return bool(d.max() >= min_dispersion)


def _hypothesis_score(
    resid: np.ndarray, inlier_mask: np.ndarray, score_mode: str, score_lambda: float,
    tau: float = 0.03, min_inliers_for_score: int = 6, eps: float = 1e-3,
    normal_dot: np.ndarray = None,
):
    """Score a RANSAC hypothesis. 'count' (default/legacy) is the raw inlier count --
    this is what let a wrong but locally-consistent pose (e.g. sliding along a flat
    fracture patch) outscore the true pose (InlierRatio > CorrPrec observed empirically).
    The residual-aware variants reward inlier count but penalize how loose those inliers
    are on average, so a tight-but-smaller consensus can beat a loose-but-larger one.

    min_inliers_for_score gates ALL modes (including 'count'): a hypothesis with fewer
    inliers than this is rejected outright (-inf), regardless of how tight its residuals
    are -- guards the ratio/quality modes against a tiny, accidentally-precise sample
    (e.g. 8 inliers at near-zero residual) outscoring a larger, more representative
    consensus (e.g. 80 inliers at moderate residual) purely by being small and lucky.
    """
    n_in = int(inlier_mask.sum())
    if n_in < min_inliers_for_score:
        return -np.inf, n_in
    mean_resid = float(resid[inlier_mask].mean())
    if score_mode == "count":
        return float(n_in), n_in
    if score_mode == "count_minus_mean_residual":
        # additive penalty -- residuals live in [0, inlier_thresh), typically << the
        # natural scale of inlier-count differences between competing hypotheses (tens of
        # points), so score_lambda has to be calibrated very high (hundreds-thousands) to
        # matter at all (confirmed empirically: lambda=50 had zero measurable effect).
        return n_in - score_lambda * mean_resid, n_in
    if score_mode == "count_minus_median_residual":
        return n_in - score_lambda * float(np.median(resid[inlier_mask])), n_in
    if score_mode == "count_over_residual":
        # Unbounded ratio -- no lambda, but no protection against a tiny ultra-precise
        # sample exploding the score either (mitigated only by min_inliers_for_score).
        return n_in / (mean_resid + 1e-4), n_in
    if score_mode == "count_over_mean_residual":
        # tau-normalized ratio: score = n_in / (mean_residual/tau + eps).
        return n_in / (mean_resid / tau + eps), n_in
    if score_mode == "count_times_quality":
        # score = n_in * (1 - mean_residual/tau), quality clipped to [0,1] so it can't
        # go negative if tau != inlier_thresh (mean_resid is otherwise always < tau when
        # tau == inlier_thresh, since inliers are filtered to resid < inlier_thresh).
        quality = np.clip(1.0 - mean_resid / tau, 0.0, 1.0)
        return n_in * quality, n_in
    if score_mode in ("count_times_normal_quality", "count_times_quality_and_normal"):
        # Phase 2C: distance-only scoring is exhausted (score_gap > 0 for count,
        # count_times_quality AND count_over_mean_residual alike -- a wrong pose
        # genuinely has tighter/larger point-distance consensus than the true one on
        # these fracture surfaces). Normals are an independent signal, validated
        # exploitable via NORMAL ORIENTATION DIAGNOSTIC (MedianDot=-0.825 on
        # oracle-correct correspondences): a genuine contact has opposed normals.
        # normal_quality in [0,1]: 1 = perfectly opposed (dot=-1), 0 = dot>=0 (not opposed).
        if normal_dot is None:
            raise ValueError(f"{score_mode} requires normal_dot")
        normal_quality = float(np.clip(-normal_dot[inlier_mask], 0.0, 1.0).mean())
        if score_mode == "count_times_normal_quality":
            return n_in * normal_quality, n_in
        dist_quality = np.clip(1.0 - mean_resid / tau, 0.0, 1.0)
        return n_in * dist_quality * normal_quality, n_in
    raise ValueError(score_mode)


def ransac_pose(
    P_cand: np.ndarray, Q_cand: np.ndarray, rng: np.random.Generator,
    sample_size: int = 3, min_dispersion: float = 0.0, max_resample_attempts: int = 10,
    score_mode: str = "count", score_lambda: float = 0.0,
    tau: float = 0.03, min_inliers_for_score: int = 6,
    normal_P_cand: np.ndarray = None, normal_Q_cand: np.ndarray = None,
    normal_tau: float = None,
):
    """RANSAC over candidate correspondences. Returns (R, t, n_inliers) or None.

    sample_size > 3 makes each hypothesis an over-determined (least-squares) Kabsch fit
    instead of an exact minimal fit -- less prone to phantom consensus from a single
    near-degenerate (coplanar/clustered) triplet on a locally flat fracture surface.
    min_dispersion additionally rejects samples that don't spatially spread across the
    patch (resampled up to max_resample_attempts times per iteration).

    normal_P_cand/normal_Q_cand (Phase 2C): per-candidate normals, 1:1 aligned with
    P_cand/Q_cand. If normal_tau is set, additionally requires
    dot(R @ n_i, n_j) < normal_tau for a point to count as inlier (hard filter) -- on
    top of the distance threshold, not instead of it. If normal_tau is None, normals are
    still computed and passed to _hypothesis_score for the soft normal-aware score modes
    (count_times_normal_quality / count_times_quality_and_normal), without filtering.
    """
    n = len(P_cand)
    if n < sample_size:
        return None
    use_normals = normal_P_cand is not None

    best_score, best_inliers, best_mask = -np.inf, -1, None
    for _ in range(RANSAC_ITERS):
        for _attempt in range(max_resample_attempts):
            sample = rng.choice(n, size=sample_size, replace=False)
            if _is_dispersed(P_cand[sample], min_dispersion):
                break
        else:
            continue  # no dispersed sample found within budget, skip this iteration
        try:
            Rmat, t = kabsch(P_cand[sample], Q_cand[sample])
        except np.linalg.LinAlgError:
            continue
        pred = (Rmat @ P_cand.T).T + t
        resid = np.linalg.norm(pred - Q_cand, axis=1)
        dist_mask = resid < RANSAC_THRESH

        normal_dot = None
        if use_normals:
            rotated_normal = (Rmat @ normal_P_cand.T).T
            normal_dot = (rotated_normal * normal_Q_cand).sum(axis=1)

        if normal_tau is not None and normal_dot is not None:
            inlier_mask = dist_mask & (normal_dot < normal_tau)
        else:
            inlier_mask = dist_mask

        score, n_in = _hypothesis_score(
            resid, inlier_mask, score_mode, score_lambda,
            tau=tau, min_inliers_for_score=min_inliers_for_score, normal_dot=normal_dot,
        )
        if score > best_score:
            best_score, best_inliers, best_mask = score, n_in, inlier_mask

    if best_mask is None or best_inliers < MIN_INLIERS:
        return None

    Rmat, t = kabsch(P_cand[best_mask], Q_cand[best_mask])
    return Rmat, t, best_inliers


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos_angle = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def build_correspondences(d_mat: np.ndarray, mode: str):
    """Filter 1-NN descriptor correspondences by mode. Returns (i_idx, j_idx) arrays of
    equal length -- a subset of i's local indices (into the Ni rows of d_mat) paired
    with their chosen j match, NOT necessarily covering all Ni rows (mutual-NN/ratio
    test reject ambiguous ones instead of forcing a match for every point).

    Modes:
      1nn               -- baseline, every i-point keeps its nearest j (no filtering).
      mutual            -- keep (i,j) only if j is also i's nearest among j's matched
                            TO it (mutual nearest neighbor) -- standard correspondence
                            pruning, rejects asymmetric "forced" matches.
      ratio<R>          -- Lowe's ratio test: keep i only if d(1st nearest)/d(2nd
                            nearest) < R. Low ratio means the match is unambiguous
                            relative to the next-best candidate.
      mutual_ratio<R>   -- intersection of both filters.
    """
    Ni, Nj = d_mat.shape
    best_j = d_mat.argmin(axis=1)

    if mode == "1nn":
        keep = np.ones(Ni, dtype=bool)
    elif mode == "mutual":
        best_i = d_mat.argmin(axis=0)
        keep = best_i[best_j] == np.arange(Ni)
    elif mode.startswith("mutual_ratio"):
        ratio_thresh = float(mode.replace("mutual_ratio", ""))
        best_i = d_mat.argmin(axis=0)
        mutual_keep = best_i[best_j] == np.arange(Ni)
        keep = mutual_keep & _ratio_test_mask(d_mat, ratio_thresh)
    elif mode.startswith("ratio"):
        ratio_thresh = float(mode.replace("ratio", ""))
        keep = _ratio_test_mask(d_mat, ratio_thresh)
    else:
        raise ValueError(mode)

    i_idx = np.where(keep)[0]
    j_idx = best_j[i_idx]
    return i_idx, j_idx


def _ratio_test_mask(d_mat: np.ndarray, ratio_thresh: float) -> np.ndarray:
    Ni, Nj = d_mat.shape
    if Nj < 2:
        return np.ones(Ni, dtype=bool)  # no second-best to compare against
    sorted_d = np.sort(d_mat, axis=1)
    d1, d2 = sorted_d[:, 0], sorted_d[:, 1]
    return (d1 / np.maximum(d2, 1e-12)) < ratio_thresh


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
    parser.add_argument(
        "--ransac_sample_size", type=int, default=3,
        help="RANSAC minimal-sample size for Kabsch (default 3, the rigid-transform "
             "minimum). >3 gives an over-determined least-squares fit per hypothesis, "
             "less prone to a single near-degenerate/coplanar sample dominating.",
    )
    parser.add_argument(
        "--ransac_min_dispersion", type=float, default=0.0,
        help="Reject RANSAC samples whose points don't spread at least this much "
             "(max pairwise distance) -- guards against near-degenerate samples on a "
             "locally flat fracture patch. 0 = disabled (default).",
    )
    parser.add_argument(
        "--corr_mode", default="1nn",
        help="Correspondence filtering mode: 1nn (baseline, default), mutual "
             "(mutual nearest neighbor), ratio<R> (Lowe's ratio test, e.g. ratio0.8), "
             "mutual_ratio<R> (both combined). See build_correspondences().",
    )
    parser.add_argument(
        "--inlier_thresh", type=float, default=0.05,
        help="Inlier distance threshold (default 0.05, was previously a fixed constant). "
             "Used consistently everywhere a 'correct' correspondence/inlier is decided: "
             "RANSAC consensus, CorrPrec/avail_rate/oracle diagnostics. Sweeping this "
             "tests whether 0.05 is too permissive and lets wrong poses pass as 'inlier'.",
    )
    parser.add_argument(
        "--score_mode", default="count",
        choices=[
            "count", "count_minus_mean_residual", "count_minus_median_residual",
            "count_over_residual", "count_over_mean_residual", "count_times_quality",
            "count_times_normal_quality", "count_times_quality_and_normal",
        ],
        help="RANSAC hypothesis scoring. 'count' (default/legacy) = raw inlier count, "
             "which let a loose-but-larger wrong consensus beat a tight-but-smaller "
             "correct one (InlierRatio > CorrPrec observed). count_minus_*_residual "
             "subtract score_lambda * residual (needs careful calibration -- residuals "
             "are tiny relative to count differences, score_lambda=50 had zero effect "
             "empirically). count_over_residual = n_in / mean_residual, a ratio score "
             "with no lambda to calibrate. Distance-only modes (count/quality/ratio) all "
             "empirically gave score_gap > 0 (RANSAC's chosen pose outscores the true GT "
             "pose under the SAME criterion) -- confirmed distance-only is exhausted. "
             "count_times_normal_quality / count_times_quality_and_normal (Phase 2C) add "
             "an independent signal: normal orientation (validated exploitable, "
             "MedianDot=-0.825 on oracle-correct correspondences -- see --normal_tau).",
    )
    parser.add_argument(
        "--score_lambda", type=float, default=50.0,
        help="Residual penalty weight for --score_mode in {count_minus_mean_residual, "
             "count_minus_median_residual} (default 50.0 -- residuals are in "
             "[0, inlier_thresh), so this needs to be calibrated much higher, e.g. "
             "hundreds-thousands, to compete with inlier-count differences of several "
             "dozen points; lambda=50 had zero measurable effect empirically).",
    )
    parser.add_argument(
        "--score_tau", type=float, default=None,
        help="Residual scale for --score_mode in {count_over_mean_residual, "
             "count_times_quality} (default: same value as --inlier_thresh).",
    )
    parser.add_argument(
        "--min_inliers_for_score", type=int, default=None,
        help="Reject any RANSAC hypothesis (all score modes, including 'count') with "
             "fewer inliers than this -- guards the ratio/quality score modes against a "
             "tiny, accidentally-precise sample (e.g. 8 inliers at ~0 residual) "
             "outscoring a larger, more representative consensus just by being small "
             "and lucky. Default: max(6, --ransac_sample_size).",
    )
    parser.add_argument(
        "--normal_tau", type=float, default=None,
        help="Phase 2C hard filter: in addition to the distance threshold, require "
             "dot(R @ n_i, n_j) < normal_tau for a point to count as inlier (e.g. -0.3, "
             "-0.5, -0.7 -- start permissive). Default None = no hard normal filter "
             "(normals are still used by the soft score modes if selected).",
    )
    args = parser.parse_args()
    score_tau = args.score_tau if args.score_tau is not None else args.inlier_thresh
    min_inliers_for_score = (
        args.min_inliers_for_score if args.min_inliers_for_score is not None
        else max(6, args.ransac_sample_size)
    )
    mask_strategies = args.strategies.split(",") if args.strategies else ALL_MASK_STRATEGIES

    global RANSAC_THRESH
    RANSAC_THRESH = args.inlier_thresh

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

    # val/test_dataloader() in module.py does NOT shuffle -- the HDF5 object list is
    # apparently alphabetically ordered, so with a capped --max_batches the first N
    # batches can be a single object family (observed: 634/634 edges were Bottle-type
    # objects with --max_batches 60, making the symmetric-vs-irregular comparison
    # impossible -- zero irregular examples). Shuffle here so a capped run still sees a
    # diverse mix of object families.
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

    # results[strategy] -> dict of lists. Same metrics also re-aggregated by
    # (strategy, symmetric-like/irregular-like) and (strategy, object_family) to test
    # whether axisymmetric objects (bottles/bowls/...) drive the rotation failure.
    results = defaultdict(lambda: defaultdict(list))
    results_by_group = defaultdict(lambda: defaultdict(list))
    results_by_family = defaultdict(lambda: defaultdict(list))

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
            names = batch["name"]  # list of str, length B

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
                family, group = object_family_and_group(names[b])
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
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "ransac_valid", False)
                                for thresh in POSE_SUCCESS_THRESHOLDS:
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, f"pose_success_{thresh}", False)
                                continue

                            # Correspondences in descriptor space, i -> j, filtered by
                            # --corr_mode (1nn baseline, or mutual-NN / ratio test to
                            # trade fewer candidates for higher precision).
                            desc_i = desc_per_k[k_i][idx_i_keep]      # (Ni, D)
                            desc_j = desc_per_k[k_j][idx_j_keep]      # (Nj, D)
                            d_mat = np.linalg.norm(
                                desc_i[:, None, :] - desc_j[None, :, :], axis=-1
                            )
                            i_idx_corr, j_idx_corr = build_correspondences(d_mat, args.corr_mode)

                            P_cand_all = raw_per_k[k_i][idx_i_keep]   # (Ni,3) -- ALL filtered i points
                            Q_full = raw_per_k[k_j][idx_j_keep]       # (Nj,3) -- ALL filtered j points
                            P_cand = P_cand_all[i_idx_corr]           # (Nc,3) -- after corr_mode filtering
                            Q_cand = Q_full[j_idx_corr]

                            # Normals aligned 1:1 with P_cand/Q_cand (Phase 2C) -- validated
                            # exploitable via the NORMAL ORIENTATION DIAGNOSTIC (MedianDot=-0.825,
                            # %dot<-0.5=78% on oracle-correct correspondences): genuine contact
                            # points have consistently opposed normals, not just close positions.
                            normal_i_full = normals_per_k[k_i][idx_i_keep]
                            normal_j_full = normals_per_k[k_j][idx_j_keep]
                            normal_P_cand = normal_i_full[i_idx_corr]
                            normal_Q_cand = normal_j_full[j_idx_corr]

                            # Diagnostic 1: is the 1-NN candidate set itself usable at all?
                            # A candidate is "correct" if it's geometrically consistent with
                            # the TRUE pose (independent of whether RANSAC/Kabsch can recover
                            # that pose from the candidate set). Distinguishes "descriptor too
                            # weak" from "no true correspondence exists among candidates"
                            # (e.g. fragment touches >1 neighbor, mask mixes multiple interfaces).
                            pred_under_gt = (R_ij_gt @ P_cand.T).T + t_ij_gt
                            resid_under_gt = np.linalg.norm(pred_under_gt - Q_cand, axis=1)
                            corr_precision = float((resid_under_gt < RANSAC_THRESH).mean())
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "correspondence_precision", corr_precision)
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "n_candidates_diag", len(P_cand))

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
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "avail_rate", float(avail_mask.mean()))
                            if avail_mask.any():
                                rank = np.argsort(d_mat, axis=1)                        # (Ni,Nj)
                                for k_top in TOPK_DIAG_LIST:
                                    k_eff = min(k_top, rank.shape[1])
                                    topk_idx = rank[:, :k_eff]
                                    hit = np.take_along_axis(correct_mat, topk_idx, axis=1).any(axis=1)
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, f"topk_recall_{k_top}",
                                           float(hit[avail_mask].mean()))

                            # Sanity check (A): oracle Kabsch on GT-pose-nearest correspondences
                            # (NOT descriptor-driven -- for every available i-point, its true
                            # nearest j-point under the ACTUAL GT pose). Tests whether plain
                            # Kabsch can recover R_ij_gt/t_ij_gt at all, independent of RANSAC
                            # and independent of whether the descriptor can find these
                            # correspondences itself. If this fails, something is wrong in the
                            # pose convention/scale/direction (Phase 0 said no, but cheap to
                            # re-verify here on the actual edge data) -- not in the matcher.
                            if avail_mask.sum() >= MIN_INLIERS:
                                oracle_best_j = resid_mat.argmin(axis=1)              # (Ni,)
                                P_oracle = P_cand_all[avail_mask]
                                Q_oracle = Q_full[oracle_best_j[avail_mask]]
                                try:
                                    R_oracle, t_oracle = kabsch(P_oracle, Q_oracle)
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, "oracle_rot_err_deg",
                                           rotation_error_deg(R_oracle, R_ij_gt))
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, "oracle_trans_err",
                                           float(np.linalg.norm(t_oracle - t_ij_gt)))
                                except np.linalg.LinAlgError:
                                    pass

                                # Normal-orientation diagnostic (Phase 2C, step 1 -- NOT a
                                # filter yet). For these oracle-correct correspondences
                                # (geometrically correct under the TRUE pose, NOT descriptor-
                                # driven), check whether normals are consistently opposed:
                                # dot(R_ij_gt @ n_i, n_j) approx -1 for a genuine contact.
                                # Decides whether normals carry exploitable signal before
                                # using them as a hard filter or a soft score term.
                                normal_i_oracle = normals_per_k[k_i][idx_i_keep][avail_mask]
                                normal_j_oracle = normals_per_k[k_j][idx_j_keep][oracle_best_j[avail_mask]]
                                rotated_normal_i = (R_ij_gt @ normal_i_oracle.T).T
                                normal_dot = (rotated_normal_i * normal_j_oracle).sum(axis=1)
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "normal_dot_mean", float(normal_dot.mean()))
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "normal_dot_median", float(np.median(normal_dot)))
                                for normal_thresh in (-0.3, -0.5, -0.7):
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, f"pct_normal_dot_below_{normal_thresh}",
                                           float((normal_dot < normal_thresh).mean()))

                            pose = ransac_pose(
                                P_cand, Q_cand, rng,
                                sample_size=args.ransac_sample_size,
                                min_dispersion=args.ransac_min_dispersion,
                                score_mode=args.score_mode,
                                score_lambda=args.score_lambda,
                                tau=score_tau,
                                min_inliers_for_score=min_inliers_for_score,
                                normal_P_cand=normal_P_cand,
                                normal_Q_cand=normal_Q_cand,
                                normal_tau=args.normal_tau,
                            )
                            if pose is None:
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "ransac_valid", False)
                                for thresh in POSE_SUCCESS_THRESHOLDS:
                                    record(results, results_by_group, results_by_family,
                                           strategy, group, family, f"pose_success_{thresh}", False)
                                continue

                            R_est, t_est, n_inliers = pose
                            rot_err = rotation_error_deg(R_est, R_ij_gt)
                            trans_err = float(np.linalg.norm(t_est - t_ij_gt))

                            # score_GT_pose vs score_RANSAC_pose: does the CURRENT scoring
                            # function (whichever --score_mode) actually rank the true pose
                            # above the one RANSAC picked? If score_gap > 0 (RANSAC's pose
                            # scores higher), the scoring criterion is still the bottleneck
                            # regardless of sampling/iterations. If score_gap <= 0 but
                            # pose_success stays low, RANSAC's exploration (not its scoring)
                            # is failing to find the hypothesis its own criterion prefers.
                            resid_at_gt = np.linalg.norm(
                                (R_ij_gt @ P_cand.T).T + t_ij_gt - Q_cand, axis=1
                            )
                            normal_dot_at_gt = (
                                (R_ij_gt @ normal_P_cand.T).T * normal_Q_cand
                            ).sum(axis=1)
                            inlier_at_gt = (
                                (resid_at_gt < RANSAC_THRESH) & (normal_dot_at_gt < args.normal_tau)
                                if args.normal_tau is not None else resid_at_gt < RANSAC_THRESH
                            )
                            score_gt, _ = _hypothesis_score(
                                resid_at_gt, inlier_at_gt,
                                args.score_mode, args.score_lambda,
                                tau=score_tau, min_inliers_for_score=min_inliers_for_score,
                                normal_dot=normal_dot_at_gt,
                            )
                            resid_at_ransac = np.linalg.norm(
                                (R_est @ P_cand.T).T + t_est - Q_cand, axis=1
                            )
                            normal_dot_at_ransac = (
                                (R_est @ normal_P_cand.T).T * normal_Q_cand
                            ).sum(axis=1)
                            inlier_at_ransac = (
                                (resid_at_ransac < RANSAC_THRESH) & (normal_dot_at_ransac < args.normal_tau)
                                if args.normal_tau is not None else resid_at_ransac < RANSAC_THRESH
                            )
                            score_ransac, _ = _hypothesis_score(
                                resid_at_ransac, inlier_at_ransac,
                                args.score_mode, args.score_lambda,
                                tau=score_tau, min_inliers_for_score=min_inliers_for_score,
                                normal_dot=normal_dot_at_ransac,
                            )
                            if np.isfinite(score_gt) and np.isfinite(score_ransac):
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "score_gt_pose", score_gt)
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "score_ransac_pose", score_ransac)
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "score_gap", score_ransac - score_gt)

                            # NormalDot mean over RANSAC's actual final inliers (distance
                            # criterion only, regardless of --normal_tau) -- diagnostic: are
                            # the inliers RANSAC settled on plausibly-opposed contact points,
                            # or just close-by points with arbitrary orientation?
                            dist_inliers_at_ransac = resid_at_ransac < RANSAC_THRESH
                            if dist_inliers_at_ransac.any():
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, "ransac_inlier_normal_dot_mean",
                                       float(normal_dot_at_ransac[dist_inliers_at_ransac].mean()))

                            # ransac_valid: RANSAC found >=3 inliers (produced *a* pose).
                            # pose_success: that pose is actually close to GT -- distinct,
                            # since 3 inliers can satisfy the residual threshold on a wrong pose.
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "ransac_valid", True)
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "rot_err_deg", rot_err)
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "trans_err", trans_err)
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "n_candidates", len(P_cand))
                            record(results, results_by_group, results_by_family,
                                   strategy, group, family, "inlier_ratio", n_inliers / len(P_cand))
                            for thresh in POSE_SUCCESS_THRESHOLDS:
                                rot_thresh, trans_thresh = thresh
                                success = (rot_err < rot_thresh) and (trans_err < trans_thresh)
                                record(results, results_by_group, results_by_family,
                                       strategy, group, family, f"pose_success_{thresh}", success)

    print(f"\nAnalyzed {n_edges} adjacent fragment pairs ({args.categories}/{args.split}).")

    print("\n" + "=" * 100)
    print(f"ORACLE KABSCH SANITY CHECK — {args.categories}/{args.split}")
    print("  Plain Kabsch (no RANSAC) on GT-pose-nearest correspondences (not descriptor-")
    print("  driven). Tests pose math/convention/scale independent of the matcher. Should")
    print("  be near-zero error -- if not, the bug is in the pose pipeline, not matching.")
    print("=" * 100)
    print(f"  {'Strategy':<12} {'OracleRotErr':>13} {'OracleTransErr':>15} {'n':>6}")
    for strategy in mask_strategies:
        d = results[strategy]
        oracle_rot = d.get("oracle_rot_err_deg", [])
        oracle_trans = d.get("oracle_trans_err", [])
        if not oracle_rot:
            continue
        print(
            f"  {strategy:<12} {np.mean(oracle_rot):>13.4f} "
            f"{np.mean(oracle_trans):>15.6f} {len(oracle_rot):>6}"
        )

    print("\n" + "=" * 100)
    print(f"PHASE 2 — GEOMETRIC BASELINE MATCHING — {args.categories}/{args.split}")
    print(f"  (corr_mode={args.corr_mode}, RANSAC sample_size={args.ransac_sample_size}, "
          f"min_dispersion={args.ransac_min_dispersion}, inlier_thresh={args.inlier_thresh}, "
          f"score_mode={args.score_mode}, score_lambda={args.score_lambda}, "
          f"score_tau={score_tau}, min_inliers_for_score={min_inliers_for_score})")
    print("  CorrPrec = inlier ratio of the candidate set under the TRUE GT pose.")
    print("  InlierRatio = inlier ratio under the pose RANSAC actually picked.")
    print("  InlierRatio > CorrPrec means RANSAC's scoring objectively prefers a WRONG")
    print("  pose over the true one -- a sampling fix alone won't help, the inlier")
    print("  threshold/scoring criterion itself needs to change.")
    print("=" * 100)
    pose_success_cols = [f"Pose@{t[0]:g}d_{t[1]:g}" for t in POSE_SUCCESS_THRESHOLDS]
    header = (
        f"  {'Strategy':<12} {'CorrPrec':>10} {'RansacValid':>12} "
        + " ".join(f"{c:>14}" for c in pose_success_cols)
        + f" {'RotErr(deg)':>12} {'TransErr':>10} {'InlierRatio':>12} {'NormalDot':>10} {'n_edges':>8}"
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
        normal_dots = d.get("ransac_inlier_normal_dot_mean", [])
        corr_str = f"{np.mean(corr_prec):>10.2%}" if corr_prec else f"{'n/a':>10}"
        rot_str = f"{np.mean(rot_errs):>12.2f}" if rot_errs else f"{'n/a':>12}"
        trans_str = f"{np.mean(trans_errs):>10.4f}" if trans_errs else f"{'n/a':>10}"
        ir_str = f"{np.mean(inlier_ratios):>12.2%}" if inlier_ratios else f"{'n/a':>12}"
        nd_str = f"{np.mean(normal_dots):>10.3f}" if normal_dots else f"{'n/a':>10}"
        pose_strs = [
            f"{np.mean(d[f'pose_success_{t}']):>14.2%}" for t in POSE_SUCCESS_THRESHOLDS
        ]
        print(
            f"  {strategy:<12} {corr_str} {ransac_valid_rate:>12.2%} "
            + " ".join(pose_strs)
            + f" {rot_str} {trans_str} {ir_str} {nd_str} {len(valid):>8}"
        )

    # --- score_GT_pose vs score_RANSAC_pose: does the CURRENT score_mode prefer the
    # true pose over the one RANSAC picked? score_gap > 0 = scoring criterion is still
    # the bottleneck (regardless of sampling/iterations); score_gap <= 0 but pose_success
    # still low = RANSAC's exploration is failing to find what its own criterion prefers.
    print("\n" + "=" * 100)
    print(f"SCORE_GT_POSE vs SCORE_RANSAC_POSE — {args.categories}/{args.split} (score_mode={args.score_mode})")
    print("  score_gap = score(RANSAC's chosen pose) - score(true GT pose), same scoring")
    print("  function/threshold for both. >0 means the criterion still prefers a wrong pose.")
    print("=" * 100)
    print(f"  {'Strategy':<12} {'ScoreGT':>14} {'ScoreRANSAC':>14} {'ScoreGap':>14} {'n':>6}")
    for strategy in mask_strategies:
        d = results[strategy]
        sgt = d.get("score_gt_pose", [])
        sransac = d.get("score_ransac_pose", [])
        sgap = d.get("score_gap", [])
        if not sgt:
            continue
        print(
            f"  {strategy:<12} {np.mean(sgt):>14.3f} {np.mean(sransac):>14.3f} "
            f"{np.mean(sgap):>+14.3f} {len(sgt):>6}"
        )

    # --- Normal-orientation diagnostic (Phase 2C step 1, NOT a filter yet) ---
    print("\n" + "=" * 100)
    print(f"NORMAL ORIENTATION DIAGNOSTIC — {args.categories}/{args.split}")
    print("  dot(R_ij_gt @ n_i, n_j) on oracle-correct correspondences (GT pose, not")
    print("  descriptor-driven). Genuine contact should give dot ~ -1 (opposed normals).")
    print("  Median strongly negative + many dot<-0.5 => normals exploitable as opposed.")
    print("  |dot| near 1 but sign unstable => normals not consistently oriented, use")
    print("  |dot| instead of a strict opposition test. No structure => too noisy, don't")
    print("  use as a hard filter.")
    print("=" * 100)
    print(f"  {'Strategy':<12} {'MeanDot':>9} {'MedianDot':>10} {'%<-0.3':>8} {'%<-0.5':>8} {'%<-0.7':>8} {'n':>6}")
    for strategy in mask_strategies:
        d = results[strategy]
        nd_mean = d.get("normal_dot_mean", [])
        if not nd_mean:
            continue
        nd_median = d.get("normal_dot_median", [])
        pct_03 = d.get("pct_normal_dot_below_-0.3", [])
        pct_05 = d.get("pct_normal_dot_below_-0.5", [])
        pct_07 = d.get("pct_normal_dot_below_-0.7", [])
        print(
            f"  {strategy:<12} {np.mean(nd_mean):>9.3f} {np.mean(nd_median):>10.3f} "
            f"{np.mean(pct_03):>8.2%} {np.mean(pct_05):>8.2%} {np.mean(pct_07):>8.2%} {len(nd_mean):>6}"
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

    # --- Symmetric-like vs irregular-like, and per object-family breakdown ---
    # Tests whether axisymmetric objects (bottle/bowl/vase/jar/cup/mug/plate/pot --
    # crude keyword heuristic, cf. SYMMETRIC_LIKE_KEYWORDS) drive the rotation failure:
    # a fracture ring around a revolution axis can be genuinely ambiguous to local
    # descriptors, independent of matching algorithm quality. Gap = InlierRatio -
    # CorrPrec, i.e. how much RANSAC's scoring prefers a wrong pose over the true one.
    def print_grouped_table(title: str, grouped: dict, group_keys):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        gheader = (
            f"  {'Group':<16} {'n_edges':>8} {'CorrPrec':>10} {'InlierRatio':>12} "
            f"{'Gap':>8} {'Pose@30d_0.1':>13} {'RotErr(deg)':>12} {'TransErr':>10}"
        )
        print(gheader)
        print("  " + "-" * (len(gheader) - 2))
        for gkey in group_keys:
            d = grouped.get(gkey)
            if d is None or not d.get("ransac_valid"):
                continue
            corr_prec = d.get("correspondence_precision", [])
            inlier_ratios = d.get("inlier_ratio", [])
            rot_errs = d.get("rot_err_deg", [])
            trans_errs = d.get("trans_err", [])
            pose30 = d.get(f"pose_success_{POSE_SUCCESS_THRESHOLDS[1]}", [])
            corr_m = np.mean(corr_prec) if corr_prec else float("nan")
            ir_m = np.mean(inlier_ratios) if inlier_ratios else float("nan")
            gap = ir_m - corr_m
            print(
                f"  {gkey[1]:<16} {len(d['ransac_valid']):>8} {corr_m:>10.2%} {ir_m:>12.2%} "
                f"{gap:>+8.2%} {np.mean(pose30) if pose30 else float('nan'):>13.2%} "
                f"{np.mean(rot_errs) if rot_errs else float('nan'):>12.2f} "
                f"{np.mean(trans_errs) if trans_errs else float('nan'):>10.4f}"
            )

    for strategy in mask_strategies:
        group_keys = sorted(
            {k for k in results_by_group if k[0] == strategy},
            key=lambda k: k[1],
        )
        if group_keys:
            print_grouped_table(
                f"SYMMETRIC-LIKE vs IRREGULAR-LIKE — strategy={strategy} — {args.categories}/{args.split}",
                results_by_group, group_keys,
            )
        family_keys = sorted(
            {k for k in results_by_family if k[0] == strategy},
            key=lambda k: -len(results_by_family[k].get("ransac_valid", [])),
        )
        if family_keys:
            print_grouped_table(
                f"PER OBJECT FAMILY — strategy={strategy} — {args.categories}/{args.split}",
                results_by_family, family_keys,
            )


if __name__ == "__main__":
    main()
