"""
scripts/phase5a_weighted_gt_check.py
=====================================
Test de la piste "sampling pondéré par aire" pour la Phase 5A — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7.

Constat du 2026-07-20 (inspection visuelle de l'objet#21,
`phase5a_visualize_pair.py`, puis stratification par MIN(n_i, n_j) dans
`phase5a_depthmap_matching.py`) : le CNN Step15 utilise un échantillonnage
`uniform` (même budget de points par fragment, peu importe sa taille), ce qui
crée un déséquilibre systématique — un grand fragment (le reste de l'objet)
échantillonne très peu de points sur sa fracture, un petit fragment (le bout
cassé) en échantillonne beaucoup. GARF utilise déjà par défaut un
échantillonnage `weighted` (proportionnel à l'aire réelle de chaque fragment,
`assembly/data/breaking_bad/weighted.py`) qui évite ce problème — mais le CNN
ne peut pas l'utiliser (son architecture de backprojection bilinéaire suppose
un nombre de points constant par fragment).

Ce script teste SI ce déséquilibre est bien une cause significative de
l'échec du matching, en comparant directement au run de référence en
`uniform` (N=511, Pose@30=27.59%, Pose@15=13.70%, RotErr=94.95° — Phase 7,
plan). Comme la stratégie "gt" n'a besoin d'AUCUNE prédiction du CNN
(`fracture_surface_gt` est un label géométrique du dataset, pas une sortie du
modèle — confirmé dans `assembly/models/cnn_segmentation_model.py`), on peut
charger le datamodule en `sample_method=weighted` et lire ce label
directement dans le batch, sans jamais charger le CNN.

Réutilise les fonctions de `scripts/phase5a_depthmap_matching.py`
(compute_pca_frame, rasterize, match_depthmaps, build_correspondences,
kabsch...) — même logique de matching, seule la SOURCE des points change.

Usage (sur le serveur — pas de --ckpt, aucun CNN chargé) :
    python scripts/phase5a_weighted_gt_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --score_mode joint \\
        --summary_json /tmp/student7/phase5a_weighted_gt.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import (
    MIN_FRAC_POINTS, MIN_OVERLAP_PIXELS, POSE_SUCCESS_THRESH,
    N_FRAC_PTS_MIN_BINS, N_FRAC_PTS_MIN_LABELS,
    compute_pca_frame, rasterize, match_depthmaps, build_correspondences,
    kabsch, rot_err_deg, trans_err, quat_wxyz_to_rotmat,
)


def stratify_by_min_pts(n_frac_pts_min, rot_err, pose30, pose15):
    """Même découpage que frac_pts_min_stratification (phase5a_depthmap_matching.py),
    sans la colonne OracleOvlp (pas calculée dans ce script simplifié gt-only).
    Sert à comparer directement la courbe Pose@30/Pose@15 vs min(n_i,n_j) entre
    ce run (weighted) et le run de référence (uniform, Phase 7 du plan) --
    comparer deux moyennes globales ne suffit pas à isoler l'effet du
    déséquilibre spécifiquement (cf. remarque utilisateur du 2026-07-20)."""
    n_frac_pts_min = np.asarray(n_frac_pts_min)
    rot_err = np.asarray(rot_err)
    pose30 = np.asarray(pose30)
    pose15 = np.asarray(pose15)
    idx = np.digitize(n_frac_pts_min, N_FRAC_PTS_MIN_BINS[1:-1])
    rows = []
    for b, label in enumerate(N_FRAC_PTS_MIN_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0})
            continue
        rows.append({
            "bin": label, "n": n,
            "rot_err_mean": float(rot_err[mask].mean()),
            "pose_30deg_0.1": 100.0 * float(pose30[mask].mean()),
            "pose_15deg_0.05": 100.0 * float(pose15[mask].mean()),
        })
    return rows


def extract_gt_variable(fracture_surface_gt_row, points_per_part_row):
    """Découpe fracture_surface_gt (1 objet, N_total,) par fragment, en
    utilisant les offsets réels de points_per_part -- PAS extract_gt_for_valid_frags
    (celle-ci suppose un reshape (B,P,num_pts) uniforme, invalide en sampling
    weighted où chaque fragment a une taille différente). Miroir exact du
    "slow path" de extract_fragment_list (projection_mapping_utils.py:61-70)
    pour garantir le même découpage / le même ordre."""
    gt_list = []
    offset = 0
    for size in points_per_part_row:
        size = int(size)
        if size > 0:
            gt_list.append(fracture_surface_gt_row[offset:offset + size])
            offset += size
    return gt_list


def process_pair_gt_only(raw_i, raw_j, gt_i, gt_j, R_ij_gt, t_ij_gt, args):
    """Version simplifiée de process_pair (phase5a_depthmap_matching.py),
    stratégie GT uniquement -- pas de CNN, pas de thresh0.3/random."""
    frac_i = raw_i[gt_i == 1]
    frac_j = raw_j[gt_j == 1]

    if len(frac_i) < MIN_FRAC_POINTS or len(frac_j) < MIN_FRAC_POINTS:
        return {"skip": "too_few_points"}

    c_i, u_i, v_i, n_i, plan_i = compute_pca_frame(frac_i)
    c_j, u_j, v_j, n_j, plan_j = compute_pca_frame(frac_j)
    if plan_i > args.max_planarity or plan_j > args.max_planarity:
        return {"skip": "too_curved"}

    ci = frac_i - c_i
    cj = frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    pixel_size = max(span_i, span_j) * 1.1 / args.resolution

    dmap_i, valid_i, u_min_i, v_min_i = rasterize(
        frac_i, c_i, u_i, v_i, n_i, args.resolution, pixel_size)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(
        frac_j, c_j, u_j, v_j, n_j, args.resolution, pixel_size)

    n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
    if n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS:
        return {"skip": "sparse_dmap"}

    best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
        dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)

    pts_i3, pts_j3 = build_correspondences(
        dmap_i, valid_i, c_i, u_i, v_i, n_i, u_min_i, v_min_i,
        dmap_j, valid_j, c_j, u_j, v_j, n_j, u_min_j, v_min_j,
        pixel_size, best_theta, best_shift, best_flip,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        return {"skip": "no_overlap_3d"}

    R_est, t_est = kabsch(pts_i3, pts_j3)
    re = rot_err_deg(R_est, R_ij_gt)
    te = trans_err(t_est, t_ij_gt)
    return {
        "rot_err": float(re), "trans_err": float(te),
        "pose_success": {f"{int(r)}deg_{t}": bool(re < r and te < t)
                         for r, t in POSE_SUCCESS_THRESH},
        "n_frac_pts": (len(frac_i), len(frac_j)),
        "n_frac_pts_min": min(len(frac_i), len(frac_j)),
        "overlap_frac": float(best_overlap_frac),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True,
                        help="Sert juste à composer la config data (catégories, etc.) — "
                             "aucun modèle n'est chargé, sample_method est forcé à weighted.")
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resolution",  type=int, default=64)
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--max_planarity", type=float, default=0.15)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          f"AUCUN CNN chargé (stratégie gt = label géométrique du dataset)")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    accum = {"rot_err": [], "trans_err": [], "n_corr_ovlp": [], "n_frac_pts_min": []}
    for key in ("30deg_0.1", "15deg_0.05"):
        accum[f"pose_{key}"] = []
    n_skip = {"too_few_points": 0, "too_curved": 0, "sparse_dmap": 0, "no_overlap_3d": 0}
    n_pairs_total = 0
    n_seen_2frag = 0
    t0 = time.time()

    print(f"\nPhase 5A -- vérification sampling pondéré (weighted), stratégie GT uniquement, "
          f"{args.categories}/{args.split}...\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx} | objets 2-frags vus={n_seen_2frag} | "
                      f"paires traitées={n_pairs_total} | {elapsed:.0f}s écoulées")

            points_per_part = batch["points_per_part"][0].numpy()   # (P,)
            pointclouds = batch["pointclouds"][0].numpy()           # (N_total, 3)
            fracture_gt = batch["fracture_surface_gt"][0].numpy()   # (N_total,)

            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            frag_pts = extract_gt_variable(pointclouds, points_per_part)   # réutilise la même logique d'offsets
            frag_gt  = extract_gt_variable(fracture_gt, points_per_part)
            if len(frag_pts) != 2:
                continue   # sécurité, ne devrait pas arriver si valid_slots==2

            p0, p1 = valid_slots[0], valid_slots[1]
            scale_np = batch["scale"][0].numpy()
            if scale_np.ndim == 1:
                scale_np = scale_np[:, None]
            quats_np = batch["quaternions"][0].numpy()
            trans_np = batch["translations"][0].numpy()

            raw_i = frag_pts[0] * scale_np[p0]
            raw_j = frag_pts[1] * scale_np[p1]
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij = R1.T @ R0
            t_ij = R1.T @ (trans_np[p0] - trans_np[p1])

            res = process_pair_gt_only(raw_i, raw_j, gt_i, gt_j, R_ij, t_ij, args)
            n_pairs_total += 1

            if "skip" in res:
                n_skip[res["skip"]] += 1
                continue

            accum["rot_err"].append(res["rot_err"])
            accum["trans_err"].append(res["trans_err"])
            accum["n_frac_pts_min"].append(res["n_frac_pts_min"])
            for k_ps, v_ps in res["pose_success"].items():
                accum[f"pose_{k_ps}"].append(float(v_ps))

    elapsed = time.time() - t0
    n_ok = len(accum["rot_err"])
    print(f"\nFini : {n_pairs_total} paires traitées ({n_seen_2frag} objets 2-frags vus) "
          f"en {elapsed:.0f}s\n")
    print(f"N valides={n_ok}  Skip={dict(n_skip)}")

    if n_ok == 0:
        print("Aucune paire valide -- vérifier la config / le dataset.")
        return

    re_mean = float(np.mean(accum["rot_err"]))
    p30 = 100.0 * float(np.mean(accum["pose_30deg_0.1"]))
    p15 = 100.0 * float(np.mean(accum["pose_15deg_0.05"]))
    npts_min_mean = float(np.mean(accum["n_frac_pts_min"]))

    print(f"\n{'Sampling':<12} {'N':>6} {'RotErr°':>8} {'Pose@30/0.1':>11} "
          f"{'Pose@15/0.05':>12} {'MinPts moyen':>13}")
    print("-" * 66)
    print(f"{'weighted':<12} {n_ok:>6} {re_mean:>8.2f} {p30:>10.2f}% {p15:>11.2f}% "
          f"{npts_min_mean:>13.1f}")
    print(f"{'uniform':<12} {'511':>6} {'94.95':>8} {'27.59':>10}% {'13.70':>11}% "
          f"{'référence Phase 7 (2026-07-20)':>13}")

    # ── Stratification par min(n_i, n_j) -- pour comparer directement la courbe
    # (pas juste une moyenne globale) au run uniform de référence. Référence
    # uniform (Phase 7, 2026-07-20) :
    #   50-100 : Pose@30=28.57% Pose@15= 9.52% (N=63)
    #   100-200: Pose@30=26.75% Pose@15=12.72% (N=228)
    #   200-500: Pose@30=28.43% Pose@15=15.69% (N=204)
    #   500-1000:Pose@30=25.00% Pose@15=18.75% (N=16)
    print(f"\nSTRATIFICATION PAR MIN(N_PTS_I, N_PTS_J) -- weighted, à comparer à "
          f"la courbe uniform ci-dessus (en commentaire dans le script)")
    strat_header = f"{'Bin':<12} {'N':>5} {'RotErr°':>8} {'Pose@30':>9} {'Pose@15':>9}"
    print(strat_header)
    print("-" * len(strat_header))
    rows = stratify_by_min_pts(accum["n_frac_pts_min"], accum["rot_err"],
                               accum["pose_30deg_0.1"], accum["pose_15deg_0.05"])
    for row in rows:
        if row["n"] == 0:
            print(f"{row['bin']:<12} {'—':>5}")
        else:
            print(f"{row['bin']:<12} {row['n']:>5} {row['rot_err_mean']:>8.2f} "
                  f"{row['pose_30deg_0.1']:>8.2f}% {row['pose_15deg_0.05']:>8.2f}%")
    print("(Lecture : si le déséquilibre était bien la cause, le sampling weighted\n"
          " devrait à la fois (a) déplacer la distribution des paires vers des min(n_i,n_j)\n"
          " plus élevés en moyenne, ET (b) faire monter Pose@30/Pose@15 À min(n_i,n_j)\n"
          " ÉGAL par rapport à la courbe uniform -- pas juste une moyenne globale\n"
          " meilleure, qui pourrait être confondue avec d'autres effets du changement\n"
          " de sampling.)")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_pairs": n_pairs_total, "n_ok": n_ok,
                "n_skip": n_skip, "rot_err_mean": re_mean,
                "pose_30deg_0.1": p30, "pose_15deg_0.05": p15,
                "n_frac_pts_min_mean": npts_min_mean,
                "n_frac_pts_min_strata": rows,
            }, f, indent=2)
        print(f"\nJSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
