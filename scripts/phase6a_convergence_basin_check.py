"""
scripts/phase6a_convergence_basin_check.py
=============================================
Test de bassin de convergence du raffinement point-à-point — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7.

Question posée (2026-07-22) : l'architecture retenue pour la suite du pipeline
est en DEUX ÉTAGES -- (1) depth-map matching pour une estimation de pose
GROSSIÈRE, (2) raffinement point-à-point (type ICP) seulement ensuite, une
fois les fragments à peu près alignés (le point-à-point direct avait déjà
échoué en Phase 3A/4A/4B faute de bonne initialisation). Mais quelle précision
l'étage 1 doit-il vraiment atteindre pour que l'étage 2 fonctionne ? On a
optimisé `Pose@30°/0.1` toute la journée sans savoir si 30° est le bon seuil,
ou si le raffinement tolère bien plus (45°, 60°) -- ou bien moins.

Protocole (oracle, indépendant du depth-map matching lui-même) :
  1. Prend la VRAIE pose relative GT (R_ij_gt, t_ij_gt, formule Phase 0) entre
     deux fragments d'un objet à 2 fragments.
  2. La PERTURBE : rotation aléatoire d'amplitude θ (axe aléatoire) composée
     avec R_ij_gt, + bruit de translation gaussien -- simule ce qu'un
     depth-map matching imparfait produirait comme pose de départ.
  3. Lance un ICP point-à-point (nearest-neighbor + Kabsch itéré, avec
     rognage des correspondances les plus éloignées à chaque itération --
     ICP "trimmed", nécessaire car les deux nuages de points de fracture ne
     se correspondent pas parfaitement point-à-point) à partir de cette pose
     perturbée.
  4. Mesure si l'ICP reconverge près de la VRAIE pose (erreur finale < seuil
     strict) -- taux de succès par amplitude de perturbation θ testée.

Restreint aux paires `n_frac_pts_min >= --min_frac_pts` (défaut 50) : la
Phase 7 a identifié une population structurellement différente à densité
CNN < 50 points (2026-07-22, ~31% du dataset, échoue même en GT sans zoom) --
ce test caractérise le raffinement pour la population "facile" (~69%) sur
laquelle on continue le pipeline actuel, pas pour cette tranche mise de côté.

Réutilise `kabsch`/`rot_err_deg`/`trans_err`/`quat_wxyz_to_rotmat` de
`phase5a_depthmap_matching.py` et `extract_gt_variable` de
`phase5a_weighted_gt_check.py` -- aucune réimplémentation. Stratégie GT
uniquement, aucun CNN chargé (même discipline oracle-first que le reste de
Phase 7) : ce test caractérise le RAFFINEMENT lui-même, indépendamment de la
qualité du masque qui l'alimentera en pratique.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase6a_convergence_basin_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --num_points_to_sample 10000 --min_frac_pts 50 \\
        --rotation_perturbations 10 20 30 45 60 90 \\
        --csv_out /tmp/student7/phase6a_convergence_basin.csv \\
        --summary_json /tmp/student7/phase6a_convergence_basin.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import (
    kabsch, rot_err_deg, trans_err, quat_wxyz_to_rotmat,
)

ABS_MIN_POINTS = 5


def random_rotation_of_magnitude(theta_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotation aléatoire d'amplitude EXACTE theta_deg, autour d'un axe uniforme
    sur la sphère. Retourne la matrice 3x3."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return R_scipy.from_rotvec(axis * np.radians(theta_deg)).as_matrix()


def trimmed_icp(pts_i, pts_j, R_init, t_init, max_iters=50, tol=1e-7, trim_ratio=0.7):
    """ICP point-à-point classique (nearest-neighbor + Kabsch itéré), avec
    rognage des correspondances les plus éloignées à chaque itération (ne
    garde que les `trim_ratio` les plus proches avant de refaire Kabsch) --
    nécessaire car les points fracture des deux fragments ne se correspondent
    jamais parfaitement un-à-un (échantillonnage indépendant de chaque côté,
    cf. plafond `oracle_overlap_frac`=0.758 mesuré le 2026-07-20).

    Retourne (R_final, t_final, converged_early: bool).
    """
    R, t = R_init.copy(), t_init.copy()
    tree = cKDTree(pts_j)
    n_keep = max(3, int(round(trim_ratio * len(pts_i))))

    for _ in range(max_iters):
        pts_i_t = (R @ pts_i.T).T + t
        dists, idx = tree.query(pts_i_t)
        keep = np.argsort(dists)[:n_keep]
        corr_i = pts_i[keep]
        corr_j = pts_j[idx[keep]]

        R_new, t_new = kabsch(corr_i, corr_j)
        change = np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t)
        R, t = R_new, t_new
        if change < tol:
            return R, t, True
    return R, t, False


def trimmed_icp_normals(pts_i, nrm_i, pts_j, nrm_j, R_init, t_init,
                         max_iters=50, tol=1e-7, trim_ratio=0.7, normal_dot_thresh=-0.3):
    """ICP avec PÉNALITÉ D'ORIENTATION (2026-07-22, suite au bassin de
    convergence étroit trouvé avec l'ICP vanille -- probablement le glissement
    dans le plan de contact sur des surfaces quasi-planes, Phase 5A.0 :
    planéité médiane 0.043, RotErr médian/moyen très écartés = signature
    bimodale classique de mauvais minimum local en ICP).

    PAS un "point-to-plane" au sens strict (minimiser la distance selon la
    normale) -- ça n'offrirait justement AUCUNE résistance au glissement
    tangentiel, c'est sa faiblesse connue sur surface plate. Ici : à chaque
    itération, après avoir trouvé les plus proches voisins, REJETTE les
    correspondances dont les normales (après rotation courante) ne sont PAS
    à peu près opposées (`dot(R@n_i, n_j) < normal_dot_thresh`) AVANT de
    refaire Kabsch sur les survivantes -- même principe que le scoring
    normal-aware qui avait fait la différence en Phase 2C
    (`count_times_quality_and_normal`, Pose@30 5%->9.6% à l'époque).

    Retourne (R_final, t_final, converged_early: bool).
    """
    R, t = R_init.copy(), t_init.copy()
    tree = cKDTree(pts_j)
    n_keep = max(3, int(round(trim_ratio * len(pts_i))))

    for _ in range(max_iters):
        pts_i_t = (R @ pts_i.T).T + t
        nrm_i_t = (R @ nrm_i.T).T
        dists, idx = tree.query(pts_i_t)

        normal_dot = np.sum(nrm_i_t * nrm_j[idx], axis=1)
        good_normal = np.where(normal_dot < normal_dot_thresh)[0]
        candidates = good_normal if len(good_normal) >= 3 else np.arange(len(pts_i))

        cand_order = np.argsort(dists[candidates])[:min(n_keep, len(candidates))]
        keep = candidates[cand_order]
        corr_i = pts_i[keep]
        corr_j = pts_j[idx[keep]]

        R_new, t_new = kabsch(corr_i, corr_j)
        change = np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t)
        R, t = R_new, t_new
        if change < tol:
            return R, t, True
    return R, t, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True,
                        help="Sert juste à composer la config data -- aucun modèle chargé, "
                             "sample_method forcé à weighted.")
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--num_points_to_sample", type=int, default=10000,
                        help="Budget PAR OBJET en weighted (cf. phase5a_weighted_gt_check.py).")
    parser.add_argument("--min_frac_pts", type=int, default=50,
                        help="N'inclut que les paires n_frac_pts_min >= ce seuil (2026-07-22 : "
                             "exclut la tranche <50 mise de côté, structurellement dure même "
                             "en GT -- ce test caractérise le raffinement sur la population "
                             "'facile' (~69%) sur laquelle le pipeline actuel continue.")
    parser.add_argument("--rotation_perturbations", type=float, nargs="+",
                        default=[10, 20, 30, 45, 60, 90],
                        help="Amplitudes de perturbation de rotation (degrés) testées.")
    parser.add_argument("--translation_noise_scale", type=float, default=0.05,
                        help="Écart-type du bruit de translation ajouté (échelle normalisée "
                             "des points, ~[-1,1] par fragment).")
    parser.add_argument("--max_icp_iters", type=int, default=50)
    parser.add_argument("--trim_ratio", type=float, default=0.7,
                        help="Fraction des correspondances les plus proches gardées à chaque "
                             "itération ICP (rognage des appariements aberrants).")
    parser.add_argument("--refiner", default="icp", choices=["icp", "icp_normals"],
                        help="'icp' = trimmed ICP vanille (plus proches voisins + Kabsch). "
                             "'icp_normals' (2026-07-22) = ajoute une pénalité d'orientation -- "
                             "rejette les correspondances dont les normales ne sont PAS à peu "
                             "près opposées avant de refaire Kabsch, pour résister au glissement "
                             "dans le plan de contact sur des surfaces quasi-planes (même "
                             "principe que le scoring normal-aware de la Phase 2C).")
    parser.add_argument("--normal_dot_thresh", type=float, default=-0.3,
                        help="Seuil de rejet pour icp_normals : dot(R@n_i, n_j) doit être en "
                             "dessous de cette valeur (normales opposées) pour qu'une "
                             "correspondance soit gardée. Phase 2C : médiane -0.825 sur les "
                             "vraies correspondances, -0.3 = permissif.")
    parser.add_argument("--success_rot_thresh", type=float, default=5.0,
                        help="Seuil de succès en degrés pour l'erreur de rotation FINALE "
                             "(après ICP) par rapport à la vraie pose -- strict, on veut "
                             "vérifier une VRAIE reconvergence, pas juste 'un peu mieux'.")
    parser.add_argument("--success_trans_thresh", type=float, default=0.02)
    parser.add_argument("--csv_out", default="", help="Dump complet, une ligne par (paire, perturbation).")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print("Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          "AUCUN CNN chargé (stratégie GT uniquement -- ce test caractérise le raffinement, "
          "pas le masque de fracture)")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    cfg.data.num_points_to_sample = args.num_points_to_sample
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    rows = []
    n_seen_2frag = 0
    n_skipped_sparse = 0
    t0 = time.time()
    print(f"\nTest de bassin de convergence -- {args.categories}/{args.split}, "
          f"perturbations={args.rotation_perturbations}°, min_frac_pts={args.min_frac_pts}...\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx} | objets 2-frags vus={n_seen_2frag} | "
                      f"{elapsed:.0f}s écoulées")

            points_per_part = batch["points_per_part"][0].numpy()
            pointclouds = batch["pointclouds"][0].numpy()
            pointclouds_normals = batch["pointclouds_normals"][0].numpy()
            fracture_gt = batch["fracture_surface_gt"][0].numpy()

            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            frag_pts  = extract_gt_variable(pointclouds, points_per_part)
            frag_nrm  = extract_gt_variable(pointclouds_normals, points_per_part)
            frag_gt   = extract_gt_variable(fracture_gt, points_per_part)
            if len(frag_pts) != 2:
                continue

            p0, p1 = valid_slots[0], valid_slots[1]
            scale_np = batch["scale"][0].numpy()
            if scale_np.ndim == 1:
                scale_np = scale_np[:, None]
            quats_np = batch["quaternions"][0].numpy()
            trans_np = batch["translations"][0].numpy()

            raw_i = frag_pts[0] * scale_np[p0]
            raw_j = frag_pts[1] * scale_np[p1]
            # Normales : PAS multipliées par scale (jamais divisées par scale dans
            # transform(), contrairement aux points -- ce sont des vecteurs unitaires).
            nrm_i = frag_nrm[0]
            nrm_j = frag_nrm[1]
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            frac_i = raw_i[gt_i == 1]
            frac_j = raw_j[gt_j == 1]
            frac_i_nrm = nrm_i[gt_i == 1]
            frac_j_nrm = nrm_j[gt_j == 1]
            n_min = min(len(frac_i), len(frac_j))
            if n_min < max(ABS_MIN_POINTS, args.min_frac_pts):
                n_skipped_sparse += 1
                continue

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

            for theta in args.rotation_perturbations:
                R_delta = random_rotation_of_magnitude(theta, rng)
                R_init = R_delta @ R_ij_gt
                t_init = t_ij_gt + rng.normal(scale=args.translation_noise_scale, size=3)

                if args.refiner == "icp_normals":
                    R_final, t_final, converged_early = trimmed_icp_normals(
                        frac_i, frac_i_nrm, frac_j, frac_j_nrm, R_init, t_init,
                        max_iters=args.max_icp_iters, trim_ratio=args.trim_ratio,
                        normal_dot_thresh=args.normal_dot_thresh,
                    )
                else:
                    R_final, t_final, converged_early = trimmed_icp(
                        frac_i, frac_j, R_init, t_init,
                        max_iters=args.max_icp_iters, trim_ratio=args.trim_ratio,
                    )
                re = rot_err_deg(R_final, R_ij_gt)
                te = trans_err(t_final, t_ij_gt)
                success = bool(re < args.success_rot_thresh and te < args.success_trans_thresh)

                rows.append({
                    "n_frac_pts_min": n_min,
                    "perturbation_deg": theta,
                    "rot_err_final": re,
                    "trans_err_final": te,
                    "converged_early": converged_early,
                    "success": success,
                })

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} essais ICP sur {n_seen_2frag} objets 2-frags vus "
          f"({n_skipped_sparse} paires sous le seuil de densité) en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    print("BASSIN DE CONVERGENCE (taux de succès ICP par amplitude de perturbation) :")
    header = f"  {'Perturbation':>13} {'N':>6} {'Succès':>8} {'RotErr moy':>11} {'RotErr méd':>11}"
    print(header)
    summary_by_theta = {}
    for theta in args.rotation_perturbations:
        theta_rows = [r for r in rows if r["perturbation_deg"] == theta]
        n = len(theta_rows)
        if n == 0:
            continue
        n_success = sum(1 for r in theta_rows if r["success"])
        re_vals = np.array([r["rot_err_final"] for r in theta_rows])
        print(f"  {theta:>11.0f}° {n:>6} {100*n_success/n:>7.1f}% "
              f"{np.mean(re_vals):>10.2f}° {np.median(re_vals):>10.2f}°")
        summary_by_theta[str(theta)] = {
            "n": n, "success_rate": 100 * n_success / n,
            "rot_err_mean": float(np.mean(re_vals)), "rot_err_median": float(np.median(re_vals)),
        }
    print("\n(Lecture : cherche l'amplitude de perturbation à partir de laquelle le taux de\n"
          " succès chute nettement -- c'est LE seuil de précision que le depth-map matching\n"
          " (étage 1) doit atteindre pour que ce raffinement (étage 2) fonctionne. Si ça tient\n"
          " encore bien au-delà de 30°, `Pose@30` était un seuil trop strict pour juger\n"
          " l'éligibilité du pipeline complet -- une bonne partie des 'échecs' Phase 7\n"
          " pourraient déjà être 'assez bons' pour ce raffinement.)")

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV complet sauvegardé ({len(rows)} lignes) : {args.csv_out}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_trials": len(rows), "n_2frag_seen": n_seen_2frag,
                "n_skipped_sparse": n_skipped_sparse, "by_perturbation": summary_by_theta,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
