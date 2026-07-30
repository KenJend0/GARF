"""
scripts/phase8_reflection_prevalence_check.py
================================================
Phase 8 — voir PLAN_REASSEMBLY_MODULE.md. Suite directe de la découverte
faite en développant `phase8_depthmap_regressor_dataset.py` : son auto-test
synthétique (`_test_3d_pipeline_roundtrip`) a révélé qu'il existe, en plus
du `flip` déjà géré par `match_depthmaps` (ambiguïté de signe de la
profondeur), une AUTRE ambiguïté indépendante -- une vraie RÉFLEXION dans
le plan (u,v) (un seul des deux axes in-plane s'inverse, pas les deux) --
que ni `fit_theta_shift_from_gt` (Kabsch 2D contraint à une rotation pure)
ni `match_depthmaps` (recherche de rotation + flip de profondeur, jamais
de miroir du plan) ne peuvent représenter.

Question : ce cas arrive-t-il souvent sur les VRAIES paires du dataset, ou
n'est-ce qu'un artefact de l'auto-test synthétique (rotation 3D totalement
aléatoire appliquée à un nuage dupliqué) ? Comme les fragments de
Breaking Bad sont désassemblés à une orientation relative arbitraire (pas
de contrainte physique particulière sur `R_ij_gt`), il n'y a a priori pas
de raison que ce cas soit rare -- à mesurer directement.

Protocole (oracle-first, GT uniquement, pas de CNN) : pour chaque paire à
2 fragments, calcule le repère PCA canonique des deux masques GT
(`canonical_pca_frame`), dérive le label (theta_gt, shift_gt) via
`fit_theta_shift_from_gt` (utilise la VRAIE pose 3D, sans recherche), puis
reconstruit la pose (`build_correspondences_nn` + `kabsch`) et compare à
la vraie pose (`rot_err_deg`). Si la dérivation est correcte ET qu'aucune
réflexion n'est nécessaire, `rot_err` doit être quasi nul. Un `rot_err`
proche de 180° signale une paire où une réflexion (u,v) aurait été
nécessaire -- la dérivation actuelle (rotation pure seulement) échoue sur
ces paires.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_reflection_prevalence_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --csv_out /tmp/student7/phase8_reflection.csv \\
        --summary_json /tmp/student7/phase8_reflection.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import (
    quat_wxyz_to_rotmat, kabsch, rot_err_deg, trans_err, build_correspondences_nn,
)
from scripts.phase8_depthmap_regressor_dataset import canonical_pca_frame, fit_theta_shift_from_gt

ABS_MIN_POINTS = 50   # même seuil que le reste de la Phase 7 (tranche <50 mise de côté)
RESOLUTION = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--contact_eps", type=float, default=0.05)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    print("Chargement du datamodule -- sample_method=weighted, AUCUN CNN chargé "
          "(stratégie GT, oracle-first)")
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
    t0 = time.time()
    print(f"\nAmpleur de l'ambiguïté de réflexion (u,v) -- {args.categories}/{args.split}...\n")

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
            fracture_gt = batch["fracture_surface_gt"][0].numpy()

            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            frag_pts = extract_gt_variable(pointclouds, points_per_part)
            frag_gt  = extract_gt_variable(fracture_gt, points_per_part)
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
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            frac_i = raw_i[gt_i == 1]
            frac_j = raw_j[gt_j == 1]
            n_min = min(len(frac_i), len(frac_j))
            if n_min < ABS_MIN_POINTS:
                continue

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

            c_i, u_i, v_i, n_i, _ = canonical_pca_frame(frac_i)
            c_j, u_j, v_j, n_j, _ = canonical_pca_frame(frac_j)

            ci = frac_i - c_i
            cj = frac_j - c_j
            span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                         float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
            span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                         float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
            pixel_size = max(span_i, span_j) * 1.1 / RESOLUTION
            u_min_i = float((ci @ u_i).min()) - 0.5 * pixel_size
            v_min_i = float((ci @ v_i).min()) - 0.5 * pixel_size
            u_min_j = float((cj @ u_j).min()) - 0.5 * pixel_size
            v_min_j = float((cj @ v_j).min()) - 0.5 * pixel_size

            theta_gt, shift_gt = fit_theta_shift_from_gt(
                frac_i, c_i, u_i, v_i, u_min_i, v_min_i,
                c_j, u_j, v_j, u_min_j, v_min_j,
                pixel_size, RESOLUTION, R_ij_gt, t_ij_gt,
            )

            pts_i3, pts_j3 = build_correspondences_nn(
                frac_i, c_i, u_i, v_i, frac_j, c_j, u_j, v_j,
                u_min_i, v_min_i, u_min_j, v_min_j,
                pixel_size, RESOLUTION, theta_gt, shift_gt,
                contact_eps=args.contact_eps,
            )
            if pts_i3 is None or len(pts_i3) < 3:
                rows.append({"n_frac_pts_min": n_min, "rot_err": None, "reflection_suspected": None})
                continue

            R_est, t_est = kabsch(pts_i3, pts_j3)
            re = rot_err_deg(R_est, R_ij_gt)
            te = trans_err(t_est, t_ij_gt)

            rows.append({
                "n_frac_pts_min": n_min,
                "rot_err": re, "trans_err": te,
                "reflection_suspected": bool(re > 90.0),
            })

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires exploitables sur {n_seen_2frag} objets "
          f"2-frags vus en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    exploitable = [r for r in rows if r["rot_err"] is not None]
    n_no_corr = len(rows) - len(exploitable)
    n_refl = sum(1 for r in exploitable if r["reflection_suspected"])
    n_ok = len(exploitable) - n_refl

    print(f"AMPLEUR DE L'AMBIGUÏTÉ DE RÉFLEXION (u,v) -- {len(exploitable)} paires "
          f"reconstruites, {n_no_corr} sans correspondance (pas mesurable) :")
    print(f"  rot_err < 90° (dérivation OK, pas de réflexion nécessaire) : "
          f"{n_ok}/{len(exploitable)} ({100*n_ok/len(exploitable):.1f}%)")
    print(f"  rot_err > 90° (RÉFLEXION SUSPECTÉE, dérivation actuelle échoue) : "
          f"{n_refl}/{len(exploitable)} ({100*n_refl/len(exploitable):.1f}%)")
    rot_errs_ok = [r["rot_err"] for r in exploitable if not r["reflection_suspected"]]
    if rot_errs_ok:
        print(f"  rot_err moyen (cas OK uniquement) : {np.mean(rot_errs_ok):.3f}°  "
              f"(devrait être proche de 0 si la dérivation est correcte)")
    rot_errs_refl = [r["rot_err"] for r in exploitable if r["reflection_suspected"]]
    if rot_errs_refl:
        print(f"  rot_err moyen (cas réflexion) : {np.mean(rot_errs_refl):.1f}°  "
              f"(proche de 180° attendu si c'est bien une réflexion pure)")

    print("\n(Lecture : si 'réflexion suspectée' est une fraction NÉGLIGEABLE (<5%), le "
          "cas est rare en pratique -- pas besoin de le gérer dans le modèle appris. Si "
          "c'est une fraction SUBSTANTIELLE (proche de 50%, comme attendu si c'est un vrai "
          "bit d'ambiguïté aléatoire indépendant par paire), le modèle appris doit tester "
          "les deux hypothèses (normal vs miroir du plan (u,v)), symétriquement au `flip` "
          "déjà géré par match_depthmaps pour la profondeur.)")

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
                "config": vars(args), "n_pairs_exploitable": len(exploitable),
                "n_no_correspondence": n_no_corr,
                "n_ok": n_ok, "n_reflection_suspected": n_refl,
                "frac_reflection_suspected": n_refl / max(len(exploitable), 1),
                "rot_err_mean_ok": float(np.mean(rot_errs_ok)) if rot_errs_ok else None,
                "rot_err_mean_reflection": float(np.mean(rot_errs_refl)) if rot_errs_refl else None,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
