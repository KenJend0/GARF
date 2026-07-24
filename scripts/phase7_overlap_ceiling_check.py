"""
scripts/phase7_overlap_ceiling_check.py
==========================================
Diagnostic 2b du plan étape par étape (2026-07-22) — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7.

Question de l'utilisateur : le plafond `oracle_overlap_frac`≈0.7 (mesuré le
2026-07-20 par rasterisation en grille) reste inexpliqué -- théoriquement, la
face la plus petite devrait être ENTIÈREMENT contenue dans la plus grande
(même cassure physique). Le sweep de résolution du 2026-07-21 a déjà montré
que ce plafond n'est PAS un invariant fixe (0.989 à R=24, 0.404 à R=128) --
donc "0.7" reflète surtout la résolution utilisée (R=64), pas une vraie
limite physique. Reste à trancher : le résidu (même sans discrétisation en
grille) vient-il du bruit d'échantillonnage indépendant de chaque côté, ou
d'une vraie non-coïncidence géométrique du dataset Breaking Bad ?

Protocole : PAS de rasterisation en grille ici (élimine toute dépendance à
une résolution) -- pour chaque paire, transforme les points fracture du côté
le plus PAUVRE dans le repère du côté le plus RICHE via la VRAIE pose GT
(formule Phase 0), puis mesure la distance au plus proche voisin (cKDTree)
vers les points fracture de l'autre côté. Fraction de points à moins de
`--contact_eps` (0.05, même tolérance que le reste du projet -- Phase 0,
Phase 2B) = overlap "vrai" continu, sans artefact de grille.

Lecture : si ce chiffre continu est déjà proche de 1.0, le ~0.7 mesuré par
grille était purement un artefact de discrétisation -- pas un problème réel.
S'il plafonne aussi nettement en dessous de 1.0, c'est soit du bruit
d'échantillonnage (indépendant de chaque côté), soit une vraie
non-coïncidence géométrique du dataset -- à distinguer par la sensibilité
à la densité de points (un test de suivi, pas fait ici).

Réutilise `extract_gt_variable` (`phase5a_weighted_gt_check.py`) et
`quat_wxyz_to_rotmat` (`phase5a_depthmap_matching.py`) -- aucune
réimplémentation. Stratégie GT uniquement, aucun CNN chargé (oracle-first).

Usage (sur le serveur) :
    python scripts/phase7_overlap_ceiling_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --num_points_to_sample 10000 \\
        --csv_out /tmp/student7/phase7_overlap_ceiling.csv \\
        --summary_json /tmp/student7/phase7_overlap_ceiling.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat

ABS_MIN_POINTS = 5


def percentile_summary(values, label):
    values = np.asarray([v for v in values if v is not None])
    if len(values) == 0:
        print(f"  {label}: —")
        return
    print(f"  {label:<28} N={len(values):>5}  p25={np.percentile(values,25):>7.3f}  "
          f"p50={np.percentile(values,50):>7.3f}  p75={np.percentile(values,75):>7.3f}  "
          f"moyenne={np.mean(values):>7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--contact_eps", type=float, default=0.05,
                        help="Tolérance de proximité (même valeur que Phase 0/2B).")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    print("Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          "AUCUN CNN chargé (stratégie GT, oracle-first)")
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
    print(f"\nDiagnostic overlap continu (sans grille) -- {args.categories}/{args.split}, "
          f"contact_eps={args.contact_eps}...\n")

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

            # ── Le côté le plus PAUVRE transformé dans le repère du côté le plus
            # RICHE via la VRAIE pose -- overlap continu, sans grille. ──────────
            if len(frac_i) <= len(frac_j):
                poor, rich = frac_i, frac_j
                poor_in_rich = (R_ij_gt @ poor.T).T + t_ij_gt
            else:
                poor, rich = frac_j, frac_i
                # Inverse de R_ij_gt/t_ij_gt : R_ji = R_ij.T, t_ji = -R_ij.T @ t_ij
                R_ji_gt = R_ij_gt.T
                t_ji_gt = -R_ij_gt.T @ t_ij_gt
                poor_in_rich = (R_ji_gt @ poor.T).T + t_ji_gt

            tree = cKDTree(rich)
            dists, _ = tree.query(poor_in_rich)
            overlap_frac_continuous = float(np.mean(dists < args.contact_eps))

            rows.append({
                "n_frac_pts_min": n_min,
                "overlap_frac_continuous": overlap_frac_continuous,
                "dist_median": float(np.median(dists)),
                "dist_p90": float(np.percentile(dists, 90)),
            })

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires exploitables sur {n_seen_2frag} objets 2-frags "
          f"vus en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    print("OVERLAP CONTINU (sans grille, distance au plus proche voisin à la VRAIE pose) :")
    percentile_summary([r["overlap_frac_continuous"] for r in rows], "overlap_frac_continuous")
    percentile_summary([r["dist_median"] for r in rows], "dist_median (au plus proche voisin)")
    percentile_summary([r["dist_p90"] for r in rows], "dist_p90")

    print(f"\n  Moyenne overlap_frac_continuous = {100*np.mean([r['overlap_frac_continuous'] for r in rows]):.1f}%")

    print("\n(Lecture : compare ce chiffre au plafond ~0.7 mesuré par grille (résolution "
          "R=64) le 2026-07-20. Si overlap_frac_continuous est proche de 1.0, le ~0.7 était "
          "un pur artefact de discrétisation -- pas un problème réel. S'il plafonne aussi "
          "nettement en dessous de 1.0, c'est du bruit d'échantillonnage ou une vraie "
          "non-coïncidence géométrique du dataset -- distinction à faire par un test de "
          "sensibilité à la densité de points, en suivi.)")

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
                "config": vars(args), "n_pairs": len(rows),
                "overlap_frac_continuous_mean": float(np.mean([r["overlap_frac_continuous"] for r in rows])),
                "overlap_frac_continuous_median": float(np.median([r["overlap_frac_continuous"] for r in rows])),
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
