"""
scripts/phase5a_skip_audit.py
===============================
Audit complet des rejets ("skips") de la Phase 5A — voir PLAN_REASSEMBLY_MODULE.md,
section Phase 7.

Constat du 2026-07-20 (discussion avec l'utilisateur) : avant d'imaginer un
fix (résolution adaptative ou autre), il faut d'abord comprendre PRÉCISÉMENT
pourquoi les paires sont rejetées. Argument de fond : deux fragments qui
viennent de la même cassure ont FORCÉMENT une vraie surface de fracture des
deux côtés — dans l'absolu, aucune paire n'est structurellement impossible à
traiter. Si une paire est rejetée, c'est qu'un de nos choix de pipeline
(budget de points, seuil de planéité, seuil de recouvrement minimum,
résolution de grille) rejette quelque chose qui existe réellement — pas que
le signal est physiquement absent. Il faut distinguer "artefact de seuil"
(qu'on peut corriger) de "cas réellement dégénéré" (contact quasi ponctuel,
où même une méthode parfaite aurait du mal), et ne pas les confondre.

Différence avec `phase5a_weighted_gt_check.py` (qui rejette une paire dès
qu'un seuil officiel est dépassé, perdant l'info qui aurait expliqué
pourquoi) : ce script ne s'arrête JAMAIS aux seuils officiels
(`MIN_FRAC_POINTS`, `max_planarity`, `MIN_OVERLAP_PIXELS`) — il pousse
CHAQUE paire aussi loin que possible dans le pipeline (jusqu'à un plancher
numérique minimal `ABS_MIN_POINTS`, sous lequel la PCA n'a plus de sens),
et enregistre à la fois :
  - si chaque seuil officiel AURAIT rejeté la paire (`would_skip_*`)
  - le résultat réel qu'on aurait obtenu si on ne l'avait pas rejetée
Ça permet de répondre directement à "si on enlève tel seuil, combien de
paires en plus réussiraient ?" au lieu de deviner.

Réutilise directement les fonctions de `phase5a_depthmap_matching.py`
(aucune réimplémentation) et `extract_gt_variable` de
`phase5a_weighted_gt_check.py`. Stratégie GT uniquement, aucun CNN chargé
(label géométrique du dataset).

Usage (sur le serveur) :
    python scripts/phase5a_skip_audit.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --score_mode joint --num_points_to_sample 10000 \\
        --csv_out /tmp/student7/phase5a_skip_audit.csv \\
        --summary_json /tmp/student7/phase5a_skip_audit.json
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
    MIN_FRAC_POINTS, MIN_OVERLAP_PIXELS,
    compute_pca_frame, rasterize, match_depthmaps, build_correspondences,
    kabsch, rot_err_deg, trans_err, quat_wxyz_to_rotmat,
)

ABS_MIN_POINTS = 5   # plancher numérique pur (PCA non-dégénérée) -- PAS le
                     # seuil officiel MIN_FRAC_POINTS=50, juste le minimum
                     # pour que compute_pca_frame ait un sens du tout.


def process_pair_audit(raw_i, raw_j, gt_i, gt_j, R_ij_gt, t_ij_gt, args):
    """Pousse la paire aussi loin que possible dans le pipeline, sans jamais
    s'arrêter aux seuils officiels -- enregistre would_skip_* (le seuil
    officiel aurait-il rejeté ?) ET le résultat réel obtenu malgré tout."""
    frac_i = raw_i[gt_i == 1]
    frac_j = raw_j[gt_j == 1]
    n_i, n_j = len(frac_i), len(frac_j)

    row = {
        "n_frac_pts_i": n_i, "n_frac_pts_j": n_j, "n_frac_pts_min": min(n_i, n_j),
        "would_skip_too_few_points": bool(n_i < MIN_FRAC_POINTS or n_j < MIN_FRAC_POINTS),
        "planarity_i": None, "planarity_j": None, "would_skip_too_curved": None,
        "n_pix_i": None, "n_pix_j": None, "would_skip_sparse_dmap": None,
        "best_overlap_frac": None, "n_corr": None, "would_skip_no_overlap_3d": None,
        "rot_err": None, "trans_err": None, "pose_30": None, "pose_15": None,
        "reached_stage": "unusable_too_few", "compute_error": None,
    }

    if n_i < ABS_MIN_POINTS or n_j < ABS_MIN_POINTS:
        return row   # vraiment rien à faire (quasi 0 point d'un côté) -- pas un artefact de seuil

    try:
        c_i, u_i, v_i, n_i_ax, plan_i = compute_pca_frame(frac_i)
        c_j, u_j, v_j, n_j_ax, plan_j = compute_pca_frame(frac_j)
        row["planarity_i"], row["planarity_j"] = float(plan_i), float(plan_j)
        row["would_skip_too_curved"] = bool(plan_i > args.max_planarity or plan_j > args.max_planarity)
        row["reached_stage"] = "pca_done"

        ci, cj = frac_i - c_i, frac_j - c_j
        span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                     float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
        span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                     float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
        pixel_size = max(span_i, span_j) * 1.1 / args.resolution

        dmap_i, valid_i, u_min_i, v_min_i = rasterize(
            frac_i, c_i, u_i, v_i, n_i_ax, args.resolution, pixel_size)
        dmap_j, valid_j, u_min_j, v_min_j = rasterize(
            frac_j, c_j, u_j, v_j, n_j_ax, args.resolution, pixel_size)
        n_pix_i, n_pix_j = int(valid_i.sum()), int(valid_j.sum())
        row["n_pix_i"], row["n_pix_j"] = n_pix_i, n_pix_j
        row["would_skip_sparse_dmap"] = bool(n_pix_i < MIN_OVERLAP_PIXELS or n_pix_j < MIN_OVERLAP_PIXELS)
        row["reached_stage"] = "rasterized"

        best_score, best_theta, best_shift, best_flip, best_overlap_frac = match_depthmaps(
            dmap_i, valid_i, dmap_j, valid_j, args.n_angles, score_mode=args.score_mode)
        row["best_overlap_frac"] = float(best_overlap_frac)
        row["reached_stage"] = "searched"

        pts_i3, pts_j3 = build_correspondences(
            dmap_i, valid_i, c_i, u_i, v_i, n_i_ax, u_min_i, v_min_i,
            dmap_j, valid_j, c_j, u_j, v_j, n_j_ax, u_min_j, v_min_j,
            pixel_size, best_theta, best_shift, best_flip,
        )
        n_corr = 0 if pts_i3 is None else len(pts_i3)
        row["n_corr"] = n_corr
        row["would_skip_no_overlap_3d"] = bool(n_corr < 3)
        if n_corr < 3:
            row["reached_stage"] = "no_correspondence"
            return row
        row["reached_stage"] = "correspondence_found"

        R_est, t_est = kabsch(pts_i3, pts_j3)
        re = rot_err_deg(R_est, R_ij_gt)
        te = trans_err(t_est, t_ij_gt)
        row["rot_err"], row["trans_err"] = float(re), float(te)
        row["pose_30"] = bool(re < 30.0 and te < 0.1)
        row["pose_15"] = bool(re < 15.0 and te < 0.05)
        row["reached_stage"] = "pose_computed"
    except Exception as e:
        row["compute_error"] = str(e)[:200]

    return row


def gate_cost_report(rows, gate_key, gate_label):
    """Parmi les paires où le seuil officiel `gate_key` AURAIT rejeté la
    paire, combien ont quand même atteint 'pose_computed', et combien de
    celles-là ont réussi (Pose@30) ? Répond directement à : si on enlève ce
    seuil, combien de succès en plus récupère-t-on ?"""
    gated = [r for r in rows if r.get(gate_key) is True]
    if not gated:
        print(f"  {gate_label} : aucune paire concernée")
        return
    n_gated = len(gated)
    reached_pose = [r for r in gated if r["reached_stage"] == "pose_computed"]
    n_reached = len(reached_pose)
    n_success = sum(1 for r in reached_pose if r["pose_30"])
    print(f"  {gate_label} : {n_gated} paires rejetées par ce seuil | "
          f"{n_reached} ({100*n_reached/n_gated:.1f}%) auraient quand même produit une pose | "
          f"{n_success} ({100*n_success/n_gated:.1f}% du total rejeté) auraient réussi Pose@30")


def percentile_summary(values, label):
    values = [v for v in values if v is not None]
    if not values:
        print(f"  {label}: —")
        return
    values = np.asarray(values)
    print(f"  {label:<28} N={len(values):>5}  p25={np.percentile(values,25):>8.2f}  "
          f"p50={np.percentile(values,50):>8.2f}  p75={np.percentile(values,75):>8.2f}  "
          f"p90={np.percentile(values,90):>8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resolution",  type=int, default=64)
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--max_planarity", type=float, default=0.15)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--num_points_to_sample", type=int, default=None,
                        help="Budget PAR OBJET en weighted (pas par fragment, cf. "
                             "phase5a_weighted_gt_check.py). None = valeur de config (5000).")
    parser.add_argument("--csv_out", default="", help="Dump complet, une ligne par paire.")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    print("Chargement du datamodule -- sample_method=weighted (forcé via model_type=garf), "
          "AUCUN CNN chargé")
    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    if args.num_points_to_sample is not None:
        cfg.data.num_points_to_sample = args.num_points_to_sample
        print(f"  num_points_to_sample override -> {args.num_points_to_sample}")
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    rows = []
    n_seen_2frag = 0
    t0 = time.time()
    print(f"\nAudit complet des rejets -- {args.categories}/{args.split}...\n")

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
            frag_gt = extract_gt_variable(fracture_gt, points_per_part)
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

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij = R1.T @ R0
            t_ij = R1.T @ (trans_np[p0] - trans_np[p1])

            row = process_pair_audit(raw_i, raw_j, gt_i, gt_j, R_ij, t_ij, args)
            rows.append(row)

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires auditées ({n_seen_2frag} objets 2-frags vus) "
          f"en {elapsed:.0f}s\n")

    # ── Funnel : où s'arrête chaque paire, en réalité (avec ou sans les seuils) ──
    stages = ["unusable_too_few", "pca_done", "rasterized", "searched",
              "no_correspondence", "correspondence_found", "pose_computed"]
    # pca_done/rasterized/searched/correspondence_found ne sont que des étapes
    # intermédiaires normalement dépassées -- ne compter que les "vrais" arrêts finaux :
    final_counts = {}
    for r in rows:
        s = r["reached_stage"]
        final_counts[s] = final_counts.get(s, 0) + 1
    print("FUNNEL (où chaque paire s'arrête RÉELLEMENT, seuils officiels ignorés) :")
    for s in ["unusable_too_few", "no_correspondence", "pose_computed"]:
        n = final_counts.get(s, 0)
        print(f"  {s:<24} {n:>6}  ({100*n/max(len(rows),1):.1f}%)")
    n_errors = sum(1 for r in rows if r["compute_error"])
    if n_errors:
        print(f"  compute_error            {n_errors:>6}  (erreurs numériques, à inspecter)")

    n_success = sum(1 for r in rows if r.get("pose_30"))
    n_pose = final_counts.get("pose_computed", 0)
    print(f"\n  Parmi les {n_pose} paires ayant produit une pose : "
          f"{n_success} ({100*n_success/max(n_pose,1):.1f}%) réussissent Pose@30")

    # ── Coût de chaque seuil officiel : combien de succès perdus si on le garde ? ──
    print("\nCOÛT DE CHAQUE SEUIL OFFICIEL (paires rejetées qui auraient quand même marché) :")
    gate_cost_report(rows, "would_skip_too_few_points", "too_few_points (seuil=50 pts)")
    gate_cost_report(rows, "would_skip_too_curved", "too_curved (seuil planarity=0.15)")
    gate_cost_report(rows, "would_skip_sparse_dmap", "sparse_dmap (seuil=20 px)")

    # ── Distributions par résultat final : où est la vraie frontière ? ──
    print("\nDISTRIBUTIONS PAR RÉSULTAT FINAL (percentiles) :")
    success_rows = [r for r in rows if r.get("pose_30") is True]
    fail_rows = [r for r in rows if r.get("pose_30") is False]
    no_corr_rows = [r for r in rows if r["reached_stage"] == "no_correspondence"]

    print(" -- n_frac_pts_min --")
    percentile_summary([r["n_frac_pts_min"] for r in success_rows], "Pose@30 = succès")
    percentile_summary([r["n_frac_pts_min"] for r in fail_rows], "Pose@30 = échec")
    percentile_summary([r["n_frac_pts_min"] for r in no_corr_rows], "no_correspondence")

    print(" -- planarity (moyenne i/j) --")
    def plan_mean(r):
        if r["planarity_i"] is None or r["planarity_j"] is None:
            return None
        return (r["planarity_i"] + r["planarity_j"]) / 2
    percentile_summary([plan_mean(r) for r in success_rows], "Pose@30 = succès")
    percentile_summary([plan_mean(r) for r in fail_rows], "Pose@30 = échec")
    percentile_summary([plan_mean(r) for r in no_corr_rows], "no_correspondence")

    print(" -- best_overlap_frac (trouvé par la recherche) --")
    percentile_summary([r["best_overlap_frac"] for r in success_rows], "Pose@30 = succès")
    percentile_summary([r["best_overlap_frac"] for r in fail_rows], "Pose@30 = échec")
    percentile_summary([r["best_overlap_frac"] for r in no_corr_rows], "no_correspondence")

    print(" -- n_pix (moyenne i/j, après rasterisation) --")
    def npix_mean(r):
        if r["n_pix_i"] is None or r["n_pix_j"] is None:
            return None
        return (r["n_pix_i"] + r["n_pix_j"]) / 2
    percentile_summary([npix_mean(r) for r in success_rows], "Pose@30 = succès")
    percentile_summary([npix_mean(r) for r in fail_rows], "Pose@30 = échec")
    percentile_summary([npix_mean(r) for r in no_corr_rows], "no_correspondence")

    print("\n(Lecture : compare les percentiles entre 'succès', 'échec' et 'no_correspondence'.\n"
          " Si les distributions se recouvrent beaucoup, la variable ne discrimine pas bien.\n"
          " Si 'no_correspondence' a des percentiles nettement plus bas que 'échec' sur\n"
          " n_pix/best_overlap_frac, ça confirme que c'est un manque de données, pas un\n"
          " problème de recherche.)")

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV complet sauvegardé ({len(rows)} lignes) : {args.csv_out}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args), "n_pairs": len(rows),
                "n_seen_2frag": n_seen_2frag, "final_counts": final_counts,
                "n_success": n_success, "n_pose_computed": n_pose,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
