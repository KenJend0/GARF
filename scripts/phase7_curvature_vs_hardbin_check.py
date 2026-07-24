"""
scripts/phase7_curvature_vs_hardbin_check.py
================================================
Diagnostic 2a du plan étape par étape (2026-07-22) — voir
PLAN_REASSEMBLY_MODULE.md, section Phase 7.

Question : la tranche `<50` (population où le CNN prédit moins de 50 points
fracture sur le côté le plus pauvre, ~31% du dataset, identifiée comme
structurellement dure même en GT — 93% d'échec même avec la vérité terrain,
2026-07-22) est-elle plus COURBÉE que le reste du dataset ? Ça répondrait
directement à la question de l'utilisateur sur la SOM (représentation
non-planaire) : justifiée seulement si le lien courbure/difficulté existe
vraiment sur CETTE population précise.

Rappel du contexte (2026-07-20) : on avait déjà trouvé que "plus de
courbure" DÉGRADE le matching sous la méthode actuelle (planarity
stratification, Pose@15 21.74%→3.39% de <0.02 à 0.20+ planéité) — cohérent
avec un problème de représentation sur faces courbées, mais jamais vérifié
si la tranche `<50` en est la cause, ou si c'est un phénomène différent
(juste peu de points, indépendamment de la courbure).

Protocole : pour chaque paire à 2 fragments, calcule le masque `thresh0.3`
(CNN), son n_min (côté le plus pauvre), et la planéité PCA (`compute_pca_frame`,
même fonction que tout le reste du projet) des deux côtés. Compare la
distribution de planéité entre le groupe `<50` et le groupe `>=50`.

Réutilise `compute_pca_frame`/`quat_wxyz_to_rotmat` de
`phase5a_depthmap_matching.py` -- aucune réimplémentation. Un seul dataset
(CNN, `sample_method=uniform`) -- pas besoin du maillage ici, juste des
points et de la planéité.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase7_curvature_vs_hardbin_check.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --csv_out /tmp/student7/phase7_curvature_hardbin.csv \\
        --summary_json /tmp/student7/phase7_curvature_hardbin.json
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import compute_pca_frame, quat_wxyz_to_rotmat
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

ABS_MIN_POINTS = 3   # minimum absolu pour que compute_pca_frame ait un sens


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
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--hard_bin_thresh", type=int, default=50,
                        help="Seuil définissant la tranche difficile (2026-07-22).")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Chargement du dataset CNN (sample_method=uniform)...")
    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit" if args.split == "val" else "test")
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.test_dataset
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    rows = []
    n_seen_2frag = 0
    t0 = time.time()
    print(f"\nDiagnostic courbure vs tranche difficile -- {args.categories}/{args.split}, "
          f"seuil={args.hard_bin_thresh}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | {elapsed:.0f}s écoulées")

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K != 2:
                continue
            n_seen_2frag += 1

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy() for k in range(K)]

            valid_pcs_np = valid_pcs.cpu().numpy()
            p_slots = [p for p in range(valid_pcs_np.shape[1]) if valid_pcs_np[0, p]]
            if len(p_slots) != 2:
                continue
            p0, p1 = p_slots

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]

            raw0 = pc_per_k[0] * scale_np[0, p0]
            raw1 = pc_per_k[1] * scale_np[0, p1]
            sc0, sc1 = score_per_k[0], score_per_k[1]

            mask0 = sc0 > args.threshold
            mask1 = sc1 > args.threshold
            frac0 = raw0[mask0]
            frac1 = raw1[mask1]
            n_min = min(len(frac0), len(frac1))
            if n_min < ABS_MIN_POINTS:
                continue

            plan0 = compute_pca_frame(frac0)[4] if len(frac0) >= 3 else None
            plan1 = compute_pca_frame(frac1)[4] if len(frac1) >= 3 else None
            if plan0 is None or plan1 is None:
                continue

            rows.append({
                "n_frac_pts_min": n_min,
                "planarity_mean": (plan0 + plan1) / 2,
                "planarity_max": max(plan0, plan1),
                "is_hard_bin": n_min < args.hard_bin_thresh,
            })

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires exploitables sur {n_seen_2frag} objets 2-frags "
          f"vus en {elapsed:.0f}s\n")

    if not rows:
        print("Aucune paire exploitable.")
        return

    hard = [r for r in rows if r["is_hard_bin"]]
    easy = [r for r in rows if not r["is_hard_bin"]]

    print(f"COURBURE : tranche difficile (<{args.hard_bin_thresh}) vs le reste :")
    print(f"  N difficile = {len(hard)} ({100*len(hard)/len(rows):.1f}% du total) | "
          f"N reste = {len(easy)}\n")
    print(" -- planarity_mean (moyenne des 2 fragments) --")
    percentile_summary([r["planarity_mean"] for r in hard], "Tranche difficile")
    percentile_summary([r["planarity_mean"] for r in easy], "Reste du dataset")
    print(" -- planarity_max (le côté le plus courbé de la paire) --")
    percentile_summary([r["planarity_max"] for r in hard], "Tranche difficile")
    percentile_summary([r["planarity_max"] for r in easy], "Reste du dataset")

    print("\n(Lecture : si la planéité est NETTEMENT plus haute [plus courbé] sur la tranche\n"
          " difficile que sur le reste, le lien courbure/difficulté est confirmé sur CETTE\n"
          " population précise -- justifie d'investiguer une représentation non-planaire\n"
          " (SOM). Si les distributions sont proches, la tranche difficile est due à autre\n"
          " chose que la courbure -- ne pas investir dans la SOM pour ce problème précis.)")

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV complet sauvegardé ({len(rows)} lignes) : {args.csv_out}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": vars(args), "n_pairs": len(rows), "n_hard": len(hard), "n_easy": len(easy),
        }
        for label, group in [("hard", hard), ("easy", easy)]:
            if group:
                summary[f"{label}_planarity_mean"] = float(np.mean([r["planarity_mean"] for r in group]))
                summary[f"{label}_planarity_median"] = float(np.median([r["planarity_mean"] for r in group]))
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
