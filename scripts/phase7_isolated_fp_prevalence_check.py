"""
scripts/phase7_isolated_fp_prevalence_check.py
=================================================
Voir PLAN_REASSEMBLY_MODULE.md, section Phase 7. Suite du 2026-07-23 :
`phase7_mask_confusion_viz.py` a montré, sur une douzaine de paires en échec
étage 1, un motif visuel net -- de petits amas ISOLÉS de faux positifs CNN
(5-30 points), spatialement séparés de la vraie fracture. Trois tentatives
de neutraliser cet effet côté pipeline de matching (`--cluster_baseline`,
`--robust_pca`, `--remove_tiny_clusters`) ont toutes échoué (net-négatif ou
neutre) -- avant d'investir dans un fine-tuning du CNN (loss de cohérence
spatiale, hard-negative mining), il faut d'abord mesurer si ce motif est
réellement RÉPANDU sur tout le dataset, ou si on l'a vu sur une poignée
d'images non représentative (leçon du 2026-0X : ne jamais généraliser à
partir d'un petit échantillon).

Protocole (UN seul dataset CNN, pas de maillage/zoom/cascade nécessaire --
diagnostic pur sur le masque, donc rapide sur un grand échantillon) : pour
chaque fragment (masque `thresh0.3`), regroupe les points prédits positifs
en composantes connexes (`cluster_points`, Phase 2D, mêmes `CLUSTER_EPS`/
`MIN_CLUSTER_SIZE` que `remove_tiny_clusters_mask`). Distingue :
  - "bruit isolé"   : points étiquetés -1 par cluster_points (composantes
                      plus petites que MIN_CLUSTER_SIZE=10) ;
  - "cluster principal" : la plus grosse composante >= MIN_CLUSTER_SIZE ;
  - "clusters secondaires" : les autres composantes >= MIN_CLUSTER_SIZE.
Pour chaque catégorie, mesure la fraction de VRAIS positifs (vs GT) --
un amas presque entièrement FAUX positif et hors du cluster principal
correspond exactement au motif vu dans les visualisations.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase7_isolated_fp_prevalence_check.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 0 \\
        --csv_out /tmp/student7/phase7_isolated_fp.csv \\
        --summary_json /tmp/student7/phase7_isolated_fp.json
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
from scripts.phase2d_interface_clustering_diagnostic import cluster_points
from scripts.phase5a_zoom_resample_thresh03_check import CLUSTER_EPS, MIN_CLUSTER_SIZE
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

ABS_MIN_POINTS = 5


def analyze_fragment(raw, mask, true_gt):
    """Regroupe les points prédits (`mask`) en composantes connexes et classe
    chaque catégorie (bruit isolé / cluster principal / clusters secondaires)
    par sa fraction de vrais positifs. Retourne None si le masque est trop
    épars pour être exploitable."""
    n_pred = int(mask.sum())
    if n_pred < 2:
        return None
    pred_pts = raw[mask]
    pred_true = true_gt[mask]

    labels = cluster_points(pred_pts, eps=CLUSTER_EPS, min_cluster_size=MIN_CLUSTER_SIZE)
    noise = labels < 0
    n_noise = int(noise.sum())
    noise_precision = float(pred_true[noise].mean()) if n_noise > 0 else None

    valid_labels = labels[labels >= 0]
    n_secondary = 0
    secondary_precision = None
    main_precision = None
    n_main = 0
    if len(valid_labels) > 0:
        sizes = np.bincount(valid_labels)
        main_label = int(np.argmax(sizes))
        is_main = labels == main_label
        is_secondary = (labels >= 0) & (~is_main)
        n_main = int(is_main.sum())
        main_precision = float(pred_true[is_main].mean())
        n_secondary = int(is_secondary.sum())
        if n_secondary > 0:
            secondary_precision = float(pred_true[is_secondary].mean())

    n_fp_total = int((~pred_true.astype(bool)).sum())
    n_fp_isolated = int((~pred_true[noise].astype(bool)).sum()) if n_noise > 0 else 0
    n_fp_secondary = 0
    if n_secondary > 0:
        n_fp_secondary = int((~pred_true[(labels >= 0) & (labels != main_label)].astype(bool)).sum())

    return {
        "n_pred": n_pred, "n_main": n_main, "n_secondary": n_secondary, "n_noise": n_noise,
        "main_precision": main_precision, "secondary_precision": secondary_precision,
        "noise_precision": noise_precision,
        "n_fp_total": n_fp_total,
        "n_fp_isolated": n_fp_isolated + n_fp_secondary,   # "isolé" = hors cluster principal
        "has_isolated_fp_cluster": bool(n_noise > 0 or n_secondary > 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--threshold",   type=float, default=0.3)
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
    print(f"\nAmpleur des amas isolés de faux positifs -- {args.categories}/{args.split}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | {elapsed:.0f}s écoulées")

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"])
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

            fracture_gt_np = batch["fracture_surface_gt"].numpy()

            for k, p in [(0, p0), (1, p1)]:
                raw = pc_per_k[k] * scale_np[0, p]
                score = score_per_k[k]
                mask = score > args.threshold
                if int(mask.sum()) < ABS_MIN_POINTS:
                    continue
                true_gt = (fracture_gt_np[0, p] == 1)
                res = analyze_fragment(raw, mask, true_gt)
                if res is not None:
                    rows.append(res)

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} fragments exploitables sur {n_seen_2frag} objets "
          f"2-frags vus en {elapsed:.0f}s\n")

    if not rows:
        print("Aucun fragment exploitable.")
        return

    n = len(rows)
    n_with_isolated = sum(1 for r in rows if r["has_isolated_fp_cluster"])
    total_fp = sum(r["n_fp_total"] for r in rows)
    total_fp_isolated = sum(r["n_fp_isolated"] for r in rows)
    main_prec = [r["main_precision"] for r in rows if r["main_precision"] is not None]
    noise_prec = [r["noise_precision"] for r in rows if r["noise_precision"] is not None]
    secondary_prec = [r["secondary_precision"] for r in rows if r["secondary_precision"] is not None]

    print(f"AMPLEUR DU MOTIF (sur {n} fragments, {n_seen_2frag} objets 2-frags vus) :")
    print(f"  Fragments avec au moins un amas isolé (bruit ou cluster secondaire) : "
          f"{n_with_isolated}/{n} ({100*n_with_isolated/n:.1f}%)")
    print(f"  Fraction des FAUX POSITIFS totaux situés dans un amas isolé "
          f"(hors cluster principal) : {100*total_fp_isolated/max(total_fp,1):.1f}% "
          f"({total_fp_isolated}/{total_fp})")
    print(f"\n  Precision (fraction de VRAIS positifs) par catégorie :")
    print(f"    Cluster principal   : N={len(main_prec):>5}  moyenne={100*np.mean(main_prec):.1f}%"
          if main_prec else "    Cluster principal   : —")
    print(f"    Clusters secondaires: N={len(secondary_prec):>5}  "
          f"moyenne={100*np.mean(secondary_prec):.1f}%" if secondary_prec else
          "    Clusters secondaires: — (aucun)")
    print(f"    Bruit isolé (<{MIN_CLUSTER_SIZE} pts): N={len(noise_prec):>5}  "
          f"moyenne={100*np.mean(noise_prec):.1f}%" if noise_prec else
          "    Bruit isolé : — (aucun)")

    print("\n(Lecture : si une fraction substantielle des fragments a un amas isolé, ET que la\n"
          " precision de ces amas est nettement plus basse que celle du cluster principal, le\n"
          " motif vu dans phase7_mask_confusion_viz.py est bien un phénomène RÉPANDU du CNN --\n"
          " justifie d'investir dans un fine-tuning ciblé (loss de cohérence spatiale,\n"
          " hard-negative mining) plutôt qu'un artefact de la poignée d'images regardées.)")

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
                "config": vars(args), "n_fragments": n, "n_2frag_seen": n_seen_2frag,
                "n_with_isolated_fp_cluster": n_with_isolated,
                "frac_with_isolated_fp_cluster": n_with_isolated / n,
                "total_fp": total_fp, "total_fp_isolated": total_fp_isolated,
                "frac_fp_isolated": total_fp_isolated / max(total_fp, 1),
                "main_precision_mean": float(np.mean(main_prec)) if main_prec else None,
                "secondary_precision_mean": float(np.mean(secondary_prec)) if secondary_prec else None,
                "noise_precision_mean": float(np.mean(noise_prec)) if noise_prec else None,
            }, f, indent=2)
        print(f"JSON résumé sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
