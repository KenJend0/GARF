"""
scripts/collect_cnn_results_inference.py
==========================================
Comme `collect_cnn_results.py`, mais calcule Acc/F1/Precision/Recall par une
VRAIE passe d'inférence sur chaque checkpoint, au lieu de lire les
`metrics.csv` de l'entraînement (dont certains sont perdus -- cf.
`HARDCODED` dans `collect_cnn_results.py`, steps 10-11).

Différence méthodologique importante (décidée avec l'utilisateur,
2026-08-04) : **un seul protocole d'éval fixe pour toutes les étapes**
(config de données de `cnn_step1_baseline` -- everyday, val, sample_method
uniform), même si certaines expériences (ex. `cnn_step14_generalization`)
définissaient à l'origine un protocole différent (everyday+artifact). Ce
choix privilégie une comparaison strictement apples-to-apples pour le
tableau du rapport, au prix de ne pas reproduire exactement les conditions
de test originales de step14/15 -- si besoin de re-tester ces étapes-là
sur artifact aussi, le faire séparément avec `analyze_errors.py --categories
artifact`.

Le jeu de données est chargé et matérialisé en mémoire (liste de batches
CPU) UNE SEULE FOIS, puis réutilisé pour chaque checkpoint -- pas de
rechargement du dataloader ni de la datamodule à chaque étape.

Métrique : réplique exactement ce que `CNNFracSeg` logge à l'entraînement
(`assembly/models/cnn_segmentation_model.py`, torchmetrics.functional sur
`coarse_seg_pred_binary`/`coarse_seg_gt`, PAR BATCH, moyenné sur tous les
batches -- même réduction que Lightning `on_epoch=True` par défaut) --
directement comparable aux valeurs déjà dans `collect_cnn_results.py`.

Usage :
    CUDA_VISIBLE_DEVICES=1 python scripts/collect_cnn_results_inference.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --output_dir output \\
        --max_batches 300
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torchmetrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from hydra.utils import instantiate

STEPS = [
    ("cnn_step1_baseline",    "Step 1 - Baseline"),
    ("cnn_step2_normals",     "Step 2 + Normals"),
    ("cnn_step3_splatting",   "Step 3 + Splatting"),
    ("cnn_step4_bilinear",    "Step 4 + Bilinear"),
    ("cnn_step5_context",     "Step 5 + Context"),
    ("cnn_step6_attention",   "Step 6 + Attention"),
    ("cnn_step7_unet",        "Step 7 + U-Net"),
    ("cnn_step8a_feat_mean",  "Step 8a + FeatFuse(mean)"),
    ("cnn_step8b_feat_max",   "Step 8b + FeatFuse(max)"),
    ("cnn_step8c_feat_concat",          "Step 8c + FeatFuse(concat)"),
    ("cnn_step8d_simplecnn_feat_mean",  "Step 8d + SimpleCNN+Fuse(mean)"),
    ("cnn_step9_geo_features",          "Step 9  + GeoFeatures"),
    ("cnn_step10_precision",            "Step 10 + Tversky+DistCentroid"),
    ("cnn_step11_hard_sampling",        "Step 11 + FocalLoss+HardSampling"),
    ("cnn_step12_overlap_aware",        "Step 12 + OverlapChannels"),
    ("cnn_step13_point_head",           "Step 13 + PointHead MLP"),
    ("cnn_step14_generalization",       "Step 14 + RandomRotate+InstanceNorm"),
    ("cnn_step15_final_model",          "Step 15 + Final model (from scratch)"),
]

VAL_METRICS = ["acc", "f1", "precision", "recall"]
SHORT_NAMES = {"acc": "Acc", "f1": "F1", "precision": "Prec", "recall": "Rec"}


def find_checkpoint(step_key: str, output_dir: str) -> str:
    """Essaie plusieurs conventions de chemin (les runs de ce projet n'ont
    pas toutes la même structure de sortie) -- retourne le premier trouvé,
    ou None."""
    candidates = [
        os.path.join(output_dir, step_key, "last.ckpt"),
        os.path.join(output_dir, step_key, "version_*", "checkpoints", "last.ckpt"),
        os.path.join(output_dir, step_key, "*.ckpt"),
    ]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]   # la version la plus récente si plusieurs
    return None


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, loader, device, max_batches: int):
    from assembly.models.cnn_segmentation_model import CNNFracSeg

    model = CNNFracSeg.load_from_checkpoint(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    per_batch = {m: [] for m in VAL_METRICS}
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
        out = model(batch)
        pred_b = out["coarse_seg_pred_binary"]
        gt_long = out["coarse_seg_gt"].long()
        if pred_b.numel() == 0:
            continue

        per_batch["acc"].append(torchmetrics.functional.accuracy(pred_b, gt_long, task="binary"))
        per_batch["recall"].append(torchmetrics.functional.recall(pred_b, gt_long, task="binary"))
        per_batch["precision"].append(torchmetrics.functional.precision(pred_b, gt_long, task="binary"))
        per_batch["f1"].append(torchmetrics.functional.f1_score(pred_b, gt_long, task="binary"))

    del model
    torch.cuda.empty_cache()

    if not per_batch["f1"]:
        return None
    return {m: float(torch.stack(per_batch[m]).mean()) for m in VAL_METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", default="output",
                         help="Racine contenant les dossiers cnn_step*/ (checkpoints)")
    parser.add_argument("--categories", default="everyday",
                         help="Fixe le protocole d'éval pour TOUTES les étapes "
                              "(cf. docstring module -- comparaison apples-to-apples)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=300, help="0 = tout le split")
    parser.add_argument("--steps", nargs="+", default=None,
                         help="Sous-ensemble de step_key à évaluer (défaut : tous)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # -- Config de données FIXE (baseline), chargée UNE SEULE FOIS ----------
    print("Chargement de la config de données fixe (cnn_step1_baseline)...")
    fake_args = argparse.Namespace(
        experiment="cnn_step1_baseline", data_root=args.data_root,
        model_type="cnn", categories=args.categories,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit" if args.split == "val" else "test")
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()
    print(f"Dataloader prêt (categories={args.categories}, split={args.split}) -- "
          f"réutilisé pour tous les checkpoints ci-dessous.\n")

    steps = STEPS if args.steps is None else [(k, l) for k, l in STEPS if k in args.steps]

    results = {}
    for step_key, step_label in steps:
        ckpt_path = find_checkpoint(step_key, args.output_dir)
        if ckpt_path is None:
            print(f"  {step_label:<38s} -- checkpoint introuvable, ignoré")
            continue
        print(f"  {step_label:<38s} -- {ckpt_path}")
        metrics = evaluate_checkpoint(ckpt_path, loader, device, args.max_batches)
        if metrics is None:
            print(f"    (aucun batch exploitable)")
            continue
        results[step_key] = (step_label, metrics)

    # -- Tableau récapitulatif ------------------------------------------
    col_w, metric_w = 38, 8
    header = f"{'Step':<{col_w}}" + "".join(f"{SHORT_NAMES[m]:>{metric_w}}" for m in VAL_METRICS)
    print(f"\n[Inférence live, checkpoint final -- protocole fixe categories={args.categories}]")
    print(header)
    print("-" * len(header))
    for step_key, step_label in steps:
        if step_key not in results:
            continue
        _, metrics = results[step_key]
        vals = "".join(f"{metrics[m]:>{metric_w}.4f}" for m in VAL_METRICS)
        print(f"{step_label:<{col_w}}{vals}")


if __name__ == "__main__":
    main()
