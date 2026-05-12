"""
scripts/compare_models_test.py
===============================
Évalue et compare GARF (PTv3 FracSeg) vs CNN Step 9/10 sur le même test set.

Produit un tableau Markdown avec :
  Model | Params | Threshold | F1 | Precision | Recall | FP% | FN% | Inference(s)

Usage:
    python scripts/compare_models_test.py \\
        --cnn_ckpt    output/cnn_step9_geo_features/last.ckpt \\
        --cnn_exp     cnn_step9_geo_features \\
        --garf_ckpt   /path/to/GARF_mini.ckpt \\
        --data_root   /path/to/breaking_bad_vol.hdf5 \\
        --split       test \\
        --thresholds  0.5,0.65 \\
        --out_dir     output/comparison_test

    # Avec un threshold calibré depuis analyze_errors.py :
    python scripts/compare_models_test.py \\
        --cnn_ckpt  output/cnn_step9_geo_features/last.ckpt \\
        --cnn_exp   cnn_step9_geo_features \\
        --garf_ckpt /path/to/GARF_mini.ckpt \\
        --data_root /path/to/breaking_bad_vol.hdf5 \\
        --split     test \\
        --threshold_file output/analysis/step9/best_threshold.json \\
        --out_dir   output/comparison_test
"""

import argparse
import json
import sys
import time
import functools
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([functools.partial])
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx], replace=True)

from assembly.models.projection_mapping_utils import extract_fragment_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(a, b):
    return a / b if b > 0 else 0.0


def evaluate_model(model, loader, threshold: float, device, max_batches: int = 0,
                   model_name: str = "model"):
    """Évalue un modèle sur un DataLoader et retourne les métriques globales."""
    model.eval().to(device)

    tp = fp = fn = tn = 0
    frag_f1s = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            frag_list, valid_pcs, K = extract_fragment_list(
                batch["pointclouds"], batch["points_per_part"]
            )
            if K == 0:
                continue

            frag_sizes = [f.shape[0] for f in frag_list]

            out = model(batch)
            pred_flat = out["coarse_seg_pred"].float()
            gt_flat   = out["coarse_seg_gt"].long()

            pred_b = (pred_flat > threshold)
            gt_b   = (gt_flat == 1)

            tp += int((pred_b & gt_b).sum())
            fp += int((pred_b & ~gt_b).sum())
            fn += int((~pred_b & gt_b).sum())
            tn += int((~pred_b & ~gt_b).sum())

            # Per-fragment F1
            offset = 0
            for sz in frag_sizes:
                p = pred_b[offset:offset+sz]
                g = gt_b[offset:offset+sz]
                offset += sz
                f_tp = int((p & g).sum())
                f_fp = int((p & ~g).sum())
                f_fn = int((~p & g).sum())
                prec = safe_div(f_tp, f_tp + f_fp)
                rec  = safe_div(f_tp, f_tp + f_fn)
                frag_f1s.append(safe_div(2 * prec * rec, prec + rec))

    elapsed = time.time() - t0
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    acc  = safe_div(tp + tn, tp + fp + fn + tn)
    fp_rate = safe_div(fp, fp + tn)
    fn_rate = safe_div(fn, fn + tp)

    return {
        "model":      model_name,
        "threshold":  threshold,
        "f1":         f1,
        "precision":  prec,
        "recall":     rec,
        "accuracy":   acc,
        "fp_rate":    fp_rate,
        "fn_rate":    fn_rate,
        "mean_frag_f1": float(np.mean(frag_f1s)) if frag_f1s else 0.0,
        "n_fragments":  len(frag_f1s),
        "inference_s":  elapsed,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def load_frac_seg_from_garf(garf_ckpt: str, device: torch.device):
    """Charge le FracSeg PTv3 depuis un checkpoint GARF complet."""
    from assembly.models.pretraining.frac_seg import FracSeg

    ckpt_data = torch.load(garf_ckpt, map_location="cpu", weights_only=False)
    keys = list(ckpt_data.get("state_dict", {}).keys())
    is_garf = any(k.startswith("feature_extractor.") for k in keys)

    if is_garf:
        frac_seg_state = {
            k.replace("feature_extractor.", ""): v
            for k, v in ckpt_data["state_dict"].items()
            if k.startswith("feature_extractor.")
        }
        import lightning as L
        tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
        tmp.close()
        torch.save({"state_dict": frac_seg_state,
                    "pytorch-lightning_version": L.__version__}, tmp.name)
        load_path = tmp.name
        is_tmp = True
    else:
        load_path = garf_ckpt
        is_tmp = False

    try:
        model = FracSeg.load_from_checkpoint(load_path, map_location=device, weights_only=False)
    finally:
        if is_tmp:
            os.unlink(load_path)

    return model


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_table(rows: list) -> str:
    header = (
        "| Model | Params | Threshold | Acc | F1 | Mean-Frag-F1 | "
        "Precision | Recall | FP% | FN% | Time(s) |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} "
            f"| {r.get('params', '?'):,} "
            f"| {r['threshold']:.2f} "
            f"| {r['accuracy']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['mean_frag_f1']:.4f} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['fp_rate']:.2%} "
            f"| {r['fn_rate']:.2%} "
            f"| {r['inference_s']:.1f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn_ckpt",       default=None, help="CNN checkpoint (.ckpt)")
    p.add_argument("--cnn_exp",        default=None, help="CNN Hydra experiment name")
    p.add_argument("--garf_ckpt",      default=None, help="GARF or FracSeg checkpoint (.ckpt)")
    p.add_argument("--data_root",      required=True)
    p.add_argument("--split",          default="test", choices=["val", "test"])
    p.add_argument("--thresholds",     default="0.5",
                   help="Comma-separated thresholds to evaluate, e.g. '0.5,0.65'")
    p.add_argument("--threshold_file", default=None,
                   help="JSON file with 'best_threshold' key (output of analyze_errors.py --sweep_threshold)")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--max_batches",    type=int, default=0, help="0 = all batches")
    p.add_argument("--out_dir",        default="output/comparison_test")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Split: {args.split}")

    # --- Thresholds ---
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    if args.threshold_file:
        with open(args.threshold_file) as fh:
            tdata = json.load(fh)
        cal_thr = tdata["best_threshold"]
        if cal_thr not in thresholds:
            thresholds.append(cal_thr)
        print(f"Calibrated threshold from {args.threshold_file}: {cal_thr:.2f}")

    print(f"Thresholds to evaluate: {thresholds}")

    # --- CNN datamodule ---
    if args.cnn_exp:
        config_dir = str(Path(__file__).resolve().parent.parent / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={args.cnn_exp}",
                    f"data.data_root={args.data_root}",
                    f"data.batch_size={args.batch_size}",
                    f"data.num_workers={args.num_workers}",
                ],
            )
        datamodule = instantiate(cfg.data)
    else:
        # Fallback: use default config
        from assembly.data.breaking_bad.module import BreakingBadDataModule
        datamodule = BreakingBadDataModule(
            data_root=args.data_root,
            categories=["everyday"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_method="uniform",
        )

    if args.split == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    all_rows = []

    # --- Evaluate CNN ---
    if args.cnn_ckpt:
        from assembly.models.cnn_segmentation_model import CNNFracSeg
        print(f"\nLoading CNN from: {args.cnn_ckpt}")
        cnn_model = CNNFracSeg.load_from_checkpoint(
            args.cnn_ckpt, map_location=device, weights_only=False
        )
        cnn_params = count_params(cnn_model)
        print(f"  CNN params: {cnn_params:,}")

        for thr in thresholds:
            print(f"  Evaluating CNN @ threshold={thr}...")
            row = evaluate_model(cnn_model, loader, thr, device,
                                 max_batches=args.max_batches,
                                 model_name=f"CNN ({Path(args.cnn_ckpt).parent.parent.name})")
            row["params"] = cnn_params
            all_rows.append(row)
            print(f"    F1={row['f1']:.4f}  Prec={row['precision']:.4f}  "
                  f"Rec={row['recall']:.4f}  FP%={row['fp_rate']:.2%}")

    # --- Evaluate GARF/FracSeg ---
    if args.garf_ckpt:
        print(f"\nLoading GARF/FracSeg from: {args.garf_ckpt}")
        garf_model = load_frac_seg_from_garf(args.garf_ckpt, device)
        garf_params = count_params(garf_model)
        print(f"  GARF params: {garf_params:,}")

        for thr in thresholds:
            print(f"  Evaluating GARF @ threshold={thr}...")
            row = evaluate_model(garf_model, loader, thr, device,
                                 max_batches=args.max_batches,
                                 model_name="GARF (PTv3 FracSeg)")
            row["params"] = garf_params
            all_rows.append(row)
            print(f"    F1={row['f1']:.4f}  Prec={row['precision']:.4f}  "
                  f"Rec={row['recall']:.4f}  FP%={row['fp_rate']:.2%}")

    # --- Output ---
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    table = format_table(all_rows)
    print(table)

    # Save Markdown
    md_path = out_dir / f"comparison_{args.split}.md"
    with open(md_path, "w") as fh:
        fh.write(f"# Model Comparison — {args.split} set\n\n")
        fh.write(table + "\n")
    print(f"\nSaved Markdown table to: {md_path}")

    # Save JSON
    json_path = out_dir / f"comparison_{args.split}.json"
    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"Saved JSON to: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
