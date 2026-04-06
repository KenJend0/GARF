"""
eval_segmentation.py
=====================
Standalone evaluation entry point for fracture segmentation models.

Handles two checkpoint formats:
  1. Standalone FracSeg/HybridFracSeg checkpoint  → load directly via Lightning
  2. Full GARF checkpoint (DenoiserFlowMatching)  → extract feature_extractor.* weights

Usage:
    # Evaluate baseline FracSeg from a full GARF checkpoint:
    python eval_segmentation.py \\
        model_type=baseline \\
        ckpt_path=/content/GARF_mini.ckpt \\
        data.data_root=/content/breaking_bad_vol.hdf5 \\
        trainer.devices=[0]

    # Evaluate baseline FracSeg from a standalone FracSeg checkpoint:
    python eval_segmentation.py \\
        model_type=baseline \\
        ckpt_path=/content/frac_seg.ckpt \\
        data.data_root=/content/breaking_bad_vol.hdf5 \\
        trainer.devices=[0]

    # Evaluate hybrid model from its own training checkpoint:
    python eval_segmentation.py \\
        model_type=hybrid \\
        ckpt_path=/content/hybrid_checkpoint.ckpt \\
        data.data_root=/content/breaking_bad_vol.hdf5 \\
        trainer.devices=[0]

    # Limit number of test batches (faster sanity check):
    python eval_segmentation.py \\
        model_type=baseline \\
        ckpt_path=/content/GARF_mini.ckpt \\
        data.data_root=/content/breaking_bad_vol.hdf5 \\
        ++trainer.limit_test_batches=100

Output metrics (logged to CSV in logs/):
    test/coarse_seg_loss      — Dice loss
    test/coarse_seg_acc       — point-wise accuracy
    test/coarse_seg_f1        — F1 score
    test/coarse_seg_recall    — recall (sensitivity to fracture points)
    test/coarse_seg_precision — precision

Comparison protocol:
    After running both baseline and hybrid, compare the CSV files in:
        logs/eval_frac_seg/    (baseline)
        logs/eval_hybrid/      (hybrid)
    The key metric to watch: coarse_seg_f1
    Secondary: coarse_seg_recall (fracture surfaces must NOT be missed)
"""

import os
import sys
import tempfile
from typing import List

import torch
import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Timer
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])


def _is_garf_checkpoint(ckpt_path: str) -> bool:
    """
    Detect whether the checkpoint is a full GARF checkpoint (DenoiserFlowMatching)
    or a standalone segmentation model checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(ckpt.get("state_dict", {}).keys())
    return any(k.startswith("feature_extractor.") for k in keys)


def _extract_frac_seg_weights(garf_ckpt_path: str, out_path: str) -> None:
    """
    Extract the FracSeg (feature_extractor) weights from a full GARF checkpoint
    and save them as a standalone FracSeg checkpoint.

    The GARF checkpoint stores FracSeg under "feature_extractor.*" keys.
    We strip the prefix so the result is loadable by FracSeg directly.
    """
    print(f"[eval_segmentation] Detected full GARF checkpoint.")
    print(f"[eval_segmentation] Extracting feature_extractor weights → {out_path}")

    ckpt = torch.load(garf_ckpt_path, map_location="cpu", weights_only=False)
    frac_seg_state = {
        k.replace("feature_extractor.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("feature_extractor.")
    }

    if not frac_seg_state:
        raise ValueError(
            "No 'feature_extractor.*' keys found in checkpoint. "
            "Are you sure this is a GARF DenoiserFlowMatching checkpoint?"
        )

    print(f"[eval_segmentation] Extracted {len(frac_seg_state)} parameter tensors.")
    torch.save({"state_dict": frac_seg_state}, out_path)


@hydra.main(version_base="1.3", config_path="./configs", config_name="eval")
def main(cfg: DictConfig):
    # ---- Determine model type from config ----
    model_type = cfg.get("model_type", "baseline")   # "baseline" or "hybrid"
    if model_type not in ("baseline", "hybrid"):
        raise ValueError(f"model_type must be 'baseline' or 'hybrid', got '{model_type}'")

    # ---- Inject the right experiment defaults if not already set ----
    # (handled by passing experiment= on the CLI, but we set sensible defaults here)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # ---- Checkpoint handling ----
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path must be provided.")

    tmp_ckpt = None
    if model_type == "baseline" and _is_garf_checkpoint(ckpt_path):
        # Extract FracSeg weights from the full GARF checkpoint into a temp file
        tmp_ckpt = tempfile.NamedTemporaryFile(suffix="_frac_seg.ckpt", delete=False)
        tmp_ckpt.close()
        _extract_frac_seg_weights(ckpt_path, tmp_ckpt.name)
        ckpt_path = tmp_ckpt.name

    # ---- Instantiate everything ----
    loggers: List[Logger] = [
        hydra.utils.instantiate(logger) for logger in cfg.get("loggers").values()
    ]
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    timer = Timer()
    callbacks.append(timer)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print(f"[eval_segmentation] Time elapsed: {timer.time_elapsed('test'):.1f}s")

    # Cleanup temp file
    if tmp_ckpt is not None:
        os.unlink(tmp_ckpt.name)


if __name__ == "__main__":
    # Inject experiment default based on model_type argument if not overridden
    argv = sys.argv[1:]
    model_type = "baseline"
    for arg in argv:
        if arg.startswith("model_type="):
            model_type = arg.split("=", 1)[1]

    if not any(arg.startswith("experiment=") for arg in argv):
        exp = "eval_frac_seg" if model_type == "baseline" else "eval_hybrid_frac_seg"
        sys.argv.insert(1, f"experiment={exp}")

    # Default log dir per model type
    if not any("loggers.csv.save_dir" in arg for arg in argv):
        log_dir = "logs/eval_frac_seg" if model_type == "baseline" else "logs/eval_hybrid"
        sys.argv.insert(1, f"++loggers.csv.save_dir={log_dir}")

    main()
