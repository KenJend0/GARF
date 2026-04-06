"""
train_hybrid_segmentation.py
=============================
Standalone entry point for the hybrid fracture segmentation experiment.

This script is a thin wrapper around train.py that pre-selects the hybrid
experiment config.  The baseline GARF train.py is NOT modified.

Usage:
    # Minimal run (local dataset path):
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5

    # Full hybrid (normals + curvature + roughness):
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5 \
        data.categories=['everyday']

    # Ablation — normals only:
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5 \
        model.use_curvature=false \
        model.use_roughness=false \
        experiment_name=hybrid_normals_only

    # Ablation — normals + curvature:
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5 \
        model.use_roughness=false \
        experiment_name=hybrid_normals_curvature

    # Resume from checkpoint:
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5 \
        ckpt_path=outputs/hybrid_frac_seg/checkpoints/last.ckpt

    # Multi-GPU:
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5 \
        trainer.devices=[0,1,2,3]

All Hydra overrides accepted by train.py are supported here.

Comparison protocol vs. baseline
---------------------------------
Run baseline pretraining first (or reuse existing checkpoint):
    python train.py experiment=pretraining_frac_seg \
        data.data_root=/path/to/breaking_bad_vol.hdf5

Then run hybrid:
    python train_hybrid_segmentation.py \
        data.data_root=/path/to/breaking_bad_vol.hdf5

Compare val/coarse_seg_f1, val/coarse_seg_acc, val/coarse_seg_recall curves
in your logger (WandB / CSV in outputs/).

Suggested metrics for comparison:
  - Point-wise accuracy   (val/coarse_seg_acc)
  - F1 score             (val/coarse_seg_f1)
  - Precision            (val/coarse_seg_precision)
  - Recall               (val/coarse_seg_recall)
  - Dice loss            (val/loss)

All metrics are logged identically to the baseline, enabling direct comparison.
"""

import sys
from hydra._internal.utils import _run_hydra
from hydra.core.global_hydra import GlobalHydra


def main():
    # Inject the experiment override before Hydra parses the rest of sys.argv.
    # This lets users still override any individual key from the command line.
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.insert(1, "experiment=hybrid_frac_seg")

    # Delegate entirely to train.py logic via its module.
    # Import after argv manipulation so Hydra sees the right defaults.
    import train as train_module
    train_module.main()


if __name__ == "__main__":
    main()
