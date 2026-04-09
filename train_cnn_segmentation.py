"""
train_cnn_segmentation.py
==========================
Standalone training entry point for the CNN fracture segmentation pipeline.
Defaults to experiment=null; pass experiment=cnn_step{1..6}_* to run an ablation.

This script is intentionally identical in structure to train_hybrid_segmentation.py
so the two pipelines have the same entry-point interface. The only difference is
that there is no encoder warm-start logic (the CNN has no PTv3 encoder to load from
a pretrained checkpoint).

Usage:
    # Step 1 — baseline:
    python train_cnn_segmentation.py \\
        experiment=cnn_step1_baseline \\
        data.data_root=/path/to/breaking_bad_vol.hdf5

    # Step 2 — + normals:
    python train_cnn_segmentation.py \\
        experiment=cnn_step2_normals \\
        data.data_root=...

    # Run all six steps sequentially:
    bash scripts/run_cnn_ablations.sh /path/to/breaking_bad_vol.hdf5

    # Multi-seed (seed override):
    python train_cnn_segmentation.py \\
        experiment=cnn_step1_baseline \\
        data.data_root=... \\
        seed=42 \\
        experiment_name=cnn_step1_baseline_seed42

    # Resume from checkpoint:
    python train_cnn_segmentation.py \\
        experiment=cnn_step1_baseline \\
        data.data_root=... \\
        ckpt_path=output/cnn_step1_baseline/last.ckpt

    # Quick smoke-test (2 batches, 1 epoch):
    python train_cnn_segmentation.py \\
        experiment=cnn_step1_baseline \\
        data.data_root=... \\
        trainer.max_epochs=1 \\
        +trainer.limit_train_batches=2 \\
        +trainer.limit_val_batches=2
"""

from typing import List

import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: List[Logger] = [
        hydra.utils.instantiate(logger)
        for logger in cfg.get("loggers", dict()).values()
    ]
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    model      = hydra.utils.instantiate(cfg.get("model"))
    datamodule = hydra.utils.instantiate(cfg.get("data"))
    callbacks  = [
        hydra.utils.instantiate(cb) for cb in cfg.get("callbacks").values()
    ]

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
