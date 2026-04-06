"""
train_hybrid_segmentation.py
=============================
Standalone entry point for the hybrid fracture segmentation experiment.
Defaults to experiment=hybrid_frac_seg; all other Hydra overrides still work.

Usage:
    # Full hybrid (normals + curvature + roughness):
    python train_hybrid_segmentation.py data.data_root=/path/to/breaking_bad_vol.hdf5

    # Ablation — normals only:
    python train_hybrid_segmentation.py data.data_root=... \\
        model.use_curvature=false model.use_roughness=false \\
        experiment_name=hybrid_normals_only

    # Ablation — normals + curvature:
    python train_hybrid_segmentation.py data.data_root=... \\
        model.use_roughness=false experiment_name=hybrid_normals_curvature

    # Resume from checkpoint:
    python train_hybrid_segmentation.py data.data_root=... \\
        ckpt_path=outputs/hybrid_frac_seg/checkpoints/last.ckpt
"""

import sys
from typing import List

import hydra
import lightning as L
import torch
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

    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    if cfg.get("ckpt_path") and cfg.get("finetuning"):
        state_dict = torch.load(cfg.get("ckpt_path"), map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.enable_lora()

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=(
            cfg.get("ckpt_path")
            if cfg.get("ckpt_path") and not cfg.get("finetuning")
            else None
        ),
    )


if __name__ == "__main__":
    # Inject hybrid experiment default if not already specified on the command line
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.insert(1, "experiment=hybrid_frac_seg")
    main()
