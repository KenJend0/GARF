"""
scripts/benchmark_inference.py
================================
Measures real inference latency/throughput of the CNN fracture segmentation
model (and optionally PTv3/FracSeg for comparison), on GPU, batch_size=1.

Reuses the exact model-loading path validated in analyze_errors.py
(load_config_and_model / load_frac_seg_from_garf) so timing reflects the
same code path used for the reported F1 numbers.

Usage:
    python scripts/benchmark_inference.py \
        --ckpt /storage/student7/teyssir/code/output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --n_warmup 10 --n_measure 100

    # + PTv3/FracSeg comparison:
    python scripts/benchmark_inference.py \
        --ckpt /storage/student7/teyssir/code/output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --garf_ckpt /path/to/GARF_mini.ckpt \
        --n_warmup 10 --n_measure 100
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.serialization
torch.serialization.add_safe_globals([functools.partial])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra.utils import instantiate

from analyze_errors import load_config_and_model, load_frac_seg_from_garf
from assembly.models.projection_mapping_utils import extract_fragment_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True,  help="CNN checkpoint (.ckpt)")
    p.add_argument("--data_root",   required=True,  help="Path to breaking_bad_vol.hdf5")
    p.add_argument("--experiment",  required=True,  help="Hydra experiment name (datamodule config only)")
    p.add_argument("--garf_ckpt",   default=None,   help="Optional PTv3/FracSeg checkpoint for comparison")
    p.add_argument("--categories",  default=None,   help="Comma-separated categories override, e.g. artifact")
    p.add_argument("--n_warmup",    type=int, default=10,  help="Warmup batches (excluded from timing)")
    p.add_argument("--n_measure",   type=int, default=100, help="Measured batches")
    return p.parse_args()


class _NS:
    """Minimal stand-in for argparse.Namespace, matches the fields
    load_config_and_model() reads off `args`."""
    def __init__(self, experiment, data_root, categories, model_type, batch_size=1, num_workers=4):
        self.experiment = experiment
        self.data_root = data_root
        self.categories = categories
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers


def _time_model(model, loader, device, n_warmup, n_measure, label):
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    times_ms = []
    n_frags_total = 0
    it = iter(loader)

    with torch.no_grad():
        for i in range(n_warmup):
            batch = next(it)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for i in range(n_measure):
            batch = next(it)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            frag_list, _, K = extract_fragment_list(batch["pointclouds"], batch["points_per_part"])
            if K == 0:
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)
            n_frags_total += K

    times_ms = np.array(times_ms)
    print(f"\n=== {label} ===")
    print(f"  Params: {n_params:,}")
    print(f"  Objects measured: {len(times_ms)}  (avg {n_frags_total / max(len(times_ms), 1):.1f} fragments/object)")
    print(f"  Per-object latency:  mean={times_ms.mean():.2f} ms  median={np.median(times_ms):.2f} ms  "
          f"p95={np.percentile(times_ms, 95):.2f} ms  min={times_ms.min():.2f} ms  max={times_ms.max():.2f} ms")
    print(f"  Per-fragment latency (mean/object / avg fragments per object): "
          f"{times_ms.mean() / max(n_frags_total / max(len(times_ms), 1), 1e-9):.3f} ms")
    print(f"  Throughput: {1000.0 / times_ms.mean():.2f} objects/sec")
    return times_ms


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- CNN ---
    cnn_args = _NS(args.experiment, args.data_root, args.categories, model_type="cnn", batch_size=1)
    cfg = load_config_and_model(cnn_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()

    from assembly.models.cnn_segmentation_model import CNNFracSeg
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    _time_model(cnn_model, loader, device, args.n_warmup, args.n_measure, label="CNN (Step 15)")

    # --- PTv3/FracSeg (optional) ---
    if args.garf_ckpt:
        garf_args = _NS(args.experiment, args.data_root, args.categories, model_type="garf", batch_size=1)
        cfg_garf = load_config_and_model(garf_args)
        datamodule_garf = instantiate(cfg_garf.data)
        datamodule_garf.setup("fit")
        loader_garf = datamodule_garf.val_dataloader()

        garf_model = load_frac_seg_from_garf(args.garf_ckpt, device)
        _time_model(garf_model, loader_garf, device, args.n_warmup, args.n_measure, label="PTv3/FracSeg")


if __name__ == "__main__":
    main()
