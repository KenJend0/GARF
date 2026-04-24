"""
scripts/count_params.py
========================
Count parameters from a Lightning checkpoint (.ckpt) or a live model.

Usage — checkpoint (no GPU, no Hydra needed):
    python scripts/count_params.py --ckpt /path/to/GARF_mini.ckpt

Usage — CNN ablation steps (loads model from config):
    python scripts/count_params.py --cnn

Both modes print a breakdown by component and a summary table.
"""

import argparse
import sys
from collections import defaultdict


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} K"
    return str(n)


def count_from_ckpt(ckpt_path: str):
    import torch

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)

    # Group params by top-level prefix (e.g. feature_extractor, frac_seg, encoder)
    groups: dict[str, int] = defaultdict(int)
    total = 0
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        n = tensor.numel()
        total += n
        prefix = key.split(".")[0]
        groups[prefix] += n

    print(f"\n{'Component':<35} {'Params':>12}")
    print("-" * 48)
    for prefix, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {prefix:<33} {_fmt(n):>12}  ({n:,})")
    print("-" * 48)
    print(f"  {'TOTAL':<33} {_fmt(total):>12}  ({total:,})")
    return total


def count_cnn_models():
    """Instantiate each CNN ablation step and count trainable parameters."""
    import torch
    from functools import partial

    sys.path.insert(0, ".")
    from assembly.models.cnn_segmentation_model import CNNFracSeg

    opt = partial(torch.optim.AdamW, lr=1e-4)

    configs = [
        ("Step 1 - Baseline",          dict()),
        ("Step 2 + Normals",           dict(use_normals=True)),
        ("Step 3 + Splatting",         dict(use_normals=True, splat_sigma=1.5)),
        ("Step 4 + Bilinear",          dict(use_normals=True, splat_sigma=1.5, use_bilinear=True)),
        ("Step 5 + Context",           dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True)),
        ("Step 6 + Attention",         dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True, use_view_attn=True)),
        ("Step 7 + U-Net",             dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True, use_view_attn=True,
                                            backbone="unet", unet_depth=4)),
        ("Step 8a FeatFuse(mean)",     dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True, use_view_attn=True,
                                            backbone="unet", unet_depth=4, feature_fusion="mean")),
        ("Step 8b FeatFuse(max)",      dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True,
                                            backbone="unet", unet_depth=4, feature_fusion="max")),
        ("Step 8c FeatFuse(concat)",   dict(use_normals=True, splat_sigma=1.5, use_bilinear=True,
                                            use_global_context=True,
                                            backbone="unet", unet_depth=4, feature_fusion="concat")),
    ]

    print(f"\n{'Step':<35} {'Params':>12}")
    print("-" * 48)
    for label, kwargs in configs:
        model = CNNFracSeg(optimizer=opt, **kwargs)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {label:<33} {_fmt(n):>12}  ({n:,})")
    print("-" * 48)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a .ckpt file (Lightning checkpoint)")
    parser.add_argument("--cnn", action="store_true",
                        help="Count params for all CNN ablation steps")
    args = parser.parse_args()

    if args.ckpt:
        count_from_ckpt(args.ckpt)

    if args.cnn:
        count_cnn_models()

    if not args.ckpt and not args.cnn:
        parser.print_help()


if __name__ == "__main__":
    main()
