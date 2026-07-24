"""
scripts/test_step16_losses.py
==============================
Standalone smoke test for the Step 16 boundary loss / spatial coherence loss
helpers added to assembly/models/cnn_segmentation_model.py. No dataset, no
checkpoint -- pure synthetic tensors, checks shapes/edge-cases only (empty
fragment, singleton fragment, no positive predictions).

Run on the remote lab (torch/lightning/scipy only need to be importable):
    CUDA_VISIBLE_DEVICES=1 python scripts/test_step16_losses.py
"""

import torch

from assembly.models.cnn_segmentation_model import (
    compute_boundary_mask,
    boundary_bce_loss,
    compute_isolated_fp_mask,
    coherence_bce_loss,
)


def main():
    torch.manual_seed(0)

    # -- Normal case: two fragments --
    frag_sizes = [50, 30]
    N = sum(frag_sizes)
    xyz    = torch.rand(N, 3)
    gt     = (torch.rand(N) > 0.7).float()
    pred   = torch.rand(N)
    pred_b = pred > 0.5

    bmask = compute_boundary_mask(xyz, gt, frag_sizes, k=5)
    assert bmask.shape == (N,) and bmask.dtype == torch.bool
    bloss = boundary_bce_loss(pred, gt, bmask)
    assert bloss.dim() == 0 and torch.isfinite(bloss)
    print(f"[normal] boundary frac={bmask.float().mean():.3f} loss={bloss.item():.4f}")

    fpmask = compute_isolated_fp_mask(xyz, pred_b, gt, frag_sizes, eps=0.3, min_cluster_size=3)
    assert fpmask.shape == (N,) and fpmask.dtype == torch.bool
    assert (fpmask & (gt >= 0.5)).sum() == 0, "isolated_fp_mask must never flag a GT-positive point"
    closs = coherence_bce_loss(pred, fpmask)
    assert closs.dim() == 0 and torch.isfinite(closs)
    print(f"[normal] isolated_fp frac={fpmask.float().mean():.3f} loss={closs.item():.4f}")

    # -- Edge cases: empty fragment, singleton fragment, no positives --
    frag_sizes2 = [0, 1, 5]
    N2 = sum(frag_sizes2)
    xyz2   = torch.rand(N2, 3)
    gt2    = torch.zeros(N2)
    pred2  = torch.rand(N2)
    predb2 = pred2 > 0.5   # random threshold at 0.5 on torch.rand -> likely some positives

    bmask2 = compute_boundary_mask(xyz2, gt2, frag_sizes2, k=5)
    assert bmask2.shape == (N2,)
    fpmask2 = compute_isolated_fp_mask(xyz2, predb2, gt2, frag_sizes2, eps=0.1, min_cluster_size=2)
    assert fpmask2.shape == (N2,)
    print("[edge] empty/singleton fragments handled without error")

    # -- Empty batch (K=0-equivalent: all frag_sizes zero) --
    bmask3 = compute_boundary_mask(torch.zeros(0, 3), torch.zeros(0), [0, 0], k=5)
    assert bmask3.shape == (0,)
    fpmask3 = compute_isolated_fp_mask(
        torch.zeros(0, 3), torch.zeros(0, dtype=torch.bool), torch.zeros(0), [0, 0]
    )
    assert fpmask3.shape == (0,)
    print("[edge] all-empty fragments handled without error")

    print("ALL OK")


if __name__ == "__main__":
    main()
