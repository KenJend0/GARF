import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))
    return loss.mean()


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Tversky loss — generalisation of Dice that independently weights FP and FN.

    T = TP / (TP + alpha*FP + beta*FN)

    alpha > 0.5 penalises FP more than FN → improves precision on imbalanced data.
    alpha = beta = 0.5 reduces to Dice loss.

    Recommended for Step 10: alpha=0.7, beta=0.3 to reduce false positives
    on fragments with low fracture surface ratio.
    """
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
