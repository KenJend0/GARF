import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))
    return loss.mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Focal loss pour segmentation binaire.

    FL(p) = -(1-p)^γ · log(p)     pour les positifs (fracture)
    FL(p) = -p^γ · log(1-p)       pour les négatifs (intact)

    γ=0 → BCE classique.
    γ=2 → les pixels bien prédits (p proche de 1 ou 0) contribuent peu,
           les pixels incertains/faux contribuent beaucoup.

    Adresse directement le problème des fragments à faible ratio fracture :
    les rares pixels fracture, souvent prédits avec peu de confiance,
    reçoivent un gradient amplifié → le modèle se force à les détecter.
    """
    pred   = pred.contiguous().view(-1).clamp(smooth, 1.0 - smooth)
    target = target.contiguous().view(-1)
    bce    = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    focal  = ((1.0 - pred) * target + pred * (1.0 - target)) ** gamma
    return (focal * bce).mean()


def dice_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Combinaison Dice + Focal Loss.

    alpha * Dice + (1-alpha) * Focal

    Dice : stabilise l'entraînement globalement.
    Focal : force l'attention sur les pixels difficiles (petites fractures).
    alpha=0.5 donne un poids égal aux deux composantes.
    """
    return alpha * dice_loss(pred, target, smooth) + (1.0 - alpha) * focal_loss(pred, target, gamma, smooth)


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
