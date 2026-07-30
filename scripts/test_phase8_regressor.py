"""
scripts/test_phase8_regressor.py
===================================
Smoke test synthétique (torch, tenseurs aléatoires -- pas besoin du
dataset/serveur) du modèle et de la loss de Phase 8
(`assembly/models/depthmap_pose_regressor.py`), même schéma que
`scripts/test_step16_losses.py` : formes, gradients, cas limites -- AVANT
d'investir un entraînement réel.

Usage (local ou serveur, torch suffit) :
    python scripts/test_phase8_regressor.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assembly.models.depthmap_pose_regressor import DepthmapPoseRegressor, pose_regressor_loss


def _random_batch(batch_size: int, resolution: int, device: str = "cpu"):
    g = torch.Generator().manual_seed(0)
    dmap_i = torch.randn(batch_size, resolution, resolution, generator=g, device=device)
    dmap_j = torch.randn(batch_size, resolution, resolution, generator=g, device=device)
    valid_i = (torch.rand(batch_size, resolution, resolution, generator=g, device=device) > 0.3).float()
    valid_j = (torch.rand(batch_size, resolution, resolution, generator=g, device=device) > 0.3).float()
    theta_gt = torch.rand(batch_size, generator=g, device=device) * 360.0
    shift_y_gt = (torch.rand(batch_size, generator=g, device=device) - 0.5) * 20
    shift_x_gt = (torch.rand(batch_size, generator=g, device=device) - 0.5) * 20
    mirror_gt = torch.rand(batch_size, generator=g, device=device) > 0.5
    return dmap_i, valid_i, dmap_j, valid_j, theta_gt, shift_y_gt, shift_x_gt, mirror_gt


def _test_forward_shapes():
    model = DepthmapPoseRegressor(feat_dim=16, hidden_dim=32)   # petit, rapide pour le test
    dmap_i, valid_i, dmap_j, valid_j, *_ = _random_batch(4, 32)
    pred = model(dmap_i, valid_i, dmap_j, valid_j)

    for branch in ("normal", "mirror"):
        for key in ("sin", "cos", "shift_y", "shift_x", "confidence_logit"):
            assert pred[branch][key].shape == (4,), f"{branch}/{key}: shape {pred[branch][key].shape}"
    assert pred["mirror_prob"].shape == (4,)
    assert torch.all((pred["mirror_prob"] >= 0) & (pred["mirror_prob"] <= 1))

    sin2cos2 = pred["normal"]["sin"] ** 2 + pred["normal"]["cos"] ** 2
    assert torch.allclose(sin2cos2, torch.ones_like(sin2cos2), atol=1e-5), (
        "sin/cos doivent être normalisés (norme 1)"
    )
    print("  [OK] _test_forward_shapes")


def _test_decode():
    model = DepthmapPoseRegressor(feat_dim=16, hidden_dim=32)
    dmap_i, valid_i, dmap_j, valid_j, *_ = _random_batch(3, 32)
    pred = model(dmap_i, valid_i, dmap_j, valid_j)
    theta_deg, shift_y, shift_x, mirror = DepthmapPoseRegressor.decode(pred)
    assert theta_deg.shape == (3,) and shift_y.shape == (3,) and mirror.shape == (3,)
    assert torch.all((theta_deg >= 0) & (theta_deg < 360.0))
    assert mirror.dtype == torch.bool
    print("  [OK] _test_decode")


def _test_loss_and_backward():
    model = DepthmapPoseRegressor(feat_dim=16, hidden_dim=32)
    dmap_i, valid_i, dmap_j, valid_j, theta_gt, sy_gt, sx_gt, mirror_gt = _random_batch(4, 32)
    pred = model(dmap_i, valid_i, dmap_j, valid_j)
    loss, parts = pose_regressor_loss(pred, theta_gt, sy_gt, sx_gt, mirror_gt)

    assert loss.dim() == 0, "la loss doit être un scalaire"
    assert torch.isfinite(loss), f"loss non finie : {loss.item()}"
    assert "regression_loss" in parts and "conf_loss" in parts

    model.zero_grad()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.any(p.grad != 0))
    assert n_grad > 0, "aucun gradient n'a circulé -- graphe de calcul cassé"
    print(f"  [OK] _test_loss_and_backward (loss={loss.item():.4f}, "
          f"{n_grad} tenseurs de paramètres avec gradient non nul)")


def _test_perfect_prediction_low_loss():
    """Cas limite : si le modèle prédit EXACTEMENT le label GT (both
    branches, avec la bonne confiance), la loss doit être quasi nulle --
    valide que la loss ne pénalise pas une prédiction parfaite par erreur
    (bug de signe/convention)."""
    theta_gt = torch.tensor([37.0, 200.0])
    sy_gt = torch.tensor([1.5, -2.0])
    sx_gt = torch.tensor([-0.5, 3.0])
    mirror_gt = torch.tensor([False, True])

    theta_rad = torch.deg2rad(theta_gt)

    def _perfect_branch(is_this_the_correct_one: torch.Tensor):
        # Hypothèse correcte -> prédit exactement le label GT + confiance haute.
        # Hypothèse incorrecte -> reste plausible mais confiance basse (pas de
        # cible de régression valide pour elle, cf. docstring pose_regressor_loss).
        sin_ = torch.where(is_this_the_correct_one, torch.sin(theta_rad), torch.zeros_like(theta_rad))
        cos_ = torch.where(is_this_the_correct_one, torch.cos(theta_rad), torch.ones_like(theta_rad))
        shift_y = torch.where(is_this_the_correct_one, sy_gt, torch.zeros_like(sy_gt))
        shift_x = torch.where(is_this_the_correct_one, sx_gt, torch.zeros_like(sx_gt))
        conf = torch.where(is_this_the_correct_one, torch.full_like(theta_gt, 10.0),
                            torch.full_like(theta_gt, -10.0))
        return {"sin": sin_, "cos": cos_, "shift_y": shift_y, "shift_x": shift_x,
                "confidence_logit": conf}

    pred = {
        "normal": _perfect_branch(~mirror_gt),
        "mirror": _perfect_branch(mirror_gt),
    }
    loss, parts = pose_regressor_loss(pred, theta_gt, sy_gt, sx_gt, mirror_gt)
    assert loss.item() < 1e-3, f"loss devrait être quasi nulle pour une prédiction parfaite : {loss.item()}"
    print(f"  [OK] _test_perfect_prediction_low_loss (loss={loss.item():.6f})")


def _test_batch_size_one():
    model = DepthmapPoseRegressor(feat_dim=16, hidden_dim=32)
    dmap_i, valid_i, dmap_j, valid_j, theta_gt, sy_gt, sx_gt, mirror_gt = _random_batch(1, 32)
    pred = model(dmap_i, valid_i, dmap_j, valid_j)
    loss, _ = pose_regressor_loss(pred, theta_gt, sy_gt, sx_gt, mirror_gt)
    assert torch.isfinite(loss)
    print("  [OK] _test_batch_size_one")


if __name__ == "__main__":
    print("Auto-tests test_phase8_regressor.py (torch, tenseurs synthétiques)...\n")
    _test_forward_shapes()
    _test_decode()
    _test_loss_and_backward()
    _test_perfect_prediction_low_loss()
    _test_batch_size_one()
    print("\nTous les auto-tests passent.")
