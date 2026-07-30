"""
scripts/train_depthmap_pose_regressor.py
===========================================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — entraîne `DepthmapPoseRegressor`
(assembly/models/depthmap_pose_regressor.py) sur le dataset précalculé par
`phase8_build_regressor_dataset.py` (.npz : dmap_i/valid_i/dmap_j/valid_j +
labels theta/shift/mirror).

Boucle d'entraînement PyTorch simple (pas Lightning) -- le modèle est petit
(~90K params) et les données sont un `.npz` précalculé, pas le pipeline
hydra/datamodule HDF5 des autres scripts du projet ; un plain training loop
est plus direct ici, cohérent avec `scripts/test_phase8_regressor.py`.

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/train_depthmap_pose_regressor.py \\
        --train_npz /tmp/student7/phase8_dataset_thresh03_train.npz \\
        --val_npz /tmp/student7/phase8_dataset_thresh03_val.npz \\
        --out_dir output/phase8_depthmap_regressor \\
        --epochs 50 --batch_size 64
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assembly.models.depthmap_pose_regressor import DepthmapPoseRegressor, pose_regressor_loss


class DepthmapPairDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.dmap_i = data["dmap_i"]
        self.valid_i = data["valid_i"]
        self.dmap_j = data["dmap_j"]
        self.valid_j = data["valid_j"]
        self.theta_gt = data["theta_gt"]
        self.shift_y_gt = data["shift_y_gt"]
        self.shift_x_gt = data["shift_x_gt"]
        self.mirror_gt = data["mirror_gt"]

    def __len__(self):
        return len(self.theta_gt)

    def __getitem__(self, idx):
        return {
            "dmap_i": torch.from_numpy(self.dmap_i[idx]),
            "valid_i": torch.from_numpy(self.valid_i[idx]),
            "dmap_j": torch.from_numpy(self.dmap_j[idx]),
            "valid_j": torch.from_numpy(self.valid_j[idx]),
            "theta_gt": torch.tensor(self.theta_gt[idx], dtype=torch.float32),
            "shift_y_gt": torch.tensor(self.shift_y_gt[idx], dtype=torch.float32),
            "shift_x_gt": torch.tensor(self.shift_x_gt[idx], dtype=torch.float32),
            "mirror_gt": torch.tensor(bool(self.mirror_gt[idx])),
        }


def _angular_error_deg(theta_pred, theta_gt):
    diff = (theta_pred - theta_gt) % 360.0
    return torch.minimum(diff, 360.0 - diff)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    angle_errs, shift_errs, mirror_correct = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["dmap_i"], batch["valid_i"], batch["dmap_j"], batch["valid_j"])
        loss, _ = pose_regressor_loss(
            pred, batch["theta_gt"], batch["shift_y_gt"], batch["shift_x_gt"], batch["mirror_gt"])
        bs = batch["theta_gt"].shape[0]
        total_loss += loss.item() * bs
        n += bs

        theta_pred, sy_pred, sx_pred, mirror_pred = DepthmapPoseRegressor.decode(pred)
        angle_errs.append(_angular_error_deg(theta_pred, batch["theta_gt"]).cpu())
        shift_err = torch.hypot(sy_pred - batch["shift_y_gt"], sx_pred - batch["shift_x_gt"])
        shift_errs.append(shift_err.cpu())
        mirror_correct.append((mirror_pred == batch["mirror_gt"]).cpu())

    angle_errs = torch.cat(angle_errs)
    shift_errs = torch.cat(shift_errs)
    mirror_correct = torch.cat(mirror_correct)
    return {
        "loss": total_loss / max(n, 1),
        "angle_err_mean_deg": float(angle_errs.mean()),
        "angle_err_median_deg": float(angle_errs.median()),
        "shift_err_mean_px": float(shift_errs.mean()),
        "mirror_acc": float(mirror_correct.float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz",   required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--feat_dim",   type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--w_angle", type=float, default=1.0)
    parser.add_argument("--w_shift", type=float, default=0.01)
    parser.add_argument("--w_conf",  type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DepthmapPairDataset(args.train_npz)
    val_ds = DepthmapPairDataset(args.val_npz)
    print(f"train: {len(train_ds)} paires | val: {len(val_ds)} paires")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DepthmapPoseRegressor(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DepthmapPoseRegressor : {n_params:,} paramètres")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["dmap_i"], batch["valid_i"], batch["dmap_j"], batch["valid_j"])
            loss, _ = pose_regressor_loss(
                pred, batch["theta_gt"], batch["shift_y_gt"], batch["shift_x_gt"],
                batch["mirror_gt"], w_angle=args.w_angle, w_shift=args.w_shift, w_conf=args.w_conf,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = batch["theta_gt"].shape[0]
            train_loss_sum += loss.item() * bs
            n_seen += bs
        scheduler.step()

        train_loss = train_loss_sum / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1:>3}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | angle_err={val_metrics['angle_err_mean_deg']:.2f}° "
              f"(médiane {val_metrics['angle_err_median_deg']:.2f}°) | "
              f"shift_err={val_metrics['shift_err_mean_px']:.2f}px | "
              f"mirror_acc={100*val_metrics['mirror_acc']:.1f}% | {elapsed:.0f}s")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "feat_dim": args.feat_dim, "hidden_dim": args.hidden_dim,
                "epoch": epoch + 1, "val_metrics": val_metrics,
            }, out_dir / "best.ckpt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "feat_dim": args.feat_dim, "hidden_dim": args.hidden_dim,
        "epoch": args.epochs, "val_metrics": val_metrics,
    }, out_dir / "last.ckpt")

    with open(out_dir / "history.json", "w") as f:
        json.dump({"config": vars(args), "n_params": n_params, "history": history}, f, indent=2)

    print(f"\nEntraînement terminé. Meilleur val_loss={best_val_loss:.4f}. "
          f"Checkpoints sauvegardés dans {out_dir}/ (best.ckpt, last.ckpt, history.json)")


if __name__ == "__main__":
    main()
