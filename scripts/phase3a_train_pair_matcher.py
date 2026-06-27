"""
scripts/phase3a_train_pair_matcher.py
========================================
Phase 3A learned matcher training (see PLAN_REASSEMBLY_MODULE.md, "Phase 3" section,
and assembly/models/pair_matching/soft_kabsch_matcher.py for the architecture).

V0 (default): correspondence-only. Trains `soft_correspondence_loss` alone. Question:
    does the network learn to predict the target correspondence matrix better than
    chance? Pose metrics (Pose@30 etc.) are still computed every step via weighted
    Kabsch for MONITORING, but are NOT part of the backward pass.
V1 (--use_kabsch): after --warmup_epochs of V0-only training, adds the geodesic
    rotation + translation L2 loss on the weighted-Kabsch pose. Kept behind a flag and
    a warmup so a not-yet-trained correspondence matrix never has to backprop through
    Kabsch's SVD with meaningless gradients (cf. plan: don't mix two failure modes).

Reuses the EXACT validated data pipeline from scripts/phase3a_pair_dataset_check.py
(`iter_positive_pairs`) -- this script only batches/trains on pairs, it does not
redefine how a pair is built (mask strategy, sampling, soft-label construction).

Positive pairs only (Protocol A, graph[i,j]=True) -- no negative-pair sampling yet
(Phase 3B, deferred).

Usage (V0):
    python scripts/phase3a_train_pair_matcher.py \
        --ckpt output/cnn_step15_final_model/last.ckpt \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --experiment cnn_step15_final_model \
        --categories everyday --train_split train --epochs 5 --max_batches 100 \
        --out_dir output/phase3a_matcher_v0 \
        --summary_json /tmp/student7/phase3a_train_v0.json

Usage (V1, after V0 shows the model learns correspondences better than chance):
    ... --use_kabsch --warmup_epochs 2 --epochs 8
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from scripts.analyze_errors import load_config_and_model
from scripts.phase3a_pair_dataset_check import iter_positive_pairs
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.hybrid_geometry_features import HybridGeometryFeatures
from assembly.models.pair_matching.soft_kabsch_matcher import (
    PairMatcherModel, soft_correspondence_loss, matching_entropy,
    weighted_kabsch, geodesic_rotation_error,
)


def build_features(sample):
    """[points(3) + normals(3) + cnn_score(1) + dist_to_centroid(1)] = 8 dims per side
    -- the agreed minimal V0 feature set, NOT the full HybridGeometryFeatures vector.
    dist_to_centroid is geom_feats[:, 3] (column order: consistency, curvature,
    roughness, dist_to_centroid -- cf. HybridGeometryFeatures with use_normals=False,
    use_curvature=True, use_roughness=True, use_dist_to_centroid=True, as configured in
    phase3a_pair_dataset_check.py's geo_extractor)."""
    dist_i = sample["geom_feats_i"][:, 3:4]
    dist_j = sample["geom_feats_j"][:, 3:4]
    feat_i = np.concatenate(
        [sample["points_i"], sample["normals_i"], sample["cnn_score_i"], dist_i], axis=-1
    )
    feat_j = np.concatenate(
        [sample["points_j"], sample["normals_j"], sample["cnn_score_j"], dist_j], axis=-1
    )
    return feat_i.astype(np.float32), feat_j.astype(np.float32)


def stack_pairs(samples: list, device) -> dict:
    """Stack a list of per-pair sample dicts (numpy, from build_pair_sample) into
    batched torch tensors on `device`."""
    feats_i, feats_j = [], []
    for s in samples:
        fi, fj = build_features(s)
        feats_i.append(fi)
        feats_j.append(fj)
    return {
        "feat_i": torch.from_numpy(np.stack(feats_i)).float().to(device),
        "feat_j": torch.from_numpy(np.stack(feats_j)).float().to(device),
        "points_i": torch.from_numpy(np.stack([s["points_i"] for s in samples])).float().to(device),
        "points_j": torch.from_numpy(np.stack([s["points_j"] for s in samples])).float().to(device),
        "valid_i": torch.from_numpy(np.stack([s["valid_i"] for s in samples])).bool().to(device),
        "valid_j": torch.from_numpy(np.stack([s["valid_j"] for s in samples])).bool().to(device),
        "target": torch.from_numpy(np.stack([s["target_ij"] for s in samples])).float().to(device),
        "R_gt": torch.from_numpy(np.stack([s["R_ij"] for s in samples])).float().to(device),
        "t_gt": torch.from_numpy(np.stack([s["t_ij"] for s in samples])).float().to(device),
    }


def compute_step(matcher_model, batch: dict, args, use_pose_loss: bool):
    """One forward pass + loss + diagnostic metrics for a stacked pair-batch. Pose
    metrics (Kabsch-based) are ALWAYS computed for monitoring; `use_pose_loss` only
    controls whether they're added to the backward graph (V1 behaviour)."""
    logits, P = matcher_model(batch["feat_i"], batch["feat_j"], batch["valid_j"])
    N = batch["points_i"].shape[1]

    l_corr = soft_correspondence_loss(
        P, batch["target"], batch["valid_i"],
        contact_row_weight=args.contact_row_weight, dustbin_row_weight=args.dustbin_row_weight,
    )
    entropy = matching_entropy(P, batch["valid_i"])

    P_points = P[..., :N]
    matched_j = torch.einsum("bnm,bmc->bnc", P_points, batch["points_j"])
    weights = P_points.sum(-1) * batch["valid_i"].float()
    R_pred, t_pred, wsum = weighted_kabsch(batch["points_i"], matched_j, weights)

    rot_err = geodesic_rotation_error(R_pred, batch["R_gt"])
    trans_err = (t_pred - batch["t_gt"]).norm(dim=-1)
    pose_valid = wsum > args.pose_min_weight

    loss = l_corr
    if use_pose_loss and pose_valid.any():
        loss = (
            loss + args.lambda_rot * rot_err[pose_valid].mean()
            + args.lambda_trans * trans_err[pose_valid].mean()
        )
    if args.lambda_entropy > 0:
        loss = loss + args.lambda_entropy * entropy

    with torch.no_grad():
        pred_label = P.argmax(dim=-1)
        target_label = batch["target"].argmax(dim=-1)
        is_dustbin_row = batch["target"][..., -1] > 0.5
        valid_mask = batch["valid_i"]
        contact_mask = (~is_dustbin_row) & valid_mask

        n_valid = valid_mask.float().sum().clamp(min=1)
        dustbin_pred_rate = ((pred_label == N) & valid_mask).float().sum() / n_valid

        n_contact = contact_mask.float().sum()
        k_eff = min(8, N)
        if n_contact > 0:
            top1_acc = (pred_label[contact_mask] == target_label[contact_mask]).float().mean()
            topk_pred = P_points.topk(k_eff, dim=-1).indices
            top8_hit = (topk_pred == target_label.unsqueeze(-1)).any(dim=-1)
            top8_recall = top8_hit[contact_mask].float().mean()

            # Random-guess baseline (closed-form expectation, not sampled -- a uniformly
            # random guess among the n_valid_j real candidates has expected top1 accuracy
            # 1/n_valid_j and expected top-k recall k/n_valid_j): without this, a value
            # like match_top8_recall=5% is uninterpretable -- could be far above or
            # actually BELOW chance depending on how many valid j points there are.
            n_valid_j_per_pair = batch["valid_j"].float().sum(dim=-1).clamp(min=1.0)  # [B]
            n_valid_j_per_row = n_valid_j_per_pair.unsqueeze(1).expand(-1, N)         # [B,N]
            random_top1 = (1.0 / n_valid_j_per_row)[contact_mask].mean()
            random_top8 = (float(k_eff) / n_valid_j_per_row).clamp(max=1.0)[contact_mask].mean()
        else:
            top1_acc = torch.tensor(float("nan"))
            top8_recall = torch.tensor(float("nan"))
            random_top1 = torch.tensor(float("nan"))
            random_top8 = torch.tensor(float("nan"))

        non_dustbin_conf = (1.0 - P[..., -1])[valid_mask].mean()
        rot_err_deg = torch.rad2deg(rot_err)
        pose_success = pose_valid & (rot_err_deg < 30.0) & (trans_err < 0.1)
        has_valid_pose = pose_valid.any()

        # Logit-scale diagnostics (cf. the dustbin-collapse fix): if dustbin_minus_max_match
        # is positive almost everywhere, the dustbin logit still structurally outscores the
        # best real match for most rows, which is exactly why dustbin_pred_rate would sit at
        # ~100% regardless of how much signal the encoder has learned.
        valid_pair_mask = batch["valid_i"].unsqueeze(-1) & batch["valid_j"].unsqueeze(1)  # [B,N,N]
        match_logits = logits[..., :N]
        match_logits_valid = match_logits[valid_pair_mask]
        max_match_per_row = match_logits.masked_fill(~valid_pair_mask, float("-inf")).amax(dim=-1)
        dustbin_logit_per_row = logits[..., -1]

        metrics = {
            "loss": float(loss.item()),
            "l_corr": float(l_corr.item()),
            "entropy": float(entropy.item()),
            "dustbin_pred_rate": float(dustbin_pred_rate.item()),
            "contact_pred_rate": float(1.0 - dustbin_pred_rate.item()),
            "match_top1_acc": float(top1_acc.item()),
            "match_top8_recall": float(top8_recall.item()),
            "random_top1_acc": float(random_top1.item()),
            "random_top8_recall": float(random_top8.item()),
            "non_dustbin_confidence": float(non_dustbin_conf.item()),
            "logit_scale": float(matcher_model.matcher.logit_scale.exp().item()),
            "dustbin_bias": float(matcher_model.matcher.dustbin_bias.item()),
            "match_logits_mean": float(match_logits_valid.mean().item()),
            "match_logits_std": float(match_logits_valid.std().item()),
            "match_logits_min": float(match_logits_valid.min().item()),
            "match_logits_max": float(match_logits_valid.max().item()),
            "dustbin_logit": float(dustbin_logit_per_row[valid_mask].mean().item()),
            "max_match_logit_mean": float(max_match_per_row[valid_mask].mean().item()),
            "dustbin_minus_max_match": float(
                (dustbin_logit_per_row - max_match_per_row)[valid_mask].mean().item()
            ),
            "pose_valid_rate": float(pose_valid.float().mean().item()),
            "rot_err_deg_mean": float(rot_err_deg[pose_valid].mean().item()) if has_valid_pose else float("nan"),
            "rot_err_deg_median": float(rot_err_deg[pose_valid].median().item()) if has_valid_pose else float("nan"),
            "trans_err_mean": float(trans_err[pose_valid].mean().item()) if has_valid_pose else float("nan"),
            "trans_err_median": float(trans_err[pose_valid].median().item()) if has_valid_pose else float("nan"),
            "pose_success_30deg_0.1": float(pose_success.float().mean().item()),
        }
    return loss, metrics


def epoch_pairs_in_chunks(loader, cnn_model, geo_extractor, device, args, rng, max_batches, pairs_per_step):
    """Group the directed positive pairs yielded by `iter_positive_pairs` into
    fixed-size chunks for batched training/eval steps."""
    chunk = []
    for event in iter_positive_pairs(loader, cnn_model, geo_extractor, device, args, rng, max_batches):
        if event["type"] != "pair":
            continue
        chunk.append(event["sample"])
        if len(chunk) >= pairs_per_step:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_epoch(loader, cnn_model, geo_extractor, device, args, rng, matcher_model,
              optimizer, max_batches, pairs_per_step, use_pose_loss, train: bool, log_every: int):
    matcher_model.train(train)
    agg = defaultdict(list)
    n_steps = 0
    for chunk in epoch_pairs_in_chunks(loader, cnn_model, geo_extractor, device, args, rng, max_batches, pairs_per_step):
        batch = stack_pairs(chunk, device)
        if train:
            optimizer.zero_grad()
            loss, metrics = compute_step(matcher_model, batch, args, use_pose_loss)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, metrics = compute_step(matcher_model, batch, args, use_pose_loss)

        for key, val in metrics.items():
            agg[key].append(val)
        n_steps += 1
        if train and log_every > 0 and n_steps % log_every == 0:
            print(f"    step {n_steps}: loss={metrics['loss']:.4f} l_corr={metrics['l_corr']:.4f} "
                  f"top1_acc={metrics['match_top1_acc']:.2%} dustbin_pred={metrics['dustbin_pred_rate']:.2%} "
                  f"pose_success={metrics['pose_success_30deg_0.1']:.2%} "
                  f"logit_scale={metrics['logit_scale']:.2f} dustbin_bias={metrics['dustbin_bias']:.2f} "
                  f"dustbin_minus_max_match={metrics['dustbin_minus_max_match']:+.3f}")

    summary = {key: float(np.nanmean(vals)) for key, vals in agg.items()}
    summary["n_steps"] = n_steps
    return summary


def print_epoch_summary(tag: str, epoch: int, summary: dict):
    print(f"\n  [{tag}] epoch {epoch}: n_steps={summary.get('n_steps', 0)}")
    print(f"    loss={summary.get('loss', float('nan')):.4f}  l_corr={summary.get('l_corr', float('nan')):.4f}  "
          f"entropy={summary.get('entropy', float('nan')):.3f}")
    print(f"    dustbin_pred_rate={summary.get('dustbin_pred_rate', float('nan')):.2%}  "
          f"contact_pred_rate={summary.get('contact_pred_rate', float('nan')):.2%}")
    print(f"    match_top1_acc={summary.get('match_top1_acc', float('nan')):.2%} "
          f"(random={summary.get('random_top1_acc', float('nan')):.2%})  "
          f"match_top8_recall={summary.get('match_top8_recall', float('nan')):.2%} "
          f"(random={summary.get('random_top8_recall', float('nan')):.2%})  "
          f"non_dustbin_confidence={summary.get('non_dustbin_confidence', float('nan')):.3f}")
    print(f"    pose_valid_rate={summary.get('pose_valid_rate', float('nan')):.2%}  "
          f"RotErr(mean/median)={summary.get('rot_err_deg_mean', float('nan')):.2f}/"
          f"{summary.get('rot_err_deg_median', float('nan')):.2f}deg  "
          f"TransErr(mean/median)={summary.get('trans_err_mean', float('nan')):.4f}/"
          f"{summary.get('trans_err_median', float('nan')):.4f}  "
          f"Pose@30deg_0.1={summary.get('pose_success_30deg_0.1', float('nan')):.2%}")
    print(f"    logit_scale={summary.get('logit_scale', float('nan')):.2f}  "
          f"dustbin_bias={summary.get('dustbin_bias', float('nan')):.2f}  "
          f"match_logits(mean/std/min/max)={summary.get('match_logits_mean', float('nan')):.2f}/"
          f"{summary.get('match_logits_std', float('nan')):.2f}/"
          f"{summary.get('match_logits_min', float('nan')):.2f}/"
          f"{summary.get('match_logits_max', float('nan')):.2f}  "
          f"dustbin_logit={summary.get('dustbin_logit', float('nan')):.2f}  "
          f"max_match_logit_mean={summary.get('max_match_logit_mean', float('nan')):.2f}  "
          f"dustbin_minus_max_match={summary.get('dustbin_minus_max_match', float('nan')):+.3f}")


def get_dataset(datamodule, split: str):
    if split == "train":
        datamodule.setup("fit")
        return datamodule.train_dataset
    if split == "val":
        datamodule.setup("fit")
        return datamodule.val_dataset
    datamodule.setup("test")
    return datamodule.test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Frozen CNN Step15 checkpoint (fracture prior).")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--train_split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--val_split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--no_val", action="store_true", help="Skip the per-epoch validation pass.")
    parser.add_argument("--batch_size", type=int, default=2, help="Dataloader batch size (objects per CNN forward).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=100, help="Object-batches per training epoch.")
    parser.add_argument("--val_max_batches", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1116)
    # Data-plumbing args -- identical names/defaults to phase3a_pair_dataset_check.py,
    # validated there (label_topk=8 fixes the diffuse-soft-label bug, cf. plan).
    parser.add_argument("--mask_strategy", default="thresh0.3")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--contact_eps", type=float, default=0.05)
    parser.add_argument("--label_sigma", type=float, default=0.02)
    parser.add_argument("--label_topk", type=int, default=8)
    parser.add_argument("--label_mode", default="soft", choices=["hard", "soft"])
    parser.add_argument("--sample_mode", default="random", choices=["random", "fps"])
    parser.add_argument("--undirected", action="store_true")
    # Model args.
    parser.add_argument("--desc_dim", type=int, default=128)
    parser.add_argument("--encoder_hidden", default="64,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_normalize_desc", action="store_true")
    parser.add_argument(
        "--init_logit_scale", type=float, default=10.0,
        help="Initial logit_scale (cf. dustbin-collapse fix). The first V0 run with the "
             "default 10.0 swung to the opposite extreme (contact_pred_rate~99.86% at "
             "epoch 0) and converged toward ~5 over 5 epochs -- starting near that "
             "converged value avoids spending epochs just on recalibration.",
    )
    parser.add_argument(
        "--init_dustbin_bias", type=float, default=0.0,
        help="Initial dustbin_bias. The first V0 run converged toward ~1.0 over 5 "
             "epochs starting from 0.0 -- same rationale as --init_logit_scale.",
    )
    # Training args.
    parser.add_argument("--pairs_per_step", type=int, default=8, help="Pairs stacked per gradient step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for the encoder.")
    parser.add_argument(
        "--scalar_lr_mult", type=float, default=1.0,
        help="LR multiplier for the two calibration scalars (logit_scale, dustbin_bias) "
             "relative to --lr. These have a much cleaner/stronger gradient signal than "
             "the per-point encoder (they affect every row uniformly), so at mult=1.0 they "
             "can dominate early training and absorb most of the loss improvement before "
             "the encoder gets a chance to learn real correspondences (observed: loss fell "
             "monotonically over 5 epochs while match_top1_acc/match_top8_recall stayed "
             "flat). <1.0 (e.g. 0.1) slows their recalibration so the encoder's signal isn't "
             "drowned out.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contact_row_weight", type=float, default=2.0)
    parser.add_argument("--dustbin_row_weight", type=float, default=1.0)
    parser.add_argument(
        "--use_kabsch", action="store_true",
        help="V1: add geodesic rotation + translation L2 loss on the weighted-Kabsch "
             "pose, after --warmup_epochs of V0 (correspondence-only). Pose metrics are "
             "always computed/logged regardless of this flag.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument("--lambda_trans", type=float, default=1.0)
    parser.add_argument(
        "--lambda_entropy", type=float, default=0.0,
        help="Entropy penalty weight (0 = log only, don't penalize -- per plan, only "
             "activate if the matching matrix is observed to stay too diffuse).",
    )
    parser.add_argument(
        "--pose_min_weight", type=float, default=3.0,
        help="Minimum total non-dustbin weight (sum of P_points per row) for a pair's "
             "Kabsch pose to be trusted (cf. Phase 2's MIN_INLIERS=3 -- same order of "
             "magnitude, not a hard equivalence).",
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--out_dir", default=None, help="Directory to save matcher checkpoints.")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    hidden = tuple(int(h) for h in args.encoder_hidden.split(","))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    train_dataset = get_dataset(datamodule, args.train_split)
    val_dataset = None if args.no_val else get_dataset(datamodule, args.val_split)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        collate_fn=datamodule.dataset_cls.collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1),
            collate_fn=datamodule.dataset_cls.collate_fn,
        )

    print(f"Loading frozen CNN checkpoint: {args.ckpt}")
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    cnn_model.eval()
    cnn_model.to(device)
    for p in cnn_model.parameters():
        p.requires_grad_(False)

    geo_extractor = HybridGeometryFeatures(
        k=16, use_normals=False, use_curvature=True,
        use_roughness=True, use_dist_to_centroid=True,
    )

    matcher_model = PairMatcherModel(
        in_dim=8, hidden=hidden, desc_dim=args.desc_dim,
        dropout=args.dropout, normalize_desc=not args.no_normalize_desc,
        init_logit_scale=args.init_logit_scale, init_dustbin_bias=args.init_dustbin_bias,
    ).to(device)

    # Two LR groups: the calibration scalars (logit_scale, dustbin_bias) affect every row
    # uniformly, giving them a much cleaner/stronger gradient than the per-point encoder --
    # at the same LR they can absorb most of the early loss improvement on their own
    # (observed in the first V0 run: loss fell monotonically while match_top1_acc/
    # match_top8_recall stayed flat). scalar_lr_mult < 1.0 slows them down so the encoder's
    # weaker signal isn't drowned out.
    scalar_params = [matcher_model.matcher.logit_scale, matcher_model.matcher.dustbin_bias]
    scalar_param_ids = {id(p) for p in scalar_params}
    encoder_params = [p for p in matcher_model.parameters() if id(p) not in scalar_param_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": scalar_params, "lr": args.lr * args.scalar_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    print(f"\nPhase 3A {'V1 (corr+pose)' if args.use_kabsch else 'V0 (corr only)'} training "
          f"on {args.categories}/{args.train_split} (N={args.num_points}, mask={args.mask_strategy}, "
          f"label_topk={args.label_topk}, pairs_per_step={args.pairs_per_step}, "
          f"init_logit_scale={args.init_logit_scale}, init_dustbin_bias={args.init_dustbin_bias}, "
          f"scalar_lr_mult={args.scalar_lr_mult})...")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    for epoch in range(args.epochs):
        use_pose_loss = args.use_kabsch and epoch >= args.warmup_epochs
        print(f"\n=== Epoch {epoch} (use_pose_loss={use_pose_loss}) ===")

        train_summary = run_epoch(
            train_loader, cnn_model, geo_extractor, device, args, rng, matcher_model,
            optimizer, args.max_batches, args.pairs_per_step, use_pose_loss,
            train=True, log_every=args.log_every,
        )
        print_epoch_summary("train", epoch, train_summary)
        history["train"].append(train_summary)

        if val_loader is not None:
            val_summary = run_epoch(
                val_loader, cnn_model, geo_extractor, device, args, rng, matcher_model,
                optimizer, args.val_max_batches, args.pairs_per_step, use_pose_loss,
                train=False, log_every=0,
            )
            print_epoch_summary("val", epoch, val_summary)
            history["val"].append(val_summary)

        if out_dir:
            torch.save(matcher_model.state_dict(), out_dir / "last.pt")
            torch.save(matcher_model.state_dict(), out_dir / f"epoch{epoch}.pt")

    if args.summary_json:
        import json as _json

        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            _json.dump(
                {
                    "config": {
                        "categories": args.categories, "train_split": args.train_split,
                        "val_split": args.val_split, "mask_strategy": args.mask_strategy,
                        "num_points": args.num_points, "label_topk": args.label_topk,
                        "label_mode": args.label_mode, "use_kabsch": args.use_kabsch,
                        "warmup_epochs": args.warmup_epochs, "epochs": args.epochs,
                        "lr": args.lr, "pairs_per_step": args.pairs_per_step,
                        "contact_row_weight": args.contact_row_weight,
                        "dustbin_row_weight": args.dustbin_row_weight,
                    },
                    "history": history,
                },
                fh, indent=2,
            )
        print(f"\nSaved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
