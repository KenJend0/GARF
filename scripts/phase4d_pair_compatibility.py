"""
scripts/phase4d_pair_compatibility.py
======================================
Phase 4D du plan de réassemblage (voir PLAN_REASSEMBLY_MODULE.md, section "Phase 4D").

Question posée : à partir des features CNN (point_features 64-dim du Step 15 figé),
peut-on distinguer une paire de fragments adjacente (graph[i,j]=True) d'une paire
non-adjacente du MÊME objet ?

Pas de matching point-à-point : on agrège les features sur les points fracture de
chaque fragment (mean + max pool → 128-dim par fragment), on concatène (256-dim pour
la paire), et un MLP binaire classe compatible / incompatible.

Ce script change de paradigme par rapport à 3A/4A/4B (Protocole A — matching
point-niveau, paires positives seulement, loss ranking/contrastive) :
  → Protocole B : classification fragment-niveau, paires positives + négatives
    intra-objet, BCE binaire.
Les deux protocoles sont indépendants — on ne compare PAS top8_gap ici mais AUC/AP.

Trois conditions (masque de fracture pour l'agrégation, identique aux phases précédentes) :
  - gt       : points fracture_surface_gt == 1  (oracle mask)
  - thresh0.3: CNN score > 0.3                  (condition réelle)
  - random   : budget identique à gt, points tirés uniformément  (contrôle)

Métriques :
  - AUC-ROC   : robuste au déséquilibre positif/négatif
  - AP (Average Precision) : idem
  - Precision@k : pour chaque fragment, ranker tous les autres fragments du même objet
    par score de compatibilité → top-k sont-ils les vrais voisins ?
    (k = degré GT du fragment dans `graph`)
  - Baseline triviale affichée : classifieur centroïde (distance inter-centroïdes,
    paires proches = compatibles), pour situer la difficulté du problème.

Architecture MLP (3 couches, paramètres ajustables) :
    input (256) → Linear → ReLU → Dropout(0.2) → Linear → ReLU → Linear → logit(1)

Usage (sur le serveur) :
    # Test rapide
    python scripts/phase4d_pair_compatibility.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --max_batches_train 100 --max_batches_val 50 \\
        --epochs 20 --summary_json /tmp/student7/phase4d_quick.json

    # Run complet
    python scripts/phase4d_pair_compatibility.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --epochs 40 \\
        --summary_json /tmp/student7/phase4d_val.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate
from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list


# ── Hyperparamètres ───────────────────────────────────────────────────────────
CNN_FEAT_DIM    = 64    # dimension de point_features (Step 15, confirmée en 3A)
AGG_DIM         = CNN_FEAT_DIM * 2   # mean + max pool → 128 par fragment
PAIR_DIM        = AGG_DIM * 2        # concat i + j → 256
HIDDEN_DIM      = 128
DROPOUT         = 0.2
MIN_FRAC_POINTS = 4     # minimum de points fracture pour agréger (sinon pooling sur rien)
NEG_RATIO       = 3     # échantillonner au plus NEG_RATIO × n_pos négatifs par objet


# ── Modèle ────────────────────────────────────────────────────────────────────

class PairCompatibilityMLP(nn.Module):
    """MLP binaire sur la représentation agrégée d'une paire de fragments."""

    def __init__(self, pair_dim=PAIR_DIM, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,) logits


# ── Agrégation des features par fragment ─────────────────────────────────────

def aggregate_fragment(feat_k: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """mean + max pool sur les points fracture sélectionnés → (AGG_DIM,) ou None."""
    pts = feat_k[mask]
    if len(pts) < MIN_FRAC_POINTS:
        return None
    return np.concatenate([pts.mean(0), pts.max(0)], axis=0)   # (AGG_DIM,)


# ── Construction des paires dans un batch ─────────────────────────────────────

def build_batch_pairs(
    cnn_feat_per_k, score_per_k, gt_per_k, pc_per_k,
    scale_np, bp_pairs, graph_np, object_to_ks, strat, rng,
):
    """Retourne (pair_vecs, labels, centroid_dists) pour toutes les paires de ce batch.

    strat : 'gt' | 'thresh0.3' | 'random'
    pair_vecs  : list[np.ndarray(PAIR_DIM,)]
    labels     : list[int]  (1=adjacent, 0=non-adjacent)
    centroid_dists : list[float] (distance inter-centroïdes, pour baseline triviale)
    """
    pair_vecs, labels, centroid_dists = [], [], []

    for b, ks_ps in object_to_ks.items():
        # Agréger les features de chaque fragment sous ce masque
        agg = {}
        centroid = {}
        n_gt = {}
        for k, p in ks_ps:
            feat_k   = cnn_feat_per_k[k]
            score_k  = score_per_k[k]
            gt_k     = gt_per_k[k]
            pc_k     = pc_per_k[k] * scale_np[b, p]

            n_gt[k] = int((gt_k == 1).sum())

            if strat == "gt":
                mask = gt_k == 1
            elif strat == "thresh0.3":
                mask = score_k > 0.3
            else:   # random
                budget = max(n_gt[k], MIN_FRAC_POINTS)
                idx    = rng.choice(len(pc_k), min(budget, len(pc_k)), replace=False)
                mask   = np.zeros(len(pc_k), dtype=bool)
                mask[idx] = True

            agg[k]      = aggregate_fragment(feat_k, mask)
            centroid[k] = pc_k.mean(0)

        # Construire toutes les paires intra-objet
        ks = [k for k, _ in ks_ps]
        ps = {k: p for k, p in ks_ps}

        pos_pairs = []
        neg_pairs = []
        for a in range(len(ks)):
            for b2 in range(a + 1, len(ks)):
                k_i, k_j = ks[a], ks[b2]
                p_i, p_j = ps[k_i], ps[k_j]
                if agg[k_i] is None or agg[k_j] is None:
                    continue
                label = int(bool(graph_np[b, p_i, p_j]))
                vec   = np.concatenate([agg[k_i], agg[k_j]], axis=0)   # (PAIR_DIM,)
                dist  = float(np.linalg.norm(centroid[k_i] - centroid[k_j]))
                if label == 1:
                    pos_pairs.append((vec, label, dist))
                else:
                    neg_pairs.append((vec, label, dist))

        # Sous-échantillonner les négatifs (NEG_RATIO × n_pos max)
        n_pos = len(pos_pairs)
        if n_pos == 0:
            continue
        if len(neg_pairs) > NEG_RATIO * n_pos:
            chosen = rng.choice(len(neg_pairs), NEG_RATIO * n_pos, replace=False)
            neg_pairs = [neg_pairs[i] for i in chosen]

        for vec, lbl, dist in pos_pairs + neg_pairs:
            pair_vecs.append(vec)
            labels.append(lbl)
            centroid_dists.append(dist)

    return pair_vecs, labels, centroid_dists


# ── Precision@k ───────────────────────────────────────────────────────────────

def precision_at_k_from_scores(all_scores_by_frag):
    """Précision@k : pour chaque fragment, ranker par score descendant, vérifier
    si les k vrais voisins sont dans le top-k.
    all_scores_by_frag : dict[frag_id → list[(score, label)]]
    Retourne precision@k moyen sur tous les fragments avec ≥1 voisin GT.
    """
    precisions = []
    for scores_labels in all_scores_by_frag.values():
        n_pos = sum(l for _, l in scores_labels)
        if n_pos == 0:
            continue
        sorted_items = sorted(scores_labels, key=lambda x: -x[0])
        top_k = sorted_items[:n_pos]
        prec = sum(l for _, l in top_k) / n_pos
        precisions.append(prec)
    return float(np.mean(precisions)) if precisions else 0.0


# ── Évaluation sur un split ───────────────────────────────────────────────────

def evaluate(loader, cnn_model, mlp, device, args, rng, max_batches=0, cnn_device=None):
    """Évalue le MLP sur un split complet.
    Retourne un dict {strat: {auc, ap, prec_at_k, baseline_auc}} pour les 3 conditions.
    """
    mlp.eval()
    # accumuler (score, label) par stratégie + per-fragment pour prec@k
    accum = {s: {"scores": [], "labels": []} for s in ("gt", "thresh0.3", "random")}
    # pour prec@k : (objet_b, frag_k) → [(score, label)]
    per_frag = {s: defaultdict(list) for s in ("gt", "thresh0.3", "random")}

    _cnn_dev = cnn_device if cnn_device is not None else device
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_gpu = {
                k: v.to(_cnn_dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            out = cnn_model(batch_gpu, return_point_features=True)
            feat_flat  = out["point_features"].float().cpu().numpy()
            score_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat    = out["coarse_seg_gt"].long().cpu().numpy()

            valid_np   = valid_pcs.cpu().numpy()
            B, P       = valid_np.shape
            bp_pairs   = [(b, p) for b in range(B) for p in range(P) if valid_np[b, p]]

            scale_np  = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            graph_np = batch["graph"].numpy()

            offsets       = np.concatenate([[0], np.cumsum(frag_sizes)])
            cnn_feat_per_k = [feat_flat [offsets[k]:offsets[k+1]] for k in range(K)]
            score_per_k    = [score_flat[offsets[k]:offsets[k+1]] for k in range(K)]
            gt_per_k       = [gt_flat   [offsets[k]:offsets[k+1]] for k in range(K)]
            pc_per_k       = [frag_list [k].cpu().numpy()          for k in range(K)]

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for strat in ("gt", "thresh0.3", "random"):
                pair_vecs, labels, _ = build_batch_pairs(
                    cnn_feat_per_k, score_per_k, gt_per_k, pc_per_k,
                    scale_np, bp_pairs, graph_np, object_to_ks, strat, rng,
                )
                if not pair_vecs:
                    continue
                x = torch.tensor(np.stack(pair_vecs), dtype=torch.float32, device=device)
                logits = mlp(x).cpu().numpy()
                scores = 1 / (1 + np.exp(-logits))   # sigmoid

                accum[strat]["scores"].extend(scores.tolist())
                accum[strat]["labels"].extend(labels)

                # per-fragment tracking (for prec@k) : frag_id = (batch_idx, k)
                # rebuild pair indices
                all_ks_ps = [(k, p) for b, ks_ps in object_to_ks.items() for k, p in ks_ps]
                ps_by_k   = {k: p for b, ks_ps in object_to_ks.items() for k, p in ks_ps}
                pair_idx  = 0
                for b_obj, ks_ps in object_to_ks.items():
                    ks_list = [k for k, _ in ks_ps]
                    ps      = {k: p for k, p in ks_ps}
                    for a in range(len(ks_list)):
                        for b2 in range(a+1, len(ks_list)):
                            k_i, k_j = ks_list[a], ks_list[b2]
                            p_i, p_j = ps[k_i], ps[k_j]
                            if pair_idx >= len(scores):
                                break
                            s_ij = scores[pair_idx]
                            l_ij = labels[pair_idx]
                            fid_i = (batch_idx, k_i)
                            fid_j = (batch_idx, k_j)
                            per_frag[strat][fid_i].append((s_ij, l_ij))
                            per_frag[strat][fid_j].append((s_ij, l_ij))
                            pair_idx += 1

    results = {}
    for strat in ("gt", "thresh0.3", "random"):
        sc = np.array(accum[strat]["scores"])
        lb = np.array(accum[strat]["labels"])
        if len(sc) == 0 or lb.sum() == 0:
            results[strat] = {"n_pairs": 0}
            continue
        auc = float(roc_auc_score(lb, sc))
        ap  = float(average_precision_score(lb, sc))
        pak = precision_at_k_from_scores(per_frag[strat])
        results[strat] = {
            "n_pairs": len(sc),
            "n_pos":   int(lb.sum()),
            "auc":     auc,
            "ap":      ap,
            "prec_at_k": pak,
        }
    return results


# ── Baseline centroïde ────────────────────────────────────────────────────────

def baseline_centroid(loader, cnn_model, device, args, rng, max_batches=0, cnn_device=None):
    """Baseline triviale : score = -distance inter-centroïdes (plus proches = plus compatibles).
    Évaluation AUC sur le val.
    """
    _cnn_dev = cnn_device if cnn_device is not None else device
    scores_all, labels_all = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_gpu = {
                k: v.to(_cnn_dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            valid_np = valid_pcs.cpu().numpy()
            B, P     = valid_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_np[b, p]]

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            graph_np = batch["graph"].numpy()

            pc_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                ks_list = [k for k, _ in ks_ps]
                ps      = {k: p for k, p in ks_ps}
                centroids = {k: (pc_per_k[k] * scale_np[b, ps[k]]).mean(0) for k in ks_list}
                for a in range(len(ks_list)):
                    for b2 in range(a+1, len(ks_list)):
                        k_i, k_j = ks_list[a], ks_list[b2]
                        dist = float(np.linalg.norm(centroids[k_i] - centroids[k_j]))
                        label = int(bool(graph_np[b, ps[k_i], ps[k_j]]))
                        scores_all.append(-dist)   # plus proche = plus compatible
                        labels_all.append(label)

    sc = np.array(scores_all)
    lb = np.array(labels_all)
    if len(sc) == 0 or lb.sum() == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": len(sc),
        "n_pos":   int(lb.sum()),
        "auc":     float(roc_auc_score(lb, sc)),
        "ap":      float(average_precision_score(lb, sc)),
    }


# ── Boucle d'entraînement ────────────────────────────────────────────────────

def train_epoch(loader, cnn_model, mlp, optimizer, device, args, rng, max_batches=0, cnn_device=None):
    mlp.train()
    total_loss = 0.0
    n_batches  = 0
    _cnn_dev   = cnn_device if cnn_device is not None else device

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch_gpu = {
            k: v.to(_cnn_dev) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]
            out = cnn_model(batch_gpu, return_point_features=True)
            feat_flat  = out["point_features"].float()
            score_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat    = out["coarse_seg_gt"].long().cpu().numpy()

        valid_np = valid_pcs.cpu().numpy()
        B, P     = valid_np.shape
        bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_np[b, p]]

        scale_np = batch["scale"].numpy()
        if scale_np.ndim == 2:
            scale_np = scale_np[:, :, None]
        graph_np = batch["graph"].numpy()

        offsets        = np.concatenate([[0], np.cumsum(frag_sizes)])
        cnn_feat_per_k = [feat_flat [offsets[k]:offsets[k+1]].cpu().numpy() for k in range(K)]
        score_per_k    = [score_flat[offsets[k]:offsets[k+1]] for k in range(K)]
        gt_per_k       = [gt_flat   [offsets[k]:offsets[k+1]] for k in range(K)]
        pc_per_k       = [frag_list [k].cpu().numpy()          for k in range(K)]

        object_to_ks = defaultdict(list)
        for k, (b, p) in enumerate(bp_pairs):
            object_to_ks[b].append((k, p))

        # Utiliser la condition 'thresh0.3' en entraînement (condition réelle)
        pair_vecs, labels, _ = build_batch_pairs(
            cnn_feat_per_k, score_per_k, gt_per_k, pc_per_k,
            scale_np, bp_pairs, graph_np, object_to_ks, args.train_strat, rng,
        )
        if not pair_vecs:
            continue

        x = torch.tensor(np.stack(pair_vecs), dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)

        # pos_weight dynamique : n_neg / n_pos
        n_pos = y.sum().clamp(min=1)
        n_neg = (1 - y).sum().clamp(min=1)
        pos_w = torch.tensor([n_neg / n_pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        optimizer.zero_grad()
        logits = mlp(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--experiment",   required=True)
    parser.add_argument("--categories",   default="everyday")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--epochs",       type=int, default=40)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden_dim",   type=int, default=HIDDEN_DIM)
    parser.add_argument("--dropout",      type=float, default=DROPOUT)
    parser.add_argument("--train_strat",  default="thresh0.3",
                        choices=["gt", "thresh0.3", "random"],
                        help="Masque fracture utilisé pendant l'entraînement.")
    parser.add_argument("--max_batches_train", type=int, default=0)
    parser.add_argument("--max_batches_val",   type=int, default=0)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device} | train_strat={args.train_strat} | epochs={args.epochs} | lr={args.lr}")

    # ── Data ──────────────────────────────────────────────────────────────────
    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    train_loader = DataLoader(
        datamodule.train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=datamodule.dataset_cls.collate_fn,
    )
    val_loader = DataLoader(
        datamodule.val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    # ── Modèle CNN figé sur CPU ───────────────────────────────────────────────
    # Le CNN (~5 GiB) dépasse la mémoire GPU disponible en cohabitation.
    # Inférence CPU (torch.no_grad) ; seul le MLP (41k params) va sur device.
    cnn_device = torch.device("cpu")
    print(f"Loading checkpoint: {args.ckpt} (CNN sur CPU, MLP sur {device})")
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=cnn_device, weights_only=False)
    cnn_model.eval()
    for p in cnn_model.parameters():
        p.requires_grad_(False)

    # ── MLP classifieur sur device (GPU) ──────────────────────────────────────
    mlp = PairCompatibilityMLP(
        pair_dim=PAIR_DIM, hidden=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP params: {n_params:,}  |  pair_dim={PAIR_DIM} → hidden={args.hidden_dim} → 1")

    # ── Baseline centroïde (avant tout entraînement) ─────────────────────────
    print("\nBaseline centroïde (val)...")
    base_res = baseline_centroid(val_loader, cnn_model, device, args, rng, args.max_batches_val,
                                 cnn_device=cnn_device)
    print(f"  Baseline centroïde : AUC={base_res.get('auc','—'):.4f}  "
          f"AP={base_res.get('ap','—'):.4f}  n_pairs={base_res.get('n_pairs',0)}")

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    history = []
    print(f"\nEntraînement — {args.epochs} epochs\n")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'AUC_gt':>8} {'AUC_t03':>8} "
          f"{'AP_gt':>7} {'AP_t03':>7} {'P@k_gt':>7} {'P@k_t03':>8}")
    print("-" * 65)

    best_auc_val  = 0.0
    best_epoch    = 0
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            train_loader, cnn_model, mlp, optimizer, device, args, rng,
            args.max_batches_train, cnn_device=cnn_device,
        )
        val_res = evaluate(
            val_loader, cnn_model, mlp, device, args, rng, args.max_batches_val,
            cnn_device=cnn_device,
        )

        auc_gt  = val_res.get("gt",         {}).get("auc",       0.0)
        auc_t03 = val_res.get("thresh0.3",  {}).get("auc",       0.0)
        ap_gt   = val_res.get("gt",         {}).get("ap",        0.0)
        ap_t03  = val_res.get("thresh0.3",  {}).get("ap",        0.0)
        pak_gt  = val_res.get("gt",         {}).get("prec_at_k", 0.0)
        pak_t03 = val_res.get("thresh0.3",  {}).get("prec_at_k", 0.0)

        print(f"{epoch:>5} {train_loss:>10.4f} {auc_gt:>8.4f} {auc_t03:>8.4f} "
              f"{ap_gt:>7.4f} {ap_t03:>7.4f} {pak_gt:>7.4f} {pak_t03:>8.4f}")

        ref_auc = auc_t03 if args.train_strat == "thresh0.3" else auc_gt
        if ref_auc > best_auc_val:
            best_auc_val = ref_auc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in mlp.state_dict().items()}

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val": {s: val_res.get(s, {}) for s in ("gt", "thresh0.3", "random")},
        })

    print(f"\nMeilleur AUC val ({args.train_strat}) : {best_auc_val:.4f} @ epoch {best_epoch}")

    # Évaluation finale avec le meilleur checkpoint
    if best_state is not None:
        mlp.load_state_dict(best_state)
    final_val = evaluate(val_loader, cnn_model, mlp, device, args, rng, args.max_batches_val,
                         cnn_device=cnn_device)

    print("\n── Résultats finaux (best checkpoint, val) ──")
    header = f"{'Strategy':<12} {'N_pairs':>8} {'AUC':>7} {'AP':>7} {'Prec@k':>8}"
    print(header)
    print("-" * len(header))
    for strat in ("gt", "thresh0.3", "random"):
        r = final_val.get(strat, {})
        if not r.get("n_pairs"):
            print(f"{strat:<12} —")
            continue
        print(f"{strat:<12} {r['n_pairs']:>8} {r['auc']:>7.4f} {r['ap']:>7.4f} {r['prec_at_k']:>8.4f}")

    print(f"\nRéférence baseline centroïde : AUC={base_res.get('auc','—'):.4f}  AP={base_res.get('ap','—'):.4f}")
    print("(AUC > baseline centroïde = signal appris réel ; AUC >> 0.5 = forte compatibilité)")

    # ── Export JSON ───────────────────────────────────────────────────────────
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config":       vars(args),
                "baseline_centroid_val": base_res,
                "best_epoch":   best_epoch,
                "best_auc_val": best_auc_val,
                "final_val":    final_val,
                "history":      history,
            }, f, indent=2)
        print(f"JSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
