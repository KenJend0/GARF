"""
scripts/phase6_0_recall_sweep.py
=================================
Phase 6.0 du plan de réassemblage (voir PLAN_REASSEMBLY_MODULE.md, "Phase 6 —
Compatibility-guided pose refinement").

Question posée : le classifieur de compatibilité 4D (AUC≈0.798, Prec@k≈0.733 sur
toutes les paires mélangées) peut-il servir de générateur de candidats à HAUT RECALL
(shortlist top-k / 2k / 3k / 5k), plutôt que de décision dure top-k ? C'est le
prérequis obligatoire avant tout raffinement de pose (Phase 6A/6B) : si le vrai
voisin n'est même pas dans une shortlist large, aucun raffinement local derrière ne
pourra le récupérer, quelle que soit sa qualité.

Différence avec Prec@k de la Phase 4D : ici le recall est mesuré PAR FRAGMENT
(pas sur toutes les paires mélangées), et stratifié par nombre de fragments de
l'objet (2 / 3-5 / 6-10 / 11+). Les objets à 2 fragments sont non-informatifs (le
seul autre fragment est nécessairement le bon voisin) : rapportés séparément,
jamais utilisés pour la décision go/no-go.

Réutilise directement scripts/phase4d_pair_compatibility.py : mêmes fonctions
d'agrégation (aggregate_fragment), même architecture MLP (PairCompatibilityMLP),
même CNN Step 15 figé (CNNFracSeg). Charge le state_dict sauvegardé par la Phase 4D
(--mlp_ckpt, écrit via --model_out du script 4D) — pas de ré-entraînement ici.

Pour chaque fragment i d'un objet à K>=2 fragments, on classe tous les autres
fragments du même objet par score de compatibilité MLP décroissant (même score
symétrique que le Prec@k de la Phase 4D, calculé une fois par paire non-ordonnée),
et on mesure si les n_pos vrais voisins GT de i sont dans le top k_gt / 2*k_gt /
3*k_gt / 5*k_gt (k_gt = n_pos = degré GT de i, capé à K-1 candidats disponibles).

Usage (serveur) :
    python scripts/phase6_0_recall_sweep.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --mlp_ckpt output/phase4d_mlp/best.pt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 200 \\
        --summary_json /tmp/student7/phase6_0_recall.json

Critère de décision (fixé dans le plan, objets 3+ fragments uniquement, stratégie
thresh0.3 = condition réelle) :
    Go  Phase 6A  si Recall@3k >= 90% OU Recall@5k >= 95%,
                  ET la shortlist ne dégénère pas vers la quasi-totalité des
                  candidats (avg_kept_ratio@5k < ~0.8 — sinon "haut recall" est
                  trivial : garder tout le monde donne recall=100%).
    No-go         si Recall@5k < 90%, ou si top-5k revient à garder quasiment tous
                  les fragments de l'objet (avg_kept_ratio@5k >= ~0.8).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate
from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list
from scripts.phase4d_pair_compatibility import (
    PairCompatibilityMLP, aggregate_fragment, MIN_FRAC_POINTS,
)

MULTIPLIERS = (1, 2, 3, 5)   # k, 2k, 3k, 5k


def frag_count_bucket(k: int) -> str:
    if k == 2:
        return "2"
    if 3 <= k <= 5:
        return "3-5"
    if 6 <= k <= 10:
        return "6-10"
    return "11+"


def aggregate_object_fragments(cnn_feat_per_k, score_per_k, gt_per_k, pc_per_k,
                                scale_np, ks_ps, strat, rng):
    """agg[k] = (AGG_DIM,) ou None, pour tous les fragments k d'un même objet."""
    agg = {}
    n_gt = {}
    for k, p in ks_ps:
        feat_k = cnn_feat_per_k[k]
        score_k = score_per_k[k]
        gt_k = gt_per_k[k]
        pc_k = pc_per_k[k] * scale_np[k]
        n_gt[k] = int((gt_k == 1).sum())

        if strat == "gt":
            mask = gt_k == 1
        elif strat == "thresh0.3":
            mask = score_k > 0.3
        else:   # random
            budget = max(n_gt[k], MIN_FRAC_POINTS)
            idx = rng.choice(len(pc_k), min(budget, len(pc_k)), replace=False)
            mask = np.zeros(len(pc_k), dtype=bool)
            mask[idx] = True

        agg[k] = aggregate_fragment(feat_k, mask)
    return agg


def build_object_pair_scores(mlp, agg, ks, device):
    """Score MLP symétrique pour chaque paire non-ordonnée (k_i, k_j) valide.
    Retourne dict[(k_i,k_j)] = score (k_i < k_j dans l'ordre de `ks`)."""
    pairs, vecs = [], []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            k_i, k_j = ks[a], ks[b]
            if agg[k_i] is None or agg[k_j] is None:
                continue
            vecs.append(np.concatenate([agg[k_i], agg[k_j]], axis=0))
            pairs.append((k_i, k_j))
    if not vecs:
        return {}
    x = torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = mlp(x).cpu().numpy()
    scores = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    return {p: float(s) for p, s in zip(pairs, scores)}


def per_fragment_recall(ks, pair_scores, graph_np, b, ps):
    """Pour chaque fragment k de l'objet, retourne
    (bucket_key, {mult: recall_at_mult, ...}, kept_ratio_at_5k) ou None si n_pos=0."""
    results = []
    n_other_total = len(ks) - 1
    for k_i in ks:
        candidates = []   # (other_k, score, label)
        for k_j in ks:
            if k_j == k_i:
                continue
            key = (k_i, k_j) if k_i < k_j else (k_j, k_i)
            if key not in pair_scores:
                continue
            p_i, p_j = ps[k_i], ps[k_j]
            label = int(bool(graph_np[b, p_i, p_j]))
            candidates.append((k_j, pair_scores[key], label))

        n_pos = sum(c[2] for c in candidates)
        if n_pos == 0 or not candidates:
            continue

        ranked = sorted(candidates, key=lambda c: -c[1])
        n_avail = len(ranked)
        recalls = {}
        kept_ratio_5k = None
        for mult in MULTIPLIERS:
            m = min(mult * n_pos, n_avail)
            top_m = ranked[:m]
            recall = sum(c[2] for c in top_m) / n_pos
            recalls[mult] = recall
            if mult == 5:
                kept_ratio_5k = m / n_other_total if n_other_total > 0 else 0.0

        results.append((frag_count_bucket(len(ks)), recalls, kept_ratio_5k))
    return results


def evaluate_recall_sweep(loader, cnn_model, mlp, device, args, rng, cnn_device):
    """Retourne dict[strat] -> dict[bucket] -> {recall@mult moyens, avg_kept_ratio, n_frags}."""
    strategies = args.strategies
    bucket_accum = {
        s: defaultdict(lambda: {m: [] for m in MULTIPLIERS} | {"kept_ratio": []})
        for s in strategies
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch_gpu = {
                k: v.to(cnn_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            out = cnn_model(batch_gpu, return_point_features=True)
            feat_flat = out["point_features"].float().cpu().numpy()
            score_flat = out["coarse_seg_pred"].float().cpu().numpy()
            gt_flat = out["coarse_seg_gt"].long().cpu().numpy()

            valid_np = valid_pcs.cpu().numpy()
            B, P = valid_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_np[b, p]]

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            graph_np = batch["graph"].numpy()

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            cnn_feat_per_k = [feat_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            score_per_k = [score_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                ks = [k for k, _ in ks_ps]
                ps = {k: p for k, p in ks_ps}
                if len(ks) < 2:
                    continue
                scale_per_k = {k: scale_np[b, ps[k]] for k in ks}

                for strat in strategies:
                    agg = aggregate_object_fragments(
                        cnn_feat_per_k, score_per_k, gt_per_k, pc_per_k,
                        {k: scale_per_k[k] for k in ks}, ks_ps, strat, rng,
                    )
                    # aggregate_object_fragments indexe scale_np par k directement
                    pair_scores = build_object_pair_scores(mlp, agg, ks, device)
                    if not pair_scores:
                        continue
                    frag_results = per_fragment_recall(ks, pair_scores, graph_np, b, ps)
                    for bucket, recalls, kept_ratio in frag_results:
                        for mult in MULTIPLIERS:
                            bucket_accum[strat][bucket][mult].append(recalls[mult])
                        if kept_ratio is not None:
                            bucket_accum[strat][bucket]["kept_ratio"].append(kept_ratio)

    # Agrégation finale
    summary = {}
    for strat in strategies:
        summary[strat] = {}
        all_multi = defaultdict(list)
        all_multi_kept = []
        for bucket, accum in bucket_accum[strat].items():
            row = {f"recall@{m}k": float(np.mean(accum[m])) if accum[m] else None
                   for m in MULTIPLIERS}
            row["avg_kept_ratio@5k"] = float(np.mean(accum["kept_ratio"])) if accum["kept_ratio"] else None
            row["n_frags"] = len(accum[1])
            summary[strat][bucket] = row
            if bucket != "2":
                for m in MULTIPLIERS:
                    all_multi[m].extend(accum[m])
                all_multi_kept.extend(accum["kept_ratio"])
        summary[strat]["All multi (3+)"] = {
            f"recall@{m}k": float(np.mean(all_multi[m])) if all_multi[m] else None
            for m in MULTIPLIERS
        } | {
            "avg_kept_ratio@5k": float(np.mean(all_multi_kept)) if all_multi_kept else None,
            "n_frags": len(all_multi[1]),
        }
    return summary


def print_summary(summary, strat):
    print(f"\n── Stratégie: {strat} ──")
    header = f"{'Group':<14} {'Recall@k':>9} {'Recall@2k':>10} {'Recall@3k':>10} {'Recall@5k':>10} {'KeptRatio@5k':>13} {'N_frags':>8}"
    print(header)
    print("-" * len(header))
    order = ["2", "3-5", "6-10", "11+", "All multi (3+)"]
    for bucket in order:
        row = summary[strat].get(bucket)
        if row is None or row.get("n_frags", 0) == 0:
            print(f"{bucket:<14} —")
            continue

        def fmt(v):
            return f"{v*100:>9.1f}%" if v is not None else f"{'—':>9}"

        print(f"{bucket:<14} {fmt(row['recall@1k']):>9} {fmt(row['recall@2k']):>10} "
              f"{fmt(row['recall@3k']):>10} {fmt(row['recall@5k']):>10} "
              f"{fmt(row['avg_kept_ratio@5k']):>13} {row['n_frags']:>8}")


def decide(summary, strat):
    row = summary[strat].get("All multi (3+)", {})
    r3, r5, kept5 = row.get("recall@3k"), row.get("recall@5k"), row.get("avg_kept_ratio@5k")
    if r3 is None or r5 is None:
        return "INDÉTERMINÉ (pas assez de fragments 3+ évalués)"
    degenerate = kept5 is not None and kept5 >= 0.8
    go = (r3 >= 0.90 or r5 >= 0.95) and not degenerate
    if degenerate:
        return (f"NO-GO ({strat}) : shortlist top-5k dégénère vers quasi-toute la liste "
                f"(kept_ratio={kept5*100:.1f}%) — un haut recall ici n'est pas informatif")
    if go:
        return f"GO Phase 6A ({strat}) : Recall@3k={r3*100:.1f}%, Recall@5k={r5*100:.1f}%"
    return (f"NO-GO ({strat}) : Recall@3k={r3*100:.1f}%, Recall@5k={r5*100:.1f}% "
            f"sous les seuils (90%/95%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint CNN Step 15 (figé).")
    parser.add_argument("--mlp_ckpt", required=True,
                        help="Checkpoint MLP 4D sauvegardé via --model_out de phase4d_pair_compatibility.py.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strategies", nargs="+", default=["thresh0.3", "gt"],
                        choices=["gt", "thresh0.3", "random"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device} | split={args.split} | strategies={args.strategies}")

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        categories=args.categories, model_type="cnn",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    dataset = datamodule.train_dataset if args.split == "train" else datamodule.val_dataset
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    cnn_device = torch.device("cpu")
    print(f"Loading CNN checkpoint: {args.ckpt} (CPU)")
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=cnn_device, weights_only=False)
    cnn_model.eval()
    for p in cnn_model.parameters():
        p.requires_grad_(False)

    print(f"Loading 4D MLP checkpoint: {args.mlp_ckpt}")
    mlp_ckpt = torch.load(args.mlp_ckpt, map_location=device, weights_only=False)
    mlp = PairCompatibilityMLP(
        pair_dim=mlp_ckpt["pair_dim"], hidden=mlp_ckpt["hidden_dim"], dropout=mlp_ckpt["dropout"],
    ).to(device)
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()
    print(f"  (best_epoch={mlp_ckpt.get('best_epoch')}, best_auc_val={mlp_ckpt.get('best_auc_val')})")

    summary = evaluate_recall_sweep(loader, cnn_model, mlp, device, args, rng, cnn_device)

    for strat in args.strategies:
        print_summary(summary, strat)

    print("\n── Décision (critère : Recall@3k>=90% OU Recall@5k>=95%, sur objets 3+ fragments) ──")
    for strat in args.strategies:
        print(f"  {decide(summary, strat)}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump({
                "config": vars(args),
                "summary": summary,
                "decisions": {s: decide(summary, s) for s in args.strategies},
            }, f, indent=2)
        print(f"\nJSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
