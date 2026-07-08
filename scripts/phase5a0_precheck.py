"""
scripts/phase5a0_precheck.py
======================================
Phase 5A.0 du plan de réassemblage (voir PLAN_REASSEMBLY_MODULE.md, section
"Phase 5A.0 -- Pre-check depth-map matching").

Vérifie deux conditions AVANT toute implémentation du matching depth-map (Phase 5A) :

  V1 — Population :
      Combien d'objets à exactement 2 fragments dans everyday/train et everyday/val ?
      Cible indicative : >= 100 paires positives (= objets 2-fragments) en val.

  V2 — Planéité des faces de fracture :
      Sur les objets à 2 fragments, calcule le score de planéité PCA par fragment :
        planarity = lambda_min / (lambda_1 + lambda_2 + lambda_3)
      où les lambda sont les valeurs propres de la matrice de covariance des points
      fracture GT dans le repère reconstruit (scale réappliqué, coordonnées brutes
      après réapplication : raw = pointclouds[i] * scale[i]).
      planarity ≈ 0 → face quasi-plane (bon pour une depth map) ;
      planarity ≈ 0.33 → isotrope (profil sphérique, mauvaise représentation 2D).
      Cible : médiane < 0.10 sur les faces des objets 2-fragments.

Arbre de décision :
    n_2frag_val >= 100  ET  median_planarity < 0.10
        OUI → Phase 5A depth-map matching prioritaire
        NON → revenir à Phase 4D compatibility classifier

Rapporte aussi :
    - Distribution de n_frac_points_per_face (pts fracture GT par fragment)
    - Taux d'objets exclus (une face a < 10 pts fracture → PCA bruitée)
    - Distribution totale des num_parts (pour situer 2-fragments dans le dataset)

Usage (sur le serveur) :
    # Rapide : val uniquement, pas de limit (rares objets 2-fragments)
    python scripts/phase5a0_precheck.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val \\
        --summary_json /tmp/student7/phase5a0_precheck_val.json

    # Complet : train + val pour avoir la distribution totale
    python scripts/phase5a0_precheck.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split train \\
        --summary_json /tmp/student7/phase5a0_precheck_train.json

Note : le modèle CNN est utilisé uniquement pour extraire les labels GT fracture
(coarse_seg_gt) dans le bon format flat — les prédictions (coarse_seg_pred) sont
ignorées. Pas de GPU requis si --device cpu, mais le forward CNN est plus rapide
sur GPU même en inférence pure GT.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate
from scripts.analyze_errors import load_config_and_model
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list
from torch.utils.data import DataLoader


MIN_FRAC_POINTS = 10   # moins de N pts fracture → face trop éparse pour une PCA fiable


def compute_planarity(pts: np.ndarray):
    """Planarity score = lambda_min / sum(lambdas), via covariance PCA.
    Returns None si < 3 points (PCA non définie).
    Proche de 0 → quasi-plan ; proche de 0.33 → isotrope/sphérique."""
    if len(pts) < 3:
        return None
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)   # trié croissant
    eigenvalues = np.maximum(eigenvalues, 0.0)  # sécurité numérique
    total = eigenvalues.sum()
    if total < 1e-12:
        return None
    return float(eigenvalues[0] / total)


def percentile_summary(values, label):
    """Print and return a dict with distribution stats."""
    if not values:
        print(f"  {label}: (aucune valeur)")
        return {}
    a = np.array(values, dtype=float)
    d = {
        "n": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p0": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p100": float(a.max()),
    }
    print(f"  {label}: n={d['n']}, "
          f"mean={d['mean']:.4f}, std={d['std']:.4f}, "
          f"p25={d['p25']:.4f}, p50={d['p50']:.4f}, "
          f"p75={d['p75']:.4f}, p90={d['p90']:.4f}, "
          f"[{d['p0']:.4f}, {d['p100']:.4f}]")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="CNN checkpoint (.ckpt)")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="0 = tout le split (recommandé : objets 2-frags sont rares)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()
    args.model_type = "cnn"  # load_config_and_model requiert cet attribut

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    cfg = load_config_and_model(args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    dataset = (
        datamodule.val_dataset if args.split == "val"
        else datamodule.train_dataset if args.split == "train"
        else datamodule.test_dataset
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,                        # pas besoin de shuffle : on veut tout compter
        collate_fn=datamodule.dataset_cls.collate_fn,
    )

    # ── Model (utilisé uniquement pour gt_flat, pred ignorée) ─────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # ── Accumulateurs ─────────────────────────────────────────────────────────
    num_parts_all_objects = []        # distribution num_parts sur TOUS les objets vus
    n_2frag_objects = 0
    n_2frag_excluded = 0              # un fragment a < MIN_FRAC_POINTS pts fracture

    planarity_scores = []             # une valeur par fragment de paire 2-frag retenu
    frac_points_counts = []           # nb pts fracture par fragment 2-frag retenu

    object_names_2frag = []

    print(f"\nScan {args.categories}/{args.split} — comptage objets 2-fragments + planéité GT...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}...")

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K == 0:
                continue
            frag_sizes = [f.shape[0] for f in frag_list]

            # GT labels uniquement — pred ignorée
            out = model(batch_gpu)
            gt_flat = out["coarse_seg_gt"].long().cpu().numpy()

            valid_pcs_np = valid_pcs.cpu().numpy()   # (B, P) bool
            B, P = valid_pcs_np.shape
            bp_pairs = [(b, p) for b in range(B) for p in range(P) if valid_pcs_np[b, p]]

            scale_np = batch["scale"].numpy()         # (B, P) ou (B, P, 1)
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]

            names = batch["name"]                     # list[str], longueur B

            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            gt_per_k = [gt_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_local_per_k = [frag_list[k].cpu().numpy() for k in range(K)]

            # Grouper les fragments k par objet b
            object_to_ks = defaultdict(list)
            for k, (b, p) in enumerate(bp_pairs):
                object_to_ks[b].append((k, p))

            for b, ks_ps in object_to_ks.items():
                n_frags = len(ks_ps)
                num_parts_all_objects.append(n_frags)

                if n_frags != 2:
                    continue

                # ── Objet à exactement 2 fragments ──────────────────────────
                n_2frag_objects += 1
                object_names_2frag.append(names[b])

                excluded = False
                for k, p in ks_ps:
                    scale_k = scale_np[b, p]                     # (1,) ou scalaire
                    raw_k = pc_local_per_k[k] * scale_k          # coordonnées scale-corrigées
                    gt_k = gt_per_k[k]                           # labels fracture GT (0/1)

                    frac_pts = raw_k[gt_k == 1]                  # points fracture de ce fragment
                    n_frac = len(frac_pts)
                    frac_points_counts.append(n_frac)

                    if n_frac < MIN_FRAC_POINTS:
                        excluded = True   # face trop éparse, on ne calcule pas la planéité
                        continue

                    plan = compute_planarity(frac_pts)
                    if plan is not None:
                        planarity_scores.append(plan)

                if excluded:
                    n_2frag_excluded += 1

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5A.0 — PRE-CHECK RÉSULTATS")
    print("=" * 60)

    # Distribution num_parts sur tout le dataset scanné
    print(f"\nDistribution num_parts (tous objets scannés, n={len(num_parts_all_objects)}) :")
    if num_parts_all_objects:
        parts_arr = np.array(num_parts_all_objects)
        for n_f in sorted(set(parts_arr.tolist())):
            cnt = int((parts_arr == n_f).sum())
            pct = 100.0 * cnt / len(parts_arr)
            tag = "  <-- cible 5A" if n_f == 2 else ""
            print(f"  {n_f} fragments : {cnt:5d} objets ({pct:5.1f}%){tag}")

    print(f"\n── V1 : Population 2-fragments ──")
    print(f"  Objets à exactement 2 fragments : {n_2frag_objects}")
    print(f"  dont exclus (une face < {MIN_FRAC_POINTS} pts fracture GT) : {n_2frag_excluded}")
    n_retained = n_2frag_objects - n_2frag_excluded
    print(f"  Objets retenus pour la planéité : {n_retained}")
    verdict_v1 = "OK" if n_2frag_objects >= 100 else "KO"
    print(f"  Critère >= 100 objets 2-frags : {verdict_v1} (n={n_2frag_objects})")

    print(f"\n── V2 : Planéité des faces de fracture (GT mask) ──")
    planarity_summary = percentile_summary(planarity_scores, "planarity")
    frac_pts_summary = percentile_summary(frac_points_counts, "n_frac_pts_per_face")
    median_plan = planarity_summary.get("p50", 1.0)
    verdict_v2 = "OK" if median_plan < 0.10 else "KO"
    print(f"  Critère médiane planéité < 0.10 : {verdict_v2} (médiane={median_plan:.4f})")

    print(f"\n── DÉCISION ──")
    if verdict_v1 == "OK" and verdict_v2 == "OK":
        decision = "Phase 5A depth-map matching PRIORITAIRE (4D en fallback)"
    else:
        reasons = []
        if verdict_v1 != "OK":
            reasons.append(f"population insuffisante (n={n_2frag_objects} < 100)")
        if verdict_v2 != "OK":
            reasons.append(f"faces trop courbes (médiane planéité={median_plan:.4f} >= 0.10)")
        decision = f"Revenir à Phase 4D ({', '.join(reasons)})"
    print(f"  → {decision}")

    # Quelques exemples d'objets 2-fragments
    if object_names_2frag:
        print(f"\n  Exemples d'objets 2-fragments : {object_names_2frag[:10]}")

    # ── Export JSON ───────────────────────────────────────────────────────────
    summary = {
        "config": vars(args),
        "n_objects_scanned": len(num_parts_all_objects),
        "num_parts_distribution": {
            str(n): int((np.array(num_parts_all_objects) == n).sum())
            for n in sorted(set(num_parts_all_objects))
        } if num_parts_all_objects else {},
        "v1_population": {
            "n_2frag_objects": n_2frag_objects,
            "n_2frag_excluded": n_2frag_excluded,
            "n_2frag_retained": n_retained,
            "verdict": verdict_v1,
            "threshold": 100,
        },
        "v2_planarity": {
            "planarity_stats": planarity_summary,
            "frac_points_stats": frac_pts_summary,
            "verdict": verdict_v2,
            "threshold_median": 0.10,
            "min_frac_points_required": MIN_FRAC_POINTS,
        },
        "decision": decision,
        "example_2frag_names": object_names_2frag[:20],
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON sauvegardé : {args.summary_json}")


if __name__ == "__main__":
    main()
