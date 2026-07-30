"""
scripts/phase8_split_npz.py
==============================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — découpe un `.npz` de
`phase8_build_regressor_dataset.py` en deux (train/val internes),
UNIQUEMENT pour la catégorie `everyday` : elle n'a pas de split `test`
dans le dataset (seul `artifact` en a un, cf. `assembly/data/breaking_bad/
base.py`), et son split `train` ne conserve pas les maillages (nécessaires
pour le zoom) -- `val` est donc le SEUL split `everyday` disponible avec
maillages. Contournement pragmatique (2026-07-30, décidé avec
l'utilisateur) : découper ce `val` 80/20 pour ce premier prototype.

Chaque ligne du `.npz` correspond à UN objet à 2 fragments (une seule paire
par objet dans ces scripts) -- découper par LIGNE revient donc à découper
par OBJET, pas de fuite au niveau fragment/pièce.

Usage :
    python scripts/phase8_split_npz.py \\
        --in_npz /tmp/student7/phase8_dataset_gt_val.npz \\
        --out_train /tmp/student7/phase8_dataset_gt_train.npz \\
        --out_val /tmp/student7/phase8_dataset_gt_valsplit.npz \\
        --train_frac 0.8 --seed 42
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_npz", required=True)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_val", required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.in_npz)
    keys = list(data.keys())
    n = len(data[keys[0]])

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(round(args.train_frac * n))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    np.savez_compressed(args.out_train, **{k: data[k][train_idx] for k in keys})
    np.savez_compressed(args.out_val, **{k: data[k][val_idx] for k in keys})

    print(f"{n} paires -> train={len(train_idx)} ({args.out_train}), "
          f"val={len(val_idx)} ({args.out_val})")


if __name__ == "__main__":
    main()
