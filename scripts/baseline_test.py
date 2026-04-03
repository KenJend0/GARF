"""
scripts/baseline_test.py
=========================
Teste GARF sur un petit sous-ensemble du dataset pour vérifier que les résultats
reproduisent ceux du papier AVANT toute modification.

Ce script sert de "baseline de référence" : si tes métriques ici correspondent
aux chiffres du papier, tu peux commencer à modifier le code en toute confiance.

Usage :
    python scripts/baseline_test.py \
        --ckpt   /path/to/garf.ckpt \
        --data   /path/to/breaking_bad.h5 \
        --n      20 \
        --split  test \
        --category everyday

Arguments :
    --ckpt      : checkpoint GARF pré-entraîné (téléchargé depuis le Model Zoo du README)
    --data      : fichier HDF5 du dataset Breaking Bad
    --n         : nombre de fragments à tester (défaut : 20, ~2 minutes)
    --split     : 'test' ou 'val'
    --category  : 'everyday', 'artifact' ou 'all'

Métriques affichées (identiques au papier) :
    Part Accuracy (PA)    : % de fragments correctement placés (seuil 0.01)
    RMSE Rotation  (°)    : erreur angulaire en degrés
    RMSE Translation      : erreur de translation (unités normalisées)
    Shape Chamfer Distance: distance de Chamfer sur la forme assemblée

Résultats attendus (papier GARF, Breaking Bad everyday, test split) :
    PA ≈ 0.72 | RMSE_R ≈ 15° | RMSE_T ≈ 0.02 | CD ≈ 0.001
    (Ces chiffres varient selon le split et la catégorie)
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# ── Permet de lancer depuis la racine du projet ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline test GARF")
    parser.add_argument("--ckpt",     required=True,  help="Chemin vers le checkpoint GARF (.ckpt)")
    parser.add_argument("--data",     required=True,  help="Chemin vers breaking_bad.h5")
    parser.add_argument("--n",        type=int, default=20, help="Nombre d'exemples à tester")
    parser.add_argument("--split",    default="test", choices=["train", "val", "test"])
    parser.add_argument("--category", default="everyday", choices=["everyday", "artifact", "all"])
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(ckpt_path: str, device: str):
    """
    Charge le modèle GARF depuis un checkpoint Lightning.
    Le checkpoint contient la config complète (hydra l'a sérialisée).
    """
    from assembly.models.denoiser.denoiser_flow_matching import DenoiserFlowMatching

    print(f"  Chargement du modèle depuis : {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Si c'est un checkpoint Lightning standard
    if "state_dict" in checkpoint:
        # On a besoin de la config pour reconstruire le modèle
        # Essaie avec hydra si la config est disponible dans le ckpt
        hparams = checkpoint.get("hyper_parameters", {})
        print(f"  Hyper-paramètres trouvés : {list(hparams.keys()) if hparams else 'aucun'}")

    model = DenoiserFlowMatching.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=False,
    )
    model.eval()
    model.to(device)
    print(f"  Modèle chargé sur {device}")
    return model


def load_dataset(data_path: str, split: str, category: str, n: int):
    """
    Charge un petit sous-ensemble du dataset Breaking Bad.
    """
    from assembly.data.breaking_bad.uniform import BreakingBadUniform

    print(f"  Chargement du dataset : {data_path}")
    print(f"  Split={split}, Category={category}, N={n}")

    dataset = BreakingBadUniform(
        split=split,
        data_root=data_path,
        category=category,
        min_parts=2,
        max_parts=20,
        num_points_to_sample=1000,   # réduit pour aller vite
    )

    # Sous-ensemble de n exemples
    total = len(dataset)
    print(f"  Dataset complet : {total} exemples — on teste sur {min(n, total)}")
    indices = list(range(min(n, total)))
    return dataset, indices


@torch.no_grad()
def run_inference(model, dataset, indices, device):
    """
    Lance l'inference du denoiser sur chaque exemple et collecte les métriques.
    """
    from torch.utils.data import DataLoader, Subset
    from assembly.data.breaking_bad.base import BreakingBadBase

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=BreakingBadBase.collate_fn,
    )

    all_pa    = []
    all_rmse_r = []
    all_rmse_t = []
    all_cd    = []

    print(f"\n  Lancement de l'inference sur {len(indices)} exemples...\n")

    for i, batch in enumerate(loader):
        # Transfère sur le device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        name = batch["name"][0]
        num_parts = batch["num_parts"][0].item()

        # Lance un step de test Lightning (gère l'inference complète + métriques)
        model.test_step(batch, i)

        # Récupère les dernières métriques accumulées
        if model.acc_list:
            pa    = model.acc_list[-1].mean().item()
            rmse_r = model.rmse_r_list[-1].mean().item()
            rmse_t = model.rmse_t_list[-1].mean().item()
            cd    = model.cd_list[-1].mean().item()

            all_pa.append(pa)
            all_rmse_r.append(rmse_r)
            all_rmse_t.append(rmse_t)
            all_cd.append(cd)

            status = "✓" if pa > 0.5 else "✗"
            print(f"  [{i+1:3d}/{len(indices)}] {status} {name:<40} "
                  f"PA={pa:.3f}  R={rmse_r:.1f}°  T={rmse_t:.4f}  CD={cd:.5f}  "
                  f"({num_parts} fragments)")

    return all_pa, all_rmse_r, all_rmse_t, all_cd


def print_summary(all_pa, all_rmse_r, all_rmse_t, all_cd):
    """Affiche le résumé des métriques avec comparaison aux chiffres du papier."""
    if not all_pa:
        print("\n  Aucun résultat collecté.")
        return

    mean_pa    = np.mean(all_pa)
    mean_rmse_r = np.mean(all_rmse_r)
    mean_rmse_t = np.mean(all_rmse_t)
    mean_cd    = np.mean(all_cd)

    print("\n" + "=" * 65)
    print("  RÉSUMÉ DES MÉTRIQUES")
    print("=" * 65)
    print(f"  {'Métrique':<30} {'Obtenu':>10}   {'Papier (everyday)':>18}")
    print("-" * 65)
    print(f"  {'Part Accuracy (PA)':<30} {mean_pa:>10.3f}   {'~0.72':>18}")
    print(f"  {'RMSE Rotation (°)':<30} {mean_rmse_r:>10.2f}   {'~15°':>18}")
    print(f"  {'RMSE Translation':<30} {mean_rmse_t:>10.4f}   {'~0.02':>18}")
    print(f"  {'Shape Chamfer Distance':<30} {mean_cd:>10.5f}   {'~0.001':>18}")
    print("-" * 65)
    print(f"  Testé sur {len(all_pa)} exemples")
    print()

    # Avertissement si les résultats semblent très différents
    if mean_pa < 0.3:
        print("  ⚠️  Part Accuracy très faible — vérifie le checkpoint ou le dataset.")
    elif mean_pa > 0.6:
        print("  ✓  Part Accuracy cohérente avec le papier.")

    # Sauvegarde CSV pour garder une trace
    out_path = Path("baseline_results.csv")
    with open(out_path, "w") as f:
        f.write("pa,rmse_r,rmse_t,cd\n")
        for pa, r, t, cd in zip(all_pa, all_rmse_r, all_rmse_t, all_cd):
            f.write(f"{pa:.4f},{r:.4f},{t:.6f},{cd:.6f}\n")
    print(f"  Résultats sauvegardés dans : {out_path.absolute()}\n")


def main():
    args = parse_args()

    print("\n========== GARF — Test de baseline ==========\n")

    # Vérifie que les fichiers existent
    if not os.path.exists(args.ckpt):
        print(f"ERREUR : checkpoint introuvable : {args.ckpt}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"ERREUR : dataset introuvable : {args.data}")
        sys.exit(1)

    print(f"  Device : {args.device}")

    # Charge le modèle
    print("\n[ Chargement du modèle ]")
    try:
        model = load_model(args.ckpt, args.device)
    except Exception as e:
        print(f"\n  ERREUR lors du chargement du modèle : {e}")
        print("  Astuce : le script attend un checkpoint Lightning avec la config hydra intégrée.")
        print("  Télécharge le checkpoint depuis le Model Zoo du README.md.")
        sys.exit(1)

    # Charge le dataset
    print("\n[ Chargement du dataset ]")
    dataset, indices = load_dataset(args.data, args.split, args.category, args.n)

    # Inference
    print("\n[ Inference ]")
    all_pa, all_rmse_r, all_rmse_t, all_cd = run_inference(
        model, dataset, indices, args.device
    )

    # Résumé
    print_summary(all_pa, all_rmse_r, all_rmse_t, all_cd)


if __name__ == "__main__":
    main()
