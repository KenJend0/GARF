"""
scripts/phase0_check_pose_convention.py
========================================
Phase 0 du plan de réassemblage léger (voir PLAN_REASSEMBLY_MODULE.md).

Vérifie empiriquement la convention de pose stockée par le dataloader
Breaking Bad, dérivée par lecture de assembly/data/transform.py :

    pointclouds_gt[part] ≈ R(quaternion) @ pointclouds[part] + translation

(formule sans le facteur d'échelle, valable pour BreakingBadUniform —
BreakingBadWeighted ajoute un facteur "scale" supplémentaire, voir
PLAN_REASSEMBLY_MODULE.md).

Pour chaque objet chargé :
  1. Reconstruit pointclouds_gt à partir de (pointclouds, quaternions, translations).
  2. Mesure le résidu point-à-point (l'ordre des points est préservé par fragment,
     donc une comparaison directe indice-à-indice est valide).
  3. Vérifie indépendamment, via nearest-neighbor (donc insensible à l'ordre), que les
     points de fracture de deux fragments adjacents (graph[i,j]=True) sont bien proches
     après reconstruction — un test géométrique qui ne dépend pas de la convention de
     correspondance d'indices.

Usage (sur le serveur, données HDF5 uniquement disponibles là-bas) :
    python scripts/phase0_check_pose_convention.py \
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
        --category everyday --split val --num_objects 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assembly.data.breaking_bad import BreakingBadUniform


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """scipy attend [x, y, z, w] ; le dataloader stocke [w, x, y, z] (scalar-first)."""
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_matrix()


def reconstruct_object(pointclouds, quaternions, translations, num_parts):
    """Applique R(quat) @ p + t à chaque fragment. Retourne une liste de (N,3) reconstruits."""
    reconstructed = []
    for part_idx in range(num_parts):
        rot_mat = quat_wxyz_to_rotmat(quaternions[part_idx])
        pc = pointclouds[part_idx]  # (N, 3), input désassemblé
        rec = (rot_mat @ pc.T).T + translations[part_idx][None, :]
        reconstructed.append(rec)
    return reconstructed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--category", default="everyday")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_objects", type=int, default=3)
    parser.add_argument("--num_points_to_sample", type=int, default=5000)
    parser.add_argument(
        "--fracture_neighbor_threshold",
        type=float,
        default=0.05,
        help="Distance max (unités normalisées) pour considérer deux points de fracture "
        "adjacents comme 'proches' après reconstruction.",
    )
    args = parser.parse_args()

    dataset = BreakingBadUniform(
        split=args.split,
        data_root=args.data_root,
        category=args.category,
        num_points_to_sample=args.num_points_to_sample,
        mesh_sample_strategy="poisson",
    )
    print(f"Dataset chargé : {len(dataset)} objets disponibles ({args.category}/{args.split})")

    for obj_idx in range(min(args.num_objects, len(dataset))):
        sample = dataset[obj_idx]
        name = sample["name"]
        num_parts = sample["num_parts"]
        pointclouds = sample["pointclouds"]              # (max_parts, N, 3) — input désassemblé
        pointclouds_gt = sample["pointclouds_gt"]          # (max_parts, N, 3) — assemblé GT
        quaternions = sample["quaternions"]                # (max_parts, 4) — [w,x,y,z]
        translations = sample["translations"]               # (max_parts, 3)
        fracture_surface_gt = sample["fracture_surface_gt"]  # (max_parts, N)
        graph = sample["graph"]                              # (max_parts, max_parts) bool

        print(f"\n=== Objet {obj_idx}: {name} ({num_parts} fragments) ===")

        # --- Test 1 : résidu de reconstruction point-à-point ---
        reconstructed = reconstruct_object(pointclouds, quaternions, translations, num_parts)
        residuals = []
        for part_idx in range(num_parts):
            err = np.linalg.norm(reconstructed[part_idx] - pointclouds_gt[part_idx], axis=1)
            residuals.append(err.mean())
        mean_residual = float(np.mean(residuals))
        print(f"  Résidu reconstruction (moyenne par fragment): {residuals}")
        print(f"  Résidu moyen objet: {mean_residual:.6f}")

        # --- Test 2 : proximité géométrique des points de fracture adjacents ---
        # Indépendant de l'ordre des points, sert de garde-fou si le test 1 a un biais subtil.
        neighbor_pairs = [
            (i, j)
            for i in range(num_parts)
            for j in range(i + 1, num_parts)
            if graph[i, j]
        ]
        for i, j in neighbor_pairs:
            frac_i = reconstructed[i][fracture_surface_gt[i][:len(reconstructed[i])].astype(bool)]
            frac_j = reconstructed[j][fracture_surface_gt[j][:len(reconstructed[j])].astype(bool)]
            if len(frac_i) == 0 or len(frac_j) == 0:
                print(f"  Paire ({i},{j}): pas de points fracture étiquetés, skip")
                continue
            tree_j = cKDTree(frac_j)
            dists, _ = tree_j.query(frac_i)
            frac_close_ratio = float((dists < args.fracture_neighbor_threshold).mean())
            print(
                f"  Paire adjacente ({i},{j}): {len(frac_i)} vs {len(frac_j)} pts fracture, "
                f"dist NN moyenne={dists.mean():.4f}, "
                f"ratio proche(<{args.fracture_neighbor_threshold})={frac_close_ratio:.2%}"
            )

        if mean_residual < 1e-3:
            print("  => Convention CONFIRMÉE (résidu quasi nul).")
        else:
            print(
                "  => Résidu élevé : convention probablement inversée ou facteur 'scale' "
                "manquant (cf. BreakingBadWeighted). Vérifier avec --sample_method weighted "
                "ou inverser la rotation (R(quat).T) avant de conclure."
            )


if __name__ == "__main__":
    main()
