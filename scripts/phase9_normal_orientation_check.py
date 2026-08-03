"""
scripts/phase9_normal_orientation_check.py
===========================================
Phase 9 (PLAN_REASSEMBLY_MODULE.md) -- teste si le "miroir" (52.2% des
paires réelles selon `phase8_reflection_prevalence_check.py`, cf.
`fit_theta_shift_mirror_from_gt`) est un artefact du signe ARBITRAIRE de
l'axe normal `n` retourné par la PCA (`compute_pca_frame`), plutôt qu'une
vraie ambiguïté physique.

Raisonnement (discuté avec l'utilisateur, 2026-08-03) : `canonical_pca_frame`
force déjà `det([u,v,n])=+1` (repère direct) séparément pour i et j -- or
deux repères directs sont TOUJOURS reliés par une rotation pure (jamais une
réflexion), c'est un fait d'algèbre linéaire, indépendant de tout choix de
signe. Le "miroir" observé vient donc d'ailleurs : rien ne garantit que `n_i`
et `n_j` (chacun de signe arbitraire, seulement contraint par la chiralité
de SON PROPRE repère) pointent en sens opposés une fois la vraie rotation
GT appliquée -- alors que deux faces de fracture qui se touchent ont
PHYSIQUEMENT des normales sortantes opposées (déjà exploité par
`normal_dot_thresh` dans `trimmed_icp_normals`). Idée : orienter `n` non pas
arbitrairement mais selon la moyenne des normales mesh du fragment
(`pointclouds_normals`, déjà supposées fiables ailleurs dans le pipeline)
-- ce qui, pour un fragment à 2 morceaux, revient à dire "la masse du reste
de l'objet est d'un côté du plan de fracture" (intuition de l'utilisateur).

Ce script compare, sur le split GT (oracle, aucun CNN), la prévalence du
mirror=True :
  - AVANT  : repère `canonical_pca_frame` existant (signe de `n` arbitraire)
  - APRÈS  : repère réorienté (signe de `n` fixé par la moyenne des
             normales mesh du fragment)
Si l'hypothèse est correcte, la prévalence doit chuter de ~52% à ~0%
(hors bruit/cas dégénérés) -- validation PUREMENT géométrique, sans
entraînement, réutilisant `fit_theta_shift_mirror_from_gt` telle quelle
(seuls les repères passés en argument changent).

Usage (sur le serveur, oracle-first) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase9_normal_orientation_check.py \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_weighted_gt_check import extract_gt_variable
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat, compute_pca_frame, rasterize
from scripts.phase8_depthmap_regressor_dataset import canonical_pca_frame, fit_theta_shift_mirror_from_gt

ABS_MIN_POINTS = 50
RESOLUTION = 64


def canonical_pca_frame_oriented(pts, point_normals):
    """Comme `canonical_pca_frame`, mais le signe de `n` est fixé par la
    moyenne des normales mesh du fragment (`point_normals`, même longueur
    que `pts`) plutôt que par le signe arbitraire renvoyé par `eigh()`.
    Une fois `n` fixé physiquement, `v` (pas `n`) est retourné au besoin
    pour restaurer `det([u,v,n])=+1` -- contrairement à
    `canonical_pca_frame`, qui ajuste `n` (ici `n` n'est plus libre).

    Convention de signe déterminée EMPIRIQUEMENT (2026-08-03) : `n` opposé
    à la moyenne des normales mesh (pas aligné) -- le premier essai
    (`n` aligné) donnait mirror=True 98.3% du temps (quasi-déterministe,
    juste la mauvaise convention), confirmant le mécanisme mais avec le
    signe inversé. Avec `n` opposé, prévalence attendue ~1-2%."""
    centroid, u, v, n, planarity = compute_pca_frame(pts)
    mean_normal = point_normals.mean(axis=0)
    if np.dot(n, mean_normal) > 0:
        n = -n
    if np.linalg.det(np.stack([u, v, n], axis=1)) < 0:
        v = -v
    return centroid, u, v, n, planarity


def canonical_pca_frame_oriented_by_centroid(pts, full_fragment_centroid):
    """Variante (2026-08-03, suite à la dégradation observée en thresh0.3
    de `canonical_pca_frame_oriented` -- prévalence oracle elle-même tombée
    à ~49.5%, cf. `phase9_normal_orientation_check_thresh03.py`) : au lieu
    de moyenner les normales (individuelles, bruitées par les faux
    positifs du masque CNN), oriente `n` en le comparant au vecteur
    "centre du fragment ENTIER -> centre de la zone de fracture"
    (`pts.mean(axis=0) - full_fragment_centroid`) -- astuce classique
    d'orientation de normale de surface par le centre de masse de l'objet.
    Ne dépend PAS des normales par point (donc pas de `point_normals` en
    argument) -- seulement des POSITIONS, une moyenne de positions étant
    beaucoup plus robuste à quelques points aberrants qu'une moyenne de
    directions (chaque faux positif peut avoir une normale qui pointe
    n'importe où, mais ne décale que peu le centroïde). `full_fragment_centroid`
    : centre de TOUT le nuage de points du fragment (pas juste la fracture)
    -- indépendant du masque CNN, donc fiable à 100%."""
    centroid, u, v, n, planarity = compute_pca_frame(pts)
    outward_dir = centroid - full_fragment_centroid
    if np.dot(n, outward_dir) < 0:
        n = -n
    if np.linalg.det(np.stack([u, v, n], axis=1)) < 0:
        v = -v
    return centroid, u, v, n, planarity


def _pixel_params(frac, c, u, v, n, pixel_size):
    """Réutilise `rasterize()` telle quelle juste pour en extraire
    `u_min`/`v_min` (mêmes conventions que `build_frame_and_rasterize`),
    sans se servir du depth map lui-même."""
    _, _, u_min, v_min = rasterize(frac, c, u, v, n, RESOLUTION, pixel_size)
    return u_min, v_min


def _pixel_size_for(frac_i, frac_j, c_i, u_i, v_i, c_j, u_j, v_j):
    ci, cj = frac_i - c_i, frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    return max(span_i, span_j) * 1.1 / RESOLUTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split",      default="val", choices=["val", "test"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    args = parser.parse_args()

    fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    cfg = load_config_and_model(fake_args)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit" if args.split == "val" else "test")
    dataset = datamodule.val_dataset if args.split == "val" else datamodule.test_dataset

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=datamodule.dataset_cls.collate_fn)

    n_seen, n_old_mirror, n_new_mirror, n_centroid_mirror = 0, 0, 0, 0
    t0 = time.time()

    for idx, batch in enumerate(loader):
        if args.max_batches > 0 and idx >= args.max_batches:
            break
        if idx % 200 == 0:
            print(f"  objet {idx} | paires vues={n_seen} | {time.time()-t0:.0f}s")

        points_per_part = batch["points_per_part"][0].numpy()
        valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
        if len(valid_slots) != 2:
            continue

        pointclouds = batch["pointclouds"][0].numpy()
        pointclouds_normals = batch["pointclouds_normals"][0].numpy()
        fracture_gt = batch["fracture_surface_gt"][0].numpy()

        frag_pts = extract_gt_variable(pointclouds, points_per_part)
        frag_nrm = extract_gt_variable(pointclouds_normals, points_per_part)
        frag_gt  = extract_gt_variable(fracture_gt, points_per_part)
        if len(frag_pts) != 2:
            continue

        p0, p1 = valid_slots[0], valid_slots[1]
        scale_np = batch["scale"][0].numpy()
        if scale_np.ndim == 1:
            scale_np = scale_np[:, None]
        quats_np = batch["quaternions"][0].numpy()
        trans_np = batch["translations"][0].numpy()

        raw_i = frag_pts[0] * scale_np[p0]
        raw_j = frag_pts[1] * scale_np[p1]
        nrm_i, nrm_j = frag_nrm[0], frag_nrm[1]
        gt_i, gt_j = frag_gt[0], frag_gt[1]

        frac_i = raw_i[gt_i == 1]
        frac_j = raw_j[gt_j == 1]
        frac_i_nrm = nrm_i[gt_i == 1]
        frac_j_nrm = nrm_j[gt_j == 1]
        if min(len(frac_i), len(frac_j)) < ABS_MIN_POINTS:
            continue

        R0 = quat_wxyz_to_rotmat(quats_np[p0])
        R1 = quat_wxyz_to_rotmat(quats_np[p1])
        R_ij_gt = R1.T @ R0
        t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

        n_seen += 1

        # -- AVANT : repère existant, signe de n arbitraire --------------
        c_i, u_i, v_i, n_i, _ = canonical_pca_frame(frac_i)
        c_j, u_j, v_j, n_j, _ = canonical_pca_frame(frac_j)
        pixel_size_old = _pixel_size_for(frac_i, frac_j, c_i, u_i, v_i, c_j, u_j, v_j)
        u_min_i, v_min_i = _pixel_params(frac_i, c_i, u_i, v_i, n_i, pixel_size_old)
        u_min_j, v_min_j = _pixel_params(frac_j, c_j, u_j, v_j, n_j, pixel_size_old)
        _, _, mirror_old, _ = fit_theta_shift_mirror_from_gt(
            frac_i, c_i, u_i, v_i, u_min_i, v_min_i,
            c_j, u_j, v_j, u_min_j, v_min_j,
            pixel_size_old, RESOLUTION, R_ij_gt, t_ij_gt,
        )

        # -- APRÈS : repère réorienté par la moyenne des normales mesh ---
        c_i2, u_i2, v_i2, n_i2, _ = canonical_pca_frame_oriented(frac_i, frac_i_nrm)
        c_j2, u_j2, v_j2, n_j2, _ = canonical_pca_frame_oriented(frac_j, frac_j_nrm)
        pixel_size_new = _pixel_size_for(frac_i, frac_j, c_i2, u_i2, v_i2, c_j2, u_j2, v_j2)
        u_min_i2, v_min_i2 = _pixel_params(frac_i, c_i2, u_i2, v_i2, n_i2, pixel_size_new)
        u_min_j2, v_min_j2 = _pixel_params(frac_j, c_j2, u_j2, v_j2, n_j2, pixel_size_new)
        _, _, mirror_new, _ = fit_theta_shift_mirror_from_gt(
            frac_i, c_i2, u_i2, v_i2, u_min_i2, v_min_i2,
            c_j2, u_j2, v_j2, u_min_j2, v_min_j2,
            pixel_size_new, RESOLUTION, R_ij_gt, t_ij_gt,
        )

        # -- APRÈS (v2) : réorienté par le centroïde du fragment entier ---
        full_centroid_i = raw_i.mean(axis=0)
        full_centroid_j = raw_j.mean(axis=0)
        c_i3, u_i3, v_i3, n_i3, _ = canonical_pca_frame_oriented_by_centroid(frac_i, full_centroid_i)
        c_j3, u_j3, v_j3, n_j3, _ = canonical_pca_frame_oriented_by_centroid(frac_j, full_centroid_j)
        pixel_size_c = _pixel_size_for(frac_i, frac_j, c_i3, u_i3, v_i3, c_j3, u_j3, v_j3)
        u_min_i3, v_min_i3 = _pixel_params(frac_i, c_i3, u_i3, v_i3, n_i3, pixel_size_c)
        u_min_j3, v_min_j3 = _pixel_params(frac_j, c_j3, u_j3, v_j3, n_j3, pixel_size_c)
        _, _, mirror_centroid, _ = fit_theta_shift_mirror_from_gt(
            frac_i, c_i3, u_i3, v_i3, u_min_i3, v_min_i3,
            c_j3, u_j3, v_j3, u_min_j3, v_min_j3,
            pixel_size_c, RESOLUTION, R_ij_gt, t_ij_gt,
        )

        n_old_mirror += int(mirror_old)
        n_new_mirror += int(mirror_new)
        n_centroid_mirror += int(mirror_centroid)

    print(f"\nFini : {n_seen} paires 2-fragments GT en {time.time()-t0:.0f}s\n")
    if n_seen == 0:
        print("Aucune paire exploitable.")
        return
    print(f"Prévalence mirror=True AVANT (repère non orienté)          : "
          f"{100*n_old_mirror/n_seen:.1f}% ({n_old_mirror}/{n_seen})")
    print(f"Prévalence mirror=True APRÈS (orienté normales, v1)        : "
          f"{100*n_new_mirror/n_seen:.1f}% ({n_new_mirror}/{n_seen})")
    print(f"Prévalence mirror=True APRÈS (orienté centroïde fragment, v2) : "
          f"{100*n_centroid_mirror/n_seen:.1f}% ({n_centroid_mirror}/{n_seen})")


if __name__ == "__main__":
    main()
