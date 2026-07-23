"""
scripts/phase5a_zoom_resample_thresh03_check.py
==================================================
Phase 5A / Phase 7 — voir PLAN_REASSEMBLY_MODULE.md.

Version `thresh0.3` (CNN) du diagnostic zoom + rééchantillonnage local
(`phase5a_zoom_resample_check.py`, validé en GT le 2026-07-22 :
`--expand_rings 0` + budget généreux améliore l'éligibilité SANS dégrader
la précision, contrairement à `--expand_rings 1`).

Question posée par l'utilisateur avant de généraliser (2026-07-22) : le CNN
n'a jamais été entraîné/évalué sur des points zoomés -- est-ce un problème ?
Réponse établie par discussion : NON directement, le CNN ne voit JAMAIS les
points zoomés dans cette architecture en deux temps (il ne fait QUE
localiser, sur son échantillonnage d'entraînement standard) :
  1. CNN tourne sur l'échantillonnage `uniform` standard (sa distribution
     d'entraînement, aucun changement) -> masque prédit `thresh0.3`.
  2. Le masque prédit sert de GRAINE pour zoomer sur le maillage (comme le
     masque GT servait de graine dans la version GT) -- le CNN n'intervient
     plus à cette étape.
  3. Cascade de résolution sur le masque zoomé, comme d'habitude.

La VRAIE question ouverte (pas résolue par l'architecture, à mesurer) : le
masque `thresh0.3` a des faux positifs/négatifs que le GT n'a jamais --
si ses points prédits-positifs sont spatialement DISPERSÉS (pas concentrés
sur la vraie fracture comme le GT l'est par construction), le zoom risque de
localiser une zone bruitée. La Phase 2D avait déjà mesuré que le clustering
spatial de `thresh0.3` est quasi identique à `gt` (`EdgeCoverage`/
`BestCorrPrec` à <1pt d'écart) -- signe encourageant, mais jamais testé pour
CE mécanisme précis.

Contrainte technique : le CNN exige `sample_method=uniform` (architecture à
backprojection bilinéaire, budget de points constant par fragment), mais
`BreakingBadUniform.transform()` NE conserve PAS `data["meshes"]` dans son
dict de retour (contrairement à `BreakingBadWeighted`, vérifié par
inspection du code) -- besoin du maillage pour le zoom. Solution : DEUX
datasets en parallèle sur le MÊME split, indexés en lockstep (pas de
DataLoader mélangé pour le second, indexation directe `dataset[idx]` --
`__getitem__` retourne `transform(get_data(idx))` sans passer par
collate_fn, donc accès direct fiable) :
  - `cnn_dataset`  (model_type="cnn",  sample_method=uniform forcé par la
    config d'expérience) -- DataLoader batch_size=1, shuffle=False, sert à
    l'inférence CNN (pointclouds, pointclouds_gt, scale/quat/trans propres
    à CET échantillon).
  - `mesh_dataset` (model_type="garf", sample_method=weighted forcé) --
    indexé directement par position (`mesh_dataset[idx]`), sert UNIQUEMENT
    à récupérer `data["meshes"]` (le maillage lui-même, identique quel que
    soit le sample_method -- seule la façon de tirer des points dessus
    change). Ses propres points échantillonnés (weighted) ne sont PAS
    utilisés ici.
Un `assert` sur le nom d'objet à chaque itération vérifie l'alignement
(`shuffle=False` sur les deux -> même ordre de `data_list`, déterministe).

Repères : les nouveaux points zoomés (maillage/GT) sont ramenés dans le
repère "input" via les quaternion/translation/scale du `cnn_dataset` (PAS
du `mesh_dataset` -- chaque `transform()` tire sa PROPRE rotation aléatoire,
les deux échantillons du même objet ne partagent pas le même repère
disassemblé). Le maillage lui-même est identique entre les deux instances
(même `meshes_max_scale`, calculé uniquement à partir de la géométrie brute).

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase5a_zoom_resample_thresh03_check.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --expand_rings 0 --extra_budget_sweep 500 1200 \\
        --resolution_sweep 128 96 64 48 32 24 20 16 12 \\
        --csv_out /tmp/student7/phase5a_zoom_thresh03.csv \\
        --summary_json /tmp/student7/phase5a_zoom_thresh03.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_errors import load_config_and_model
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat
from scripts.phase5a_zoom_resample_check import (
    ABS_MIN_POINTS, zoom_resample, to_input_frame, run_cascade, report_zoom_sweep,
)
from scripts.phase2d_interface_clustering_diagnostic import cluster_points
from assembly.models.cnn_segmentation_model import CNNFracSeg
from assembly.models.projection_mapping_utils import extract_fragment_list

# Mêmes valeurs que la Phase 2D (meilleur compromis trouvé le 2026-06-27 :
# EdgeCoverage/BestCorrPrec plafonnent dès eps=0.02, descendre plus bas
# n'apporte rien, juste plus de bruit).
CLUSTER_EPS = 0.02
MIN_CLUSTER_SIZE = 10


def dominant_cluster_mask(points):
    """Isole le plus GROS cluster spatial (composantes connexes par proximité,
    réutilise `cluster_points` de la Phase 2D) parmi `points` -- ajouté le
    2026-07-22 après un résultat catastrophique du zoom sur `thresh0.3` : le
    masque prédit par le CNN a des faux positifs dispersés que le GT n'a
    jamais, et les utiliser TOUS comme graines de zoom (sans filtrage)
    localisait des zones bruitées un peu partout sur le fragment plutôt que la
    vraie fracture, corrompant le repère PCA même pour les paires qui
    marchaient déjà en baseline. Retourne un masque bool (True = dans le
    cluster dominant) ; tout-vrai si le clustering n'est pas exploitable
    (< 2 points, ou 100% classé bruit)."""
    if len(points) < 2:
        return np.ones(len(points), dtype=bool)
    labels = cluster_points(points, eps=CLUSTER_EPS, min_cluster_size=MIN_CLUSTER_SIZE)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return np.ones(len(points), dtype=bool)
    dominant = np.bincount(valid).argmax()
    return labels == dominant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["val", "test"],
                        help="Nécessite les meshes (conservés en val/test) -- pas 'train'.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3,
                        help="Seuil de probabilité CNN pour le masque fracture prédit.")
    parser.add_argument("--resolution_sweep", type=int, nargs="+",
                        default=[128, 96, 64, 48, 32, 24, 20, 16, 12])
    parser.add_argument("--n_angles",    type=int, default=36)
    parser.add_argument("--score_mode",  default="joint", choices=["relief", "overlap_only", "joint"])
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--extra_budget", type=int, default=500,
                        help="Ignoré si --extra_budget_sweep est donné.")
    parser.add_argument("--extra_budget_sweep", type=int, nargs="+", default=[],
                        help="Teste plusieurs budgets dans la même passe -- voir "
                             "phase5a_zoom_resample_check.py pour le détail du mécanisme.")
    parser.add_argument("--expand_rings", type=int, default=0,
                        help="Défaut 0 (pas d'expansion) -- validé en GT le 2026-07-22 : "
                             "l'expansion à 1 anneau dégradait la précision sans nécessité.")
    parser.add_argument("--max_n_min_base", type=int, default=0,
                        help="Ne traite QUE les paires dont n_frac_pts_min (baseline, "
                             "AVANT zoom) est < ce seuil (2026-07-22, diagnostic ciblé sur "
                             "la tranche '<50' -- évite de dépenser du calcul sur les "
                             "paires déjà bien servies). 0 = pas de filtre (défaut).")
    parser.add_argument("--csv_out", default="", help="Dump complet, une ligne par paire.")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dilate_px = 0
    args.gaussian_sigma_px = 0.0

    budgets = sorted(args.extra_budget_sweep) if args.extra_budget_sweep else [args.extra_budget]
    max_budget = budgets[-1]

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── Deux datasets sur le même split, indexés en lockstep (cf. docstring) ───
    print("Chargement du dataset CNN (sample_method=uniform, forcé par la config "
          "d'expérience)...")
    cnn_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cnn_cfg = load_config_and_model(cnn_fake_args)
    cnn_datamodule = instantiate(cnn_cfg.data)
    cnn_datamodule.setup("fit" if args.split == "val" else "test")
    cnn_dataset = cnn_datamodule.val_dataset if args.split == "val" else cnn_datamodule.test_dataset

    print("Chargement du dataset mesh (sample_method=weighted, forcé via model_type=garf) "
          "-- UNIQUEMENT pour data['meshes'], ses propres points échantillonnés ne sont "
          "pas utilisés...")
    mesh_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    mesh_cfg = load_config_and_model(mesh_fake_args)
    mesh_datamodule = instantiate(mesh_cfg.data)
    mesh_datamodule.setup("fit" if args.split == "val" else "test")
    mesh_dataset = mesh_datamodule.val_dataset if args.split == "val" else mesh_datamodule.test_dataset

    assert len(cnn_dataset) == len(mesh_dataset), (
        f"Tailles de dataset différentes ({len(cnn_dataset)} vs {len(mesh_dataset)}) -- "
        f"le lockstep par position suppose la même liste d'objets dans le même ordre."
    )

    cnn_loader = DataLoader(
        cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=cnn_datamodule.dataset_cls.collate_fn,
    )

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    t0 = time.time()
    print(f"\nDiagnostic zoom + rééchantillonnage local (thresh0.3/CNN) -- "
          f"{args.categories}/{args.split}, threshold={args.threshold}, budgets={budgets}...\n")

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | "
                      f"zoom échoué={n_zoom_failed} | {elapsed:.0f}s écoulées")

            mesh_data = mesh_dataset[idx]
            assert batch["name"][0] == mesh_data["name"], (
                f"Désalignement lockstep à l'index {idx} : "
                f"cnn={batch['name'][0]!r} vs mesh={mesh_data['name']!r}"
            )

            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"]
            )
            if K != 2:
                continue   # objets à 2 fragments uniquement (convention Phase 5A)
            n_seen_2frag += 1

            gt_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_gt"], batch_gpu["points_per_part"]
            )

            out = model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()    for k in range(K)]
            gt_per_k    = [gt_frag_list[k].cpu().numpy() for k in range(K)]

            valid_pcs_np = valid_pcs.cpu().numpy()   # (1, P)
            p_slots = [p for p in range(valid_pcs_np.shape[1]) if valid_pcs_np[0, p]]
            if len(p_slots) != 2:
                continue
            p0, p1 = p_slots

            scale_np = batch["scale"].numpy()
            if scale_np.ndim == 2:
                scale_np = scale_np[:, :, None]
            quats_np = batch["quaternions"].numpy()
            trans_np = batch["translations"].numpy()

            raw0 = pc_per_k[0] * scale_np[0, p0]
            raw1 = pc_per_k[1] * scale_np[0, p1]
            sc0, sc1 = score_per_k[0], score_per_k[1]
            gtgt0, gtgt1 = gt_per_k[0], gt_per_k[1]   # repère maillage/GT, aligné à pc_per_k

            R0 = quat_wxyz_to_rotmat(quats_np[0, p0])
            R1 = quat_wxyz_to_rotmat(quats_np[0, p1])
            R_ij = R1.T @ R0
            t_ij = R1.T @ (trans_np[0, p0] - trans_np[0, p1])

            mask0 = sc0 > args.threshold
            mask1 = sc1 > args.threshold
            frac_i_base = raw0[mask0]
            frac_j_base = raw1[mask1]
            n_min_base = min(len(frac_i_base), len(frac_j_base))
            if n_min_base < ABS_MIN_POINTS:
                continue
            if args.max_n_min_base > 0 and n_min_base >= args.max_n_min_base:
                continue   # diagnostic ciblé sur une tranche de densité (2026-07-22)

            baseline_res = run_cascade(frac_i_base, frac_j_base, R_ij, t_ij, args)

            # ── Comparaison GT sur les MÊMES objets (2026-07-22, demande de
            # l'utilisateur) : la tranche '<50 côté CNN' est-elle aussi dure pour
            # le GT (fracture physiquement petite/ambiguë pour tout le monde), ou
            # est-ce spécifique à la localisation du CNN ? `fracture_surface_gt`
            # est déjà dans le batch `uniform` (pas besoin du mesh_dataset ici,
            # simple label par point, même alignement que pc_per_k). ──────────
            fracture_gt_np = batch["fracture_surface_gt"].numpy()   # (1, P, N)
            gt_label0 = fracture_gt_np[0, p0]
            gt_label1 = fracture_gt_np[0, p1]
            frac_i_gt = raw0[gt_label0 == 1]
            frac_j_gt = raw1[gt_label1 == 1]
            gt_res = run_cascade(frac_i_gt, frac_j_gt, R_ij, t_ij, args)

            # ── Zoom : graines = points prédits fracture par le CNN (PAS le GT),
            # en repère maillage (gtgt0/gtgt1, aligné index-à-index avec pc_per_k
            # puisque extract_fragment_list découpe pointclouds ET pointclouds_gt
            # avec les mêmes offsets points_per_part). Maillage venant du dataset
            # `mesh_dataset` (même géométrie, sample_method n'affecte pas le mesh
            # lui-même). Repère de sortie : quaternion/translation du CNN_DATASET
            # (celui qui possède frac_i_base), pas du mesh_dataset.
            #
            # FILTRAGE PAR CLUSTERING (2026-07-22, suite à l'échec du run brut) :
            # le masque thresh0.3 a des faux positifs dispersés que le GT n'a
            # jamais -- ne garder QUE le cluster spatial dominant (repère input,
            # invariant à la pose) comme graine, pas tous les points prédits.
            # Le masque `frac_i_base`/baseline reste INCHANGÉ (comparaison
            # apples-to-apples avec le pipeline réel actuel) -- seul le choix
            # des graines de zoom est purifié.
            #
            # CORRECTION DE REPÈRE (2026-07-22, bug trouvé après l'échec du run
            # avec clustering) : `BreakingBadUniform.transform()` applique une
            # rotation aléatoire supplémentaire à TOUT L'OBJET assemblé
            # (`rotate_whole_part`, avant la rotation par fragment) -- que
            # `BreakingBadWeighted.transform()` N'APPLIQUE JAMAIS (commentée
            # dans son code). Donc `gtgt0`/`gtgt1` (venant du dataset CNN,
            # uniform) sont dans un repère tourné par `init_rot` relativement
            # au maillage (venant du dataset mesh, weighted, jamais tourné
            # globalement) -- chercher "quelle face du maillage est la plus
            # proche" sans corriger ça revient à comparer des points dans deux
            # repères différents, donnant des faces essentiellement aléatoires.
            # Fix : ramener les graines au repère canonique du maillage avant
            # la recherche (`@ R_init.T`), puis reconvertir les nouveaux points
            # zoomés vers le repère "tourné" AVANT `to_input_frame` (qui,
            # lui, suppose déjà ce repère) via `@ R_init`. ───────────────────
            R_init = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
            cluster_mask0 = dominant_cluster_mask(raw0[mask0])
            cluster_mask1 = dominant_cluster_mask(raw1[mask1])
            seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init.T
            seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init.T
            meshes = mesh_data["meshes"]
            new_i_gt_max = zoom_resample(meshes[p0], seed_i_gt, max_budget,
                                          args.expand_rings, rng)
            new_j_gt_max = zoom_resample(meshes[p1], seed_j_gt, max_budget,
                                          args.expand_rings, rng)
            if new_i_gt_max is None or new_j_gt_max is None:
                n_zoom_failed += 1
                continue

            row = {
                "n_frac_pts_min_base": n_min_base,
                "base_stage": baseline_res["reached_stage"],
                "base_pose_30": baseline_res.get("pose_30", False),
                # Tracking des graines avant/après clustering (2026-07-22, demande de
                # l'utilisateur -- "comment avec 200pts de plus on ne trouve toujours
                # pas 3 correspondances, c'est un mystère") : distingue "le CNN ne
                # prédit presque rien" (n_seed_raw bas) de "le clustering détruit ce
                # qu'il y avait" (n_seed_clustered << n_seed_raw).
                "n_seed_raw_i": int(mask0.sum()), "n_seed_raw_j": int(mask1.sum()),
                "n_seed_clustered_i": int(cluster_mask0.sum()),
                "n_seed_clustered_j": int(cluster_mask1.sum()),
                "gt_stage": gt_res["reached_stage"],
                "gt_pose_30": gt_res.get("pose_30", False),
                "zoom_by_budget": {},
            }
            for k in budgets:
                new_i_rotated = new_i_gt_max[:k] @ R_init
                new_j_rotated = new_j_gt_max[:k] @ R_init
                new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
                new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
                frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
                frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)
                zoomed_res = run_cascade(frac_i_zoom, frac_j_zoom, R_ij, t_ij, args)
                row["zoom_by_budget"][k] = {
                    "n_frac_pts_min": zoomed_res["n_frac_pts_min"],
                    "stage": zoomed_res["reached_stage"],
                    "pose_30": zoomed_res.get("pose_30", False),
                }
            rows.append(row)

    elapsed = time.time() - t0

    if rows:
        print("\nGRAINES AVANT/APRÈS CLUSTERING (le côté le plus pauvre des deux "
              "fragments, par paire) :")
        n_raw_min = np.array([min(r["n_seed_raw_i"], r["n_seed_raw_j"]) for r in rows])
        n_clu_min = np.array([min(r["n_seed_clustered_i"], r["n_seed_clustered_j"]) for r in rows])
        n_dropped_to_zero = int(np.sum(n_clu_min == 0))
        for label, vals in [("n_seed_raw (avant clustering)", n_raw_min),
                             ("n_seed_clustered (après clustering)", n_clu_min)]:
            print(f"  {label:<38} N={len(vals):>5}  "
                  f"p25={np.percentile(vals,25):>6.1f}  p50={np.percentile(vals,50):>6.1f}  "
                  f"p75={np.percentile(vals,75):>6.1f}  moyenne={np.mean(vals):>6.1f}")
        print(f"  Paires où le clustering réduit un côté à 0 graine : "
              f"{n_dropped_to_zero}/{len(rows)} ({100*n_dropped_to_zero/len(rows):.1f}%)")
        print("(Lecture : si n_seed_clustered est beaucoup plus bas que n_seed_raw, le "
              "clustering (MIN_CLUSTER_SIZE=10) détruit une bonne partie des graines déjà "
              "rares. Si les deux sont proches et bas, le CNN ne prédit tout simplement "
              "presque rien -- le clustering n'est pas en cause.)\n")

        # ── GT sur les MÊMES objets (2026-07-22) : cette tranche est-elle dure
        # pour tout le monde (fracture physiquement petite/ambiguë), ou
        # spécifique à la localisation du CNN ? ─────────────────────────────
        n = len(rows)
        gt_no_corr = sum(1 for r in rows if r["gt_stage"] == "no_correspondence")
        gt_pose    = sum(1 for r in rows if r["gt_stage"] == "pose_computed")
        gt_p30     = sum(1 for r in rows if r["gt_pose_30"])
        cnn_no_corr = sum(1 for r in rows if r["base_stage"] == "no_correspondence")
        cnn_pose    = sum(1 for r in rows if r["base_stage"] == "pose_computed")
        cnn_p30     = sum(1 for r in rows if r["base_pose_30"])
        print(f"GT vs CNN SUR LES MÊMES {n} OBJETS (tranche sélectionnée par le CNN) :")
        print(f"  {'':<20} {'CNN (thresh0.3)':>16} {'GT (oracle)':>14}")
        print(f"  {'no_correspondence':<20} {100*cnn_no_corr/n:>15.1f}% {100*gt_no_corr/n:>13.1f}%")
        print(f"  {'pose_computed':<20} {100*cnn_pose/n:>15.1f}% {100*gt_pose/n:>13.1f}%")
        print(f"  {'Pose@30':<20} {100*cnn_p30/n:>15.1f}% {100*gt_p30/n:>13.1f}%")
        print("(Lecture : si GT réussit largement mieux que CNN sur CES MÊMES objets, la\n"
              " tranche n'est pas intrinsèquement dure -- c'est la localisation du CNN qui\n"
              " est en cause, pas la sparsité ni le clustering. Si GT échoue presque autant,\n"
              " ces objets ont une fracture physiquement petite/ambiguë pour tout le monde,\n"
              " indépendamment du CNN -- un problème plus profond que la Phase 7 actuelle.)\n")

    report_zoom_sweep(rows, budgets, args, n_seen_2frag, n_zoom_failed, elapsed)


if __name__ == "__main__":
    main()
