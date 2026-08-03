"""
scripts/phase9_normal_orientation_check_thresh03.py
====================================================
Phase 9 (PLAN_REASSEMBLY_MODULE.md) -- même question que
`phase9_normal_orientation_check.py` (le miroir peut-il être prédit par une
règle géométrique déterministe, orientation de `n` via les normales mesh,
plutôt que par le `mirror_head` appris ?), mais sur les points CNN
`thresh0.3` + zoom -- EXACTEMENT le jeu de points utilisé par
`run_learned_stage1(..., mirror_mode="geometric")` dans
`phase8_pipeline_learned_check.py` -- au lieu des points GT propres.

Motivation (2026-08-03) : sur GT, la règle géométrique donnait 98.2% de
prévalence quasi-déterministe (`phase9_normal_orientation_check.py`), mais
branchée dans le pipeline réel (`--mirror_mode geometric`, thresh0.3), le
résultat bout-en-bout était STATISTIQUEMENT IDENTIQUE au `mirror_head`
appris (Pose@30 30.1%->30.4%, succès strict 18.0%->17.6%). Hypothèse à
tester : le masque CNN thresh0.3 (bruité -- précision/rappel imparfaits,
cf. diagnostic 2026-07-22) dégrade `nrm_i.mean(axis=0)` (calculée sur les
points RETENUS, potentiellement pollués par des faux positifs et le zoom
autour de graines imparfaites), ce qui ferait chuter l'exactitude de la
règle géométrique loin des 98.2% observés sur GT propre.

Contrairement à `phase9_normal_orientation_check.py` (qui ne mesurait que
la PRÉVALENCE de mirror=True après réorientation, sans référence), ce
script mesure l'EXACTITUDE : compare la prédiction géométrique (SANS accès
à `R_ij_gt`, comme à l'inférence réelle) à `mirror_gt` (calculé lui via
`R_ij_gt`/`t_ij_gt`, oracle -- uniquement pour la validation, jamais pour
la prédiction elle-même) -- comparable directement aux 86.4% de précision
miroir du `mirror_head` appris (`phase8_eligibility_diagnostic.py`,
2026-07-31).

Mise à jour 2026-08-03 (résultat v1 : 86.2%, quasi égal au modèle appris,
donc rien gagné) : compare aussi une v2, `geometric_mirror_prediction_by_centroid`,
qui n'utilise PAS les normales par point (bruitées par les faux positifs
du masque CNN) mais seulement le vecteur "centre du fragment ENTIER ->
centre de la zone de fracture" -- une moyenne de POSITIONS, plus robuste
au bruit qu'une moyenne de DIRECTIONS (idée de l'utilisateur). Le centre
du fragment entier ne dépend PAS du masque CNN (calculé sur tous les
points du fragment).

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase9_normal_orientation_check_thresh03.py \\
        --ckpt output/cnn_step15_final_model/last.ckpt \\
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
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat
from scripts.phase5a_zoom_resample_check import zoom_resample_with_normals, to_input_frame, normals_to_input_frame
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase8_depthmap_regressor_dataset import fit_theta_shift_mirror_from_gt
from scripts.phase8_build_regressor_dataset import build_frame_and_rasterize, RESOLUTION

ABS_MIN_POINTS = 50


def geometric_mirror_prediction(frame, nrm_i, nrm_j):
    """Même règle que `run_learned_stage1(..., mirror_mode="geometric")`
    (`phase8_pipeline_learned_check.py`) -- dupliquée ici pour ne pas faire
    dépendre ce script de diagnostic du modèle appris (pas de checkpoint
    régresseur nécessaire, juste la géométrie). v1 : orientation par la
    moyenne des normales mesh de la zone de fracture (bruitée sous
    thresh0.3, cf. docstring module)."""
    sign_i = 1.0 if np.dot(frame["n_i"], nrm_i.mean(axis=0)) > 0 else -1.0
    sign_j = 1.0 if np.dot(frame["n_j"], nrm_j.mean(axis=0)) > 0 else -1.0
    return bool(sign_i == sign_j)


def geometric_mirror_prediction_by_centroid(frame, frac_i, frac_j, full_centroid_i, full_centroid_j):
    """v2 (2026-08-03) : au lieu de moyenner des normales individuelles
    (bruitées par les faux positifs du masque CNN), oriente `n_i`/`n_j` en
    comparant le centre de la zone de fracture (`frac_i.mean(axis=0)`,
    déjà = `frame["c_i"]`) au centre du fragment ENTIER
    (`full_centroid_i`, indépendant du masque CNN -- calculé sur TOUS les
    points du fragment, pas seulement ceux retenus par le seuillage).
    Une moyenne de POSITIONS encaisse mieux quelques points aberrants
    qu'une moyenne de DIRECTIONS (v1)."""
    outward_i = frame["c_i"] - full_centroid_i
    outward_j = frame["c_j"] - full_centroid_j
    sign_i = 1.0 if np.dot(frame["n_i"], outward_i) > 0 else -1.0
    sign_j = 1.0 if np.dot(frame["n_j"], outward_j) > 0 else -1.0
    return bool(sign_i == sign_j)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True, help="Checkpoint CNN (segmentation)")
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--categories", default="everyday")
    parser.add_argument("--split",      default="val", choices=["val", "test"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    from torch.utils.data import DataLoader
    from assembly.models.cnn_segmentation_model import CNNFracSeg
    from assembly.models.projection_mapping_utils import extract_fragment_list

    print("Chargement du dataset CNN (sample_method=uniform)...")
    cnn_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
    )
    cnn_cfg = load_config_and_model(cnn_fake_args)
    cnn_datamodule = instantiate(cnn_cfg.data)
    cnn_datamodule.setup("fit" if args.split == "val" else "test")
    cnn_dataset = cnn_datamodule.val_dataset if args.split == "val" else cnn_datamodule.test_dataset

    print("Chargement du dataset mesh (sample_method=weighted) -- meshes uniquement...")
    mesh_fake_args = argparse.Namespace(
        experiment=args.experiment, data_root=args.data_root,
        batch_size=1, num_workers=4, categories=args.categories, model_type="garf",
    )
    mesh_cfg = load_config_and_model(mesh_fake_args)
    mesh_datamodule = instantiate(mesh_cfg.data)
    mesh_datamodule.setup("fit" if args.split == "val" else "test")
    mesh_dataset = mesh_datamodule.val_dataset if args.split == "val" else mesh_datamodule.test_dataset

    assert len(cnn_dataset) == len(mesh_dataset)
    cnn_loader = DataLoader(cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=cnn_datamodule.dataset_cls.collate_fn)

    print(f"Chargement du checkpoint CNN : {args.ckpt}")
    cnn_model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    cnn_model.eval()
    cnn_model.to(device)

    n_seen, n_correct, n_correct_centroid, n_geom_true, n_gt_true, n_zoom_failed = 0, 0, 0, 0, 0, 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | paires vues={n_seen} | {time.time()-t0:.0f}s")

            mesh_data = mesh_dataset[idx]
            assert batch["name"][0] == mesh_data["name"]

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            frag_list, valid_pcs, K = extract_fragment_list(
                batch_gpu["pointclouds"], batch_gpu["points_per_part"])
            if K != 2:
                continue

            gt_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_gt"], batch_gpu["points_per_part"])
            nrm_frag_list, _, _ = extract_fragment_list(
                batch_gpu["pointclouds_normals"], batch_gpu["points_per_part"])

            out = cnn_model(batch_gpu)
            pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
            frag_sizes = [f.shape[0] for f in frag_list]
            offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
            score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
            pc_per_k    = [frag_list[k].cpu().numpy()     for k in range(K)]
            gt_per_k    = [gt_frag_list[k].cpu().numpy()  for k in range(K)]
            nrm_per_k   = [nrm_frag_list[k].cpu().numpy() for k in range(K)]

            valid_pcs_np = valid_pcs.cpu().numpy()
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
            gtgt0, gtgt1 = gt_per_k[0], gt_per_k[1]
            nrm0, nrm1 = nrm_per_k[0], nrm_per_k[1]

            R0 = quat_wxyz_to_rotmat(quats_np[0, p0])
            R1 = quat_wxyz_to_rotmat(quats_np[0, p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[0, p0] - trans_np[0, p1])

            mask0 = sc0 > args.threshold
            mask1 = sc1 > args.threshold
            n_min_raw = min(int(mask0.sum()), int(mask1.sum()))
            if n_min_raw < ABS_MIN_POINTS:
                continue

            cluster_mask0 = dominant_cluster_mask(raw0[mask0])
            cluster_mask1 = dominant_cluster_mask(raw1[mask1])
            R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
            seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init_rot.T
            seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init_rot.T
            meshes = mesh_data["meshes"]
            new_i_gt, new_i_nrm_gt = zoom_resample_with_normals(
                meshes[p0], seed_i_gt, args.extra_budget, args.expand_rings, rng)
            new_j_gt, new_j_nrm_gt = zoom_resample_with_normals(
                meshes[p1], seed_j_gt, args.extra_budget, args.expand_rings, rng)
            if new_i_gt is None or new_j_gt is None:
                n_zoom_failed += 1
                continue
            new_i_rotated = new_i_gt @ R_init_rot
            new_j_rotated = new_j_gt @ R_init_rot
            new_i_nrm_rotated = new_i_nrm_gt @ R_init_rot
            new_j_nrm_rotated = new_j_nrm_gt @ R_init_rot
            new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
            new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
            new_i_nrm_input = normals_to_input_frame(new_i_nrm_rotated, quats_np[0, p0])
            new_j_nrm_input = normals_to_input_frame(new_j_nrm_rotated, quats_np[0, p1])
            frac_i_base, frac_j_base = raw0[mask0], raw1[mask1]
            frac_i_nrm, frac_j_nrm = nrm0[mask0], nrm1[mask1]
            frac_i_zoom = np.concatenate([frac_i_base, new_i_input], axis=0)
            frac_j_zoom = np.concatenate([frac_j_base, new_j_input], axis=0)
            nrm_i_zoom = np.concatenate([frac_i_nrm, new_i_nrm_input], axis=0)
            nrm_j_zoom = np.concatenate([frac_j_nrm, new_j_nrm_input], axis=0)

            if len(frac_i_zoom) < 3 or len(frac_j_zoom) < 3:
                continue

            # -- même repère que le pipeline réel (arbitraire, PAS orienté) --
            _, _, _, _, frame = build_frame_and_rasterize(frac_i_zoom, frac_j_zoom)

            # -- prédiction géométrique v1, SANS R_ij_gt (comme à l'inférence) --
            mirror_pred = geometric_mirror_prediction(frame, nrm_i_zoom, nrm_j_zoom)

            # -- prédiction géométrique v2 (centroïde du fragment ENTIER,
            #    indépendant du masque CNN -- raw0/raw1 = nuage complet) ---
            mirror_pred_centroid = geometric_mirror_prediction_by_centroid(
                frame, frac_i_zoom, frac_j_zoom, raw0.mean(axis=0), raw1.mean(axis=0))

            # -- vrai label, AVEC R_ij_gt (oracle, validation seulement) -----
            _, _, mirror_gt, _ = fit_theta_shift_mirror_from_gt(
                frac_i_zoom, frame["c_i"], frame["u_i"], frame["v_i"],
                frame["u_min_i"], frame["v_min_i"],
                frame["c_j"], frame["u_j"], frame["v_j"],
                frame["u_min_j"], frame["v_min_j"],
                frame["pixel_size"], RESOLUTION, R_ij_gt, t_ij_gt,
            )

            n_seen += 1
            n_correct += int(mirror_pred == mirror_gt)
            n_correct_centroid += int(mirror_pred_centroid == mirror_gt)
            n_geom_true += int(mirror_pred)
            n_gt_true += int(mirror_gt)

    elapsed = time.time() - t0
    print(f"\nFini : {n_seen} paires 2-fragments thresh0.3+zoom exploitables en {elapsed:.0f}s "
          f"({n_zoom_failed} zooms impossibles)\n")
    if n_seen == 0:
        print("Aucune paire exploitable.")
        return
    print(f"Exactitude règle géométrique v1 (moyenne normales)     (vs mirror_gt oracle) : "
          f"{100*n_correct/n_seen:.1f}% ({n_correct}/{n_seen})")
    print(f"Exactitude règle géométrique v2 (centroïde fragment)   (vs mirror_gt oracle) : "
          f"{100*n_correct_centroid/n_seen:.1f}% ({n_correct_centroid}/{n_seen})")
    print(f"  Prévalence prédite v1 (mirror=True) : {100*n_geom_true/n_seen:.1f}%")
    print(f"  Prévalence vraie    (mirror_gt=True) : {100*n_gt_true/n_seen:.1f}%")
    print(f"\n(Référence GT propre, phase9_normal_orientation_check.py : prévalence ~98.2%)")
    print(f"(Référence mirror_head appris, phase8_eligibility_diagnostic.py : précision 86.4%)")


if __name__ == "__main__":
    main()
