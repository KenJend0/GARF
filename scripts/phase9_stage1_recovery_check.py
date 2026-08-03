"""
scripts/phase9_stage1_recovery_check.py
========================================
Phase 9 (PLAN_REASSEMBLY_MODULE.md) -- chantier (2) : pourquoi certaines
paires restent INÉLIGIBLES (`no_correspondence`) avec l'étage 1 appris
(`assembly/models/depthmap_pose_regressor.py`), et si on peut en récupérer
une partie SANS réentraînement.

Contexte (2026-08-03, discuté avec l'utilisateur) : le hand-crafted
récupère beaucoup d'éligibilité via une cascade multi-résolution
(no_correspondence 73.6% -> 48.1%, Phase 1/3). Le modèle appris tourne
aujourd'hui à résolution FIXE (64, `phase8_build_regressor_dataset.RESOLUTION`),
sans filet de secours. Mais attention : `build_correspondences_nn`
(`phase5a_depthmap_matching.py`) ne fait PAS un arrondi de grille comme
l'ancienne version (bug corrigé le 2026-07-22) -- c'est un plus-proche-
voisin CONTINU avec une tolérance physique absolue (`contact_eps`,
défaut 0.05), donc la résolution n'agit pas directement sur la tolérance
de correspondance, seulement sur la qualité du (theta, shift) que le
MODÈLE prédit (une carte plus grossière regroupe les points épars en
moins de cases, signal moins troué à l'entrée du réseau).

Deux mécanismes de récupération testés ici, INDÉPENDAMMENT l'un de
l'autre, sur les paires actuellement `no_correspondence` à la config par
défaut (résolution 64, contact_eps 0.05) :

1. **Cascade de résolution** : `_SiameseEncoder` utilise `AdaptiveAvgPool2d(1)`
   (vérifié dans le code), donc le modèle accepte n'importe quelle taille
   de carte SANS erreur de forme -- mais il n'a jamais rien vu d'autre que
   64x64 à l'entraînement (vrai décalage de distribution possible, pas
   juste une question de forme). Recalcule depth/normal maps + profil FFT
   à une résolution plus grossière (48, 32, 24), refait tourner le MÊME
   checkpoint (pas de réentraînement), retente `build_correspondences_nn`
   à cette nouvelle résolution.
2. **Assouplissement de `contact_eps`** : à résolution 64 fixe, RÉUTILISE
   le (theta, shift, mirror) déjà prédit (pas de nouveau forward, gratuit)
   et retente juste `build_correspondences_nn` avec une tolérance plus
   généreuse (0.08, 0.1, 0.15, 0.2) -- légitime ici car l'étage 1 n'a
   besoin que d'être grossier, l'ICP (étage 2) raffine derrière.

Sélection du miroir : `geometric_centroid` (v2, validée 2026-08-03,
90.0% d'exactitude vs 86.4% appris, moins de paramètres) -- cf.
`phase8_pipeline_learned_check.py`.

Pour chaque mécanisme, mesure : combien de paires `no_correspondence` sont
récupérées, et parmi elles, combien atteignent Pose@30 (comparé à la
vraie pose GT, pour vérifier qu'on ne récupère pas juste "une pose",
mais des poses PLAUSIBLES) -- sinon on aurait juste déplacé le problème
de l'éligibilité vers le groupe C (Pose@30 raté).

Usage (sur le serveur) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase9_stage1_recovery_check.py \\
        --regressor_ckpt output/phase8_depthmap_regressor_thresh03_mirrorhead/best.ckpt \\
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
from scripts.phase5a_depthmap_matching import (
    quat_wxyz_to_rotmat, rasterize, rasterize_normal_channels, kabsch, rot_err_deg, trans_err,
    build_correspondences_nn, angle_correlation_profile, DEFAULT_N_ANGLES,
)
from scripts.phase5a_zoom_resample_check import (
    zoom_resample_with_normals, to_input_frame, normals_to_input_frame,
)
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase8_depthmap_regressor_dataset import canonical_pca_frame
from assembly.models.depthmap_pose_regressor import DepthmapPoseRegressor

ABS_MIN_POINTS = 50
BASE_RESOLUTION = 64
RESOLUTION_CASCADE = [48, 32, 24]
CONTACT_EPS_SCHEDULE = [0.08, 0.1, 0.15, 0.2]
BASE_CONTACT_EPS = 0.05


def load_regressor(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DepthmapPoseRegressor(feat_dim=ckpt["feat_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _frame_at_resolution(frac_i, frac_j, resolution):
    """Copie de `build_frame_and_rasterize` (`phase8_build_regressor_dataset.py`),
    dupliquée avec `resolution` EXPLICITE (la fonction d'origine lit la
    constante module `RESOLUTION=64`, on ne la modifie pas pour ne rien
    risquer sur le dataset d'entraînement/l'éval Phase 8 déjà validés)."""
    c_i, u_i, v_i, n_i, _ = canonical_pca_frame(frac_i)
    c_j, u_j, v_j, n_j, _ = canonical_pca_frame(frac_j)

    ci, cj = frac_i - c_i, frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    pixel_size = max(span_i, span_j) * 1.1 / resolution

    dmap_i, valid_i, u_min_i, v_min_i = rasterize(frac_i, c_i, u_i, v_i, n_i, resolution, pixel_size)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(frac_j, c_j, u_j, v_j, n_j, resolution, pixel_size)

    frame = {
        "c_i": c_i, "u_i": u_i, "v_i": v_i, "n_i": n_i, "u_min_i": u_min_i, "v_min_i": v_min_i,
        "c_j": c_j, "u_j": u_j, "v_j": v_j, "n_j": n_j, "u_min_j": u_min_j, "v_min_j": v_min_j,
        "pixel_size": pixel_size,
    }
    return dmap_i, valid_i, dmap_j, valid_j, frame


@torch.no_grad()
def run_stage1_at_resolution(model, frac_i, frac_j, nrm_i, nrm_j, device, resolution,
                              full_centroid_i, full_centroid_j):
    """Un forward du modèle appris à une résolution donnée -- retourne
    (theta, shift, mirror, frame) ou None si trop peu de points. Décision
    miroir : `geometric_centroid` (v2, cf. `phase8_pipeline_learned_check.py`),
    pas de dépendance au `mirror_head` appris."""
    if len(frac_i) < 3 or len(frac_j) < 3:
        return None

    dmap_i, valid_i, dmap_j, valid_j, frame = _frame_at_resolution(frac_i, frac_j, resolution)
    nmap_i = rasterize_normal_channels(
        frac_i, nrm_i, frame["c_i"], frame["u_i"], frame["v_i"], frame["n_i"],
        resolution, frame["pixel_size"], frame["u_min_i"], frame["v_min_i"])
    nmap_j = rasterize_normal_channels(
        frac_j, nrm_j, frame["c_j"], frame["u_j"], frame["v_j"], frame["n_j"],
        resolution, frame["pixel_size"], frame["u_min_j"], frame["v_min_j"])

    profile_normal = angle_correlation_profile(dmap_i, valid_i, dmap_j, valid_j, DEFAULT_N_ANGLES)
    profile_mirror = angle_correlation_profile(
        dmap_i, valid_i, np.flip(dmap_j, axis=0), np.flip(valid_j, axis=0), DEFAULT_N_ANGLES)

    t_dmap_i = torch.from_numpy(dmap_i[None]).float().to(device)
    t_valid_i = torch.from_numpy(valid_i[None]).float().to(device)
    t_dmap_j = torch.from_numpy(dmap_j[None]).float().to(device)
    t_valid_j = torch.from_numpy(valid_j[None]).float().to(device)
    t_nmap_i = torch.from_numpy(nmap_i[None].astype(np.float32)).to(device)
    t_nmap_j = torch.from_numpy(nmap_j[None].astype(np.float32)).to(device)
    t_profile_normal = torch.from_numpy(profile_normal[None].astype(np.float32)).to(device)
    t_profile_mirror = torch.from_numpy(profile_mirror[None].astype(np.float32)).to(device)

    pred = model(t_dmap_i, t_valid_i, t_dmap_j, t_valid_j, t_nmap_i, t_nmap_j,
                 t_profile_normal, t_profile_mirror)

    outward_i = frame["c_i"] - full_centroid_i
    outward_j = frame["c_j"] - full_centroid_j
    sign_i = 1.0 if np.dot(frame["n_i"], outward_i) > 0 else -1.0
    sign_j = 1.0 if np.dot(frame["n_j"], outward_j) > 0 else -1.0
    mirror = bool(sign_i == sign_j)
    mirror_t = torch.tensor([mirror], device=device)
    chosen = {k: torch.where(mirror_t, pred["mirror"][k], pred["normal"][k])
              for k in ("sin", "cos", "shift_y", "shift_x")}
    theta_t = torch.rad2deg(torch.atan2(chosen["sin"], chosen["cos"])) % 360.0
    theta = float(theta_t.item())
    shift = (float(chosen["shift_y"].item()), float(chosen["shift_x"].item()))

    return theta, shift, mirror, frame


def _try_correspondences(frac_i, frac_j, frame, resolution, theta, shift, mirror, contact_eps,
                          R_ij_gt, t_ij_gt):
    v_j_use = -frame["v_j"] if mirror else frame["v_j"]
    pts_i3, pts_j3 = build_correspondences_nn(
        frac_i, frame["c_i"], frame["u_i"], frame["v_i"],
        frac_j, frame["c_j"], frame["u_j"], v_j_use,
        frame["u_min_i"], frame["v_min_i"], frame["u_min_j"], frame["v_min_j"],
        frame["pixel_size"], resolution, theta, shift, contact_eps=contact_eps,
    )
    if pts_i3 is None or len(pts_i3) < 3:
        return None
    R_est, t_est = kabsch(pts_i3, pts_j3)
    re = rot_err_deg(R_est, R_ij_gt)
    te = trans_err(t_est, t_ij_gt)
    return bool(re < 30.0 and te < 0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor_ckpt", required=True)
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
    regressor = load_regressor(args.regressor_ckpt, device)

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

    n_seen = 0
    n_baseline_eligible = 0
    n_no_correspondence = 0
    n_rescued_by_resolution = 0
    n_rescued_by_resolution_pose30 = 0
    n_rescued_by_eps = 0
    n_rescued_by_eps_pose30 = 0
    n_zoom_failed = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(cnn_loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | paires vues={n_seen} | no_corr={n_no_correspondence} | "
                      f"{time.time()-t0:.0f}s")

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

            full_centroid_i = raw0.mean(axis=0)
            full_centroid_j = raw1.mean(axis=0)

            n_seen += 1

            # -- Baseline : résolution 64, contact_eps 0.05 (config actuelle) --
            base = run_stage1_at_resolution(regressor, frac_i_zoom, frac_j_zoom,
                                             nrm_i_zoom, nrm_j_zoom, device,
                                             BASE_RESOLUTION, full_centroid_i, full_centroid_j)
            if base is None:
                continue
            theta_b, shift_b, mirror_b, frame_b = base
            pose30_b = _try_correspondences(frac_i_zoom, frac_j_zoom, frame_b, BASE_RESOLUTION,
                                             theta_b, shift_b, mirror_b, BASE_CONTACT_EPS,
                                             R_ij_gt, t_ij_gt)
            if pose30_b is not None:
                n_baseline_eligible += 1
                continue   # déjà éligible, rien à récupérer sur cette paire
            n_no_correspondence += 1

            # -- Mécanisme 1 : cascade de résolution (nouveau forward à chaque R) --
            rescued_resolution = False
            for R in RESOLUTION_CASCADE:
                res = run_stage1_at_resolution(regressor, frac_i_zoom, frac_j_zoom,
                                                nrm_i_zoom, nrm_j_zoom, device,
                                                R, full_centroid_i, full_centroid_j)
                if res is None:
                    continue
                theta_r, shift_r, mirror_r, frame_r = res
                pose30_r = _try_correspondences(frac_i_zoom, frac_j_zoom, frame_r, R,
                                                 theta_r, shift_r, mirror_r, BASE_CONTACT_EPS,
                                                 R_ij_gt, t_ij_gt)
                if pose30_r is not None:
                    n_rescued_by_resolution += 1
                    n_rescued_by_resolution_pose30 += int(pose30_r)
                    rescued_resolution = True
                    break

            # -- Mécanisme 2 : contact_eps plus généreux, MÊME (theta,shift,mirror) --
            for eps in CONTACT_EPS_SCHEDULE:
                pose30_e = _try_correspondences(frac_i_zoom, frac_j_zoom, frame_b, BASE_RESOLUTION,
                                                 theta_b, shift_b, mirror_b, eps, R_ij_gt, t_ij_gt)
                if pose30_e is not None:
                    n_rescued_by_eps += 1
                    n_rescued_by_eps_pose30 += int(pose30_e)
                    break

    elapsed = time.time() - t0
    print(f"\nFini : {n_seen} paires exploitables en {elapsed:.0f}s "
          f"({n_zoom_failed} zooms impossibles)\n")
    if n_seen == 0:
        print("Aucune paire exploitable.")
        return

    print(f"Éligible dès la config par défaut (R=64, eps=0.05) : "
          f"{n_baseline_eligible}/{n_seen} ({100*n_baseline_eligible/n_seen:.1f}%)")
    print(f"Non éligible (no_correspondence)                   : "
          f"{n_no_correspondence}/{n_seen} ({100*n_no_correspondence/n_seen:.1f}%)\n")

    if n_no_correspondence == 0:
        print("Aucune paire no_correspondence à tenter de récupérer.")
        return

    print(f"Mécanisme 1 -- cascade de résolution ({RESOLUTION_CASCADE}) :")
    print(f"  récupérées : {n_rescued_by_resolution}/{n_no_correspondence} "
          f"({100*n_rescued_by_resolution/n_no_correspondence:.1f}%)")
    if n_rescued_by_resolution:
        print(f"  dont Pose@30 correct : {n_rescued_by_resolution_pose30}/{n_rescued_by_resolution} "
              f"({100*n_rescued_by_resolution_pose30/n_rescued_by_resolution:.1f}%)")

    print(f"\nMécanisme 2 -- contact_eps plus généreux ({CONTACT_EPS_SCHEDULE}) :")
    print(f"  récupérées : {n_rescued_by_eps}/{n_no_correspondence} "
          f"({100*n_rescued_by_eps/n_no_correspondence:.1f}%)")
    if n_rescued_by_eps:
        print(f"  dont Pose@30 correct : {n_rescued_by_eps_pose30}/{n_rescued_by_eps} "
              f"({100*n_rescued_by_eps_pose30/n_rescued_by_eps:.1f}%)")


if __name__ == "__main__":
    main()
