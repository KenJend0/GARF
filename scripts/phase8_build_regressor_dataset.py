"""
scripts/phase8_build_regressor_dataset.py
=============================================
Phase 8 (PLAN_REASSEMBLY_MODULE.md) — précalcule le dataset d'entraînement
du modèle appris (`assembly/models/depthmap_pose_regressor.py`) : pour
chaque paire à 2 fragments, génère `(dmap_i, valid_i, dmap_j, valid_j,
theta_gt, shift_y_gt, shift_x_gt, mirror_gt)` et sauvegarde le tout dans un
`.npz` (précalcul UNE fois -- l'inférence CNN + le zoom sur maillage sont
coûteux, pas la peine de les refaire à chaque epoch).

Réutilise tel quel (aucune réimplémentation) :
  - Zoom/rééchantillonnage + correction `init_rot` + clustering des graines
    (`phase5a_zoom_resample_check.py`, `phase5a_zoom_resample_thresh03_check.py`)
    -- MÊME mécanisme que le pipeline hand-crafted déjà validé, on ne
    change QUE l'étape "trouver theta/shift" en aval.
  - `canonical_pca_frame`/`fit_theta_shift_mirror_from_gt`
    (`phase8_depthmap_regressor_dataset.py`) pour dériver les labels.
  - `rasterize()` (`phase5a_depthmap_matching.py`) pour générer les depth
    maps, résolution FIXE (contrairement à la cascade multi-résolution du
    hand-crafted -- un seul `RESOLUTION` pour ce modèle).

`--strategy gt` (oracle-first, à valider EN PREMIER, comme toute la
Phase 7) : masque de fracture GT, un seul dataset (`weighted`).
`--strategy thresh03` (condition réelle, le vrai objectif d'entraînement --
apprendre la robustesse au bruit du masque CNN, PAS un cas idéal) : masque
prédit par le CNN Step 15, deux datasets en lockstep (CNN `uniform` +
mesh `weighted`), clustering pour les graines de zoom, correction
`init_rot` -- même plomberie que `phase6b_pipeline_thresh03_check.py`.

Usage (sur le serveur) :
    # Oracle-first : valider la génération sur GT
    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_build_regressor_dataset.py \\
        --strategy gt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 3000 \\
        --out /tmp/student7/phase8_dataset_gt_val.npz

    # Condition réelle (CNN thresh0.3) -- --split train donne BEAUCOUP plus
    # de paires que val/test, mais SANS zoom (meshes indisponibles en train
    # -- désactivé automatiquement, masque brut utilisé directement) :
    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_build_regressor_dataset.py \\
        --strategy thresh03 --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split train --max_batches 0 \\
        --out /tmp/student7/phase8_dataset_thresh03_train.npz

    CUDA_VISIBLE_DEVICES=1 python scripts/phase8_build_regressor_dataset.py \\
        --strategy thresh03 --ckpt output/cnn_step15_final_model/last.ckpt \\
        --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \\
        --experiment cnn_step15_final_model \\
        --categories everyday --split val --max_batches 0 \\
        --out /tmp/student7/phase8_dataset_thresh03_val.npz
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
from scripts.phase5a_depthmap_matching import quat_wxyz_to_rotmat, rasterize
from scripts.phase5a_zoom_resample_check import zoom_resample, to_input_frame
from scripts.phase5a_zoom_resample_thresh03_check import dominant_cluster_mask
from scripts.phase8_depthmap_regressor_dataset import canonical_pca_frame, fit_theta_shift_mirror_from_gt

ABS_MIN_POINTS = 50
RESOLUTION = 64


def build_frame_and_rasterize(frac_i, frac_j):
    """Repère PCA canonique + rasterisation à résolution fixe, pour LES
    DEUX fragments d'une paire -- même convention (pixel_size partagé)
    que `run_match_at_resolution`/`process_pair`."""
    c_i, u_i, v_i, n_i, _ = canonical_pca_frame(frac_i)
    c_j, u_j, v_j, n_j, _ = canonical_pca_frame(frac_j)

    ci, cj = frac_i - c_i, frac_j - c_j
    span_i = max(float((ci @ u_i).max() - (ci @ u_i).min()),
                 float((ci @ v_i).max() - (ci @ v_i).min()), 1e-8)
    span_j = max(float((cj @ u_j).max() - (cj @ u_j).min()),
                 float((cj @ v_j).max() - (cj @ v_j).min()), 1e-8)
    pixel_size = max(span_i, span_j) * 1.1 / RESOLUTION

    dmap_i, valid_i, u_min_i, v_min_i = rasterize(frac_i, c_i, u_i, v_i, n_i, RESOLUTION, pixel_size)
    dmap_j, valid_j, u_min_j, v_min_j = rasterize(frac_j, c_j, u_j, v_j, n_j, RESOLUTION, pixel_size)

    frame = {
        "c_i": c_i, "u_i": u_i, "v_i": v_i, "u_min_i": u_min_i, "v_min_i": v_min_i,
        "c_j": c_j, "u_j": u_j, "v_j": v_j, "u_min_j": u_min_j, "v_min_j": v_min_j,
        "pixel_size": pixel_size,
    }
    return dmap_i, valid_i, dmap_j, valid_j, frame


def process_pair(frac_i, frac_j, R_ij_gt, t_ij_gt):
    """Retourne un dict prêt à empiler dans le `.npz`, ou None si la paire
    n'est pas exploitable (masque trop épars)."""
    n_min = min(len(frac_i), len(frac_j))
    if n_min < ABS_MIN_POINTS:
        return None

    dmap_i, valid_i, dmap_j, valid_j, frame = build_frame_and_rasterize(frac_i, frac_j)

    theta_gt, shift_gt, mirror_gt, residual = fit_theta_shift_mirror_from_gt(
        frac_i, frame["c_i"], frame["u_i"], frame["v_i"], frame["u_min_i"], frame["v_min_i"],
        frame["c_j"], frame["u_j"], frame["v_j"], frame["u_min_j"], frame["v_min_j"],
        frame["pixel_size"], RESOLUTION, R_ij_gt, t_ij_gt,
    )
    sy, sx = shift_gt
    return {
        "dmap_i": dmap_i.astype(np.float32), "valid_i": valid_i.astype(np.float32),
        "dmap_j": dmap_j.astype(np.float32), "valid_j": valid_j.astype(np.float32),
        "theta_gt": np.float32(theta_gt), "shift_y_gt": np.float32(sy), "shift_x_gt": np.float32(sx),
        "mirror_gt": bool(mirror_gt), "residual": np.float32(residual),
        "n_frac_pts_min": int(n_min),
    }


def save_npz(rows, out_path):
    keys = ["dmap_i", "valid_i", "dmap_j", "valid_j", "theta_gt", "shift_y_gt",
            "shift_x_gt", "mirror_gt", "residual", "n_frac_pts_min"]
    arrays = {k: np.stack([r[k] for r in rows]) for k in keys}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["gt", "thresh03"])
    parser.add_argument("--ckpt", default="", help="Requis si --strategy thresh03")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--categories",  default="everyday")
    parser.add_argument("--split",       default="val", choices=["train", "val", "test"],
                        help="'train' -- BEAUCOUP plus de paires que val/test, mais SANS "
                             "zoom (meshes indisponibles en train pour la variante "
                             "weighted, `BreakingBadWeighted.transform()` plante même sans "
                             "les utiliser explicitement -- vérifié le 2026-07-30). Le zoom "
                             "est donc désactivé AUTOMATIQUEMENT pour 'train' (utilise le "
                             "masque brut directement) ; 'val'/'test' gardent le zoom.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--extra_budget", type=int, default=1200)
    parser.add_argument("--expand_rings", type=int, default=0)
    parser.add_argument("--num_points_to_sample", type=int, default=10000)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = tout le split")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.strategy == "thresh03" and not args.ckpt:
        parser.error("--ckpt est requis avec --strategy thresh03")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    n_seen_2frag = 0
    n_zoom_failed = 0
    t0 = time.time()

    if args.strategy == "gt":
        # `model_type="cnn"` (sample_method=uniform) plutôt que "garf"
        # (weighted) -- cette branche n'a JAMAIS utilisé le zoom/les meshes
        # (juste le masque GT brut), donc rien ne justifie le dataset
        # weighted, qui en plus PLANTE sur --split train (son transform()
        # essaie d'attacher les meshes inconditionnellement, même si
        # l'appelant ne les utilise pas). "cnn" fonctionne sur les 3 splits
        # et expose les mêmes champs GT (fracture_surface_gt, quaternions,
        # etc.) -- aucun compromis, juste un bug évité.
        print("Chargement du dataset CNN (sample_method=uniform), AUCUN CNN chargé "
              "(juste le format de données -- stratégie GT, oracle-first)")
        fake_args = argparse.Namespace(
            experiment=args.experiment, data_root=args.data_root,
            batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
        )
        cfg = load_config_and_model(fake_args)
        cfg.data.num_points_to_sample = args.num_points_to_sample
        datamodule = instantiate(cfg.data)
        datamodule.setup("fit" if args.split != "test" else "test")
        dataset = {"train": datamodule.train_dataset, "val": datamodule.val_dataset,
                   "test": datamodule.test_dataset}[args.split]

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=datamodule.dataset_cls.collate_fn)

        for idx, batch in enumerate(loader):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            if idx % 200 == 0:
                print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | paires générées="
                      f"{len(rows)} | {time.time()-t0:.0f}s écoulées")

            points_per_part = batch["points_per_part"][0].numpy()
            valid_slots = [p for p in range(len(points_per_part)) if points_per_part[p] > 0]
            if len(valid_slots) != 2:
                continue
            n_seen_2frag += 1

            pointclouds = batch["pointclouds"][0].numpy()
            fracture_gt = batch["fracture_surface_gt"][0].numpy()
            frag_pts = extract_gt_variable(pointclouds, points_per_part)
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
            gt_i, gt_j = frag_gt[0], frag_gt[1]

            frac_i = raw_i[gt_i == 1]
            frac_j = raw_j[gt_j == 1]

            R0 = quat_wxyz_to_rotmat(quats_np[p0])
            R1 = quat_wxyz_to_rotmat(quats_np[p1])
            R_ij_gt = R1.T @ R0
            t_ij_gt = R1.T @ (trans_np[p0] - trans_np[p1])

            row = process_pair(frac_i, frac_j, R_ij_gt, t_ij_gt)
            if row is not None:
                rows.append(row)

    else:   # thresh03
        from torch.utils.data import DataLoader
        from assembly.models.cnn_segmentation_model import CNNFracSeg
        from assembly.models.projection_mapping_utils import extract_fragment_list

        # Zoom désactivé automatiquement sur --split train : le dataset mesh
        # (weighted) plante sur ce split (meshes indisponibles, cf. --split
        # help ci-dessus) -- 'train' utilise donc le masque thresh0.3 BRUT
        # directement, sans rééchantillonnage sur maillage.
        use_zoom = args.split != "train"

        print("Chargement du dataset CNN (sample_method=uniform)...")
        cnn_fake_args = argparse.Namespace(
            experiment=args.experiment, data_root=args.data_root,
            batch_size=1, num_workers=4, categories=args.categories, model_type="cnn",
        )
        cnn_cfg = load_config_and_model(cnn_fake_args)
        cnn_datamodule = instantiate(cnn_cfg.data)
        cnn_datamodule.setup("fit" if args.split != "test" else "test")
        cnn_dataset = {"train": cnn_datamodule.train_dataset, "val": cnn_datamodule.val_dataset,
                       "test": cnn_datamodule.test_dataset}[args.split]

        mesh_dataset = None
        if use_zoom:
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
        else:
            print("--split train : zoom désactivé (meshes indisponibles), masque brut utilisé directement.")

        cnn_loader = DataLoader(cnn_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=cnn_datamodule.dataset_cls.collate_fn)

        print(f"Chargement du checkpoint CNN : {args.ckpt}")
        model = CNNFracSeg.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
        model.eval()
        model.to(device)

        with torch.no_grad():
            for idx, batch in enumerate(cnn_loader):
                if args.max_batches > 0 and idx >= args.max_batches:
                    break
                if idx % 200 == 0:
                    print(f"  objet {idx} | 2-frags vus={n_seen_2frag} | paires générées="
                          f"{len(rows)} | {time.time()-t0:.0f}s écoulées")

                mesh_data = None
                if use_zoom:
                    mesh_data = mesh_dataset[idx]
                    assert batch["name"][0] == mesh_data["name"]

                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                frag_list, valid_pcs, K = extract_fragment_list(
                    batch_gpu["pointclouds"], batch_gpu["points_per_part"])
                if K != 2:
                    continue
                n_seen_2frag += 1

                gt_frag_list, _, _ = extract_fragment_list(
                    batch_gpu["pointclouds_gt"], batch_gpu["points_per_part"])

                out = model(batch_gpu)
                pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
                frag_sizes = [f.shape[0] for f in frag_list]
                offsets = np.concatenate([[0], np.cumsum(frag_sizes)])
                score_per_k = [pred_flat[offsets[k]:offsets[k + 1]] for k in range(K)]
                pc_per_k    = [frag_list[k].cpu().numpy()    for k in range(K)]
                gt_per_k    = [gt_frag_list[k].cpu().numpy() for k in range(K)]

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

                R0 = quat_wxyz_to_rotmat(quats_np[0, p0])
                R1 = quat_wxyz_to_rotmat(quats_np[0, p1])
                R_ij_gt = R1.T @ R0
                t_ij_gt = R1.T @ (trans_np[0, p0] - trans_np[0, p1])

                mask0 = sc0 > args.threshold
                mask1 = sc1 > args.threshold
                n_min_raw = min(int(mask0.sum()), int(mask1.sum()))
                if n_min_raw < ABS_MIN_POINTS:
                    continue

                if use_zoom:
                    cluster_mask0 = dominant_cluster_mask(raw0[mask0])
                    cluster_mask1 = dominant_cluster_mask(raw1[mask1])
                    R_init_rot = quat_wxyz_to_rotmat(batch["init_rot"].numpy()[0])
                    seed_i_gt = (gtgt0[mask0][cluster_mask0]) @ R_init_rot.T
                    seed_j_gt = (gtgt1[mask1][cluster_mask1]) @ R_init_rot.T
                    meshes = mesh_data["meshes"]
                    new_i_gt = zoom_resample(meshes[p0], seed_i_gt, args.extra_budget,
                                              args.expand_rings, rng)
                    new_j_gt = zoom_resample(meshes[p1], seed_j_gt, args.extra_budget,
                                              args.expand_rings, rng)
                    if new_i_gt is None or new_j_gt is None:
                        n_zoom_failed += 1
                        continue
                    new_i_rotated = new_i_gt @ R_init_rot
                    new_j_rotated = new_j_gt @ R_init_rot
                    new_i_input = to_input_frame(new_i_rotated, quats_np[0, p0], trans_np[0, p0])
                    new_j_input = to_input_frame(new_j_rotated, quats_np[0, p1], trans_np[0, p1])
                    frac_i = np.concatenate([raw0[mask0], new_i_input], axis=0)
                    frac_j = np.concatenate([raw1[mask1], new_j_input], axis=0)
                else:
                    # --split train : pas de meshes -- masque thresh0.3 brut directement.
                    frac_i = raw0[mask0]
                    frac_j = raw1[mask1]

                row = process_pair(frac_i, frac_j, R_ij_gt, t_ij_gt)
                if row is not None:
                    rows.append(row)

    elapsed = time.time() - t0
    print(f"\nFini : {len(rows)} paires générées sur {n_seen_2frag} objets 2-frags vus "
          f"({n_zoom_failed} zooms impossibles) en {elapsed:.0f}s")

    if not rows:
        print("Aucune paire exploitable -- rien à sauvegarder.")
        return

    n_mirror = sum(1 for r in rows if r["mirror_gt"])
    residuals = [r["residual"] for r in rows]
    print(f"  mirror_gt=True : {n_mirror}/{len(rows)} ({100*n_mirror/len(rows):.1f}%)")
    print(f"  résidu de fit (RMS pixels) : moyenne={np.mean(residuals):.4f}, "
          f"p95={np.percentile(residuals,95):.4f} (devrait être proche de 0 -- "
          f"sinon la dérivation du label a échoué sur certaines paires)")

    save_npz(rows, args.out)
    print(f"\nDataset sauvegardé : {args.out} ({len(rows)} paires, résolution {RESOLUTION})")


if __name__ == "__main__":
    main()
