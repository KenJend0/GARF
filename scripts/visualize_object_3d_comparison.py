"""
scripts/visualize_object_3d_comparison.py
==========================================
Visualisation 3D interactive d'un objet complet de Breaking Bad.

Compare côte à côte :
  - Ground Truth   (surfaces fracturées réelles)
  - Notre modèle   (CNN Step 9/10)
  - GARF baseline  (PTv3 FracSeg, optionnel)

Code couleur des points :
  gris   (#D8D8D8) = surface intacte (TN)
  jaune  (#FFD600) = fracture GT uniquement
  vert   (#4CAF50) = TP  (fracture détectée correctement)
  rouge  (#F44336) = FP  (fausse alarme)
  bleu   (#2196F3) = FN  (fracture manquée)

Usage:
    # Par nom d'objet dans le HDF5 :
    python scripts/visualize_object_3d_comparison.py \\
        --name everyday/BeerBottle/BeerBottle-1 \\
        --ckpt output/cnn_step9_geo_features/version_0/checkpoints/last.ckpt \\
        --data_root /path/to/breaking_bad_vol.hdf5

    # Par index dans le split val :
    python scripts/visualize_object_3d_comparison.py \\
        --index 42 \\
        --ckpt output/cnn_step9_geo_features/version_0/checkpoints/last.ckpt \\
        --data_root /path/to/breaking_bad_vol.hdf5 \\
        --split val

    # Avec comparaison GARF + export HTML Plotly :
    python scripts/visualize_object_3d_comparison.py \\
        --index 42 \\
        --ckpt output/cnn_step9_geo_features/version_0/checkpoints/last.ckpt \\
        --garf_ckpt /path/to/GARF_mini.ckpt \\
        --data_root /path/to/breaking_bad_vol.hdf5 \\
        --threshold 0.65 \\
        --export_html \\
        --out_dir output/viz_3d

    # Affichage GT seul (sans modèle) :
    python scripts/visualize_object_3d_comparison.py \\
        --index 42 \\
        --data_root /path/to/breaking_bad_vol.hdf5 \\
        --gt_only
"""

import argparse
import sys
import functools
from pathlib import Path

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([functools.partial])
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLOR_TN     = np.array([0.847, 0.847, 0.847])   # gris  — surface intacte
COLOR_GT     = np.array([1.000, 0.839, 0.000])   # jaune — GT fracture (sans prediction)
COLOR_TP     = np.array([0.298, 0.686, 0.314])   # vert  — TP
COLOR_FP     = np.array([0.957, 0.263, 0.212])   # rouge — FP
COLOR_FN     = np.array([0.129, 0.588, 0.953])   # bleu  — FN


def colorize_fragment(pred_b: np.ndarray, gt_b: np.ndarray) -> np.ndarray:
    """Retourne un tableau (N, 3) de couleurs RGB [0,1] selon TP/FP/FN/TN."""
    colors = np.tile(COLOR_TN, (len(gt_b), 1))
    colors[gt_b & pred_b]   = COLOR_TP
    colors[~gt_b & pred_b]  = COLOR_FP
    colors[gt_b & ~pred_b]  = COLOR_FN
    return colors


def colorize_gt_only(gt_b: np.ndarray) -> np.ndarray:
    """Colorie uniquement selon GT (jaune = fracture, gris = intact)."""
    colors = np.tile(COLOR_TN, (len(gt_b), 1))
    colors[gt_b] = COLOR_GT
    return colors


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_object_data(data_root: str, name: str = None, index: int = None,
                     split: str = "val", num_points: int = 5000,
                     category: str = "everyday"):
    """
    Charge les données brutes d'un objet (meshes + labels fracture).

    Retourne:
        raw     : dict de get_data() (nuages de points GT, labels, meshes)
        dataset : instance BreakingBadUniform (pour accéder à transform si besoin)
        obj_name: clé HDF5 de l'objet
    """
    from assembly.data.breaking_bad.uniform import BreakingBadUniform

    dataset = BreakingBadUniform(
        split=split,
        data_root=data_root,
        category=category,
        num_points_to_sample=num_points,
        mesh_sample_strategy="uniform",
    )

    if name is not None:
        if name not in dataset.data_list:
            raise ValueError(f"Object '{name}' not found in split='{split}'. "
                             f"First objects: {dataset.data_list[:5]}")
        idx = dataset.data_list.index(name)
    elif index is not None:
        idx = index
        name = dataset.data_list[idx]
    else:
        raise ValueError("Provide --name or --index.")

    print(f"Loading object: {name}  (index={idx})")
    raw = dataset.get_data(idx)
    return raw, dataset, name


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _make_cnn_batch_no_rotation(raw: dict, dataset, device: torch.device):
    """
    Prépare un batch CNN sans rotation aléatoire ni shuffle.
    → Chaque fragment est centré puis normalisé, mais dans l'ordre original des points.
    → Les coordonnées GT retournées sont celles de raw["pointclouds_gt"] (assemblées, non tournées).

    Returns:
        batch   : dict prêt pour CNNFracSeg.forward()
        xyz_list: list de (N, 3) — coordonnées GT originales (non tournées) par fragment
        gt_list : list de (N,) bool — labels fracture dans le même ordre de points
    """
    num_parts = raw["num_parts"]
    N         = dataset.num_points_to_sample
    max_parts = dataset.max_parts

    xyz_list = [raw["pointclouds_gt"][i].astype(np.float32)          for i in range(num_parts)]
    nrm_list = [raw["pointclouds_normals_gt"][i].astype(np.float32)  for i in range(num_parts)]
    gt_list  = [raw["fracture_surface_gt"][i].astype(bool)           for i in range(num_parts)]

    # Centrer + normaliser chaque fragment pour le modèle (sans rotation ni shuffle)
    pointclouds_model = []
    normals_model     = []
    for i in range(num_parts):
        xyz = xyz_list[i].copy()
        xyz -= xyz.mean(axis=0)           # centrage
        s = np.max(np.abs(xyz))
        if s > 0:
            xyz /= s                      # normalisation
        pointclouds_model.append(xyz)
        normals_model.append(nrm_list[i].copy())

    # Padding à max_parts
    pc_pad  = np.zeros((max_parts, N, 3), dtype=np.float32)
    nm_pad  = np.zeros((max_parts, N, 3), dtype=np.float32)
    fr_pad  = np.zeros((max_parts, N),    dtype=np.int8)
    ppp     = np.zeros(max_parts,         dtype=np.int64)
    graph   = raw["graph"].astype(np.float32)

    for i in range(num_parts):
        pc_pad[i] = pointclouds_model[i]
        nm_pad[i] = normals_model[i]
        fr_pad[i] = gt_list[i].astype(np.int8)
        ppp[i]    = N

    batch = {
        "pointclouds":         torch.tensor(pc_pad).unsqueeze(0).to(device),
        "pointclouds_normals": torch.tensor(nm_pad).unsqueeze(0).to(device),
        "points_per_part":     torch.tensor(ppp).unsqueeze(0).to(device),
        "fracture_surface_gt": torch.tensor(fr_pad.reshape(1, -1)).to(device),
        "graph":               torch.tensor(graph).unsqueeze(0).to(device),
        "name":                [raw.get("name", "")],
        "removal_pieces":      [raw.get("removal_pieces", "")],
        "redundant_pieces":    [raw.get("redundant_pieces", "")],
    }
    return batch, xyz_list, gt_list


def run_cnn_inference(raw: dict, dataset, ckpt: str, threshold: float, device: torch.device):
    """
    Exécute CNNFracSeg sans rotation aléatoire.
    Les coordonnées xyz retournées sont les coordonnées GT originales (assemblées).
    """
    from assembly.models.cnn_segmentation_model import CNNFracSeg

    model = CNNFracSeg.load_from_checkpoint(ckpt, map_location=device, weights_only=False)
    model.eval().to(device)

    batch, xyz_list, gt_list = _make_cnn_batch_no_rotation(raw, dataset, device)

    with torch.no_grad():
        out = model(batch)

    pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
    gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()

    num_parts = raw["num_parts"]
    N         = dataset.num_points_to_sample

    preds_b, gts_b = [], []
    for i in range(num_parts):
        start, end = i * N, (i + 1) * N
        preds_b.append(pred_flat[start:end] > threshold)
        gts_b.append(gt_flat[start:end].astype(bool))

    return preds_b, gts_b, xyz_list


def run_garf_inference(raw: dict, dataset, ckpt: str, threshold: float, device: torch.device):
    """
    Exécute FracSeg (PTv3) sans rotation aléatoire.
    Utilise les MÊMES points que CNN (coordonnées GT brutes) en subsamplant à 5000 pts total.
    → Tous les panels (GT, CNN, GARF) partagent le même système de coordonnées.
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate as hydra_instantiate
    from hydra.core.global_hydra import GlobalHydra

    num_parts = raw["num_parts"]
    N_uniform = dataset.num_points_to_sample   # 5000 pts/fragment (CNN)
    max_parts = dataset.max_parts
    N_total_garf = 5000                        # total pts pour GARF (distribution entraînement)

    # Nombre de points par fragment pour GARF (répartition uniforme entre fragments)
    pts_per_frag = max(1, N_total_garf // num_parts)
    pts_per_frag = min(pts_per_frag, N_uniform)  # ne pas dépasser ce qu'on a

    # Sous-échantillonner les points GT (même base que CNN, subsample pour GARF)
    rng = np.random.default_rng(seed=0)  # seed fixe → reproducible
    xyz_list, gt_list, pc_model, nm_model, n_per_frag = [], [], [], [], []
    for i in range(num_parts):
        xyz_full = raw["pointclouds_gt"][i].astype(np.float32)
        nrm_full = raw["pointclouds_normals_gt"][i].astype(np.float32)
        gt_full  = raw["fracture_surface_gt"][i].astype(bool)

        # Subsample à pts_per_frag
        idx = rng.choice(len(xyz_full), size=pts_per_frag, replace=False)
        xyz = xyz_full[idx]
        nrm = nrm_full[idx]
        gt  = gt_full[idx]

        xyz_list.append(xyz)
        gt_list.append(gt)
        n_per_frag.append(pts_per_frag)

        # Centrer + normaliser pour le modèle (sans rotation)
        xyz_c = xyz - xyz.mean(axis=0)
        s = np.max(np.abs(xyz_c))
        if s > 0:
            xyz_c /= s
        pc_model.append(xyz_c)
        nm_model.append(nrm.copy())

    # Concaténer en format FracSeg (flat, pas stacké)
    pc_concat  = np.concatenate(pc_model).astype(np.float32)
    nm_concat  = np.concatenate(nm_model).astype(np.float32)
    gt_concat  = np.concatenate(gt_list).astype(np.int64)

    ppp = np.zeros(max_parts, dtype=np.int64)
    for i in range(num_parts):
        ppp[i] = n_per_frag[i]

    # Extraire les poids feature_extractor
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    keys = list(ckpt_data.get("state_dict", {}).keys())
    is_garf = any(k.startswith("feature_extractor.") for k in keys)
    state_dict = {
        k.replace("feature_extractor.", ""): v
        for k, v in ckpt_data["state_dict"].items()
        if k.startswith("feature_extractor.")
    } if is_garf else ckpt_data.get("state_dict", ckpt_data)

    # Instancier FracSeg via Hydra
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="eval", overrides=["experiment=eval_frac_seg"])
    GlobalHydra.instance().clear()

    model = hydra_instantiate(cfg.model)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    batch = {
        "pointclouds":         torch.tensor(pc_concat).unsqueeze(0).to(device),
        "pointclouds_normals": torch.tensor(nm_concat).unsqueeze(0).to(device),
        "points_per_part":     torch.tensor(ppp).unsqueeze(0).to(device),
        "fracture_surface_gt": torch.tensor(gt_concat).unsqueeze(0).to(device),
        "graph":               torch.tensor(raw["graph"].astype(np.float32)).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        out = model(batch)

    pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
    gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()

    preds_b, gts_b = [], []
    offset = 0
    for i in range(num_parts):
        n_i = n_per_frag[i]
        preds_b.append(pred_flat[offset:offset + n_i] > threshold)
        gts_b.append(gt_flat[offset:offset + n_i].astype(bool))
        offset += n_i

    return preds_b, gts_b, xyz_list


# ---------------------------------------------------------------------------
# Open3D visualization
# ---------------------------------------------------------------------------

def build_open3d_pcd(xyz_list, colors_list, offset_x=0.0):
    """Construit un PointCloud Open3D à partir de plusieurs fragments."""
    import open3d as o3d

    all_xyz    = np.concatenate(xyz_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)

    all_xyz[:, 0] += offset_x

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))
    return pcd


def show_open3d(panels: list, title: str = "Fracture Visualization"):
    """
    Affiche plusieurs PointClouds côte à côte dans une fenêtre Open3D interactive.

    panels : list de (label, pcd_o3d) — les nuages à afficher côte à côte
    """
    import open3d as o3d

    print(f"\nOpening Open3D viewer: '{title}'")
    print("  Controls: mouse drag = rotate, scroll = zoom, Ctrl+drag = pan, Q/Esc = quit")

    geometries = []
    labels_text = []

    for i, (label, pcd) in enumerate(panels):
        # Espacer les objets horizontalement
        pts = np.asarray(pcd.points)
        extent_x = pts[:, 0].max() - pts[:, 0].min() if len(pts) > 0 else 1.0
        shift = i * (extent_x + 0.3)
        shifted = o3d.geometry.PointCloud(pcd)
        shifted.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) + np.array([shift, 0, 0])
        )
        geometries.append(shifted)
        labels_text.append((label, shift))
        print(f"  Panel {i+1}: {label}  ({len(pts):,} points)")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1400,
        height=700,
        point_show_normal=False,
    )


# ---------------------------------------------------------------------------
# Plotly HTML export
# ---------------------------------------------------------------------------

def export_plotly_html(panels: list, out_path: Path, title: str = "Fracture 3D"):
    """Exporte une visualisation Plotly 3D interactive en HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[warn] plotly not installed — skipping HTML export (pip install plotly)")
        return

    n_panels = len(panels)
    specs = [[{"type": "scatter3d"}] * n_panels]
    fig = make_subplots(rows=1, cols=n_panels, specs=specs,
                        subplot_titles=[p[0] for p in panels])

    for col, (label, pcd) in enumerate(panels, start=1):
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        rgb_str = [
            f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
            for c in cols
        ]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=rgb_str, opacity=0.8),
                name=label,
                showlegend=True,
            ),
            row=1, col=col,
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        height=700,
    )
    fig.write_html(str(out_path))
    print(f"  Saved HTML: {out_path}")


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def print_fragment_metrics(label: str, preds_b_list, gts_b_list):
    tp = fp = fn = tn = 0
    for p, g in zip(preds_b_list, gts_b_list):
        tp += int((p & g).sum())
        fp += int((p & ~g).sum())
        fn += int((~p & g).sum())
        tn += int((~p & ~g).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    fdr  = fp / max(fp + tp, 1)
    print(f"  [{label}]  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}"
          f"  FDR={fdr:.3f}  TP={tp:,}  FP={fp:,}  FN={fn:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3D interactive fracture visualization")
    p.add_argument("--name",       default=None,  help="HDF5 object name (e.g. everyday/BeerBottle/BeerBottle-1)")
    p.add_argument("--index",      type=int, default=None, help="Object index in the dataset split")
    p.add_argument("--data_root",  required=True, help="Path to breaking_bad_vol.hdf5")
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--category",   default="everyday", choices=["everyday", "artifact", "all"])
    p.add_argument("--ckpt",       default=None,  help="CNN checkpoint (.ckpt)")
    p.add_argument("--garf_ckpt",  default=None,  help="GARF or FracSeg checkpoint (.ckpt)")
    p.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold")
    p.add_argument("--num_points", type=int, default=5000, help="Points per fragment")
    p.add_argument("--gt_only",    action="store_true", help="Show GT only, no model inference")
    p.add_argument("--export_html", action="store_true", help="Export Plotly HTML instead of Open3D")
    p.add_argument("--out_dir",    default="output/viz_3d", help="Output directory for HTML")
    p.add_argument("--no_display", action="store_true", help="Skip Open3D display (only export HTML)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load object ---
    raw, dataset, obj_name = load_object_data(
        data_root=args.data_root,
        name=args.name,
        index=args.index,
        split=args.split,
        num_points=args.num_points,
        category=args.category,
    )

    num_parts = raw["num_parts"]
    print(f"Object: {obj_name}  ({num_parts} fragments)")

    panels = []

    try:
        import open3d as o3d
        has_o3d = True
    except ImportError:
        print("[warn] open3d not installed — install with: pip install open3d")
        has_o3d = False

    # Coordonnées GT brutes (assemblées, aucune rotation) — base commune pour tous les panels
    xyz_raw  = [raw["pointclouds_gt"][i].astype(np.float32)   for i in range(num_parts)]
    gt_raw   = [raw["fracture_surface_gt"][i].astype(bool)    for i in range(num_parts)]

    # --- Panel GT ---
    if has_o3d:
        gt_colors = [colorize_gt_only(g) for g in gt_raw]
        gt_pcd = build_open3d_pcd(xyz_raw, gt_colors)
        panels.append(("Ground Truth", gt_pcd))
    else:
        gt_pcd = None

    # --- Panel CNN ---
    cnn_preds = cnn_gts = None
    if args.ckpt and not args.gt_only:
        print(f"\nRunning CNN inference (threshold={args.threshold})...")
        cnn_preds, cnn_gts, _ = run_cnn_inference(raw, dataset, args.ckpt,
                                                   args.threshold, device)
        print_fragment_metrics("CNN", cnn_preds, cnn_gts)

    if cnn_preds is not None:
        cnn_colors = [colorize_fragment(p, g) for p, g in zip(cnn_preds, gt_raw)]
        if has_o3d:
            cnn_pcd = build_open3d_pcd(xyz_raw, cnn_colors)  # même xyz que GT
            panels.append((f"CNN (thr={args.threshold})", cnn_pcd))
    else:
        if not args.gt_only and not args.ckpt:
            print("[info] No --ckpt provided — showing GT only.")

    # --- Panel GARF ---
    if args.garf_ckpt and not args.gt_only:
        print(f"\nRunning GARF/FracSeg inference (threshold={args.threshold})...")
        try:
            garf_preds, garf_gts, garf_xyz = run_garf_inference(raw, dataset, args.garf_ckpt,
                                                                  args.threshold, device)
            print_fragment_metrics("GARF", garf_preds, garf_gts)
            # gt_raw_sub : sous-ensemble de gt_raw (mêmes indices que garf_xyz)
            # garf_xyz est déjà un sous-ensemble de xyz_raw (même seed=0, même indices)
            gt_raw_sub = garf_gts   # GT labels dans le même ordre que garf_xyz
            garf_colors = [colorize_fragment(p, g) for p, g in zip(garf_preds, gt_raw_sub)]
            if has_o3d:
                garf_pcd = build_open3d_pcd(garf_xyz, garf_colors)
                panels.append((f"GARF (thr={args.threshold})", garf_pcd))
        except Exception as e:
            print(f"[warn] GARF inference failed: {e}")

    # --- Display ---
    safe_name = obj_name.replace("/", "_")

    if args.export_html and panels:
        html_path = out_dir / f"{safe_name}_comparison.html"
        export_plotly_html(panels, html_path, title=f"Fracture 3D — {obj_name}")

    if not args.no_display and panels:
        show_open3d(panels, title=f"Fracture 3D — {obj_name}")
    elif not panels:
        print("[warn] No panels to display. Provide --ckpt or check Open3D installation.")

    print("\nDone.")


if __name__ == "__main__":
    main()
