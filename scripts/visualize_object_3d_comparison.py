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

def run_cnn_inference(raw: dict, dataset, ckpt: str, threshold: float, device: torch.device):
    """
    Exécute le modèle CNN sur tous les fragments d'un objet.

    Retourne:
        preds_b : list de (N,) bool — prédiction binaire par fragment
        gt_b    : list de (N,) bool — GT binaire par fragment
        xyz_list: list de (N, 3) float — coordonnées 3D par fragment
    """
    from assembly.models.cnn_segmentation_model import CNNFracSeg

    model = CNNFracSeg.load_from_checkpoint(ckpt, map_location=device, weights_only=False)
    model.eval().to(device)

    data = dataset.transform(raw.copy())

    # Build a batch of size 1
    batch = {
        k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }
    # String keys
    for k in ["name", "removal_pieces", "redundant_pieces"]:
        if k in data:
            batch[k] = [data[k]]

    with torch.no_grad():
        out = model(batch)

    pred_flat = out["coarse_seg_pred"].float().cpu().numpy()   # (N_total,)
    gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()      # (N_total,)

    # Split into per-fragment arrays
    num_parts = raw["num_parts"]
    N = dataset.num_points_to_sample

    preds_b, gts_b, xyz_list = [], [], []
    for i in range(num_parts):
        start, end = i * N, (i + 1) * N
        p = pred_flat[start:end]
        g = gt_flat[start:end].astype(bool)
        preds_b.append(p > threshold)
        gts_b.append(g)
        xyz_list.append(raw["pointclouds_gt"][i])

    return preds_b, gts_b, xyz_list


def run_garf_inference(raw: dict, dataset, ckpt: str, threshold: float, device: torch.device):
    """
    Exécute le modèle FracSeg (PTv3) extrait d'un checkpoint GARF.
    Retourne preds_b, gts_b, xyz_list (même format que run_cnn_inference).
    """
    import tempfile, os
    from assembly.models.pretraining.frac_seg import FracSeg

    # Détecter si c'est un checkpoint GARF complet
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    keys = list(ckpt_data.get("state_dict", {}).keys())
    is_garf = any(k.startswith("feature_extractor.") for k in keys)

    if is_garf:
        frac_seg_state = {
            k.replace("feature_extractor.", ""): v
            for k, v in ckpt_data["state_dict"].items()
            if k.startswith("feature_extractor.")
        }
        tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
        tmp.close()
        import lightning as L
        torch.save({"state_dict": frac_seg_state,
                    "pytorch-lightning_version": L.__version__}, tmp.name)
        load_path = tmp.name
    else:
        load_path = ckpt
        tmp = None

    try:
        model = FracSeg.load_from_checkpoint(load_path, map_location=device, weights_only=False)
        model.eval().to(device)

        data = dataset.transform(raw.copy())
        batch = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }
        for k in ["name", "removal_pieces", "redundant_pieces"]:
            if k in data:
                batch[k] = [data[k]]

        with torch.no_grad():
            out = model(batch)

        pred_flat = out["coarse_seg_pred"].float().cpu().numpy()
        gt_flat   = out["coarse_seg_gt"].long().cpu().numpy()

        num_parts = raw["num_parts"]
        N = dataset.num_points_to_sample
        preds_b, gts_b, xyz_list = [], [], []
        for i in range(num_parts):
            start, end = i * N, (i + 1) * N
            p = pred_flat[start:end]
            g = gt_flat[start:end].astype(bool)
            preds_b.append(p > threshold)
            gts_b.append(g)
            xyz_list.append(raw["pointclouds_gt"][i])

        return preds_b, gts_b, xyz_list
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


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

    # GT arrays
    gt_list  = [raw["fracture_surface_gt"][i].astype(bool) for i in range(num_parts)]
    xyz_list = [raw["pointclouds_gt"][i] for i in range(num_parts)]

    panels = []

    # --- Panel GT ---
    gt_colors = [colorize_gt_only(g) for g in gt_list]
    try:
        import open3d as o3d
        gt_pcd = build_open3d_pcd(xyz_list, gt_colors)
        panels.append(("Ground Truth", gt_pcd))
    except ImportError:
        print("[warn] open3d not installed — install with: pip install open3d")
        gt_pcd = None

    # --- Panel CNN ---
    if args.ckpt and not args.gt_only:
        print(f"\nRunning CNN inference (threshold={args.threshold})...")
        cnn_preds, cnn_gts, _ = run_cnn_inference(raw, dataset, args.ckpt,
                                                   args.threshold, device)
        print_fragment_metrics("CNN", cnn_preds, cnn_gts)
        cnn_colors = [colorize_fragment(p, g) for p, g in zip(cnn_preds, cnn_gts)]
        if gt_pcd is not None:
            cnn_pcd = build_open3d_pcd(xyz_list, cnn_colors)
            panels.append((f"CNN (thr={args.threshold})", cnn_pcd))
    else:
        if not args.gt_only and not args.ckpt:
            print("[info] No --ckpt provided — showing GT only.")

    # --- Panel GARF ---
    if args.garf_ckpt and not args.gt_only:
        print(f"\nRunning GARF/FracSeg inference (threshold={args.threshold})...")
        try:
            garf_preds, garf_gts, _ = run_garf_inference(raw, dataset, args.garf_ckpt,
                                                          args.threshold, device)
            print_fragment_metrics("GARF", garf_preds, garf_gts)
            garf_colors = [colorize_fragment(p, g) for p, g in zip(garf_preds, garf_gts)]
            if gt_pcd is not None:
                garf_pcd = build_open3d_pcd(xyz_list, garf_colors)
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
