#!/usr/bin/env python3
"""
scripts/create_presentation.py
Présentation GARF — CNN ablation study.
1 slide = 1 idée. Max 4 bullets. Visuels larges.

Usage:
    pip install python-pptx matplotlib scipy
    python scripts/create_presentation.py --out presentation.pptx
"""

import argparse, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ───────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1A, 0x2E, 0x4A)
TEAL   = RGBColor(0x00, 0x7B, 0x9E)
LBLUE  = RGBColor(0xD6, 0xEA, 0xF8)
GREEN  = RGBColor(0x1E, 0x8B, 0x4C)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
RED    = RGBColor(0xBF, 0x2B, 0x2B)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LGRAY  = RGBColor(0xF2, 0xF3, 0xF4)
DGRAY  = RGBColor(0x5D, 0x6D, 0x7E)
DARK   = RGBColor(0x17, 0x20, 0x2A)
GOLD   = RGBColor(0xD4, 0xAC, 0x0D)
_BG    = "#F2F3F4"

SW = Inches(13.33)
SH = Inches(7.50)

# ── Results data ──────────────────────────────────────────────────────────
RESULTS = [
    {"label": "Step 1 — Baseline",    "acc": 0.7897, "f1": 0.6805, "prec": 0.6029, "rec": 0.8107},
    {"label": "Step 2 + Normals",     "acc": 0.7988, "f1": 0.7090, "prec": 0.6139, "rec": 0.8732},
    {"label": "Step 3 + Splatting",   "acc": 0.8240, "f1": 0.7193, "prec": 0.6338, "rec": 0.8715},
    {"label": "Step 4 + Bilinear",    "acc": 0.8115, "f1": 0.7087, "prec": 0.6093, "rec": 0.8851},
    {"label": "Step 5 + Context",     "acc": 0.8248, "f1": 0.7262, "prec": 0.6286, "rec": 0.8882},
    {"label": "Step 6 + Attention",   "acc": None,   "f1": None,   "prec": None,   "rec": None  },
    {"label": "GARF (PTv3 baseline)", "acc": 0.9257, "f1": 0.9094, "prec": 0.9143, "rec": 0.9115},
]

# ── PPTX helpers ──────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def rect(sl, l, t, w, h, fill=TEAL, line=None, lw=Pt(0)):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line: s.line.color.rgb = line; s.line.width = lw
    else:    s.line.color.rgb = fill; s.line.width = Pt(0)
    return s

def tb(sl, text, l, t, w, h, size=18, bold=False, italic=False,
       color=DARK, align=PP_ALIGN.LEFT):
    b = sl.shapes.add_textbox(l, t, w, h)
    b.text_frame.word_wrap = True
    p = b.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = color
    return b

def img(sl, buf, l, t, w, h):
    sl.shapes.add_picture(buf, l, t, w, h)

def header(sl, title, subtitle=None):
    rect(sl, 0, 0, SW, Inches(1.1), fill=NAVY)
    tb(sl, title, Inches(0.5), Inches(0.1), Inches(11), Inches(0.75),
       size=30, bold=True, color=WHITE)
    if subtitle:
        tb(sl, subtitle, Inches(0.5), Inches(0.8), Inches(11), Inches(0.32),
           size=13, color=RGBColor(0xAA, 0xBB, 0xCC))

def bullets(sl, items, l, t, w, size=17, gap=0.65):
    for i, txt in enumerate(items):
        tb(sl, "•  " + txt, l, t + Inches(i * gap), w, Inches(gap),
           size=size, color=DARK)

def badge(sl, text, l, t, w=2.5, col=TEAL):
    rect(sl, Inches(l), Inches(t), Inches(w), Inches(0.44), fill=col)
    tb(sl, text, Inches(l), Inches(t + 0.04), Inches(w), Inches(0.38),
       size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ── Synthetic fragment ────────────────────────────────────────────────────

def _make_fragment(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2*np.pi, n)
    r  = np.sqrt(rng.uniform(0, 1, n))
    x, y = r*np.cos(th), r*np.sin(th)
    z = 0.28*(x**2+y**2-0.5) + rng.normal(0, 0.022, n)
    pts = np.column_stack([x, y, z])
    nrm = np.column_stack([-0.56*x, -0.56*y, np.ones(n)])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    frac = (r > 0.73) | (np.abs(z - z.min()) < 0.06)
    return pts, nrm, frac.astype(bool)

_PTS, _NRM, _FRAC = _make_fragment()

def _n01(a): return (a-a.min())/(a.max()-a.min()+1e-8)
def _pix(pts, ax, H, W):
    ui, vi = [(0,2),(1,2),(0,1)][ax]
    px = np.clip((_n01(pts[:,ui])*(W-1)).astype(int), 0, W-1)
    py = np.clip((_n01(pts[:,vi])*(H-1)).astype(int), 0, H-1)
    return px, py

def proj_floor(pts, ax, H=80, W=80):
    px, py = _pix(pts, ax, H, W)
    di = [1,0,2][ax]; d = _n01(pts[:,di])
    depth = np.full((H,W), np.nan)
    for i in range(len(pts)):
        if np.isnan(depth[py[i],px[i]]) or d[i] > depth[py[i],px[i]]:
            depth[py[i],px[i]] = d[i]
    occ = ~np.isnan(depth)
    return np.where(occ, depth, 0), occ

def proj_splat(pts, ax, H=80, W=80, sigma=1.5):
    px, py = _pix(pts, ax, H, W)
    di = [1,0,2][ax]; d = _n01(pts[:,di])
    acc = np.zeros((H,W)); cnt = np.zeros((H,W))
    for i in range(len(pts)):
        acc[py[i],px[i]] += d[i]; cnt[py[i],px[i]] += 1
    avg = np.where(cnt>0, acc/np.where(cnt>0,cnt,1), 0)
    return gaussian_filter(avg, sigma), gaussian_filter(cnt, sigma)>0.05

def proj_bilinear(pts, ax, H=80, W=80):
    ui, vi = [(0,2),(1,2),(0,1)][ax]; di = [1,0,2][ax]
    u = _n01(pts[:,ui])*(W-1); v = _n01(pts[:,vi])*(H-1); d = _n01(pts[:,di])
    acc = np.zeros((H,W)); wa = np.zeros((H,W))
    for i in range(len(pts)):
        x0,y0 = int(u[i]),int(v[i]); x1,y1 = min(x0+1,W-1),min(y0+1,H-1)
        wx,wy = u[i]-x0, v[i]-y0
        for bx,by,bw in [(x0,y0,(1-wx)*(1-wy)),(x1,y0,wx*(1-wy)),
                          (x0,y1,(1-wx)*wy),(x1,y1,wx*wy)]:
            acc[by,bx]+=bw*d[i]; wa[by,bx]+=bw
    return np.where(wa>0, acc/np.where(wa>0,wa,1), 0), wa>0.01

def nmap(pts, nrm, ax, H=80, W=80):
    px, py = _pix(pts, ax, H, W)
    nm = np.zeros((H,W,3)); cnt = np.zeros((H,W))
    for i in range(len(pts)):
        nm[py[i],px[i]] += (nrm[i]+1)/2; cnt[py[i],px[i]] += 1
    return np.where(cnt[:,:,None]>0, nm/np.where(cnt[:,:,None]>0,cnt[:,:,None],1), 0)

def _save(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig); return buf

def _clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_facecolor(_BG)

# ── Figures ───────────────────────────────────────────────────────────────

def fig_fragment_projections():
    fig = plt.figure(figsize=(11, 3.0), facecolor=_BG)
    ax3 = fig.add_subplot(1,4,1, projection='3d', facecolor=_BG)
    c = ['#e74c3c' if f else '#3498db' for f in _FRAC]
    ax3.scatter(_PTS[:,0], _PTS[:,1], _PTS[:,2], c=c, s=2, alpha=0.7, linewidths=0)
    ax3.set_title("3D fragment\n(red = fracture)", fontsize=9)
    ax3.set_axis_off()
    for i, name in enumerate(["Front view", "Side view", "Top view"]):
        ax = fig.add_subplot(1,4,i+2); d,o = proj_floor(_PTS, ax=i)
        ax.imshow(np.where(o,d,np.nan), cmap='plasma', origin='lower',
                  vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(name, fontsize=9); ax.axis('off'); ax.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _save(fig)

def fig_normals_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor=_BG)
    d, o = proj_floor(_PTS, ax=0); nm = nmap(_PTS, _NRM, ax=0)
    axes[0].imshow(np.where(o,d,np.nan), cmap='gray', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title("Depth only — Step 1", fontsize=10, pad=4); axes[0].axis('off')
    axes[1].imshow(nm, origin='lower', interpolation='nearest')
    axes[1].set_title("Normal map  (nx→R  ny→G  nz→B)", fontsize=10, pad=4); axes[1].axis('off')
    combined = np.clip(0.4*np.stack([np.where(o,d,0)]*3,-1) + 0.6*nm, 0, 1)
    axes[2].imshow(combined, origin='lower', interpolation='nearest')
    axes[2].set_title("Depth + normals — Step 2 input", fontsize=10, pad=4); axes[2].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    plt.tight_layout(pad=0.5)
    return _save(fig)

def fig_splatting_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), facecolor=_BG)
    d_fl, o_fl = proj_floor(_PTS, ax=2, H=64, W=64)
    d_sp, o_sp = proj_splat(_PTS, ax=2, H=64, W=64, sigma=1.5)
    axes[0].imshow(np.where(o_fl,d_fl,np.nan), cmap='plasma', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title("Floor rasterisation\n(sparse, aliased)", fontsize=11); axes[0].axis('off')
    axes[1].imshow(np.where(o_sp,d_sp,np.nan), cmap='plasma', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title("Gaussian splatting\n(dense, smooth)", fontsize=11); axes[1].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    plt.tight_layout(pad=0.5)
    return _save(fig)

def fig_bilinear_comparison():
    rng = np.random.default_rng(99)
    pts_s = _PTS[rng.choice(len(_PTS), 120, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), facecolor=_BG)
    d_fl, o_fl = proj_floor(pts_s, ax=2, H=40, W=40)
    d_bi, o_bi = proj_bilinear(pts_s, ax=2, H=40, W=40)
    axes[0].imshow(np.where(o_fl,d_fl,np.nan), cmap='plasma', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title("Floor  (blocky)", fontsize=11); axes[0].axis('off')
    axes[1].imshow(np.where(o_bi,d_bi,np.nan), cmap='plasma', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title("Bilinear  (smooth)", fontsize=11); axes[1].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    plt.tight_layout(pad=0.5)
    return _save(fig)

def fig_context_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.0), facecolor=_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off'); ax.set_facecolor(_BG)
    def box(x, y, w, h, txt, col):
        ax.add_patch(plt.Rectangle((x,y),w,h,color=col,zorder=2))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',
                fontsize=10,color='white',fontweight='bold',zorder=3)
    def arr(x1, y1, x2, y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=2.0))
    box(0.2, 1.5, 2.2, 1.0, "CNN pixel\nfeatures", '#007B9E')
    arr(2.4, 2.0, 2.9, 2.0)
    box(2.9, 1.5, 1.8, 1.0, "Mean\nPool", '#1A2E4A')
    arr(4.7, 2.0, 5.2, 2.0)
    box(5.2, 1.5, 2.2, 1.0, "Global\ndescriptor", '#1A2E4A')
    ax.annotate('',xy=(1.3, 1.5),xytext=(6.3, 1.5),
                arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=2.0,
                                connectionstyle='arc3,rad=-0.35'))
    ax.text(3.8, 0.5, "broadcast to all pixels", ha='center', fontsize=9,
            color='#5D6D7E', style='italic')
    box(0.2, 0.1, 2.2, 0.9, "Concat → prediction", '#1E8B4C')
    ax.set_title("Global Context Injection (Step 5)", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _save(fig)

def fig_attention_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.0), facecolor=_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis('off'); ax.set_facecolor(_BG)
    for i, (lbl, col, w) in enumerate([("Front view", '#007B9E', 0.55),
                                        ("Side view",  '#1A2E4A', 0.30),
                                        ("Top view",   '#7F8C8D', 0.15)]):
        y = 3.5 - i * 1.4
        ax.add_patch(plt.Rectangle((0.2, y-0.38), 2.6, 0.85, color=col, alpha=0.9))
        ax.text(1.5, y+0.05, lbl, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
        ax.annotate('', xy=(3.2, y+0.05), xytext=(2.8, y+0.05),
                    arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=1.8))
        ax.text(3.6, y+0.05, f"α = {w:.2f}", va='center',
                fontsize=10, color=col, fontweight='bold')
    ax.text(5.8, 1.8, "Softmax\nweighted\nsum", ha='center', va='center', fontsize=9,
            color='#5D6D7E',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCCCCC'))
    ax.annotate('', xy=(7.7, 1.8), xytext=(7.0, 1.8),
                arrowprops=dict(arrowstyle='->', color='#5D6D7E', lw=2.0))
    ax.add_patch(plt.Rectangle((7.7, 1.2), 2.0, 1.2, color='#1E8B4C', alpha=0.9))
    ax.text(8.7, 1.8, "Fused\nfeature", ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    ax.set_title("View Attention (Step 6)", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _save(fig)

def fig_results_table_chart():
    labels = [r["label"] for r in RESULTS]
    f1s    = [r["f1"] if r["f1"] else 0.0 for r in RESULTS]
    colors = ['#95A5A6','#007B9E','#007B9E','#1A3C5E','#1A3C5E','#CCCCCC','#D4AC0D']
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=_BG)
    bars = ax.barh(labels, f1s, color=colors, edgecolor='white', linewidth=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score (val, best epoch)", fontsize=11)
    ax.set_title("Fracture Surface F1 — CNN Ablation vs GARF baseline", fontsize=12, pad=8)
    for bar, v in zip(bars, f1s):
        label = f"{v:.3f}" if v > 0 else "pending"
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left', fontsize=10)
    ax.axvline(RESULTS[-1]["f1"], color='#D4AC0D', linestyle='--', lw=1.5, alpha=0.8,
               label=f'GARF baseline ({RESULTS[-1]["f1"]:.3f})')
    ax.legend(fontsize=10, loc='lower right')
    _clean(ax)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.5)
    return _save(fig)

def fig_f1_progression():
    steps = [1, 2, 3, 4, 5]
    f1s   = [0.6805, 0.7090, 0.7193, 0.7087, 0.7262]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=_BG)
    ax.plot(steps, f1s, 'o-', color='#007B9E', lw=2.5, ms=8, zorder=3)
    ax.fill_between(steps, f1s, alpha=0.10, color='#007B9E')
    ax.axhline(0.9094, color='#D4AC0D', linestyle='--', lw=2.0,
               label='GARF baseline (90.9%)')
    for x, y in zip(steps, f1s):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10)
    ax.set_xticks(steps)
    ax.set_xticklabels(["S1\nBaseline","S2\n+Normals","S3\n+Splatting",
                         "S4\n+Bilinear","S5\n+Context"], fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0.55, 1.05)
    ax.legend(fontsize=10)
    ax.set_title("F1 Progression across Ablation Steps", fontsize=12, pad=8)
    _clean(ax)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.5)
    return _save(fig)

# ── Slide builders ────────────────────────────────────────────────────────

def slide_01_title(prs):
    sl = blank(prs)
    rect(sl, 0, 0, Inches(5.5), SH, fill=NAVY)
    rect(sl, Inches(5.5), 0, SW - Inches(5.5), SH, fill=LGRAY)
    tb(sl, "CNN-based Fracture\nSurface Segmentation",
       Inches(0.4), Inches(1.2), Inches(5.0), Inches(2.8),
       size=36, bold=True, color=WHITE)
    tb(sl, "A Progressive Ablation Study",
       Inches(0.4), Inches(4.1), Inches(5.0), Inches(0.55),
       size=18, color=RGBColor(0xAA, 0xBB, 0xCC))
    tb(sl, "Compared against GARF (PTv3) baseline",
       Inches(0.4), Inches(4.7), Inches(5.0), Inches(0.45),
       size=14, italic=True, color=RGBColor(0x88, 0x99, 0xAA))
    tb(sl, "Teyssir Aissi  ·  USTH  ·  April 2026",
       Inches(0.4), Inches(6.7), Inches(5.0), Inches(0.4),
       size=13, color=DGRAY)
    img(sl, fig_fragment_projections(),
        Inches(5.7), Inches(2.0), Inches(7.3), Inches(2.6))
    tb(sl, "3D fragment  →  3 orthographic depth maps  →  CNN",
       Inches(5.7), Inches(4.7), Inches(7.3), Inches(0.4),
       size=11, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)

def slide_02_motivation(prs):
    sl = blank(prs)
    header(sl, "Why This Work?")
    # Left panel
    rect(sl, Inches(0.5), Inches(1.3), Inches(5.6), Inches(4.8),
         fill=LBLUE, line=TEAL, lw=Pt(2))
    tb(sl, "GARF works well", Inches(0.7), Inches(1.5), Inches(5.2), Inches(0.5),
       size=16, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    bullets(sl, ["F1 = 90.9%  on fracture segmentation",
                 "Point Transformer V3 (PTv3) encoder",
                 "End-to-end, strong 3D features"],
            Inches(0.8), Inches(2.15), Inches(5.0), size=15)
    # Right panel
    rect(sl, Inches(7.2), Inches(1.3), Inches(5.6), Inches(4.8),
         fill=RGBColor(0xFD, 0xED, 0xEC), line=RED, lw=Pt(2))
    tb(sl, "But — open questions", Inches(7.4), Inches(1.5), Inches(5.2), Inches(0.5),
       size=16, bold=True, color=RED, align=PP_ALIGN.CENTER)
    bullets(sl, ["Heavy 3D ops (hard to interpret)",
                 "Which geometry matters?",
                 "Can 2D projections compete?"],
            Inches(7.5), Inches(2.15), Inches(5.0), size=15)
    tb(sl, "→", Inches(6.1), Inches(3.5), Inches(1.0), Inches(0.6),
       size=32, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

def slide_03_idea(prs):
    sl = blank(prs)
    header(sl, "Core Idea")
    rect(sl, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.0), fill=NAVY)
    tb(sl, '"Project each 3D fragment onto 2D views,\nthen classify fracture points with a CNN."',
       Inches(1.0), Inches(1.65), Inches(11.3), Inches(1.7),
       size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    bullets(sl, ["Simple, interpretable, no 3D ops",
                 "3 orthographic views: front / side / top",
                 "Back-project CNN scores → per-point prediction"],
            Inches(1.5), Inches(3.8), Inches(10.5), size=16, gap=0.7)
    tb(sl, "→  We study this systematically with a 6-step ablation",
       Inches(1.5), Inches(6.0), Inches(10.5), Inches(0.55),
       size=15, italic=True, color=TEAL, align=PP_ALIGN.CENTER)

def slide_04_pipeline(prs):
    sl = blank(prs)
    header(sl, "Pipeline Overview")
    boxes = [("3D Fragment\nxyz + normals", TEAL),
             ("Ortho\nProjection\n(3 planes)", NAVY),
             ("ResNet-18\nCNN", NAVY),
             ("Back-\nprojection", TEAL),
             ("Fracture\nScore ∈ [0,1]", GREEN)]
    BW, BH, BY = Inches(2.2), Inches(1.5), Inches(1.6)
    xs = [Inches(0.3), Inches(2.75), Inches(5.2), Inches(7.65), Inches(10.1)]
    for (lbl, col), x in zip(boxes, xs):
        rect(sl, x, BY, BW, BH, fill=col)
        tb(sl, lbl, x, BY + Inches(0.15), BW, BH - Inches(0.15),
           size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i in range(len(xs)-1):
        tb(sl, "→", xs[i] + BW, BY + Inches(0.4), Inches(0.35), Inches(0.6),
           size=22, bold=True, color=DGRAY, align=PP_ALIGN.CENTER)
    subs = ["5 000 pts/frag", "128×128 depth maps", "shared weights", "NN lookup", "sigmoid > 0.5"]
    for sub, x in zip(subs, xs):
        tb(sl, sub, x, BY + BH + Inches(0.1), BW, Inches(0.35),
           size=10, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)
    rect(sl, Inches(0.3), Inches(3.5), Inches(12.7), Inches(0.38), fill=LBLUE)
    tb(sl, "Back-projection: each 3D point inherits the score of its nearest projected pixel",
       Inches(0.5), Inches(3.52), Inches(12.3), Inches(0.34),
       size=12, italic=True, color=NAVY, align=PP_ALIGN.CENTER)
    img(sl, fig_fragment_projections(),
        Inches(0.5), Inches(4.0), Inches(12.3), Inches(3.1))

def slide_05_ablation_principle(prs):
    sl = blank(prs)
    header(sl, "Ablation Study Design",
           "One component added per step — contribution isolated")
    rows = [
        ("Step 1", "Depth only",              "baseline CNN",           "✓ done", DGRAY),
        ("Step 2", "+ Surface Normals",        "extra input channels",   "✓ done", TEAL),
        ("Step 3", "+ Gaussian Splatting",     "denser projection",      "✓ done", TEAL),
        ("Step 4", "+ Bilinear Interpolation", "smoother projection",    "✓ done", NAVY),
        ("Step 5", "+ Global Context",         "fragment-level feature", "✓ done", NAVY),
        ("Step 6", "+ View Attention",         "per-point view weights", "running…", GREEN),
    ]
    col_x = [Inches(0.3), Inches(1.6), Inches(4.5), Inches(9.2), Inches(11.2)]
    col_w = [Inches(1.2), Inches(2.8), Inches(4.6), Inches(1.9), Inches(1.9)]
    hdrs  = ["", "Step", "What changes", "Status", ""]
    Y0 = Inches(1.35)
    for hdr, cx, cw in zip(hdrs, col_x, col_w):
        rect(sl, cx, Y0, cw - Inches(0.05), Inches(0.42), fill=NAVY)
        tb(sl, hdr, cx + Inches(0.05), Y0 + Inches(0.06), cw - Inches(0.1), Inches(0.32),
           size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for i, (sid, name, what, status, col) in enumerate(rows):
        y = Inches(1.82 + i * 0.72)
        bg = LGRAY if i % 2 == 0 else WHITE
        rect(sl, Inches(0.3), y, Inches(12.8), Inches(0.68),
             fill=bg, line=RGBColor(0xDD, 0xDD, 0xDD), lw=Pt(0.5))
        scol = GREEN if status == "✓ done" else RED
        for txt, cx, cw, tc, al in [
            (sid,  col_x[0], col_w[0], col,   PP_ALIGN.CENTER),
            (name, col_x[1], col_w[1], col,   PP_ALIGN.LEFT),
            (what, col_x[2], col_w[2], DARK,  PP_ALIGN.LEFT),
            (status, col_x[3], col_w[3], scol, PP_ALIGN.CENTER),
        ]:
            tb(sl, txt, cx + Inches(0.05), y + Inches(0.17),
               cw - Inches(0.1), Inches(0.42),
               size=13, bold=(txt == name), color=tc, align=al)

def slide_06_normals(prs):
    sl = blank(prs)
    header(sl, "Step 2 — Surface Normals",
           "Adding surface orientation to the CNN input")
    bullets(sl, ["Add a normal map (nx, ny, nz) alongside the depth map",
                 "CNN input: 4 channels per view instead of 1",
                 "Hypothesis: orientation reveals fracture boundaries"],
            Inches(0.6), Inches(1.3), Inches(5.4), size=15, gap=0.68)
    badge(sl, "Result:  F1  68.1% → 70.9%  (+2.8 pp)",
          l=0.6, t=3.5, w=5.2, col=TEAL)
    badge(sl, "Recall  81.1% → 87.3%  (+6.3 pp)", l=0.6, t=4.1, w=5.2, col=TEAL)
    img(sl, fig_normals_comparison(),
        Inches(6.2), Inches(1.2), Inches(6.9), Inches(3.6))

def slide_07_splatting(prs):
    sl = blank(prs)
    header(sl, "Step 3 — Gaussian Splatting",
           "Denser projection coverage")
    bullets(sl, ["Each 3D point splat as a 2D Gaussian blob",
                 "Fills holes: fewer empty pixels in depth map",
                 "σ calibrated to mean inter-point distance"],
            Inches(0.6), Inches(1.3), Inches(5.4), size=15, gap=0.68)
    badge(sl, "Result:  F1  70.9% → 71.9%  (+1.0 pp)", l=0.6, t=3.5, w=5.2, col=NAVY)
    badge(sl, "Precision  61.4% → 63.4%  (+2.0 pp)",   l=0.6, t=4.1, w=5.2, col=NAVY)
    img(sl, fig_splatting_comparison(),
        Inches(6.2), Inches(1.3), Inches(6.9), Inches(3.5))

def slide_08_bilinear(prs):
    sl = blank(prs)
    header(sl, "Step 4 — Bilinear Interpolation",
           "Smoother point-to-pixel mapping")
    bullets(sl, ["Each point contributes to 4 surrounding pixels (weighted)",
                 "Sub-pixel accuracy — smoother depth gradients",
                 "Hypothesis: less aliasing → better generalisation"],
            Inches(0.6), Inches(1.3), Inches(5.4), size=15, gap=0.68)
    badge(sl, "Result:  F1  71.9% → 70.9%  (−1.0 pp)", l=0.6, t=3.5, w=5.2,
          col=RGBColor(0xBF, 0x2B, 0x2B))
    tb(sl, "→ Splatting already handles coverage; bilinear adds no benefit here.",
       Inches(0.6), Inches(4.3), Inches(5.2), Inches(0.6),
       size=13, italic=True, color=DGRAY)
    img(sl, fig_bilinear_comparison(),
        Inches(6.2), Inches(1.3), Inches(6.9), Inches(3.5))

def slide_09_context(prs):
    sl = blank(prs)
    header(sl, "Step 5 — Global Context",
           "Fragment-level awareness injected into every pixel")
    bullets(sl, ["Mean-pool all CNN pixel features per fragment",
                 "Concatenate this global descriptor back into every pixel",
                 "Lets the model reason globally, not just locally"],
            Inches(0.6), Inches(1.3), Inches(5.4), size=15, gap=0.68)
    badge(sl, "Result:  F1  70.9% → 72.6%  (+1.7 pp)", l=0.6, t=3.5, w=5.2, col=GREEN)
    badge(sl, "Best CNN result overall", l=0.6, t=4.1, w=5.2,
          col=RGBColor(0x1E, 0x6B, 0x3C))
    img(sl, fig_context_diagram(),
        Inches(6.2), Inches(1.4), Inches(6.9), Inches(3.4))

def slide_10_attention(prs):
    sl = blank(prs)
    header(sl, "Step 6 — View Attention",
           "Learning which view matters per point")
    bullets(sl, ["MLP scores each of the 3 views per point",
                 "Softmax → weighted combination of view features",
                 "Occluded or uninformative views get lower weight"],
            Inches(0.6), Inches(1.3), Inches(5.4), size=15, gap=0.68)
    badge(sl, "Status: training still running…", l=0.6, t=3.5, w=5.2,
          col=RGBColor(0xE6, 0x7E, 0x22))
    img(sl, fig_attention_diagram(),
        Inches(6.2), Inches(1.4), Inches(6.9), Inches(3.4))

def slide_11_results_table(prs):
    sl = blank(prs)
    header(sl, "Results — Full Ablation Table",
           "Best val-F1 checkpoint  ·  seed 1116  ·  50 epochs")
    hdrs = ["", "Model", "Acc ↑", "F1 ↑", "Prec ↑", "Recall ↑"]
    col_x = [Inches(0.3), Inches(1.2), Inches(4.8), Inches(6.3), Inches(7.8), Inches(9.3)]
    col_w = [Inches(0.88), Inches(3.55), Inches(1.45), Inches(1.45), Inches(1.45), Inches(1.45)]
    Y0 = Inches(1.3)
    for hdr, cx, cw in zip(hdrs, col_x, col_w):
        rect(sl, cx, Y0, cw - Inches(0.04), Inches(0.44), fill=NAVY)
        tb(sl, hdr, cx + Inches(0.04), Y0 + Inches(0.06), cw - Inches(0.08), Inches(0.32),
           size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    def fmt(v): return f"{v:.3f}" if v is not None else "—"
    best_cnn = 4
    for i, row in enumerate(RESULTS):
        y = Inches(1.79 + i * 0.67)
        is_best = (i == best_cnn); is_garf = (i == len(RESULTS) - 1)
        if is_garf:
            rect(sl, Inches(0.3), y - Inches(0.06), Inches(10.8), Inches(0.08), fill=TEAL)
        bg = (RGBColor(0xFF, 0xF5, 0xCC) if is_best
              else RGBColor(0xEF, 0xF9, 0xF4) if is_garf
              else (LGRAY if i % 2 == 0 else WHITE))
        rect(sl, Inches(0.3), y, Inches(10.8), Inches(0.63),
             fill=bg, line=RGBColor(0xCC, 0xCC, 0xCC), lw=Pt(0.5))
        step_lbl = f"S{i+1}" if i < 6 else "REF"
        vals = [step_lbl, row["label"],
                fmt(row["acc"]), fmt(row["f1"]), fmt(row["prec"]), fmt(row["rec"])]
        for j, (val, cx, cw) in enumerate(zip(vals, col_x, col_w)):
            c = (GOLD if is_best and j >= 2
                 else TEAL if is_garf
                 else DARK)
            tb(sl, val, cx + Inches(0.04), y + Inches(0.14),
               cw - Inches(0.08), Inches(0.42),
               size=12, bold=(is_best and j >= 2) or is_garf, color=c,
               align=PP_ALIGN.LEFT if j == 1 else PP_ALIGN.CENTER)
    tb(sl, "★  Best CNN (Step 5) highlighted    |    GARF baseline in green",
       Inches(0.3), Inches(6.8), Inches(10.8), Inches(0.35),
       size=10, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)

def slide_12_results_chart(prs):
    sl = blank(prs)
    header(sl, "Results — F1 Score Comparison")
    img(sl, fig_results_table_chart(),
        Inches(0.4), Inches(1.2), Inches(8.5), Inches(5.5))
    img(sl, fig_f1_progression(),
        Inches(8.9), Inches(1.2), Inches(4.2), Inches(5.5))

def slide_13_insights(prs):
    sl = blank(prs)
    header(sl, "Key Insights")
    insights = [
        (TEAL,   "Normals help recall  (+6.3 pp)",
                 "81.1% → 87.3%  —  the CNN detects fracture orientation, not just depth."),
        (NAVY,   "Splatting improves precision  (+2.0 pp)",
                 "Fewer empty pixels → fewer false positives on sparse projections."),
        (GREEN,  "Global context gives best F1  (72.6%)",
                 "Fragment-level pooling is the single biggest gain on the model side."),
        (RED,    "Large gap vs GARF  (−18 pp F1)",
                 "3D reasoning from PTv3 is hard to replace with fixed 2D projections."),
    ]
    for i, (col, title, desc) in enumerate(insights):
        y = Inches(1.4 + i * 1.45)
        rect(sl, Inches(0.3), y, Inches(0.22), Inches(1.05), fill=col)
        tb(sl, title, Inches(0.65), y + Inches(0.05), Inches(12.2), Inches(0.5),
           size=18, bold=True, color=col)
        tb(sl, desc, Inches(0.65), y + Inches(0.58), Inches(12.2), Inches(0.55),
           size=14, color=DARK)

def slide_14_limitations(prs):
    sl = blank(prs)
    header(sl, "Limitations")
    items = [
        (ORANGE, "Bilinear interpolation was not effective",
                 "Step 4 shows marginal loss vs Step 3. Splatting already fills coverage; bilinear adds noise."),
        (RED,    "3D → 2D projection is inherently lossy",
                 "Some fracture patterns are only visible from angles not in our 3 fixed views."),
        (DGRAY,  "Limited training budget (50 epochs vs 500 for GARF)",
                 "CNN results are preliminary — more epochs could close some of the gap."),
    ]
    for i, (col, title, desc) in enumerate(items):
        y = Inches(1.5 + i * 1.7)
        rect(sl, Inches(0.3), y, Inches(0.22), Inches(1.2), fill=col)
        tb(sl, title, Inches(0.65), y + Inches(0.05), Inches(12.2), Inches(0.5),
           size=17, bold=True, color=col)
        tb(sl, desc, Inches(0.65), y + Inches(0.58), Inches(12.2), Inches(0.6),
           size=14, color=DARK)

def slide_15_nextsteps(prs):
    sl = blank(prs)
    header(sl, "Next Steps")
    cols = [
        ("Short term",  TEAL,  ["Complete Step 6 (view attention)",
                                 "Run GARF baseline eval on same val split",
                                 "Collect 3D/2D visualisations"]),
        ("Medium term", NAVY,  ["Train 500 epochs for fair comparison",
                                 "Hybrid model: PTv3 + geometric features",
                                 "Learnable projection angles"]),
        ("Open question", GREEN, ["Can 2D CNN match PTv3 on recall?",
                                   "Do projections complement 3D embeddings?"]),
    ]
    CW = Inches(3.9)
    for i, (term, col, items_) in enumerate(cols):
        cx = Inches(0.4 + i * 4.3)
        rect(sl, cx, Inches(1.3), CW, Inches(5.7), fill=LGRAY, line=col, lw=Pt(2))
        rect(sl, cx, Inches(1.3), CW, Inches(0.55), fill=col)
        tb(sl, term, cx + Inches(0.1), Inches(1.36), CW - Inches(0.2), Inches(0.44),
           size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        for j, item in enumerate(items_):
            tb(sl, "→  " + item, cx + Inches(0.15), Inches(2.1 + j * 1.4),
               CW - Inches(0.3), Inches(1.3), size=14, color=DARK)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="presentation.pptx")
    args = ap.parse_args()

    import os; os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("Generating visuals…")
    prs = new_prs()
    for fn in [slide_01_title, slide_02_motivation, slide_03_idea, slide_04_pipeline,
               slide_05_ablation_principle, slide_06_normals, slide_07_splatting,
               slide_08_bilinear, slide_09_context, slide_10_attention,
               slide_11_results_table, slide_12_results_chart,
               slide_13_insights, slide_14_limitations, slide_15_nextsteps]:
        fn(prs)
        print(f"  ✓ {fn.__name__}")

    prs.save(args.out)
    print(f"\nSaved → {args.out}  ({len(prs.slides)} slides)")

if __name__ == "__main__":
    main()
