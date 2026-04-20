#!/usr/bin/env python3
"""
scripts/create_presentation.py  — v2 with matplotlib visuals
All visuals generated synthetically — no dataset required.

Usage:
    pip install python-pptx matplotlib scipy
    python scripts/create_presentation.py --out presentation.pptx

On Colab:
    !pip install python-pptx -q
    !python /content/GARFw/scripts/create_presentation.py --out /content/presentation.pptx
    from google.colab import files; files.download('/content/presentation.pptx')
"""

import argparse
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
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

_BG = "#F2F3F4"
W = Inches(13.33)
H = Inches(7.50)


# ── PPTX helpers ──────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def rect(slide, l, t, w, h, fill=TEAL, line=None, line_w=Pt(1)):
    shp = slide.shapes.add_shape(1, l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line:
        shp.line.color.rgb = line
        shp.line.width = line_w
    else:
        shp.line.color.rgb = fill
        shp.line.width = Pt(0)
    return shp

def tb(slide, text, l, t, w, h,
       size=18, bold=False, italic=False,
       color=DARK, align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb

def header_bar(slide, title, subtitle=None):
    rect(slide, 0, 0, W, Inches(1.1), fill=NAVY)
    tb(slide, title,
       Inches(0.4), Inches(0.1), Inches(11), Inches(0.85),
       size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        tb(slide, subtitle,
           Inches(0.4), Inches(0.82), Inches(11), Inches(0.35),
           size=13, color=RGBColor(0xAA, 0xBB, 0xCC), align=PP_ALIGN.LEFT)

def panel(slide, l, t, w, h, title, items, accent=TEAL, item_size=14):
    rect(slide, l, t, w, h, fill=LGRAY, line=accent, line_w=Pt(2))
    tb(slide, title, l + Inches(0.15), t + Inches(0.08),
       w - Inches(0.3), Inches(0.55),
       size=15, bold=True, color=accent)
    for i, item in enumerate(items):
        tb(slide, "• " + item,
           l + Inches(0.2), t + Inches(0.75 + i * 0.75),
           w - Inches(0.4), Inches(0.7),
           size=item_size, color=DARK)

def addimg(slide, buf, l, t, w, h):
    slide.shapes.add_picture(buf, l, t, w, h)


# ── Synthetic fragment ────────────────────────────────────────────────────

def _make_fragment(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r     = np.sqrt(rng.uniform(0, 1, n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0.28 * (x**2 + y**2 - 0.5) + rng.normal(0, 0.022, n)
    pts = np.column_stack([x, y, z])
    nrm = np.column_stack([-0.56 * x, -0.56 * y, np.ones(n)])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    frac = (r > 0.73) | (np.abs(z - z.min()) < 0.06)
    return pts, nrm, frac.astype(bool)

_PTS, _NRM, _FRAC = _make_fragment()


# ── Projection utilities ──────────────────────────────────────────────────

def _n01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def _pix(pts, ax, H, W):
    ui, vi = [(0,2),(1,2),(0,1)][ax]
    px = np.clip((_n01(pts[:,ui])*(W-1)).astype(int), 0, W-1)
    py = np.clip((_n01(pts[:,vi])*(H-1)).astype(int), 0, H-1)
    return px, py

def proj_floor(pts, ax, H=80, W=80):
    px, py = _pix(pts, ax, H, W)
    di = [1,0,2][ax]
    d  = _n01(pts[:,di])
    depth = np.full((H,W), np.nan)
    for i in range(len(pts)):
        if np.isnan(depth[py[i],px[i]]) or d[i] > depth[py[i],px[i]]:
            depth[py[i],px[i]] = d[i]
    occ = ~np.isnan(depth)
    return np.where(occ, depth, 0), occ

def proj_splat(pts, ax, H=80, W=80, sigma=1.5):
    px, py = _pix(pts, ax, H, W)
    di = [1,0,2][ax]
    d  = _n01(pts[:,di])
    acc = np.zeros((H,W)); cnt = np.zeros((H,W))
    for i in range(len(pts)):
        acc[py[i],px[i]] += d[i]; cnt[py[i],px[i]] += 1
    avg = np.where(cnt>0, acc/np.where(cnt>0,cnt,1), 0)
    return gaussian_filter(avg, sigma), gaussian_filter(cnt,sigma)>0.05

def proj_bilinear(pts, ax, H=80, W=80):
    ui, vi = [(0,2),(1,2),(0,1)][ax]
    di = [1,0,2][ax]
    u = _n01(pts[:,ui])*(W-1); v = _n01(pts[:,vi])*(H-1); d = _n01(pts[:,di])
    acc = np.zeros((H,W)); w_acc = np.zeros((H,W))
    for i in range(len(pts)):
        x0,y0 = int(u[i]), int(v[i])
        x1,y1 = min(x0+1,W-1), min(y0+1,H-1)
        wx,wy = u[i]-x0, v[i]-y0
        for bx,by,bw in [(x0,y0,(1-wx)*(1-wy)),(x1,y0,wx*(1-wy)),
                          (x0,y1,(1-wx)*wy),(x1,y1,wx*wy)]:
            acc[by,bx]+=bw*d[i]; w_acc[by,bx]+=bw
    return np.where(w_acc>0, acc/np.where(w_acc>0,w_acc,1), 0), w_acc>0.01

def normal_map(pts, nrm, ax, H=80, W=80):
    px, py = _pix(pts, ax, H, W)
    nmap = np.zeros((H,W,3)); cnt = np.zeros((H,W))
    for i in range(len(pts)):
        nmap[py[i],px[i]] += (nrm[i]+1)/2; cnt[py[i],px[i]]+=1
    return np.where(cnt[:,:,None]>0, nmap/np.where(cnt[:,:,None]>0,cnt[:,:,None],1), 0)


# ── Figure helpers ────────────────────────────────────────────────────────

def _bytes(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

def _ax_clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(_BG)


# ── Figure: 3D fragment + 3 projections ──────────────────────────────────

def fig_3d_projections():
    fig = plt.figure(figsize=(10, 3.2), facecolor=_BG)
    ax3 = fig.add_subplot(1,4,1, projection='3d', facecolor=_BG)
    c = ['#e74c3c' if f else '#3498db' for f in _FRAC]
    ax3.scatter(_PTS[:,0], _PTS[:,1], _PTS[:,2], c=c, s=1.5, alpha=0.7, linewidths=0)
    ax3.set_title("3D fragment\n(red=fracture)", fontsize=8, pad=2)
    ax3.set_axis_off()
    for i, name in enumerate(["Front (XZ)","Side (YZ)","Top (XY)"]):
        ax2 = fig.add_subplot(1,4,i+2)
        d,o = proj_floor(_PTS, ax=i)
        ax2.imshow(np.where(o,d,np.nan), cmap='plasma', origin='lower',
                   vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title(name, fontsize=8, pad=2)
        ax2.axis('off'); ax2.set_facecolor(_BG)
    plt.tight_layout(pad=0.3)
    return _bytes(fig)


# ── Figure: splatting comparison ─────────────────────────────────────────

def fig_splatting():
    fig, axes = plt.subplots(1,2, figsize=(5.5,2.8), facecolor=_BG)
    d_fl,o_fl = proj_floor(_PTS, ax=2, H=64, W=64)
    d_sp,o_sp = proj_splat(_PTS, ax=2, H=64, W=64, sigma=1.5)
    axes[0].imshow(np.where(o_fl,d_fl,np.nan), cmap='plasma', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[0].set_title("Floor rasterisation\n(sparse, aliased)", fontsize=9)
    axes[0].axis('off')
    axes[1].imshow(np.where(o_sp,d_sp,np.nan), cmap='plasma', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[1].set_title("Gaussian splatting\n(dense, smooth)", fontsize=9)
    axes[1].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _bytes(fig)


# ── Figure: normals comparison ────────────────────────────────────────────

def fig_normals():
    fig, axes = plt.subplots(1,3, figsize=(7.5,2.8), facecolor=_BG)
    d,o = proj_floor(_PTS, ax=0)
    nm  = normal_map(_PTS, _NRM, ax=0)
    axes[0].imshow(np.where(o,d,np.nan), cmap='gray', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[0].set_title("Depth only\n(Step 1)", fontsize=9); axes[0].axis('off')
    axes[1].imshow(nm, origin='lower', interpolation='nearest')
    axes[1].set_title("Normal map\n(nx→R  ny→G  nz→B)", fontsize=9); axes[1].axis('off')
    combined = np.clip(0.45*np.stack([np.where(o,d,0)]*3,-1) + 0.55*nm, 0, 1)
    axes[2].imshow(combined, origin='lower', interpolation='nearest')
    axes[2].set_title("Depth + normals\n(Step 2 input)", fontsize=9); axes[2].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _bytes(fig)


# ── Figure: bilinear comparison ───────────────────────────────────────────

def fig_bilinear():
    rng = np.random.default_rng(99)
    pts_s = _PTS[rng.choice(len(_PTS), 120, replace=False)]
    fig, axes = plt.subplots(1,2, figsize=(5.5,2.8), facecolor=_BG)
    d_fl,o_fl = proj_floor(pts_s, ax=2, H=40, W=40)
    d_bi,o_bi = proj_bilinear(pts_s, ax=2, H=40, W=40)
    axes[0].imshow(np.where(o_fl,d_fl,np.nan), cmap='plasma', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[0].set_title("Floor  (blocky)", fontsize=9); axes[0].axis('off')
    axes[1].imshow(np.where(o_bi,d_bi,np.nan), cmap='plasma', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[1].set_title("Bilinear  (smooth)", fontsize=9); axes[1].axis('off')
    for ax in axes: ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _bytes(fig)


# ── Figure: pipeline strip ────────────────────────────────────────────────

def fig_pipeline_strip():
    fig, axes = plt.subplots(1,5, figsize=(12,2.4), facecolor=_BG)
    d_sp,o_sp = proj_splat(_PTS, ax=0, sigma=2.0)
    nm = normal_map(_PTS, _NRM, ax=0)

    # 1. 3D scatter
    axes[0].scatter(_PTS[:,0], _PTS[:,2],
                    c=['#e74c3c' if f else '#3498db' for f in _FRAC],
                    s=1.0, alpha=0.6, linewidths=0)
    axes[0].set_aspect('equal'); axes[0].axis('off')
    axes[0].set_title("3D Input", fontsize=8)
    # 2. Depth
    d,o = proj_floor(_PTS, ax=0)
    axes[1].imshow(np.where(o,d,np.nan), cmap='gray', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[1].set_title("Depth map", fontsize=8); axes[1].axis('off')
    # 3. Normals
    axes[2].imshow(nm, origin='lower', interpolation='nearest')
    axes[2].set_title("Normal map", fontsize=8); axes[2].axis('off')
    # 4. Splatted
    axes[3].imshow(np.where(o_sp,d_sp,np.nan), cmap='plasma', origin='lower',
                   vmin=0,vmax=1, interpolation='nearest')
    axes[3].set_title("After splatting", fontsize=8); axes[3].axis('off')
    # 5. Simulated score
    rng = np.random.default_rng(7)
    score = np.zeros((80,80)); cnt = np.zeros((80,80))
    px,py = _pix(_PTS, 0, 80, 80)
    for i in range(len(_PTS)):
        score[py[i],px[i]] = rng.uniform(0.7,1.0) if _FRAC[i] else rng.uniform(0.0,0.3)
        cnt[py[i],px[i]] = 1
    score = gaussian_filter(score*cnt, sigma=1.2)
    axes[4].imshow(score, cmap='RdYlGn', origin='lower', vmin=0, vmax=0.6,
                   interpolation='nearest')
    axes[4].set_title("CNN score\n(fracture prob)", fontsize=8); axes[4].axis('off')

    for ax in axes: ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.2)
    return _bytes(fig)


# ── Figure: results bar chart ─────────────────────────────────────────────

def fig_results_chart(data=None):
    labels = ["Step 1\nBaseline","Step 2\n+Normals","Step 3\n+Splatting",
              "Step 4\n+Bilinear","Step 5\n+Context","Step 6\n+Attention","GARF\n(PTv3)"]
    f1s = [d.get("f1",0) for d in data] if data else [0.6805,0.7090,0.7193,0.7087,0.7262,0.0,0.9094]
    colors = ['#7F8C8D','#007B9E','#007B9E','#1A3C5E','#1A3C5E','#1E8B4C','#D4AC0D']
    fig, ax = plt.subplots(figsize=(5.5,3.5), facecolor=_BG)
    bars = ax.barh(labels, f1s, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 Score", fontsize=9)
    ax.set_title("Fracture F1 by Step", fontsize=10, pad=5)
    for bar, v in zip(bars, f1s):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}" if v>0 else "?", va='center', ha='left', fontsize=8)
    ax.axvline(f1s[-1], color='#D4AC0D', linestyle='--', lw=1.2, alpha=0.7)
    _ax_clean(ax)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _bytes(fig)


# ── Figure: global context diagram ───────────────────────────────────────

def fig_context_diagram():
    fig, ax = plt.subplots(figsize=(5,2.8), facecolor=_BG)
    ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off'); ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    def box(x,y,w,h,txt,col):
        ax.add_patch(plt.Rectangle((x,y),w,h,color=col,zorder=2))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',
                fontsize=8,color='white',fontweight='bold',zorder=3)
    def arr(x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=1.5))
    box(0.2,3.2,2.4,1.0,"Per-pixel\nfeatures",'#007B9E')
    arr(2.6,3.7,3.1,3.7)
    box(3.1,3.2,2.0,1.0,"Mean\nPool",'#1A2E4A')
    arr(5.1,3.7,5.6,3.7)
    box(5.6,3.2,2.4,1.0,"Global\ndescriptor",'#1A2E4A')
    arr(6.8,3.2,6.8,2.4)
    ax.annotate('',xy=(1.4,2.4),xytext=(6.8,2.4),
                arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=1.5))
    ax.text(4.1,2.1,"broadcast to all pixels",ha='center',fontsize=7.5,
            color='#5D6D7E',style='italic')
    box(0.2,1.0,2.4,1.0,"Concat+MLP\n→ prediction",'#1E8B4C')
    ax.set_title("Step 5 — Global Context Injection", fontsize=9, pad=4)
    plt.tight_layout(pad=0.2)
    return _bytes(fig)


# ── Figure: view attention ────────────────────────────────────────────────

def fig_attention_diagram():
    fig, ax = plt.subplots(figsize=(5,2.8), facecolor=_BG)
    ax.set_xlim(0,10); ax.set_ylim(0,5.5); ax.axis('off'); ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    weights = [0.55,0.30,0.15]; cols = ['#007B9E','#1A2E4A','#7F8C8D']
    labels  = ["Front","Side","Top"]
    for i,(w,col,lbl) in enumerate(zip(weights,cols,labels)):
        y = 4.3 - i*1.5
        ax.add_patch(plt.Rectangle((0.3,y-0.4),2.8,0.9,color=col,alpha=0.85))
        ax.text(1.7,y,f"{lbl} view",ha='center',va='center',
                fontsize=9,color='white',fontweight='bold')
        ax.annotate('',xy=(3.5,y),xytext=(3.1,y),
                    arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=1.5))
        ax.text(3.9,y,f"α={w:.2f}",va='center',fontsize=9,color=col,fontweight='bold')
    ax.text(5.8,2.7,"Softmax\n→\nweighted\nsum",ha='center',va='center',fontsize=8,
            color='#5D6D7E',bbox=dict(boxstyle='round',facecolor='white',edgecolor='#CCCCCC'))
    ax.annotate('',xy=(7.8,2.7),xytext=(7.0,2.7),
                arrowprops=dict(arrowstyle='->',color='#5D6D7E',lw=1.5))
    ax.add_patch(plt.Rectangle((7.8,2.1),1.9,1.2,color='#1E8B4C',alpha=0.9))
    ax.text(8.75,2.7,"Fused\nfeat.",ha='center',va='center',
            fontsize=8.5,color='white',fontweight='bold')
    ax.set_title("Step 6 — View Attention", fontsize=9, pad=4)
    plt.tight_layout(pad=0.2)
    return _bytes(fig)


# ── Figure: insights ──────────────────────────────────────────────────────

def fig_insights():
    fig, axes = plt.subplots(1,3, figsize=(9,2.6), facecolor=_BG)
    for ax,(cats,vals,ylabel,title,colors) in zip(axes,[
        (["No normals","+ Normals"],[0.8107,0.8732],"Recall","Normals → recall ↑",
         ['#7F8C8D','#007B9E']),
        (["No splatting","+ Splatting"],[0.6029,0.6338],"Precision","Splatting → precision ↑",
         ['#7F8C8D','#007B9E']),
        (None,None,"F1","F1 progression by step",None),
    ]):
        if cats:
            bars = ax.bar(cats,vals,color=colors,width=0.45,edgecolor='white')
            ax.set_ylim(0,1.0); ax.set_ylabel(ylabel,fontsize=8)
            ax.set_title(title,fontsize=9)
            for b,v in zip(bars,vals):
                ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",
                        ha='center',fontsize=8)
        else:
            steps=[1,2,3,4,5]; f1s=[0.6805,0.7090,0.7193,0.7087,0.7262]
            ax.plot(steps,f1s,'o-',color='#1A2E4A',lw=1.8,ms=5)
            ax.fill_between(steps,f1s,alpha=0.12,color='#1A2E4A')
            ax.set_xlabel("Step",fontsize=8); ax.set_ylabel("F1",fontsize=8)
            ax.set_title(title,fontsize=9); ax.set_ylim(0.4,0.85)
            ax.set_xticks(steps)
        _ax_clean(ax)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(pad=0.4)
    return _bytes(fig)


# ── Figure: timeline ──────────────────────────────────────────────────────

def fig_timeline():
    fig, ax = plt.subplots(figsize=(9,1.8), facecolor=_BG)
    ax.set_xlim(-0.5,9.5); ax.set_ylim(-0.5,2.5); ax.axis('off'); ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    ax.plot([0,9.5],[1,1],color='#CCCCCC',lw=2,zorder=1)
    for x,lbl,col in [(0,"Complete\nStep 6",'#BF2B2B'),(2.5,"Eval GARF\nbaseline",'#E67E22'),
                       (5,"500 epoch\ntraining",'#007B9E'),(7.5,"Hybrid\nmodel",'#1E8B4C'),
                       (9.5,"Full\ncomparison",'#1A2E4A')]:
        ax.scatter([x],[1],s=200,color=col,zorder=3,edgecolors='white',lw=1.5)
        ax.text(x,1.45,lbl,ha='center',va='bottom',fontsize=7.5,
                color=col,fontweight='bold',linespacing=1.3)
    plt.tight_layout(pad=0.2)
    return _bytes(fig)


# ── Slide builders ────────────────────────────────────────────────────────

def slide_title(prs):
    sl = blank(prs)
    rect(sl, 0, 0, Inches(5.8), H, fill=NAVY)
    rect(sl, Inches(5.8), 0, W-Inches(5.8), H, fill=LGRAY)
    tb(sl,"CNN-based Fracture\nSurface Segmentation",
       Inches(0.4),Inches(1.6),Inches(5.2),Inches(2.4),
       size=34,bold=True,color=WHITE,align=PP_ALIGN.LEFT)
    tb(sl,"A Progressive Ablation Study",
       Inches(0.4),Inches(4.0),Inches(5.2),Inches(0.55),
       size=17,color=RGBColor(0xAA,0xBB,0xCC))
    tb(sl,"Compared against GARF (PTv3) baseline",
       Inches(0.4),Inches(4.55),Inches(5.2),Inches(0.45),
       size=14,italic=True,color=RGBColor(0x88,0x99,0xAA))
    tb(sl,"Teyssir Aissi  ·  USTH  ·  April 2026",
       Inches(0.4),Inches(6.6),Inches(5.2),Inches(0.45),size=13,color=DGRAY)
    # Right: 3D+projections image
    addimg(sl, fig_3d_projections(), Inches(5.9), Inches(1.5), Inches(7.1), Inches(2.8))
    tb(sl,"3D fragment  →  3 orthographic depth maps  →  CNN classification",
       Inches(5.9),Inches(4.4),Inches(7.1),Inches(0.45),
       size=11,italic=True,color=DGRAY,align=PP_ALIGN.CENTER)
    return sl


def slide_problem(prs):
    sl = blank(prs)
    header_bar(sl,"Why This Work?","Motivation")
    rect(sl,Inches(0.4),Inches(1.3),Inches(5.7),Inches(4.5),fill=LBLUE,line=TEAL,line_w=Pt(2))
    tb(sl,"GARF is powerful",Inches(0.6),Inches(1.4),Inches(5.3),Inches(0.5),
       size=16,bold=True,color=TEAL,align=PP_ALIGN.CENTER)
    for i,s in enumerate(["✔  State-of-the-art on Breaking Bad",
                           "✔  Point Transformer V3 encoder",
                           "✔  Strong 3D feature learning",
                           "✔  End-to-end differentiable"]):
        tb(sl,s,Inches(0.7),Inches(2.0+i*0.62),Inches(5.1),Inches(0.55),size=14,color=DARK)
    tb(sl,"BUT →",Inches(6.2),Inches(3.4),Inches(1.0),Inches(0.5),
       size=22,bold=True,color=RED,align=PP_ALIGN.CENTER)
    rect(sl,Inches(7.3),Inches(1.3),Inches(5.6),Inches(4.5),
         fill=RGBColor(0xFD,0xED,0xEC),line=RED,line_w=Pt(2))
    tb(sl,"Open questions",Inches(7.5),Inches(1.4),Inches(5.2),Inches(0.5),
       size=16,bold=True,color=RED,align=PP_ALIGN.CENTER)
    for i,s in enumerate(["✗  Complex 3D ops (torch_scatter)",
                           "✗  Implicit geometric reasoning",
                           "✗  Hard to inspect / ablate",
                           "✗  Can a 2D approach compete?"]):
        tb(sl,s,Inches(7.5),Inches(2.0+i*0.62),Inches(5.2),Inches(0.55),size=14,color=DARK)
    # Bottom: projection preview
    addimg(sl, fig_3d_projections(), Inches(0.4), Inches(5.9), Inches(12.5), Inches(1.35))
    return sl


def slide_idea(prs):
    sl = blank(prs)
    header_bar(sl,"Core Idea")
    rect(sl,Inches(0.8),Inches(1.6),Inches(11.7),Inches(2.1),fill=NAVY)
    tb(sl,"\"Project 3D fragments onto 2D views,\nthen classify fracture surfaces with a CNN.\"",
       Inches(1.0),Inches(1.8),Inches(11.3),Inches(1.8),
       size=24,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    tb(sl,"Hypothesis: 2D geometric cues (shape, depth, normals) carry enough signal "
          "for fracture detection — without full 3D ops.",
       Inches(1.5),Inches(3.9),Inches(10.0),Inches(0.65),
       size=15,color=DARK,align=PP_ALIGN.CENTER)
    tb(sl,"Bonus: interpretable  ·  lightweight  ·  easy to ablate step by step",
       Inches(1.5),Inches(4.6),Inches(10.0),Inches(0.45),
       size=14,italic=True,color=TEAL,align=PP_ALIGN.CENTER)
    addimg(sl, fig_3d_projections(), Inches(0.8), Inches(5.15), Inches(11.7), Inches(2.1))
    return sl


def slide_pipeline(prs):
    sl = blank(prs)
    header_bar(sl,"Method Overview — Full Pipeline")
    boxes = [("3D\nFragment",Inches(0.3),TEAL),
             ("Ortho\nProjection",Inches(2.85),NAVY),
             ("3 Views\n(XZ/YZ/XY)",Inches(5.45),TEAL),
             ("ResNet-18\nCNN",Inches(8.0),NAVY),
             ("Fracture\nScore",Inches(10.55),GREEN)]
    BW,BH,BY = Inches(2.3),Inches(1.3),Inches(1.25)
    for lbl,bx,col in boxes:
        rect(sl,bx,BY,BW,BH,fill=col)
        tb(sl,lbl,bx,BY+Inches(0.2),BW,BH-Inches(0.2),
           size=14,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    for i in range(len(boxes)-1):
        tb(sl,"→",boxes[i][1]+BW,BY+BH/2-Inches(0.25),Inches(0.35),Inches(0.5),
           size=20,bold=True,color=DGRAY,align=PP_ALIGN.CENTER)
    subs=["xyz+normals","3 planes","depth+occ.","per-px feats","per-pt label"]
    for (_,bx,__),sub in zip(boxes,subs):
        tb(sl,sub,bx,BY+BH+Inches(0.08),BW,Inches(0.35),
           size=10,italic=True,color=DGRAY,align=PP_ALIGN.CENTER)
    rect(sl,Inches(0.3),Inches(2.85),Inches(12.7),Inches(0.45),fill=LBLUE)
    tb(sl,"Back-projection: each 3D point inherits the CNN score of its projected pixel",
       Inches(0.5),Inches(2.9),Inches(12.3),Inches(0.38),
       size=12,italic=True,color=NAVY,align=PP_ALIGN.CENTER)
    # Pipeline strip image
    addimg(sl, fig_pipeline_strip(), Inches(0.3), Inches(3.4), Inches(12.7), Inches(3.8))
    return sl


def slide_method(prs):
    sl = blank(prs)
    header_bar(sl,"Methodology — Controlled Ablation",
               "One component added per step — isolates each contribution")
    steps=[("Step 1","Baseline CNN","Depth only",DGRAY,"✓ done"),
           ("Step 2","+ Normals","Normal map channels",TEAL,"✓ done"),
           ("Step 3","+ Splatting","Gaussian splat",TEAL,"✓ done"),
           ("Step 4","+ Bilinear","Bilinear projection",NAVY,"✓ done"),
           ("Step 5","+ Global Context","scatter_add global feat",NAVY,"✓ done"),
           ("Step 6","+ View Attention","MLP attention over views",GREEN,"running")]
    col_x=[Inches(0.3),Inches(1.55),Inches(4.0),Inches(8.55),Inches(10.8)]
    col_w=[Inches(1.2),Inches(2.4),Inches(4.5),Inches(2.2),Inches(2.1)]
    Y0=Inches(1.25)
    for hdr,cx,cw in zip(["","Model","Change","Status",""],col_x,col_w):
        rect(sl,cx,Y0,cw-Inches(0.04),Inches(0.42),fill=NAVY)
        tb(sl,hdr,cx+Inches(0.04),Y0+Inches(0.04),cw-Inches(0.08),Inches(0.34),
           size=12,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    for i,(step,name,what,col,status) in enumerate(steps):
        y=Inches(1.72+i*0.62)
        bg=LGRAY if i%2==0 else WHITE
        rect(sl,Inches(0.3),y,Inches(12.5),Inches(0.58),
             fill=bg,line=RGBColor(0xDD,0xDD,0xDD),line_w=Pt(0.5))
        scol=GREEN if status=="✓ done" else RED
        for txt,cx,cw,tcol in [(step,col_x[0],col_w[0],col),
                                (name,col_x[1],col_w[1],col),
                                (what,col_x[2],col_w[2],DARK),
                                (status,col_x[3],col_w[3],scol)]:
            tb(sl,txt,cx+Inches(0.05),y+Inches(0.12),cw-Inches(0.1),Inches(0.38),
               size=12,bold=(txt==name),color=tcol,
               align=PP_ALIGN.LEFT if txt==what else PP_ALIGN.CENTER)
    # Mini F1 progression chart on right
    addimg(sl, fig_results_chart(), Inches(9.0), Inches(5.5), Inches(4.1), Inches(1.85))
    return sl


def slide_input(prs):
    sl = blank(prs)
    header_bar(sl,"Input Improvements","Steps 2 & 3 — What the CNN sees")
    panel(sl,Inches(0.4),Inches(1.3),Inches(5.9),Inches(3.5),
          "Step 2 — Surface Normals",
          ["Add normal map alongside depth",
           "Each pixel: (nx, ny, nz)  →  RGB",
           "CNN sees surface orientation",
           "Input: 4 channels  (depth + 3 normals)"],
          accent=TEAL)
    panel(sl,Inches(7.0),Inches(1.3),Inches(5.9),Inches(3.5),
          "Step 3 — Gaussian Splatting",
          ["Each 3D point → 2D Gaussian blob",
           "Fills gaps: denser coverage",
           "σ = 0.5 px — soft footprint",
           "Reduces empty-pixel artefacts"],
          accent=NAVY)
    # Images
    addimg(sl, fig_normals(),  Inches(0.4), Inches(4.9), Inches(5.9), Inches(2.35))
    addimg(sl, fig_splatting(), Inches(7.0), Inches(4.9), Inches(5.9), Inches(2.35))
    return sl


def slide_projection(prs):
    sl = blank(prs)
    header_bar(sl,"Projection Improvement","Step 4 — Bilinear Interpolation")
    rect(sl,Inches(0.5),Inches(1.4),Inches(5.7),Inches(3.5),
         fill=LGRAY,line=DGRAY,line_w=Pt(1.5))
    tb(sl,"Floor Projection  (Steps 1–3)",
       Inches(0.7),Inches(1.5),Inches(5.3),Inches(0.5),
       size=15,bold=True,color=DGRAY,align=PP_ALIGN.CENTER)
    for i,s in enumerate(["Point → nearest pixel","Fast, deterministic",
                           "✗  Aliasing / staircasing","✗  Empty pixels between points"]):
        tb(sl,s,Inches(0.8),Inches(2.1+i*0.65),Inches(5.0),Inches(0.6),size=14,color=DARK)
    tb(sl,"vs",Inches(6.3),Inches(3.4),Inches(0.7),Inches(0.5),
       size=20,bold=True,color=DGRAY,align=PP_ALIGN.CENTER)
    rect(sl,Inches(7.1),Inches(1.4),Inches(5.7),Inches(3.5),
         fill=LBLUE,line=TEAL,line_w=Pt(2))
    tb(sl,"Bilinear Projection  (Step 4)",
       Inches(7.3),Inches(1.5),Inches(5.3),Inches(0.5),
       size=15,bold=True,color=TEAL,align=PP_ALIGN.CENTER)
    for i,s in enumerate(["Point → 4 surrounding pixels","Weighted by sub-pixel distance",
                           "✔  Smoother, denser coverage","✔  Sub-pixel accuracy"]):
        tb(sl,s,Inches(7.3),Inches(2.1+i*0.65),Inches(5.2),Inches(0.6),size=14,color=DARK)
    addimg(sl, fig_bilinear(), Inches(1.5), Inches(5.1), Inches(10.0), Inches(2.15))
    return sl


def slide_model(prs):
    sl = blank(prs)
    header_bar(sl,"Model Improvements","Steps 5 & 6")
    panel(sl,Inches(0.4),Inches(1.3),Inches(5.9),Inches(3.3),
          "Step 5 — Global Context",
          ["Mean-pool all pixel features per fragment",
           "Concat back into every pixel",
           "Global scene awareness per pixel",
           "Inspired by squeeze-and-excite"],
          accent=NAVY)
    panel(sl,Inches(7.0),Inches(1.3),Inches(5.9),Inches(3.3),
          "Step 6 — View Attention",
          ["3 views → 3 feature maps",
           "MLP scores each view",
           "Softmax → weighted sum",
           "Best view dominates"],
          accent=GREEN)
    addimg(sl, fig_context_diagram(),  Inches(0.4), Inches(4.7), Inches(5.9), Inches(2.55))
    addimg(sl, fig_attention_diagram(), Inches(7.0), Inches(4.7), Inches(5.9), Inches(2.55))
    return sl


def slide_results(prs, data=None):
    sl = blank(prs)
    header_bar(sl,"Results — Ablation Table",
               "Breaking Bad val set  ·  50 epochs  ·  batch 32")
    rows=[("Step 1","Baseline CNN","—"),("Step 2","+ Normals","normals"),
          ("Step 3","+ Splatting","splatting"),("Step 4","+ Bilinear","bilinear"),
          ("Step 5","+ Context","context"),("Step 6","+ Attention","attention"),
          ("Base","GARF (PTv3)","full 3D")]
    def v(i,k):
        return f"{data[i][k]:.3f}" if data and i<len(data) and k in data[i] else "??.???"
    col_x=[Inches(0.2),Inches(1.0),Inches(3.1),Inches(5.3),
           Inches(6.7),Inches(8.1),Inches(9.5)]
    col_w=[Inches(0.78),Inches(2.05),Inches(2.15),Inches(1.35),
           Inches(1.35),Inches(1.35),Inches(1.35)]
    hdrs=["","Model","Feature","Loss↓","Acc↑","F1↑","Rec↑"]
    Y0=Inches(1.28)
    for hdr,cx,cw in zip(hdrs,col_x,col_w):
        rect(sl,cx,Y0,cw-Inches(0.04),Inches(0.42),fill=NAVY)
        tb(sl,hdr,cx+Inches(0.04),Y0+Inches(0.05),cw-Inches(0.1),Inches(0.32),
           size=11,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    for i,(step,name,feat) in enumerate(rows):
        y=Inches(1.75+i*0.64)
        is_best=(i==4); is_garf=(i==6)
        if is_garf:
            rect(sl,Inches(0.2),y-Inches(0.05),Inches(10.8),Inches(0.07),fill=TEAL)
        bg=RGBColor(0xFF,0xF9,0xE6) if is_best else (LGRAY if i%2==0 else WHITE)
        rect(sl,Inches(0.2),y,Inches(10.8),Inches(0.60),
             fill=bg,line=RGBColor(0xCC,0xCC,0xCC),line_w=Pt(0.5))
        vals=[step,name,feat,v(i,"loss"),v(i,"acc"),v(i,"f1"),v(i,"recall")]
        for j,(val,cx,cw) in enumerate(zip(vals,col_x,col_w)):
            c=GOLD if(is_best and j>=3) else (TEAL if is_garf else DARK)
            tb(sl,val,cx+Inches(0.04),y+Inches(0.13),cw-Inches(0.08),Inches(0.36),
               size=11,bold=(is_best and j>=3),color=c,
               align=PP_ALIGN.LEFT if j==1 else PP_ALIGN.CENTER)
    addimg(sl, fig_results_chart(data), Inches(11.0), Inches(1.28), Inches(2.1), Inches(5.3))
    tb(sl,"★ Best CNN highlighted  |  Fill in real values after eval",
       Inches(0.2),Inches(6.72),Inches(10.8),Inches(0.4),
       size=10,italic=True,color=DGRAY,align=PP_ALIGN.CENTER)
    return sl


def slide_insights(prs):
    sl = blank(prs)
    header_bar(sl,"Key Insights")
    insights=[(TEAL,"Normals boost recall  (+6.3 pp)",
               "Recall jumps 81.1% → 87.3% — fewer fracture points missed. "
               "The CNN sees surface orientation, not just depth."),
              (NAVY,"Splatting improves accuracy  (+2.5 pp F1)",
               "Gaussian splatting fills projection gaps → denser coverage, "
               "fewer false positives from empty pixels."),
              (GREEN,"Global context gives best F1  (72.6%)",
               "Injecting a fragment-level descriptor into each pixel gives the biggest "
               "single model gain — local + global reasoning wins.")]
    for i,(col,title,desc) in enumerate(insights):
        y=Inches(1.45+i*1.7)
        rect(sl,Inches(0.3),y,Inches(0.22),Inches(1.2),fill=col)
        tb(sl,title,Inches(0.65),y+Inches(0.05),Inches(7.6),Inches(0.5),
           size=19,bold=True,color=col)
        tb(sl,desc,Inches(0.65),y+Inches(0.58),Inches(7.6),Inches(0.6),
           size=14,color=DARK)
    addimg(sl, fig_insights(), Inches(8.5), Inches(1.3), Inches(4.6), Inches(5.8))
    return sl


def slide_limitations(prs):
    sl = blank(prs)
    header_bar(sl,"Limitations")
    items=[(ORANGE,"Bilinear projection not effective",
            "Step 4 showed marginal / negative gains. Splatting already handles "
            "coverage; bilinear may add complexity without benefit."),
           (RED,"Projection is inherently lossy",
            "3D → 2D discards depth ordering and point density. Some fracture "
            "patterns are only visible from angles not covered by 3 fixed views."),
           (DGRAY,"Limited training budget (50 epochs)",
            "GARF uses 500 epochs. Our budget was reduced for feasibility — "
            "results may underestimate final CNN convergence.")]
    for i,(col,title,desc) in enumerate(items):
        y=Inches(1.45+i*1.65)
        rect(sl,Inches(0.3),y,Inches(0.22),Inches(1.2),fill=col)
        tb(sl,title,Inches(0.65),y+Inches(0.05),Inches(7.8),Inches(0.5),
           size=18,bold=True,color=col)
        tb(sl,desc,Inches(0.65),y+Inches(0.58),Inches(7.8),Inches(0.6),
           size=14,color=DARK)
    addimg(sl, fig_bilinear(), Inches(9.2), Inches(1.4), Inches(3.9), Inches(1.9))
    addimg(sl, fig_splatting(), Inches(9.2), Inches(3.5), Inches(3.9), Inches(1.9))
    return sl


def slide_nextsteps(prs):
    sl = blank(prs)
    header_bar(sl,"Next Steps")
    cols=[("Short term",TEAL,["Complete Step 6 — fill results table",
                               "Evaluate GARF baseline (same val set)",
                               "Collect 3D/2D visualisations"]),
          ("Medium term",NAVY,["Train 500 epochs for fair comparison",
                                "Adaptive view selection (learnable angle)",
                                "Test hybrid: GARF encoder + geo features"]),
          ("Research Q",GREEN,["Can 2D CNN match 3D PTv3 on recall?",
                                "Do projections complement 3D embeddings?"])]
    CW=Inches(3.7)
    for i,(term,col,items) in enumerate(cols):
        cx=Inches(0.4+i*4.3)
        rect(sl,cx,Inches(1.3),CW,Inches(5.5),fill=LGRAY,line=col,line_w=Pt(2))
        rect(sl,cx,Inches(1.3),CW,Inches(0.55),fill=col)
        tb(sl,term,cx+Inches(0.1),Inches(1.35),CW-Inches(0.2),Inches(0.45),
           size=15,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
        for j,item in enumerate(items):
            tb(sl,"→  "+item,cx+Inches(0.15),Inches(2.0+j*1.1),CW-Inches(0.3),Inches(1.0),
               size=13,color=DARK)
    addimg(sl, fig_timeline(), Inches(0.4), Inches(6.5), Inches(12.5), Inches(0.82))
    return sl


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="presentation.pptx")
    args = ap.parse_args()

    print("Generating synthetic visuals…")
    prs = new_prs()
    slide_title(prs)
    slide_problem(prs)
    slide_idea(prs)
    slide_pipeline(prs)
    slide_method(prs)
    slide_input(prs)
    slide_projection(prs)
    slide_model(prs)
    # Real results — best val F1 checkpoint, seed 1116, 50 epochs
    real_data = [
        {"acc": 0.7897, "f1": 0.6805, "loss": None, "recall": 0.8107},  # Step 1
        {"acc": 0.7988, "f1": 0.7090, "loss": None, "recall": 0.8732},  # Step 2
        {"acc": 0.8240, "f1": 0.7193, "loss": None, "recall": 0.8715},  # Step 3
        {"acc": 0.8115, "f1": 0.7087, "loss": None, "recall": 0.8851},  # Step 4
        {"acc": 0.8248, "f1": 0.7262, "loss": None, "recall": 0.8882},  # Step 5
        {},                                                               # Step 6 pending
        {"acc": 0.9257, "f1": 0.9094, "loss": 0.0906, "recall": 0.9115},# GARF baseline
    ]
    slide_results(prs, data=real_data)
    slide_insights(prs)
    slide_limitations(prs)
    slide_nextsteps(prs)
    prs.save(args.out)
    print(f"Saved → {args.out}  ({len(prs.slides)} slides)")

if __name__ == "__main__":
    main()
