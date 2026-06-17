"""
scripts/inject_stats_html.py
----------------------------
Lit les couleurs directement dans le JSON embarqué du HTML Plotly (Python pur),
calcule les stats, injecte un div HTML statique — aucune dépendance JavaScript.

Usage:
    python scripts/inject_stats_html.py --viz_dir viz
"""

import argparse
import re
from collections import Counter
from pathlib import Path


# ── Classification par teinte dominante ──────────────────────────────────────

def classify(r: int, g: int, b: int) -> str:
    if r > 220 and g > 170 and b < 30:           return 'gt'   # jaune  GT
    if g > 140 and r < 130 and b < 130:           return 'tp'   # vert   TP
    if r > 220 and g < 100 and b < 100:           return 'fp'   # rouge  FP
    if b > 200 and r < 80:                        return 'fn'   # bleu   FN
    if abs(r - g) < 20 and abs(g - b) < 20:      return 'tn'   # gris   TN
    return 'other'


def parse_trace_colors(html: str) -> list[Counter]:
    """
    Extrait les tableaux de couleurs de chaque trace Plotly depuis le HTML brut.
    Retourne une liste de Counter (un par trace avec > 50 points).
    """
    results = []
    # Trouver chaque bloc "color":["rgb(...", ...] dans le JSON embarqué
    for m in re.finditer(r'"color":\s*\["rgb\(', html):
        bracket_pos = html.index('[', m.start())
        # Avancer jusqu'à la fermeture du tableau
        depth, i = 0, bracket_pos + 1
        while i < len(html):
            c = html[i]
            if c == '[':
                depth += 1
            elif c == ']':
                if depth == 0:
                    break
                depth -= 1
            i += 1
        arr_text = html[bracket_pos: i + 1]
        # Compter les couleurs directement depuis le texte
        tuples = re.findall(r'rgb\((\d+),(\d+),(\d+)\)', arr_text)
        if len(tuples) < 50:          # ignorer tableaux minuscules (legend, etc.)
            continue
        counter = Counter()
        for r_s, g_s, b_s in tuples:
            cat = classify(int(r_s), int(g_s), int(b_s))
            counter[cat] += 1
        results.append(counter)
    return results


# ── Génération du HTML statique ───────────────────────────────────────────────

DOT = (
    '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
    '<span style="width:10px;height:10px;border-radius:50%;background:{color};'
    'flex-shrink:0;border:1px solid rgba(0,0,0,.1)"></span>{label}</span>'
)


def dot(color: str, label: str) -> str:
    return DOT.format(color=color, label=label)


def fmt(n: int) -> str:
    return "{:,}".format(n).replace(",", " ").replace(",", " ")   # espace fine comme séparateur


def gt_block(c: Counter) -> str:
    total = sum(c.values())
    return (
        dot("#FFD600", f"<b>{fmt(c['gt'])}</b> fracture pts")
        + dot("#D8D8D8", f"<b>{fmt(c['tn'])}</b> intact pts")
        + f'<span style="color:#aaa;margin-left:6px">Total: <b>{fmt(total)}</b></span>'
    )


def pred_block(c: Counter) -> str:
    total = sum(c.values())
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fdr  = fp / max(fp + tp, 1)
    return (
        f'<div style="margin-bottom:7px">'
        + dot("#4CAF50", f"<b>{fmt(tp)}</b> TP")
        + dot("#F44336", f"<b>{fmt(fp)}</b> FP")
        + dot("#2196F3", f"<b>{fmt(fn)}</b> FN")
        + dot("#D8D8D8", f"<b>{fmt(c['tn'])}</b> TN")
        + f'<span style="color:#aaa;margin-left:6px">Total: <b>{fmt(total)}</b></span>'
        + "</div>"
        + f'<div style="display:flex;gap:20px;font-size:13px;color:#444">'
        + f'<span>F1 <b style="font-size:15px;color:#111">{f1:.3f}</b></span>'
        + f'<span>Precision <b>{prec:.3f}</b></span>'
        + f'<span>Recall <b>{rec:.3f}</b></span>'
        + f'<span>FDR <b>{fdr:.3f}</b></span>'
        + f'<span style="color:#bbb;font-size:12px">pred={fmt(tp+fp)} / gt={fmt(tp+fn)}</span>'
        + "</div>"
    )


def build_stats_bar(traces: list[Counter], names: list[str]) -> str:
    n = len(traces)
    col_w = f"{100/n:.2f}%"
    cells = ""
    for c, name in zip(traces, names):
        is_gt = c["gt"] > 0 and c["tp"] == 0 and c["fp"] == 0 and c["fn"] == 0
        content = gt_block(c) if is_gt else pred_block(c)
        cells += (
            f'<td style="width:{col_w};vertical-align:top;padding:14px 18px;'
            f'border-right:1px solid #E5E7EB">'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;color:#999;margin-bottom:8px">{name}</div>'
            f'{content}</td>'
        )
    return (
        '<div id="fracture-stats" style="background:#FAFAFA;border-top:2px solid #E5E7EB;'
        'font-family:Inter,system-ui,sans-serif;font-size:13px;color:#222;box-sizing:border-box">'
        f'<table style="width:100%;border-collapse:collapse"><tr>{cells}</tr></table>'
        '</div>'
    )


# ── Extraction des noms de traces ─────────────────────────────────────────────

def extract_trace_names(html: str) -> list[str]:
    """Extrait les noms des traces dans l'ordre d'apparition."""
    return re.findall(r'"name"\s*:\s*"([^"]+)"', html)


# ── Injection / nettoyage ─────────────────────────────────────────────────────

MARKER_START = '<!-- fracture-stats-start -->'
MARKER_END   = '<!-- fracture-stats-end -->'


def clean_previous(content: str) -> str:
    """Supprime une injection précédente."""
    # Supprimer ancien script JS
    content = re.sub(
        r'\n?<script[^>]*id="fracture-stats-script"[^>]*>.*?</script>',
        '', content, flags=re.DOTALL
    )
    # Supprimer ancienne div statique
    content = re.sub(
        MARKER_START + r'.*?' + MARKER_END,
        '', content, flags=re.DOTALL
    )
    return content


def inject(html_path: Path):
    content = html_path.read_text(encoding="utf-8")
    content = clean_previous(content)

    traces = parse_trace_colors(content)
    if not traces:
        print(f"  [warn] {html_path.name}  — no color traces found")
        return

    names = extract_trace_names(content)
    # garder seulement les noms qui correspondent à des traces avec données
    # (parfois il y a des noms de traces "dummy" dans la config)
    while len(names) < len(traces):
        names.append(f"Trace {len(names)+1}")
    names = names[:len(traces)]

    stats_html = (
        f'\n{MARKER_START}\n'
        + build_stats_bar(traces, names)
        + f'\n{MARKER_END}\n'
    )

    if "</body>" in content:
        content = content.replace("</body>", stats_html + "</body>", 1)
    else:
        content += stats_html

    html_path.write_text(content, encoding="utf-8")

    # Résumé console
    print(f"  [ok]   {html_path.name}")
    for c, name in zip(traces, names):
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        total = sum(c.values())
        is_gt = c["gt"] > 0 and tp == 0 and fp == 0 and fn == 0
        if is_gt:
            print(f"         {name}: fracture={fmt(c['gt'])} intact={fmt(c['tn'])} total={fmt(total)}")
        else:
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"         {name}: TP={fmt(tp)} FP={fmt(fp)} FN={fmt(fn)} TN={fmt(c['tn'])} "
                  f"F1={f1:.3f} Prec={prec:.3f} Rec={rec:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_dir", default="viz")
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    html_files = sorted(viz_dir.glob("*.html"))
    if not html_files:
        print(f"Aucun fichier HTML dans {viz_dir}")
        return

    print(f"Traitement de {len(html_files)} fichier(s)...\n")
    for f in html_files:
        inject(f)
    print("\nDone.")


if __name__ == "__main__":
    main()
