"""
scripts/phase9a_analyze_rows.py
================================
Phase 9A (PLAN_REASSEMBLY_MODULE.md) -- agrège le CSV par-paire produit par
`phase8_pipeline_learned_check.py --rows_csv ...` : médiane/p25/p75 par
groupe (A/B/C), et croisement rot_err_stage1 (binné) x icp_status au sein
du groupe C -- pour vérifier si l'ICP dégrade surtout les cas où l'étage 1
est déjà loin de la GT (hypothèse : hors bassin de convergence, cf. Phase
6A) plutôt que de façon homogène.

Usage :
    python scripts/phase9a_analyze_rows.py /tmp/student7/phase9a_rows_thresh03.csv
"""

import sys

import pandas as pd

NUMERIC_COLS = [
    "rot_err_stage1", "trans_err_stage1", "rot_err_final", "trans_err_final",
    "delta_rot_icp", "delta_trans_icp", "n_corr", "mirror_prob",
    "icp_energy_final", "overlap_frac_final", "normal_consistency_final",
    "n_frac_pts_min",
]

ROT_ERR_BINS = [0, 30, 60, 90, 180]
ROT_ERR_LABELS = ["0-30", "30-60", "60-90", "90-180"]


def main():
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    print(f"Total paires étage 2 : {len(df)}\n")

    print("=== Médiane / p25 / p75 par groupe (A = Pose@30+strict, "
          "B = Pose@30 sans strict, C = Pose@30 raté) ===")
    for g in ("A", "B", "C"):
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        print(f"\n-- Groupe {g} (n={len(sub)}) --")
        stats = sub[NUMERIC_COLS].agg(
            lambda s: f"{s.median():.3g} [{s.quantile(0.25):.3g}, {s.quantile(0.75):.3g}]"
        )
        for col, val in stats.items():
            print(f"  {col:28s}: {val}")

    print("\n=== ICP status x tranche d'erreur angle étage1, au sein de chaque groupe ===")
    df["rot_err_stage1_bin"] = pd.cut(df["rot_err_stage1"], bins=ROT_ERR_BINS,
                                       labels=ROT_ERR_LABELS, include_lowest=True)
    for g in ("A", "B", "C"):
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        print(f"\n-- Groupe {g} --")
        table = pd.crosstab(sub["rot_err_stage1_bin"], sub["icp_status"],
                             normalize="index") * 100
        print(table.round(1))
        counts = sub["rot_err_stage1_bin"].value_counts().sort_index()
        print("  (n par tranche)")
        print(counts)

    print("\n=== mirror_prob (confiance miroir) x icp_status, groupe C uniquement ===")
    sub_c = df[df["group"] == "C"].copy()
    sub_c["mirror_conf_bin"] = pd.cut(
        (sub_c["mirror_prob"] - 0.5).abs(), bins=[0, 0.1, 0.3, 0.5],
        labels=["proche 0.5 (incertain)", "moyen", "confiant"], include_lowest=True)
    table = pd.crosstab(sub_c["mirror_conf_bin"], sub_c["icp_status"], normalize="index") * 100
    print(table.round(1))

    print("\n=== 9A-bis : split du groupe B par rot_ok/trans_ok (seuil strict "
          "5deg/0.02), avant ET après ICP ===")
    rot_thresh, trans_thresh = 5.0, 0.02
    sub_b = df[df["group"] == "B"].copy()

    def _split(rot_col, trans_col):
        rot_ok = sub_b[rot_col] < rot_thresh
        trans_ok = sub_b[trans_col] < trans_thresh
        cat = pd.Series("both_fail", index=sub_b.index)
        cat[rot_ok & ~trans_ok] = "trans_fail_only"
        cat[~rot_ok & trans_ok] = "rot_fail_only"
        cat[rot_ok & trans_ok] = "both_ok"
        return cat

    sub_b["fail_mode_before_icp"] = _split("rot_err_stage1", "trans_err_stage1")
    sub_b["fail_mode_after_icp"] = _split("rot_err_final", "trans_err_final")

    print("\n-- Avant ICP (n=%d) --" % len(sub_b))
    print((sub_b["fail_mode_before_icp"].value_counts(normalize=True) * 100).round(1))
    print("\n-- Après ICP (n=%d) --" % len(sub_b))
    print((sub_b["fail_mode_after_icp"].value_counts(normalize=True) * 100).round(1))

    print("\n-- Transition avant -> après ICP (lignes=avant, colonnes=après) --")
    transition = pd.crosstab(sub_b["fail_mode_before_icp"], sub_b["fail_mode_after_icp"])
    print(transition)

    print("\n=== Groupe C : sparsité (n_frac_pts_min) x catastrophique "
          "(rot_err_stage1 >= 90°) ===")
    print("(teste si la majorité du groupe C non expliquée par le miroir "
          "s'explique par un signal trop pauvre pour l'étage 1, indépendamment "
          "du miroir -- cf. PLAN_REASSEMBLY_MODULE.md, 2026-08-03)")
    sub_c2 = df[df["group"] == "C"].copy()
    sub_c2["catastrophic"] = sub_c2["rot_err_stage1"] >= 90
    density_bins = [0, 25, 50, 100, 200, 10**6]
    density_labels = ["<25", "25-50", "50-100", "100-200", "200+"]
    sub_c2["density_bin"] = pd.cut(sub_c2["n_frac_pts_min"], bins=density_bins,
                                    labels=density_labels, include_lowest=True)
    table = pd.crosstab(sub_c2["density_bin"], sub_c2["catastrophic"], normalize="index") * 100
    print(table.round(1))
    counts = sub_c2["density_bin"].value_counts().sort_index()
    print("  (n par tranche)")
    print(counts)
    print("\n  Médiane n_frac_pts_min, catastrophique vs pas (au sein de C) :")
    print(sub_c2.groupby("catastrophic")["n_frac_pts_min"].median())


if __name__ == "__main__":
    main()
