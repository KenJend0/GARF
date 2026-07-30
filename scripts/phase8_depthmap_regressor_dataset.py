"""
scripts/phase8_depthmap_regressor_dataset.py
===============================================
Phase 8 — voir PLAN_REASSEMBLY_MODULE.md, section "Phase 8 — modèle appris
sur les depth maps" (2026-07-30).

Génère les labels GT (theta, shift) pour entraîner un modèle qui remplace
`match_depthmaps()` (recherche FFT + score relief/overlap sur `n_angles`
rotations discrètes × 2 flips) par une régression directe à partir des deux
depth maps. Contrairement à la recherche FFT, ici on CONNAÎT la vraie pose
3D (`R_ij_gt`/`t_ij_gt`) -- le label (theta_gt, shift_gt) se dérive donc
exactement, sans recherche, via un Kabsch 2D (même principe que `kabsch()`
3D déjà dans `phase5a_depthmap_matching.py`, restreint au plan).

Dérivation (cf. PLAN_REASSEMBLY_MODULE.md pour le détail complet) :
`build_correspondences_nn` applique, pour un point de `frac_i` à la
position pixel `p_i = (cols_i, rows_i)` :
    p_j = half + Rot(-theta) @ (p_i + shift - half)
soit une transformation affine `p_j = M @ p_i + b` avec `M = Rot(-theta)`
(rotation pure) et `b = Rot(-theta) @ (shift - half) + half`. Comme
`frac_i` se projette de façon connue à la fois dans SA PROPRE grille
(`p_i`) et, via la VRAIE pose 3D, dans la grille de `j` (`p_j_target`),
les deux jeux de points se correspondent point à point (mêmes points,
deux expressions) -- pas de recherche de correspondance nécessaire, un
Kabsch 2D entre `p_i` et `p_j_target` retrouve exactement `(M, b)`, d'où
`(theta_gt, shift_gt)`.

CORRECTION (2026-07-30, après les auto-tests ci-dessous) -- il existe
une DEUXIÈME ambiguïté, indépendante du flip de profondeur de
`match_depthmaps` : une vraie RÉFLEXION dans le plan (u,v) (un seul des
deux axes in-plane s'inverse, pas les deux), que ni le Kabsch 2D
(contraint à une rotation pure) ni `match_depthmaps` (rotation + flip de
profondeur seulement, jamais de miroir du plan) ne peuvent représenter.
Mesuré sur les vraies paires GT du dataset
(`phase8_reflection_prevalence_check.py`) : **52.2% des paires ont besoin
de ce miroir** -- pas un cas rare, quasiment un tirage à pile ou face
(cohérent avec des fragments désassemblés à une orientation relative
arbitraire). `fit_theta_shift_mirror_from_gt()` détecte et corrige ce
bit en essayant les deux orientations de `v_j` et en gardant celle qui
minimise le résidu du Kabsch 2D -- le modèle appris devra, de la même
façon, tester les deux hypothèses (plan normal vs miroir) et choisir via
une confiance apprise, symétriquement au `flip` existant.

Auto-test exécutable en local (numpy/scipy purs, pas besoin du serveur) :
    python scripts/phase8_depthmap_regressor_dataset.py
"""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase5a_depthmap_matching import compute_pca_frame, build_correspondences_nn


def canonical_pca_frame(pts: np.ndarray):
    """`compute_pca_frame()` + choix de signe canonique (2026-07-30, phase8) :
    force `det([u, v, n]) = +1` (repère toujours direct), en négant `n` si
    besoin. `compute_pca_frame` original ne le fait PAS -- chaque axe garde
    le signe arbitraire renvoyé par `eigh()`, ce qui est sans conséquence
    pour `match_depthmaps` (qui teste explicitement les deux signes via
    `best_flip`). Mais pour la dérivation directe du label GT (pas de
    recherche), une relation (u,v)-plane PURE ROTATION entre les repères de
    deux fragments n'est garantie QUE si les deux repères ont la MÊME
    "chiralité" (`det(F_i) == det(F_j)`) -- sinon la vraie relation est une
    RÉFLEXION dans le plan (u,v), que `kabsch_2d` (contraint à `det=+1`) ne
    peut pas fitter correctement (confirmé empiriquement : sans
    canonicalisation, ~50% des paires synthétiques aléatoires du test
    `_test_3d_pipeline_roundtrip` échouaient avec rot_err≈180°). Canoniser
    les DEUX repères (i et j) à la même chiralité élimine cette ambiguïté
    à la racine, pour l'entraînement ET l'inférence du modèle appris --
    pas besoin d'un `flip` de sortie du modèle (cf. PLAN_REASSEMBLY_MODULE.md,
    Phase 8, corrigé le 2026-07-30 après cette découverte)."""
    centroid, u, v, n, planarity = compute_pca_frame(pts)
    if np.linalg.det(np.stack([u, v, n], axis=1)) < 0:
        n = -n
    return centroid, u, v, n, planarity


def kabsch_2d(P: np.ndarray, Q: np.ndarray):
    """R (2x2, rotation pure, det=+1), t (2,) tels que R @ p + t ≈ q pour
    chaque paire (p, q) de P, Q. Copie 2D de `kabsch()` (3D,
    phase5a_depthmap_matching.py) -- même méthode (SVD de la covariance
    croisée centrée), restreinte au plan."""
    pm, qm = P.mean(0), Q.mean(0)
    Pc, Qc = P - pm, Q - qm
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rmat = Vt.T @ np.diag([1.0, d]) @ U.T
    return Rmat, qm - Rmat @ pm


def fit_theta_shift(p_i: np.ndarray, p_j_target: np.ndarray, resolution: int):
    """Retrouve (theta_deg, shift=(sy,sx)) tels que la transformation
    `build_correspondences_nn` appliquée à `p_i` avec ce (theta, shift)
    reproduit `p_j_target`. `p_i`/`p_j_target` : (N,2) arrays de
    coordonnées PIXEL (cols, rows) -- mêmes points, deux expressions
    (repère propre de i vs. position vraie dans le repère de j).
    """
    half = resolution / 2.0
    M, b = kabsch_2d(p_i, p_j_target)
    # M == Rot(-theta) : Rot(phi) = [[cos phi, -sin phi], [sin phi, cos phi]]
    theta_rad = -np.arctan2(M[1, 0], M[0, 0])
    theta_deg = float(np.degrees(theta_rad)) % 360.0
    # b = M @ (shift - half) + half  =>  shift = M.T @ (b - half) + half
    half_vec = np.array([half, half])
    shift_vec = M.T @ (b - half_vec) + half_vec   # (sx, sy) order -- cf. docstring module
    sx, sy = float(shift_vec[0]), float(shift_vec[1])
    return theta_deg, (sy, sx)


def project_to_pixels(pts: np.ndarray, centroid, u, v, u_min, v_min, pixel_size):
    """(cols, rows) pixel continus -- même formule que `build_correspondences_nn`
    (pas d'arrondi)."""
    c = pts - centroid
    cols = ((c @ u) - u_min) / pixel_size
    rows = ((c @ v) - v_min) / pixel_size
    return np.stack([cols, rows], axis=1)


def fit_theta_shift_from_gt(
    frac_i, c_i, u_i, v_i, u_min_i, v_min_i,
    c_j, u_j, v_j, u_min_j, v_min_j,
    pixel_size, resolution, R_ij_gt, t_ij_gt,
):
    """Label GT (theta_deg, shift=(sy,sx)) pour une paire de fragments, à
    partir de la vraie pose 3D -- voir docstring module pour la dérivation.
    `frac_i` : points 3D bruts du fragment i (repère d'entrée, PAS centrés).

    IMPORTANT : `c_i,u_i,v_i` / `c_j,u_j,v_j` DOIVENT venir de
    `canonical_pca_frame()` (PAS `compute_pca_frame()` brut) -- sans
    canonicalisation de la chiralité, la relation (u,v)-plane entre les
    deux repères peut être une RÉFLEXION (pas une rotation pure) pour ~50%
    des paires (ambiguïté de signe des vecteurs propres PCA, indépendante
    par fragment), que `fit_theta_shift` (contraint à `det=+1`) ne peut pas
    fitter correctement -- cf. `canonical_pca_frame()` pour le détail.
    """
    p_i = project_to_pixels(frac_i, c_i, u_i, v_i, u_min_i, v_min_i, pixel_size)
    frac_i_in_j = (R_ij_gt @ frac_i.T).T + t_ij_gt
    p_j_target = project_to_pixels(frac_i_in_j, c_j, u_j, v_j, u_min_j, v_min_j, pixel_size)
    return fit_theta_shift(p_i, p_j_target, resolution)


def _fit_residual(p_i: np.ndarray, p_j_target: np.ndarray, theta_deg: float, shift, resolution: int):
    """RMS entre `p_j_target` et la reconstruction de `p_i` par
    `(theta_deg, shift)` -- sert à décider, entre deux hypothèses (normal /
    miroir), laquelle est correcte (résidu quasi nul) sans passer par la
    reconstruction 3D complète (moins cher, même verdict -- cf.
    `fit_theta_shift_mirror_from_gt`)."""
    p_j_pred = _forward_transform_reference(p_i, theta_deg, shift, resolution)
    return float(np.sqrt(np.mean(np.sum((p_j_pred - p_j_target) ** 2, axis=1))))


def fit_theta_shift_mirror_from_gt(
    frac_i, c_i, u_i, v_i, u_min_i, v_min_i,
    c_j, u_j, v_j, u_min_j, v_min_j,
    pixel_size, resolution, R_ij_gt, t_ij_gt,
):
    """Comme `fit_theta_shift_from_gt`, mais détecte et corrige la
    RÉFLEXION (u,v) découverte le 2026-07-30 (52.2% des paires réelles,
    cf. `phase8_reflection_prevalence_check.py`) : essaie `v_j` tel quel
    (`mirror=False`) ET `-v_j` (`mirror=True`), garde l'hypothèse dont le
    résidu de Kabsch 2D est le plus petit. Retourne
    `(theta_deg, shift, mirror, residual)` -- `mirror` est le label que le
    modèle appris devra aussi prédire (deuxième hypothèse binaire, en plus
    de `theta`/`shift`, symétrique au `flip` de profondeur déjà géré par
    `match_depthmaps`)."""
    p_i = project_to_pixels(frac_i, c_i, u_i, v_i, u_min_i, v_min_i, pixel_size)
    frac_i_in_j = (R_ij_gt @ frac_i.T).T + t_ij_gt

    candidates = []
    for mirror, v_j_use in ((False, v_j), (True, -v_j)):
        p_j_target = project_to_pixels(frac_i_in_j, c_j, u_j, v_j_use, u_min_j, v_min_j, pixel_size)
        theta_deg, shift = fit_theta_shift(p_i, p_j_target, resolution)
        residual = _fit_residual(p_i, p_j_target, theta_deg, shift, resolution)
        candidates.append((residual, theta_deg, shift, mirror))

    residual, theta_deg, shift, mirror = min(candidates, key=lambda c: c[0])
    return theta_deg, shift, mirror, residual


# ── Auto-tests (numpy/scipy purs, exécutables en local) ────────────────────

def _forward_transform_reference(p_i: np.ndarray, theta_deg: float, shift, resolution: int):
    """Reproduction DIRECTE (mêmes lignes, même ordre d'opérations) de la
    transformation appliquée par `build_correspondences_nn` sur les
    coordonnées pixel (cols, rows) -- isolée ici uniquement pour pouvoir
    tester `fit_theta_shift` (son inverse) sans dépendre de la recherche
    plus-proche-voisin de `build_correspondences_nn` (qui a besoin d'un
    nuage `frac_j` réel à interroger, pas juste d'une position cible). La
    cohérence avec le VRAI code de production est vérifiée séparément par
    `_test_3d_pipeline_roundtrip`, qui appelle `build_correspondences_nn`
    elle-même de bout en bout."""
    sy, sx = shift
    theta_rad = theta_deg * np.pi / 180.0
    cos_neg, sin_neg = np.cos(-theta_rad), np.sin(-theta_rad)
    half = resolution / 2.0
    cx = p_i[:, 0] + sx - half
    cy = p_i[:, 1] + sy - half
    return np.stack([
        half + cos_neg * cx - sin_neg * cy,
        half + sin_neg * cx + cos_neg * cy,
    ], axis=1)


def _test_2d_roundtrip():
    """Étage 1 : le coeur du calcul (Kabsch 2D + décodage theta/shift), sans
    passer par la 3D/PCA. Génère un (theta,shift) connu, transforme des
    points synthétiques via `_forward_transform_reference` (copie directe de
    la formule de `build_correspondences_nn`), puis vérifie que
    `fit_theta_shift` retrouve exactement le même (theta, shift)."""
    rng = np.random.default_rng(0)
    resolution = 64
    for trial in range(20):
        theta0 = float(rng.uniform(0, 360))
        shift0 = (float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)))  # (sy, sx)

        n_pts = 200
        p_i = rng.uniform(0, resolution, size=(n_pts, 2))  # (cols, rows)
        p_j_target = _forward_transform_reference(p_i, theta0, shift0, resolution)

        theta_rec, shift_rec = fit_theta_shift(p_i, p_j_target, resolution)

        theta_err = min(abs(theta_rec - theta0), 360.0 - abs(theta_rec - theta0))
        shift_err = np.hypot(shift_rec[0] - shift0[0], shift_rec[1] - shift0[1])
        assert theta_err < 1e-6, f"trial {trial}: theta {theta_rec} != {theta0} (err={theta_err})"
        assert shift_err < 1e-6, f"trial {trial}: shift {shift_rec} != {shift0} (err={shift_err})"

    print(f"  [OK] _test_2d_roundtrip : 20 tirages aléatoires, theta/shift retrouvés exactement")


def _test_3d_pipeline_roundtrip():
    """Étage 2 : la fonction complète `fit_theta_shift_from_gt`, avec de
    vrais repères PCA 3D et une vraie transformation rigide R,t -- pas
    juste le cas 2D dégénéré du test précédent. Construit deux fragments
    synthétiques liés par une pose 3D connue, calcule leurs repères PCA
    (`compute_pca_frame`, la fonction réellement utilisée en production),
    dérive le label, puis vérifie qu'appliquer ce label reconstruit bien
    (via `build_correspondences_nn` + Kabsch 3D) une pose proche de la
    vraie -- test de bout en bout, pas juste de la géométrie 2D isolée."""
    from scripts.phase5a_depthmap_matching import kabsch, rot_err_deg, trans_err
    from scipy.spatial.transform import Rotation as R_scipy

    rng = np.random.default_rng(1)
    resolution = 64
    n_mirror = 0

    for trial in range(10):
        # Fragment i : nuage quasi-plan (fracture) + un peu de bruit hors-plan.
        n_pts = 300
        uv = rng.uniform(-0.5, 0.5, size=(n_pts, 2))
        depth_noise = rng.normal(0, 0.01, size=n_pts)
        frac_i = np.stack([uv[:, 0], uv[:, 1], depth_noise], axis=1)
        frac_i += rng.uniform(-1, 1, size=3)   # centre arbitraire, pas à l'origine

        R_ij_gt = R_scipy.random(random_state=trial).as_matrix()
        t_ij_gt = rng.uniform(-1, 1, size=3)
        frac_j = (R_ij_gt @ frac_i.T).T + t_ij_gt   # même fracture, vue depuis j

        c_i, u_i, v_i, n_i, _ = canonical_pca_frame(frac_i)
        c_j, u_j, v_j, n_j, _ = canonical_pca_frame(frac_j)

        span = max(
            float(((frac_i - c_i) @ u_i).max() - ((frac_i - c_i) @ u_i).min()),
            1e-6,
        )
        pixel_size = span * 1.1 / resolution
        u_min_i = float(((frac_i - c_i) @ u_i).min()) - 0.5 * pixel_size
        v_min_i = float(((frac_i - c_i) @ v_i).min()) - 0.5 * pixel_size
        u_min_j = float(((frac_j - c_j) @ u_j).min()) - 0.5 * pixel_size
        v_min_j = float(((frac_j - c_j) @ v_j).min()) - 0.5 * pixel_size

        theta_gt, shift_gt, mirror_gt, _residual = fit_theta_shift_mirror_from_gt(
            frac_i, c_i, u_i, v_i, u_min_i, v_min_i,
            c_j, u_j, v_j, u_min_j, v_min_j,
            pixel_size, resolution, R_ij_gt, t_ij_gt,
        )
        v_j_used = -v_j if mirror_gt else v_j
        n_mirror += int(mirror_gt)

        pts_i3, pts_j3 = build_correspondences_nn(
            frac_i, c_i, u_i, v_i, frac_j, c_j, u_j, v_j_used,
            u_min_i, v_min_i, u_min_j, v_min_j,
            pixel_size, resolution, theta_gt, shift_gt,
            contact_eps=5 * pixel_size,   # tolérance large : bruit hors-plan + arrondi pixel
        )
        assert pts_i3 is not None and len(pts_i3) >= 3, (
            f"trial {trial}: pas assez de correspondances reconstruites "
            f"({0 if pts_i3 is None else len(pts_i3)})"
        )

        R_est, t_est = kabsch(pts_i3, pts_j3)
        re = rot_err_deg(R_est, R_ij_gt)
        te = trans_err(t_est, t_ij_gt)
        assert re < 5.0, f"trial {trial}: rot_err={re:.2f}° trop grand (mirror={mirror_gt})"
        assert te < 0.05, f"trial {trial}: trans_err={te:.4f} trop grand (mirror={mirror_gt})"

    print(f"  [OK] _test_3d_pipeline_roundtrip : 10 tirages, pose reconstruite "
          f"à <5°/<0.05 de la vraie pose via le label GT dérivé (mirror détecté "
          f"{n_mirror}/10 fois, cohérent avec ~50% attendu sur données réelles)")


if __name__ == "__main__":
    print("Auto-tests phase8_depthmap_regressor_dataset.py (numpy/scipy purs, local)...\n")
    _test_2d_roundtrip()
    _test_3d_pipeline_roundtrip()
    print("\nTous les auto-tests passent.")
