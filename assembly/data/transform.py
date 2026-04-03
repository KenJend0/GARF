"""
assembly/data/transform.py
==========================
Utilitaires de transformation géométrique pour les nuages de points.
Ces fonctions sont utilisées en DATA AUGMENTATION pendant l'entraînement :
  - on applique des rotations et shuffles aléatoires sur les fragments
    pour que le modèle apprenne à être invariant à l'orientation initiale.
"""

import random

import numpy as np
from scipy.spatial.transform import Rotation as R


def recenter_pc(pc):
    """
    Centre un nuage de points autour de l'origine.

    Pourquoi ? Tous les fragments doivent être dans un repère cohérent.
    Si on ne centre pas, le centroïde influence la translation prédite.

    Args:
        pc: tableau numpy de forme [N, 3]  (N points, 3 coordonnées xyz)

    Returns:
        (pc_centered, centroid)
        - pc_centered : nuage recentré [N, 3]
        - centroid    : le vecteur de translation retiré [3]  (utile pour retrouver la pose originale)
    """
    centroid = np.mean(pc, axis=0)       # moyenne sur tous les points → point central [3]
    return pc - centroid[None], centroid  # [None] ajoute une dim pour le broadcast


def rotate_pc(pc, normal=None, numpy_rng=None):
    """
    Applique une rotation aléatoire uniforme sur SO(3) à un nuage de points.

    Pourquoi SO(3) uniforme ? Pour que le modèle soit entraîné sur toutes
    les orientations possibles de façon équilibrée (pas de biais vers l'axe Z par ex.).

    La convention de quaternion utilisée ici est SCALAR-FIRST : [w, x, y, z]
    (scipy retourne [x, y, z, w] par défaut, d'où le re-indexage [[3, 0, 1, 2]])

    Args:
        pc         : nuage de points [N, 3]
        normal     : normales de surface [N, 3] ou None
                     (les normales doivent être rotées avec les points !)
        numpy_rng  : générateur aléatoire numpy pour la reproductibilité

    Returns:
        (rotated_pc, rotated_normal, quat_gt)
        - rotated_pc     : nuage roté [N, 3]
        - rotated_normal : normales rotées [N, 3] (ou None)
        - quat_gt        : quaternion de la rotation INVERSE [4]
                           (inverse car on voudra prédire la rotation pour revenir à l'original)
    """
    # Génère une rotation aléatoire uniforme sur SO(3) → matrice 3x3
    rot_mat = R.random(random_state=numpy_rng).as_matrix()

    # Applique la rotation : pc_roté = (R @ pc^T)^T
    # On transpose car les points sont en ligne [N,3] mais @ attend des colonnes
    rotated_pc = (rot_mat @ pc.T).T

    # La GT est la rotation INVERSE (rot_mat.T = inverse d'une matrice orthogonale)
    # → c'est la rotation que le modèle doit prédire pour "défaire" la rotation appliquée
    quat_gt = R.from_matrix(rot_mat.T).as_quat()

    # scipy donne [x, y, z, w] → on veut [w, x, y, z] (scalar-first)
    quat_gt = quat_gt[[3, 0, 1, 2]]

    if normal is None:
        return rotated_pc, None, quat_gt

    # Les normales sont des vecteurs directionnels : même rotation que les points
    rotated_normal = (rot_mat @ normal.T).T
    return rotated_pc, rotated_normal, quat_gt


def shuffle_pc(pc, normal=None):
    """
    Mélange aléatoirement l'ordre des points d'un nuage de points.

    Pourquoi ? Les réseaux comme PointTransformerV3 traitent les points
    comme un ensemble non-ordonné (set). Le shuffle garantit que le modèle
    ne mémorise pas un ordre particulier dans les données.

    Args:
        pc     : nuage de points [N, 3]
        normal : normales [N, 3] ou None

    Returns:
        (shuffled_pc, shuffled_normal, order)
        - shuffled_pc     : nuage mélangé [N, 3]
        - shuffled_normal : normales dans le même ordre [N, 3] (ou None)
        - order           : indices de permutation [N] (pour retrouver l'ordre original si besoin)
    """
    order = np.arange(pc.shape[0])  # indices 0, 1, 2, ..., N-1
    random.shuffle(order)            # permutation aléatoire in-place
    shuffled_pc = pc[order]          # réordonne les points selon la permutation

    if normal is None:
        return shuffled_pc, None, order

    shuffled_normal = normal[order]  # même permutation pour les normales
    return shuffled_pc, shuffled_normal, order


def rotate_whole_part(pc, normal=None):
    """
    Applique la MÊME rotation aléatoire à TOUS les fragments d'un objet.

    Pourquoi ? En entraînement multi-pièces, on veut parfois simuler une
    orientation globale de l'objet sans modifier les poses relatives entre fragments.
    Cette fonction est différente de rotate_pc qui tourne chaque fragment séparément.

    Args:
        pc     : nuage de tous les fragments [P, N, 3]
                 P = nombre de pièces, N = points par pièce, 3 = xyz
        normal : normales [P, N, 3] ou None

    Returns:
        (rotated_pc, rotated_normal, quat_gt)
        - rotated_pc     : tous les fragments rotés avec la MÊME rotation [P, N, 3]
        - rotated_normal : normales rotées [P, N, 3] (ou None)
        - quat_gt        : quaternion de la rotation inverse appliquée [4] (scalar-first)
    """
    P, N, _ = pc.shape

    # Aplatit en (P*N, 3) pour appliquer une seule opération matricielle
    pc = pc.reshape(-1, 3)

    # Une seule rotation aléatoire pour toutes les pièces
    rot_mat = R.random().as_matrix()
    pc = (rot_mat @ pc.T).T

    # Quaternion inverse (scalar-first) → ce que le modèle doit prédire pour "défaire"
    quat_gt = R.from_matrix(rot_mat.T).as_quat()
    quat_gt = quat_gt[[3, 0, 1, 2]]  # [x,y,z,w] → [w,x,y,z]

    if normal is None:
        return pc.reshape(P, N, 3), None, quat_gt

    # Même rotation pour les normales, puis reshape vers la forme d'origine
    rotated_normal = (rot_mat @ normal.reshape(-1, 3).T).T.reshape(P, N, 3)
    return pc.reshape(P, N, 3), rotated_normal, quat_gt
