# Plan — Module de réassemblage léger à partir du prior CNN (Step 15)

Décidé le 2026-06-26. Suite directe de l'ablation CNN (`AVANCEES_POST_PRESENTATION.md`),
checkpoint final : `output/cnn_step15_final_model/last.ckpt`.

## Contexte et constat de départ

GARF (`assembly/models/denoiser/`) ne fait **pas** de matching pair-à-pair explicite : il
régresse une pose SE(3) globale par fragment (flow matching), les fragments interagissant
via de l'attention globale (`denoiser_transformer.py`), pas via une étape de correspondance
de points. Donc utiliser le filtre CNN pour ne garder que les points de fracture, en amont
d'un module de matching pair-à-pair, est un changement de paradigme (vers la registration
classique par correspondances), pas un simple remplacement de backbone.

**Question centrale :** est-ce que les points prédits comme fracture par le CNN conservent
assez de vrais points de contact pour permettre un matching pairwise fiable et moins coûteux
que de tout faire tourner sur tous les points ?

Ce qu'on NE fait PAS pour l'instant (réserve, seulement si les phases ci-dessous valident
l'hypothèse) : pas de "GARF-lite" global, pas de flow matching, pas de Sinkhorn, pas de
cycle consistency, pas de réseau de matching avant validation des phases 0/1.

## Phase 0 — Vérification réelle du dataloader

Avant tout code de matching, confirmer empiriquement (pas juste "le champ existe") :
- `graph` : adjacence fragment-fragment (qui touche qui)
- `quaternions` / `translations` : poses stockées par fragment
- `pointclouds` (input désassemblé) vs `pointclouds_gt` (assemblé)
- `fracture_surface_gt` : labels GT de fracture par point
- noms de fragments / objets

**Convention de pose dérivée du code** (`assembly/data/transform.py`, `uniform.py`,
`weighted.py`) :
- `recenter_pc` : `pointcloud_centré = pointclouds_gt[part] - centroid`, retourne
  `(pointcloud_centré, centroid)`. `translations` stocké = `centroid`.
- `rotate_pc` : applique une rotation aléatoire `rot_mat` à `pointcloud_centré` →
  `pointclouds` (l'input désassemblé). Le quaternion stocké est la rotation **inverse**
  (`rot_mat.T`), c'est-à-dire la rotation que le modèle doit prédire pour "défaire" la
  rotation et retrouver l'orientation assemblée.
- Donc, **formule de reconstruction** : `pointclouds_gt[part] ≈ R(quaternion) @ pointclouds[part] * scale + translation`
  (le facteur `scale` n'existe que dans `BreakingBadWeighted`, qui renormalise l'input
  après rotation ; absent dans `BreakingBadUniform`).
- Pose relative entre deux fragments i,j adjacents (dans le repère assemblé commun) :
  `R_ij = R_j^{-1} @ R_i`, `t_ij = R_j^{-1} @ (t_i - t_j)` (mappe l'input du fragment i vers
  le repère de l'input du fragment j).

Script de vérification : `scripts/phase0_check_pose_convention.py` — charge 2-3 objets
réels, reconstruit l'objet assemblé à partir des poses stockées, mesure le résidu
(distance moyenne point-à-point après reconstruction, puisque l'ordre des points est
préservé par fragment) et vérifie que les points de fracture de deux fragments adjacents
(`graph[i,j]=True`) sont bien proches après reconstruction (test géométrique indépendant
de l'ordre des points, via nearest-neighbor).

**Critère de validation :** résidu quasi nul / points de fracture adjacents proches →
convention confirmée. Sinon, inverser rotation/translation et re-tester.

## Phase 1 — Recall@K du CNN Step 15

But : mesurer si le filtre CNN garde les vrais points de fracture (checkpoint
`output/cnn_step15_final_model/last.ckpt`).

À tester :
- Filtrage : top-256 / top-512 / top-1024, threshold 0.2/0.3/0.5, top-percent 5/10/20/30%
- Splits : everyday (val) et artifact (zero-shot)
- Découpage par complexité : 2-5 fragments / 6-10 fragments / 11+ fragments
  (le recall peut s'effondrer sur les objets complexes même s'il est bon sur les objets
  simples — or c'est justement là que le prior serait le plus utile)

Tableau attendu : Split × Fragments × Filtrage → Recall fracture, Precision fracture,
Points gardés, Reduction ratio.

Décision : Recall@512 ou @1024 > 90% → bon, passer à la Phase 2. Entre 80-90% →
exploitable mais garder plus de points. < 80% → corriger le filtre avant de continuer.

## Phase 2 — Baseline géométrique (seulement si Phase 0 + Phase 1 passent)

Pipeline minimal, sans réseau appris :
CNN Step 15 → top-K points fracture → features géométriques simples → correspondances
candidates → RANSAC → Kabsch/SVD → score de paire.

Comparaison à 3 conditions obligatoire (pour isoler la responsabilité d'un échec) :
- masque fracture GT → matching
- masque fracture prédit par le CNN → matching
- points aléatoires / tous les points → matching

Lecture des résultats :
- GT marche, prédit échoue → le problème vient du CNN/filtrage
- GT échoue aussi → le matching géométrique naïf est insuffisant
- prédit marche correctement → l'hypothèse du prior CNN est validée

## Phase 3 — Matcher appris (en réserve, seulement si Phase 2 montre un vrai signal)

Architecture gardée en réserve :
- Input : top-K points de fracture des fragments i et j
- Encodeur léger partagé : PointNet / EdgeConv
- Interaction de paire : corrélation de descripteurs, top-k correspondances
- Pose : weighted Kabsch / RANSAC
- Sorties : score de paire, correspondances, pose relative

Supervision (une fois Phase 0 confirmée) : `graph[i,j]` (label de paire), pose relative
GT dérivée de `quaternions`/`translations` (formule ci-dessus), `fracture_surface_gt`.

Échantillonnage à l'entraînement : toutes les paires positives + 2-3x négatifs
aléatoires (hard negatives seulement plus tard).

## Métriques d'évaluation déjà disponibles (ne pas réécrire)

Dans `assembly/models/denoiser/modules/evaluation/evaluator.py` :
- `rot_metrics` — RMSE/MAE rotation en degrés
- `trans_metrics` — RMSE/MAE translation (L2)
- `calc_part_acc` / `calc_part_acc_weighted` — accuracy par fragment (Chamfer < 0.01)
- `calc_shape_cd` / `calc_shape_cd_weighted` — Chamfer distance bidirectionnelle de l'objet
  assemblé complet

## Granularité

Module additionnel, pas un nouveau pipeline complet : on charge le checkpoint Step 15
figé (inférence seule, pas de fine-tuning conjoint pour l'instant) en pré-traitement, on
filtre les points, puis on nourrit le module de matching. L'ablation CNN reste intacte et
indépendante.

## Prochaine action concrète

Lancer `scripts/phase0_check_pose_convention.py` sur le serveur (données HDF5 uniquement
disponibles là-bas) pour confirmer empiriquement la convention de pose dérivée ci-dessus,
avant d'écrire le moindre code de matching.
