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
- Donc, **formule de reconstruction** : `pointclouds_gt[part] ≈ R(quaternion) @ (pointclouds[part] * scale[part]) + translation`.
  Le facteur `scale` (champ `sample["scale"]`) existe dans **les deux** classes (`BreakingBadUniform`
  ET `BreakingBadWeighted`) : `transform()` divise les points par-fragment par
  `max(abs(points après rotation))` en toute fin de pipeline, donc il faut le réappliquer
  avant d'inverser la rotation, sinon le résidu de reconstruction reste de l'ordre de la
  taille de l'objet (~0.5-0.8) au lieu d'être quasi nul (erreur trouvée lors du premier
  run de `scripts/phase0_check_pose_convention.py` sur le serveur, corrigée).
- Pose relative entre deux fragments i,j adjacents : `R_ij = R_j^{-1} @ R_i`,
  `t_ij = R_j^{-1} @ (t_i - t_j)` (mappe l'input du fragment i vers le repère de l'input
  du fragment j). **Domaine de validité : cette formule s'applique aux coordonnées input
  après réapplication du facteur `scale`**, c'est-à-dire `q_i = pointclouds[i] * scale[i]`,
  `q_j = pointclouds[j] * scale[j]`, et alors `q_j ≈ R_ij @ q_i + t_ij`. Sans réappliquer
  `scale`, la transformation entre fragments normalisés n'est pas strictement rigide si
  leurs facteurs d'échelle diffèrent (chaque fragment a son propre `scale`, indépendant).

Script de vérification : `scripts/phase0_check_pose_convention.py` — charge 2-3 objets
réels, reconstruit l'objet assemblé à partir des poses stockées, mesure le résidu
(distance moyenne point-à-point après reconstruction, puisque l'ordre des points est
préservé par fragment) et vérifie que les points de fracture de deux fragments adjacents
(`graph[i,j]=True`) sont bien proches après reconstruction (test géométrique indépendant
de l'ordre des points, via nearest-neighbor).

**Critère de validation :** résidu quasi nul / points de fracture adjacents proches →
convention confirmée. Sinon, inverser rotation/translation et re-tester.

**Statut : CONFIRMÉ (2026-06-26)**, sur `everyday/val` et `artifact/val` (le split
`test` n'existe pas pour `artifact`) — résidu de reconstruction ~1e-8 (bruit numérique)
sur tous les fragments testés, une fois le facteur `scale` réappliqué (voir script).

## Phase 1 — Recall@K du CNN Step 15

But : mesurer si le filtre CNN garde les vrais points de fracture (checkpoint
`output/cnn_step15_final_model/last.ckpt`).

À tester :
- Filtrage : top-256 / top-512 / top-1024, threshold 0.2/0.3/0.5, top-percent 5/10/20/30%
- Splits : everyday (val) et artifact (zero-shot)
- Découpage par complexité : 2-5 fragments / 6-10 fragments / 11+ fragments
  (le recall peut s'effondrer sur les objets complexes même s'il est bon sur les objets
  simples — or c'est justement là que le prior serait le plus utile)

**Deux niveaux de recall** (ajout après relecture des résultats Phase 0) :
- `fragment_fracture_recall@K` : recall sur tous les points fracture GT d'un fragment.
  `fracture_surface_gt` est un label **fragment-level**, pas pair-specific — un fragment
  touchant plusieurs voisins a des points fracture vers chacun d'eux, donc ce recall seul
  ne dit pas si le filtre garde les BONS points pour une paire (i,j) donnée.
- `edge_contact_recall@K` : pour chaque arête `graph[i,j]=True`, reconstruit les fragments
  en pose GT (Phase 0 confirmée), définit `contact_i_to_j` = points fracture de i dont le
  NN dans j est `< eps`, et mesure si le filtre garde ces points spécifiquement. C'est la
  métrique pertinente pour le matching pair-à-pair (on veut les bons points de contact
  pour le bon voisin, pas juste "des points cassés"). Tester deux eps : 0.02 (strict) et
  0.05 (tolérant, déjà utilisé en Phase 0) — si les conclusions diffèrent fortement entre
  les deux, la définition du contact est sensible, à noter pour l'évaluation pairwise.

Tableau attendu : Split × Fragments × Filtrage → fragment_fracture_recall,
edge_contact_recall@eps∈{0.02,0.05}, Precision, `kept_ratio` (proportion de points
gardés par le filtre — *pas* la réduction ; réduction effective = `1 - kept_ratio`).
Implémenté dans `scripts/phase1_recall_at_k.py`.

Décision (sur fragment_fracture_recall@512/1024 ET edge_contact_recall@512/1024) :
- Les deux bons (>90%) → go Phase 2 (baseline géométrique).
- Fracture recall bon mais edge contact recall mauvais → le CNN détecte la fracture mais
  pas forcément les zones utiles pour chaque paire spécifique ; matching plus difficile,
  garder plus de points ou faire du filtrage pair-specific.
- Fracture recall mauvais → stop, retravailler le filtre avant le matching.

**Résultat (2026-06-26, `scripts/phase1_recall_at_k.py`, everyday/val + artifact/val) :**
top-K et top-percent (budget fixe) **échouent** le critère — recall 13-73% selon la
complexité, parce qu'un budget absolu par fragment ne suit pas la quantité réelle de
surface de fracture (qui croît avec le nombre de fragments/voisins). En revanche, les
filtres **par seuil de probabilité** (0.2/0.3/0.5) réussissent largement : recall
fragment ET edge dans 84-98%, stable même sur 11+ fragments, et quasi insensible à eps
(0.02 vs 0.05) — la définition du contact n'est pas un point fragile. `kept_ratio` plus
modeste (~30-65% gardés, pas 5-20%) mais le signal du CNN (ranking de probabilité) est
validé : le problème n'était pas le modèle, mais la stratégie de filtrage à budget fixe.

**Critère de décision mis à jour** : on ne cherche plus un budget top-K agressif, mais un
masque fracture **adaptatif par seuil**, à haut recall, pour construire la baseline
géométrique. Configuration retenue pour la Phase 2 (le seul *oracle* est le masque GT —
les seuils CNN restent des prédictions, pas des références) :
- threshold 0.3 — configuration principale, compromis recall/precision
- threshold 0.5 — version plus compacte : moins de points gardés, recall plus faible,
  précision plus élevée
- threshold 0.2 — version high-recall **prédite** : recall maximal, précision plus faible,
  plus de points gardés (PAS un oracle)
- masque fracture GT — oracle / référence haute (isole la responsabilité CNN vs matching,
  cf. plan Phase 2 ci-dessous)

## Phase 2 — Baseline géométrique (seulement si Phase 0 + Phase 1 passent)

Pipeline minimal, sans réseau appris :
CNN Step 15 → masque fracture par seuil de probabilité (threshold 0.2/0.3/0.5, cf. Phase 1
— PAS top-K, rejeté en Phase 1) → features géométriques simples → correspondances
candidates → RANSAC → Kabsch/SVD → score de paire.

Comparaison à 3 conditions obligatoire (pour isoler la responsabilité d'un échec) :
- masque fracture GT (oracle) → matching
- masque fracture prédit par le CNN (threshold) → matching
- points aléatoires / tous les points → matching

Lecture des résultats :
- GT marche, prédit échoue → le problème vient du CNN/filtrage
- GT échoue aussi → **pas** "le prior CNN ne marche pas" — la conclusion correcte est que
  le matching géométrique naïf (ces descripteurs simples) est insuffisant. Les descripteurs
  choisis sont volontairement faibles pour une baseline minimale ; un échec sur GT teste le
  matching, pas le CNN.
- prédit marche correctement → l'hypothèse du prior CNN est validée

Implémenté dans `scripts/phase2_geometric_baseline.py`. Descripteurs : les 3 scalaires
rotation/translation-invariants de `HybridGeometryFeatures` (consistency, curvature,
roughness — pas les normales brutes, non-invariantes entre fragments non-alignés) +
distance au centroïde du fragment. Correspondances candidates par 1-NN en espace
descripteur, RANSAC (500 itérations, 3 points, seuil inlier 0.05) + Kabsch pondéré sur
les inliers. Stratégies comparées : `gt`, `thresh0.2/0.3/0.5` (issus de la Phase 1),
`random` (même budget que le masque GT), `all`. Travaille sur le repère d'entrée
non-assemblé après réapplication de `scale` (cf. Phase 0 — pas une fuite de label, `scale`
est recalculable directement depuis les points : `scale = max(abs(points))`).

**Trois métriques distinctes (à ne pas confondre)** :
- `correspondence_precision` (diagnostic) : fraction des candidats 1-NN géométriquement
  corrects sous la vraie pose GT, calculée indépendamment de RANSAC. Isole "existe-t-il de
  vraies correspondances parmi les candidats ?" d'un échec RANSAC/Kabsch en aval — si ~0%,
  RANSAC ne peut structurellement pas réussir, quel que soit le nombre d'itérations.
- `ransac_valid_rate` : fraction des paires où RANSAC trouve ≥3 inliers. Dit seulement
  que RANSAC a produit *une* pose, pas qu'elle est bonne (3 inliers peuvent correspondre à
  une pose fausse par coïncidence géométrique).
- `pose_success_rate@(seuil_rot, seuil_trans)` : fraction des paires où la pose estimée a
  rotation_error < seuil_rot ET translation_error < seuil_trans (ex: `pose_success@15deg_0.05`,
  `pose_success@30deg_0.1`). C'est la métrique qui compte réellement pour juger le matching.

**Résultat Phase 2A (2026-06-26, everyday/val, `scripts/phase2_geometric_baseline.py`) :**
`ransac_valid_rate=100%` partout y compris `random` (confirme que ce critère est
trivialement satisfait, cf. ci-dessus — non-informatif) ; `pose_success` quasi nul
(0-1.6%) **y compris sur le masque GT**, erreur de rotation moyenne ~126-128° (proche de
la moyenne attendue entre deux rotations SO(3) indépendantes, ~120°). Diagnostic
`correspondence_precision` : **gt/thresh0.2/0.3/0.5 ≈ 6.4-6.6%, random/all ≈ 2.7%** — soit
~2.4x plus de bonnes correspondances avec le masque fracture qu'avec des points
quelconques (le filtre fracture *aide*), et les masques CNN sont quasiment au niveau de
l'oracle GT (le CNN n'est pas le facteur limitant ici). Mais en absolu, même à l'oracle
GT, 93.4% des correspondances 1-NN restent fausses — bien trop bas pour un RANSAC à
échantillon minimal (3 points) : `P(triplet tout correct) ≈ 0.065³ ≈ 0.03%` par tirage,
~13% sur 500 itérations en théorie, mais en pratique RANSAC peut aussi verrouiller sur un
faux consensus géométriquement cohérent plutôt que sur le bon triplet, ce qui explique le
quasi-zéro observé.

**Conclusion Phase 2A officielle :** le filtrage fracture améliore bien la qualité des
correspondances candidates, et les masques CNN se comportent presque comme le masque GT
— **le CNN n'est pas le goulot d'étranglement**. En revanche, un matching 1-NN basé sur
4 descripteurs scalaires invariants (consistency/curvature/roughness/dist_to_centroid)
est insuffisant pour produire une pose fiable. L'échec vient du module de matching
géométrique naïf, pas du prior CNN ni d'un bug de convention de pose (code revérifié,
RANSAC/Kabsch/formule R_ij-t_ij confirmés corrects). Conforme à la branche prévue du plan
("GT échoue aussi → matching naïf insuffisant").

**Phase 2B — améliorer la génération de correspondances avant de refaire RANSAC/Kabsch**
(ordre à respecter, chaque étape diagnostique avant d'agir) :
1. **Diagnostic top-K** (implémenté dans `phase2_geometric_baseline.py`) : `avail_rate`
   (fraction de points i ayant ≥1 vraie correspondance disponible dans le masque de j —
   isole le confound multi-voisins) et `topk_recall_{5,10,20}` (la bonne correspondance
   apparaît-elle dans le top-K du classement descripteur, même si pas en top-1 ?). Si la
   bonne correspondance n'apparaît même pas en top-20, le descripteur n'a quasi aucun
   signal ; si elle y apparaît souvent, le 1-NN est juste trop strict.

   **Résultat (2026-06-26, everyday/val, 1h05 pour 60 batches/634 arêtes/6 stratégies —
   coût élevé, à garder en tête)** : `avail_rate` ≈ 38% sur `gt`/`thresh*` (vs 19% sur
   `random`/`all`) — donc **62% des points filtrés n'ont structurellement aucune vraie
   correspondance disponible** dans le voisin évalué (confound multi-voisins confirmé,
   pas juste une hypothèse). Mais parmi les 38% qui ont une vraie correspondance dispo,
   `topk_recall_20` ≈ 60% (`top5`≈42%, `top10`≈51%) — le descripteur porte un **vrai
   signal**, nettement mieux que le hasard, juste insuffisant pour un 1-NN strict.
   Cohérent avec `CorrPrec`≈6.7% (≈ `avail_rate` × `top1_recall`, les deux causes se
   combinent). Ajout d'une stratégie `gt_edge` (oracle restreint aux points de contact
   pair-specific, NN<eps en repère assemblé reconstruit — comme Phase 1, *pas* en repère
   local brut, qui aurait été une comparaison sans sens entre deux repères indépendants)
   pour isoler l'effet du confound de celui du descripteur. **Pour contrôler le coût**,
   un flag `--strategies` permet de ne lancer qu'un sous-ensemble (chaque stratégie coûte
   ~le même travail O(Ni×Nj) ; restreindre est le levier principal pour réduire le temps).
2. Mutual nearest neighbor / ratio test (type Lowe) pour réduire le nombre de
   correspondances tout en augmentant `correspondence_precision` (cible indicative :
   15-25%, pas besoin de 80%).
3. Descripteurs plus riches si 1-2 ne suffisent pas : FPFH/SHOT/spin-images,
   eigenvalues multi-échelle, histogrammes de patch local, ou fusionner avec les features
   CNN du Step 15 (le plus prometteur — relie le prior appris directement au matching).
4. RANSAC plus strict après hypothèse de pose : contraintes géométriques additionnelles
   (normales opposées après transformation, absence de pénétration/overlap absurde,
   score restreint aux points fracture uniquement) pour éviter le faux consensus.

**Protocole en deux temps** (ne pas mélanger les deux questions) :
- **A. Registration sur paires positives** (`graph[i,j]=True` uniquement, ce que fait déjà
  `phase2_geometric_baseline.py`) : rotation/translation error, inlier_ratio,
  ransac_valid_rate, pose_success_rate. Répond à "si deux fragments vont vraiment ensemble,
  la baseline retrouve-t-elle leur pose relative ?"
- **B. Discrimination positives/négatives** (à ajouter dans une itération suivante,
  seulement après A) : échantillonner aussi des paires `graph[i,j]=False`, et mesurer si le
  score de paire (ex: inlier_ratio ou résidu inverse) sépare vraies et fausses paires
  (precision/recall du score, ou AUC/AP). Répond à une question différente : "le score de
  paire permet-il de reconnaître qu'une paire est vraie ?" Ne pas mélanger A et B dans la
  même mesure.

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

Phase 0 confirmée. Lancer `scripts/phase1_recall_at_k.py` sur le serveur avec le
checkpoint `output/cnn_step15_final_model/last.ckpt`, sur `everyday/val` puis
`artifact/val`, pour mesurer le Recall@K et décider si on passe à la Phase 2.
