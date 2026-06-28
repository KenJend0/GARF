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

   **Résultat `gt_edge` (2026-06-27)** : `CorrPrec` 6.56%→28.82% (×4.4), `AvailRate`
   38%→100% comme attendu (par construction). Mais `pose_success`/`RotErr` ne bougent
   quasiment pas (`Pose@30°/0.1` 1.26%→4.26%, `RotErr` reste ~127-128°) malgré un
   `InlierRatio` qui double (20.8%→47.4%). **Test oracle décisif, déductible directement
   des colonnes déjà calculées** (pas besoin de relancer) : `CorrPrec` est exactement
   l'inlier ratio sous la VRAIE pose GT (même calcul, même seuil) ; `InlierRatio` est
   l'inlier ratio sous la pose que RANSAC a choisie. Sur `gt_edge` : `InlierRatio`
   (47.41%) **>** `CorrPrec` (28.82%) → le critère de score de RANSAC préfère
   objectivement une pose fausse à la vraie. Donc le sampling seul ne suffira pas, il
   faut aussi revoir le critère de score. Hypothèse retenue : les surfaces de fracture
   sont localement quasi-planes — une pose fausse qui glisse/tourne dans le plan de
   contact peut accumuler autant ou plus d'inliers apparents qu'une pose correcte (faux
   consensus géométriquement cohérent), surtout avec un triplet minimal proche de la
   dégénérescence coplanaire.

   **Mitigation testée** : `ransac_pose()` accepte maintenant `sample_size` (>3 → fit
   Kabsch sur-déterminé par hypothèse, moins sensible à un triplet quasi-coplanaire) et
   `min_dispersion` (rejette les échantillons trop peu dispersés spatialement). CLI :
   `--ransac_sample_size`, `--ransac_min_dispersion`. Le tableau de sortie affiche
   désormais explicitement la légende `CorrPrec` vs `InlierRatio` pour éviter d'avoir à
   recalculer la comparaison à la main. À tester : 3 (baseline) → 4 dispersé → 6 dispersé
   (pas direct 8-12 : `CorrPrec³ ≈ 0.024` pour un triplet, donc un échantillon de 8 points
   tous corrects devient très improbable — il faut un fit robuste qui tolère une partie
   de faux dans l'échantillon, pas un échantillon plus grand mais toujours minimal/exact).

   **Résultat sample_size/dispersion (2026-06-27, `gt_edge`)** : 3→4→6 avec
   `min_dispersion=0.1` réduit légèrement l'écart `InlierRatio - CorrPrec` (18.6→14.4→14.5
   points) et améliore un peu `Pose@30°/0.1` (4.26%→4.57%→6.78%), mais le rendement
   décroît vite (4→6 quasi sans effet) — **le sampling n'est pas le facteur principal**,
   confirmé. Le scoring (comptage brut d'inliers à un seuil généreux) reste le verrou.

   **Hypothèse de symétrie de révolution** : beaucoup d'objets "everyday" (bouteilles,
   bols, vases, jarres...) ont une fracture en anneau autour d'un axe de révolution — la
   rotation autour de cet axe peut être intrinsèquement ambiguë géométriquement (la
   surface se ressemble tout le long de l'anneau), indépendamment de la qualité du
   matching. Ajout dans `phase2_geometric_baseline.py` d'une ré-agrégation des résultats
   déjà calculés par `(stratégie, famille d'objet)` et `(stratégie, groupe symmetric-like
   vs irregular-like)` — heuristique grossière par mot-clé sur le nom d'objet (`Bottle`,
   `Bowl`, `Vase`, `Jar`, `Cup`, `Mug`, `Plate`, `Pot`), pas un vrai détecteur de symétrie.
   Lecture prévue : si `RotErr`/`Pose@30` sont nettement pires sur `symmetric-like`,
   l'ambiguïté de rotation est en partie structurelle (le scoring devra intégrer une
   contrainte plus globale — normales, couverture spatiale, résidu pondéré — pas
   seulement un comptage d'inliers) ; si les deux groupes sont comparables, le problème
   est le scoring/descripteur en général, pas la symétrie des objets.

   **Artefact d'échantillonnage trouvé (2026-06-27)** : le premier run avec
   `--max_batches 60` a donné **634/634 arêtes = objets `BeerBottle`/`Bottle`**, aucun
   objet irrégulier — donc aucune comparaison possible. Cause : `val_dataloader()`
   (`module.py`) ne shuffle pas (`shuffle` non spécifié dans le `DataLoader`, donc
   `False` par défaut), et la liste d'objets HDF5 semble triée alphabétiquement — les 60
   premiers batches ne couvrent que les noms commençant par "B". **Corrigé** dans
   `phase2_geometric_baseline.py` : construction manuelle du `DataLoader` avec
   `shuffle=True` (seed fixe, `--seed`) au lieu d'utiliser `datamodule.val_dataloader()`/
   `test_dataloader()` directement — nécessaire pour que tout run plafonné par
   `--max_batches` voie un mix représentatif de familles d'objets.

   **Résultat symétrie (2026-06-27, après correction du shuffle, n=717 arêtes, mix
   représentatif) : hypothèse NON confirmée (Cas B).** `irregular-like` (n=201) :
   RotErr=131.3°, Pose@30=5.47%, Gap=13.23 ; `symmetric-like` (n=516) : RotErr=126.3°,
   Pose@30=4.84%, Gap=15.44 — quasi équivalents, et si différence il y a, c'est
   `irregular-like` qui est légèrement *pire* (l'inverse de ce que prédirait
   l'hypothèse). Confirmé par famille : `ToyFigure` (n=75, clairement pas symétrique)
   ne fait que 9.33% de `Pose@30` ; `Statue` (n=21) a `CorrPrec`=61.72% et `Gap` négatif
   (RANSAC ne préfère même pas une pose fausse) mais `Pose@30=0%` quand même. **Donc
   l'échec n'est pas structurel à la géométrie des objets — c'est le matching
   (descripteur + scoring RANSAC) qui est insuffisant en général**, indépendamment de la
   symétrie. Écarte l'explication "ambiguïté de révolution" et recentre sur le scoring/
   descripteur (étapes 2-4 du plan Phase 2B), pas sur une contrainte géométrique
   spécifique aux objets symétriques.
2. Mutual nearest neighbor / ratio test (type Lowe) pour réduire le nombre de
   correspondances tout en augmentant `correspondence_precision` (cible indicative :
   15-25%, pas besoin de 80%).
3. Descripteurs plus riches si 1-2 ne suffisent pas : FPFH/SHOT/spin-images,
   eigenvalues multi-échelle, histogrammes de patch local, ou fusionner avec les features
   CNN du Step 15 (le plus prometteur — relie le prior appris directement au matching).
4. RANSAC plus strict après hypothèse de pose : contraintes géométriques additionnelles
   (normales opposées après transformation, absence de pénétration/overlap absurde,
   score restreint aux points fracture uniquement) pour éviter le faux consensus.

**Sanity check oracle (A, ajouté avant le ratio test, 2026-06-27)** : Kabsch simple (sans
RANSAC) sur des correspondances NON pilotées par le descripteur (plus proche voisin sous
la VRAIE pose GT, via `resid_mat.argmin`). Implémenté dans `phase2_geometric_baseline.py`
(tableau `ORACLE KABSCH SANITY CHECK`, métriques `oracle_rot_err_deg`/`oracle_trans_err`).
Vérifie que la convention de pose/scale (confirmée en Phase 0) et l'implémentation Kabsch
elle-même peuvent récupérer `R_ij_gt`/`t_ij_gt` quand on leur donne des correspondances
parfaites — motivé par le cas `Statue` (CorrPrec=61.72%, Gap négatif, mais Pose@30=0%) qui
méritait une vérification indépendante du matching avant d'aller plus loin.

**Mutual-NN / ratio test (B, étape 2 du plan)** : ajout de `build_correspondences()` et du
flag `--corr_mode` (`1nn` baseline, `mutual`, `ratio<R>` type Lowe, `mutual_ratio<R>`) pour
filtrer les correspondances 1-NN avant RANSAC — moins de candidats mais plus précis,
objectif indicatif `CorrPrec` 15-25%+ (pas besoin de 80%). À tester dans cet ordre (sur
`gt_edge`, le masque le plus propre) : `1nn` (référence déjà connue) → `mutual` →
`ratio0.8` → `ratio0.7` → `ratio0.6` → `mutual_ratio0.8`. Lecture attendue : si `CorrPrec`
monte fortement mais `Pose@30` reste plat, le problème est le score RANSAC (étape 4) plus
que les correspondances ; si `CorrPrec` ne bouge presque pas, les descripteurs sont trop
faibles pour qu'un filtrage de correspondances aide (passer direct à l'étape 3).

**Résultat sanity check + mutual-NN (2026-06-27)** : `OracleRotErr`≈11.8°/`OracleTransErr`
≈0.035 (Kabsch simple sur NN sous la vraie pose GT, pas piloté par le descripteur) — bien
loin du ~125° quasi-aléatoire observé partout ailleurs. **Confirme que le pipeline de
pose/scale/Kabsch fonctionne** : le verrou est bien dans la recherche de correspondances +
le scoring RANSAC, pas dans la convention de pose (résidu non-nul normal, dû à la
tolérance eps=0.05 de l'oracle, pas un bug). `mutual` vs `1nn` sur la même population
(717 arêtes) : amélioration marginale (`CorrPrec` 34.09%→35.19%, `Pose@30` 5.02%→5.16%,
`RotErr` 127.58°→124.20°) — pas assez pour justifier le sweep complet `ratio0.8/0.7/0.6`.
**Décision : saut direct vers l'étape 4 (scoring RANSAC), sweep `ratio0.7` repoussé à plus
tard comme mini-ablation complémentaire si besoin.**

**Étape 4 (scoring RANSAC) — implémenté dans `phase2_geometric_baseline.py` :**
- `--inlier_thresh` (remplace l'ancienne constante fixe `RANSAC_THRESH=0.05`) — appliqué
  partout de façon cohérente (consensus RANSAC, `CorrPrec`/`avail_rate`/oracle) pour que
  `Gap = InlierRatio - CorrPrec` reste interprétable au même seuil. À tester : 0.05
  (référence) → 0.03 → 0.02 → 0.01.
- `--score_mode` : `count` (comptage brut, défaut/historique — favorise un faux consensus
  large mais peu précis) vs `count_minus_mean_residual` / `count_minus_median_residual`
  (pénalise les inliers "lâches" : `score = n_in - λ * résidu_moyen_ou_médian_des_inliers`).
  `--score_lambda` (défaut 50.0) règle le poids de la pénalité.
- Ordre de test recommandé : `gt_edge`, `sample_size=6`/`min_dispersion=0.1` comme base,
  sweep `--inlier_thresh` d'abord (seul, `score_mode=count`), puis `--score_mode` une fois
  un bon seuil trouvé. Bon signe attendu : `Gap` baisse fortement, `Pose@30` 5%→10-20%,
  `RotErr` nettement sous 100° (même imparfait, ça validerait que le verrou était le score).

**Résultat sweep `--inlier_thresh` (2026-06-27, `gt_edge`, `score_mode=count`) :**
`Gap` se referme presque complètement avec le seuil (14.52→6.46→3.24→0.73 pour
0.05/0.03/0.02/0.01), `OracleRotErr` baisse aussi (11.8°→8.2°→4.7°→3.1°, prévisible :
seuil plus strict = correspondances oracle plus proches). **Mais `Pose@30`/`RotErr`
restent quasi plats (4.5-6.6%, 122-128°), sans tendance monotone** — `Pose@30` baisse
même au seuil le plus strict (0.01). Explication : serrer le seuil élimine le biais de
score, mais élimine aussi une grande partie des vraies correspondances (`CorrPrec`
34%→8%) ; à seuil=0.01 avec `sample_size=6`, `P(échantillon tout correct) ≈ 0.083⁶ ≈ 2e-6`
— RANSAC n'a quasi plus rien de propre à trouver, même sans biais. **Le seuil global seul
est un compromis, pas un levier** : les deux effets (moins de biais, moins de matière) se
neutralisent à peu près.

**Résultat `count_minus_mean_residual` à `inlier_thresh=0.03`, `score_lambda=50` :
AUCUN effet mesurable** (résultats identiques à `score_mode=count` au même seuil, écarts
dans le bruit du tirage RANSAC). Diagnostic : `score_lambda=50` est beaucoup trop faible
— le résidu vit dans `[0, 0.03)`, pénalité max `50×0.03=1.5`, alors que l'écart de
comptage d'inliers entre deux hypothèses concurrentes peut être de plusieurs dizaines de
points sur un pool de centaines de candidats. Le terme résiduel est noyé, pas un échec de
l'approche. **Plutôt que deviner `λ` par essais successifs, ajout de `count_over_residual`**
(`score = n_in / (résidu_moyen + ε)`, dans `_hypothesis_score()`) — score en ratio sans
paramètre à caler, qui équilibre naturellement comptage et précision quelle que soit
l'échelle absolue. `count_minus_*_residual` conservés mais nécessiteraient `λ` de l'ordre
de plusieurs centaines/milliers pour avoir un effet, à éviter sauf besoin spécifique.

**Affinement (2026-06-27)** : un ratio non borné (`count_over_residual`) peut être
trompé par un petit échantillon accidentellement très précis (ex: 8 inliers à résidu
quasi nul battant 80 inliers à résidu modéré). Ajout de :
- `min_inliers_for_score` (défaut `max(6, sample_size)`) — rejette **toute** hypothèse
  (y compris en mode `count`) sous ce plancher, avant même de calculer le score. Garde-fou
  appliqué uniformément, pas seulement aux modes ratio/qualité.
- `count_over_mean_residual` : `score = n_in / (résidu_moyen/τ + ε)` — version normalisée
  par `τ` (défaut = `--inlier_thresh`) du ratio, moins sensible à l'échelle absolue du
  résidu que `count_over_residual`.
- `count_times_quality` : `score = n_in × clip(1 - résidu_moyen/τ, 0, 1)` — formulation
  multiplicative simple, sans risque d'explosion numérique (`quality` borné dans [0,1]).
- Nouvelle mesure clé `score_gap = score(pose RANSAC) - score(pose GT)`, calculée avec la
  même fonction de score que celle utilisée pendant la recherche RANSAC (pas juste
  `InlierRatio - CorrPrec`, qui ne reflète que le mode `count`). Lecture : `score_gap > 0`
  → le critère de score actuel préfère encore une pose fausse, peu importe le sampling ;
  `score_gap ≤ 0` mais `Pose@30` toujours bas → le score est correct mais l'exploration
  RANSAC ne trouve pas l'hypothèse que son propre critère préférerait (problème
  d'échantillonnage/itérations, pas de scoring).

Prochain test recommandé : comparer `count` / `count_over_mean_residual` /
`count_times_quality` à `inlier_thresh=0.03`, `tau=0.03` (par défaut), `sample_size=6`,
`min_dispersion=0.1`, sur `gt_edge`. Si aucun des deux nouveaux scores ne fait dépasser
`Pose@30`≈10%, passer à une contrainte plus informative (normales opposées, score
Chamfer symétrique) plutôt que continuer à itérer sur des fonctions de score purement
distance.

**Résultat (2026-06-27) : distance-only épuisé, confirmé sur 3 formules de score.**
`Pose@30` reste ~5.7-6.3% et `RotErr` ~122-127° pour `count`, `count_times_quality` ET
`count_over_mean_residual` — quasi identiques. Plus important : **`score_gap` est positif
dans les trois cas** (`count`: +51.8, `count_times_quality`: +23.4,
`count_over_mean_residual`: +92.8) — la pose choisie par RANSAC score systématiquement
plus haut que la vraie pose GT, **sous le même critère**, peu importe la formule. Donc le
problème n'est pas la formule de score : la distance pure entre points ne suffit pas à
discriminer une pose fausse d'une pose vraie sur ces surfaces de fracture (une pose fausse
a réellement un ensemble de points plus proches, en nombre et en qualité moyenne). Chaîne
de preuve complète : (1) Oracle Kabsch confirme le pipeline de pose sain (RotErr~8.2° à
tau=0.03) ; (2) `gt_edge` donne un signal de correspondance réel (`CorrPrec`~22.8%, pas
une absence totale) ; (3) aucune reformulation distance-only ne corrige le biais de score.
**Décision : passer à une information indépendante de la distance point-point — les
normales (Phase 2C).**

## Phase 2C — Contrainte sur les normales (en cours)

Étape 1 (diagnostic, pas de filtre dur) — implémenté dans `phase2_geometric_baseline.py`,
nouveau tableau `NORMAL ORIENTATION DIAGNOSTIC` : sur les correspondances oracle-correctes
de `gt_edge` (correctes sous la vraie pose GT, pas pilotées par le descripteur), calcule
`dot(R_ij_gt @ n_i, n_j)` — une vraie zone de contact devrait donner `dot ≈ -1`. Mesure
`MeanDot`/`MedianDot`/`%dot<-0.3`/`%dot<-0.5`/`%dot<-0.7`. Lecture : médiane nettement
négative + beaucoup de `dot<-0.5` → normales exploitables comme opposées ; `|dot|` proche
de 1 mais signe instable → utiliser `|dot|` plutôt qu'un test d'opposition strict ;
dispersion sans structure → normales trop bruitées, ne pas filtrer dur dessus.

**Résultat diagnostic (2026-06-27)** : `MeanDot=-0.665`, `MedianDot=-0.825`,
`%dot<-0.5=78.31%`, `%dot<-0.7=73.42%` (n=566) — **normales clairement exploitables**,
signal cohérent (médiane nettement négative, pas de dispersion sans structure). Go étapes
2-3.

Étapes 2-3 — implémentées dans `phase2_geometric_baseline.py` :
- `--normal_tau` : filtre dur optionnel, en plus du seuil de distance (pas à sa place) —
  `inlier = (distance < inlier_thresh) ET (dot(R @ n_i, n_j) < normal_tau)`. `None` par
  défaut (désactivé). Tester `-0.3` (permissif) → `-0.5` → `-0.7`.
- Deux nouveaux `--score_mode` : `count_times_normal_quality` (`score = n_in ×
  mean(clip(-dot, 0, 1))` sur les inliers) et `count_times_quality_and_normal`
  (`score = n_in × dist_quality × normal_quality`) — fonctionnent même sans
  `--normal_tau` (normales utilisées en scoring doux, pas en filtre).
- `score_gap` (déjà existant) recalculé de façon cohérente avec le filtre/score normal
  choisi (la pose GT et la pose RANSAC sont toutes les deux réévaluées avec le même
  critère normal-aware si applicable).
- Nouvelle colonne `NormalDot` dans le tableau principal : moyenne de `dot(R_est @ n_i,
  n_j)` sur les inliers finaux de RANSAC (au seuil de distance, indépendamment de
  `--normal_tau`) — diagnostic indépendant pour voir si les inliers retenus sont
  plausiblement des points de contact (négatif) ou des points proches sans orientation
  cohérente.

Bon signe attendu : `score_gap` diminue fortement (idéalement négatif), `Pose@30` passe
au-dessus de 10-15%, `RotErr` nettement sous 120°. `InlierRatio`/`RansacValid` peuvent
baisser — acceptable, on préfère moins de poses candidates mais plus fiables.

**Résultat (2026-06-27)** : `count_times_quality_and_normal` (score doux, sans filtre dur)
sur `gt_edge` donne la meilleure config — `Pose@30`=9.62%, `RotErr`=103.3° (vs 127.0°
distance-only), `score_gap`=+10.46 dans son propre mode de scoring (pas comparable en
valeur absolue à `count` — mais nettement réduit *dans son propre référentiel*, cohérent
avec l'amélioration de `Pose@30`/`RotErr`). Combiner filtre dur (`--normal_tau -0.3/-0.5`)
+ score doux n'apporte rien de plus (`score_gap` même légèrement pire) — **le signal
normal doit rester continu, pas être un filtre dur** ; config retenue : score doux seul,
sans `--normal_tau`.

**Test décisif sur les masques réels** (`gt`, `thresh0.2/0.3/0.5`, même config) : effondrement
net — `CorrPrec` 22.79%→2.6-3.2%, `Pose@30` 9.6%→1.3-2.0%. `thresh0.2/0.3/0.5` ≈ `gt`
(le CNN n'est toujours pas en cause). **Conclusion : le gain du scoring normal-aware ne se
transfère pas aux masques réels, parce que `gt_edge` est un oracle pair-specific
(restreint via la pose GT, irréalisable en pratique) alors que `gt`/`thresh` mélangent les
points fracture de TOUS les voisins d'un fragment.** Le verrou n'est plus "trouver la
bonne pose à partir de points fracture" mais "identifier quelle partie de la fracture
d'un fragment correspond à quel voisin" — un problème de segmentation d'interface, pas de
matching point-à-point. Aucune amélioration du descripteur/scoring ne peut compenser un
pool de candidats dominé par du bruit structurel (points d'autres interfaces).

## Phase 2D — Diagnostic clustering d'interfaces (proto, pas un pipeline complet)

Teste si les points fracture d'un fragment se découpent naturellement en patches
spatiaux correspondant chacun à un voisin, **sans connaître la pose GT** (contrairement à
`gt_edge`) — si oui, un clustering spatial naïf pourrait approximer `gt_edge` en
pratique. Implémenté dans `scripts/phase2d_interface_clustering_diagnostic.py`. PAS de
RANSAC, PAS de descripteurs — diagnostic pur :
1. Clustering par connectivité (graphe de proximité radius + composantes connexes,
   scipy uniquement, pas de dépendance sklearn) sur les points fracture (masque `gt` ou
   `thresh<T>`) d'un fragment, dans son repère local (structure invariante à la pose).
2. Étiquetage GT (diagnostic uniquement) : voisin réel le plus proche en repère assemblé
   (formule Phase 0), si distance `< 0.05` (même tolérance que Phase 1).
3. Métriques : `mean_clusters_per_fragment`, `noise_rate`, `cluster_purity` (pondérée par
   taille de cluster), `mixed_cluster_rate` (`<50%` pureté), `edge_coverage` (existe-t-il
   un cluster majoritairement étiqueté `j` pour l'arête `(i,j)` ?),
   `best_cluster_corrprec_upper_bound` (pureté du meilleur cluster par arête — borne
   supérieure de `CorrPrec` atteignable si on ne donnait au matcher que ce cluster).

Lecture : `purity`/`edge_coverage` > 70% et `best_cluster_corrprec` >> `CorrPrec` global
(2.6-3.2%) → interfaces spatialement séparables, un module cluster→matching est une
suite crédible. Pureté faible / couverture basse → la séparation spatiale naïve ne
suffit pas, il faudra un modèle appris pair-specific ou des features plus riches.

Ne pas repartir sur FPFH/SHOT ou un matcher appris avant ce diagnostic : le dernier
résultat montre que le problème est **avant** le descripteur point-à-point (mauvais
voisinage de candidats) — un meilleur descripteur sur un pool contaminé par plusieurs
interfaces ne résoudra pas le problème de fond.

**Résultat sweep `--cluster_eps` (2026-06-27, `gt`, `min_cluster_size=10`) :** `0.04 →
0.02 → 0.015 → 0.01` fragmente de plus en plus (`Clusters/Frag` 1.21→4.48,
`Cluster/Deg` 0.62→1.89 — sur-segmentation à 0.01), mais **`EdgeCoverage` et
`BestCorrPrec` plafonnent autour de 43-44%/37-38% dès `eps=0.02`** et ne progressent
plus en dessous, alors que le bruit explose (`Noise` 11%→24.6%). `eps=0.02` est le
meilleur compromis ; descendre plus bas n'apporte rien, juste plus de bruit.
**Confirmé sur `thresh0.3` à `eps=0.02`** : résultats quasi identiques à `gt` (écarts
< 1 point sur toutes les métriques — `EdgeCov` 43.35% vs 43.57%, `BestCorrPrec` 36.80%
vs 37.23%) → le CNN préserve la structure spatiale des interfaces aussi bien que le
masque GT. **Encore une fois, le CNN n'est pas le facteur limitant.**

### Conclusion finale Phase 2D

> La segmentation fracture binaire (CNN ou GT) est utile mais insuffisante pour le
> matching pairwise. Le verrou n'est pas le matching point-à-point ni le prior CNN, mais
> l'absence de séparation pair-specific des interfaces : un fragment touchant plusieurs
> voisins a tous ses points fracture mélangés dans un même masque. Le clustering spatial
> non supervisé (composantes connexes par proximité) récupère une partie réelle de ce
> signal — `BestCorrPrec`≈37-38%, contre 2.6-3.2% sur le masque global non filtré, soit
> ~12x mieux — mais son `EdgeCoverage` plafonne à ~44% quel que soit le réglage de `eps`,
> ce qui montre que les interfaces ne sont pas entièrement séparables par simple
> proximité spatiale : un peu plus de la moitié des arêtes (voisinages réels) ne sont
> couvertes par aucun cluster dominant. Le CNN ne dégrade pas ce résultat par rapport au
> masque GT (`thresh0.3` ≈ `gt` à `eps=0.02`), donc la limite est structurelle au
> problème, pas à la qualité du prior de segmentation.

**Décision (2026-06-27) : on part sur le module cluster-based léger.** Cohérent avec le
diagnostic Phase 2D, rapide à implémenter, basé sur le CNN réel (`thresh0.3`, pas un
oracle), suffisant pour une conclusion de stage solide. Le module appris de
séparation/matching d'interfaces reste une perspective Phase 3 / future work — plus
ambitieux (labels d'interface, architecture, entraînement, ablations) mais trop risqué à
ce stade.

## Phase 2E — Matching pairwise cluster-based (prototype court et ciblé)

Implémenté dans `scripts/phase2e_cluster_based_matching.py`. Sur les paires positives
uniquement (Protocole A, `graph[i,j]=True`), compare deux variantes sur les MÊMES arêtes :
- **`global`** (référence) : matching (1-NN descripteur + RANSAC `count_times_quality_and_normal`)
  sur tout le masque fracture `thresh0.3` de chaque fragment, comme en Phase 2C.
- **`cluster_based`** : clustering par connectivité (mêmes paramètres Phase 2D,
  `eps=0.02`) sur les points fracture de CHAQUE fragment séparément, dans son repère
  local — **aucune connaissance de la pose GT**, contrairement à l'étiquetage GT utilisé
  en Phase 2D qui ne servait qu'au diagnostic (le clustering lui-même est
  rotation/translation-invariant, donc clusterer en repère local ou assemblé donne
  exactement les mêmes clusters pour un fragment pris isolément). Teste *chaque paire*
  de clusters (cluster de `i` × cluster de `j`), garde l'hypothèse de pose la mieux
  notée (même critère de score que RANSAC) parmi toutes les paires de clusters testées.
  Si aucune paire de clusters ne produit de pose valide → `no_cluster_match`.

Métriques : `RansacValid`, `Pose@30°/0.1`, `RotErr`, `TransErr`, `no_cluster_match_rate`,
pour `global` vs `cluster_based` sur les mêmes arêtes. Attendu réaliste (pas un objectif
de performance absolue) : `Pose@30` global ≈ 1.5-2% (cohérent avec le run précédent sur
masques réels), `cluster_based` pourrait monter vers 4-8% si la séparation d'interface
aide effectivement le matching réel, pas seulement la précision de correspondance
théorique mesurée en Phase 2D.

**Résultat (2026-06-27) — contre-intuitif :** `cluster_based` fait **moins bien** que
`global` (`Pose@30` 1.82% vs 3.18%, `RotErr` quasi identique 124.6° vs 123.5°,
`RansacValid` 78.62% vs 87.59%, `no_cluster_match_rate`=23.57%). Le clustering isole bien
des patches individuellement plus purs (Phase 2D : `BestCorrPrec`≈37%), mais en testant
*toutes* les paires de clusters (4-16 par arête) et en gardant la mieux notée, on
multiplie les occasions que le critère de score (déjà connu pour préférer parfois une
pose fausse, `score_gap`>0 en Phase 2C) sélectionne une mauvaise paire — sur un pool plus
restreint (un cluster), un mauvais alignement peut sembler "propre" par coïncidence plus
facilement que sur le pool plus large du masque global.

**Diagnostic complémentaire ajouté : `oracle_cluster_pair`**, implémenté dans le même
script. Utilise la pose GT **uniquement pour choisir** quelle paire de clusters matcher
(recouvrement mutuel en repère assemblé, `CONTACT_EPS=0.05`) — le matching/RANSAC réel
sur cette paire reste identique à `cluster_based`, aucune triche dans cette étape. Tranche
: si `oracle_cluster_pair` >> `cluster_based`, le pipeline échoue surtout à **choisir** la
bonne paire (soutient un module de compatibilité cluster-cluster appris, Phase 3) ; si
`oracle_cluster_pair` reste mauvais aussi, la limite est plus profonde que la sélection
(descripteurs/RANSAC eux-mêmes insuffisants même sur la bonne paire).

**Résultat (2026-06-27) — Cas B confirmé :** `oracle_cluster_pair` `Pose@30`=2.47%,
`RotErr`=119.7°, `TransErr`=0.245 — seulement une amélioration modeste par rapport à
`cluster_based` (2.01%/127.1°/0.342) et `global` (1.26%/126.2°/0.345), très loin du saut
spectaculaire (10-20%) qui aurait indiqué que la sélection de paire était le vrai
goulot. `no_oracle_match`=49.23% (cohérent avec `EdgeCoverage`≈43-44% de la Phase 2D) :
même en cherchant la meilleure paire possible, ~50% des arêtes n'ont structurellement
aucune paire de clusters qui se recouvre géométriquement. **Même avec la bonne paire
(quand elle existe), le matching reste très imparfait** — limite plus profonde que la
sélection de clusters.

### Conclusion finale Phase 2 (toutes sous-phases)

> Le verrou n'est plus seulement "quels points donner au matcher" — le CNN n'est jamais
> le facteur limitant (confirmé à chaque étape : recall en Phase 1, précision de
> correspondance en Phase 2A-C, structure spatiale en Phase 2D), et le clustering
> spatial récupère une partie réelle du problème de séparation d'interface. Mais le
> **matcher géométrique lui-même** (descripteurs faits main + RANSAC, même avec le
> meilleur scoring normal-aware trouvé) a un plafond de performance bas et largement
> indépendant de la qualité du pool de candidats : `gt_edge` (oracle pair-specific sur
> tout le masque fracture) culmine à ~9.6% de `Pose@30`, et `oracle_cluster_pair`
> (oracle de sélection sur des clusters plus petits) reste à ~2.5%. Aucun raffinement du
> filtrage ou de la sélection de candidats ne lèvera cette limite — elle est intrinsèque
> au pipeline de matching par descripteurs géométriques + RANSAC. **La Phase 3 (module
> appris) n'est donc pas une amélioration optionnelle, mais une nécessité démontrée
> empiriquement** pour dépasser ce plafond.

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

## Phase 3 — Matcher appris (cadrage décidé le 2026-06-27, suite directe de la conclusion Phase 2)

**Pourquoi pas juste un meilleur descripteur.** La Phase 2 a isolé deux causes
indépendantes du plafond, pas une seule :
1. Descripteurs faibles : même à l'oracle `gt_edge`, `CorrPrec` (top-1) ≈ 22.8-28.8%,
   mais `topk_recall_20` ≈ 60% — le signal existe, le 1-NN strict est trop strict.
2. Scoring RANSAC structurellement biaisé : `score_gap > 0` confirmé sur 3 formules de
   score différentes (`count`, `count_times_quality`, `count_over_mean_residual`) — le
   critère préfère objectivement une pose fausse, quelle que soit la formule de
   comptage/qualité essayée (Phase 2C).

Remplacer seulement `HybridGeometryFeatures` par un encodeur appris, en gardant un
RANSAC à seuil dur derrière, ne corrigerait que la cause 1. **Décision : la Phase 3 doit
apprendre à la fois de meilleurs descripteurs ET à pondérer/sélectionner les
correspondances utiles à la pose** — pas seulement mieux décrire les points, puisque la
Phase 2 a montré que le scoring géométrique heuristique échoue même quand un signal de
correspondance existe.

**Architecture cible (positive pairs only, Phase 3A) :**
```
CNN Step 15 figé → points fracture thresh0.3 (échantillon borné, N=512 ou 1024)
→ encodeur partagé léger (PointNet / EdgeConv), features = coords locales + normales
  + score CNN + features géométriques simples (HybridGeometryFeatures en complément,
  pas en remplacement total — à valider empiriquement si elles aident en input)
→ interaction cross-fragment (corrélation de descripteurs i×j)
→ matrice de correspondance souple : row-softmax + dustbin (PAS Sinkhorn au départ —
  Sinkhorn impose une structure quasi one-to-one équilibrée, alors que les surfaces de
  fracture sont échantillonnées/bruitées et les correspondances peuvent être
  many-to-one ou partielles ; un dustbin laisse les points sans contact réel ne
  matcher avec rien plutôt que d'être forcés dans une permutation)
→ weighted Kabsch différentiable (poids = lignes de la matrice de correspondance)
→ pose relative R_ij, t_ij
```

**Loss composite (pas pose loss seule — risque d'instabilité : matrice diffuse,
concentration sur quelques points faciles, solution dégénérée donnant parfois une bonne
pose sans correspondances interprétables) :**
```
L = L_pose + α·L_correspondance + β·L_contact + γ·L_regularization
```
- `L_pose` : geodesic rotation loss + L2 translation loss (R_ij/t_ij GT, formule déjà
  dérivée et confirmée en Phase 0 : `R_ij = R_j^{-1} @ R_i`, `t_ij = R_j^{-1} @ (t_i - t_j)`,
  repère après réapplication de `scale`).
- `L_correspondance` : supervision auxiliaire de la matrice souple, label dérivé
  directement de la logique `gt_edge` déjà implémentée (`phase2_geometric_baseline.py`,
  reconstruction GT + NN < `CONTACT_EPS`=0.05) — même principe que les diagnostics
  `CorrPrec` de la Phase 2, réutilisé ici comme signal d'apprentissage plutôt que comme
  métrique d'évaluation seule.
- `L_contact` : pénalise les poids de correspondance élevés sur des points sans contact
  réel (complète `L_correspondance`, cible spécifiquement le confound multi-voisins
  identifié en Phase 2B/2D — un fragment touchant plusieurs voisins a des points
  fracture vers chacun d'eux dans le même masque `thresh0.3`).
- `L_regularization` : à définir en implémentation (ex. entropie de la matrice pour
  éviter une diffusion totale) — détail d'implémentation, pas un choix de cadrage.

**Phase 3A — scope : positive pairs only (`graph[i,j]=True`).** Question posée : "si
deux fragments vont vraiment ensemble, le module retrouve-t-il leur pose relative ?"
Pas de paires négatives à ce stade — mélanger pair/non-pair et qualité de pose dans le
même entraînement initial risque de produire un modèle qui apprend surtout à
discriminer "paire/non-paire" sans bien apprendre la pose.

Évaluation : réutiliser les métriques déjà définies en Phase 2
(`Pose@30°/0.1`, `Pose@15°/0.05`, `RotErr`, `TransErr`) pour comparaison directe avec
`global` (Phase 2C, ~1.3-3.2%), `gt_edge` oracle (~9.6%, plafond Phase 2 toutes
conditions confondues) et l'oracle Kabsch sanity check (~11.8° RotErr, borne haute du
pipeline pose/Kabsch lui-même, indépendante du matching).

**Phase 3B (seulement après 3A, si le signal de pose est validé) :** ajouter les paires
négatives (`graph[i,j]=False`) pour apprendre un score de compatibilité de paire —
question différente ("le modèle reconnaît-il quelles paires vont ensemble ?"), à ne pas
mélanger avec 3A (cf. distinction Protocole A/B déjà établie en fin de Phase 2).
Échantillonnage : toutes les paires positives + 2-3x négatifs aléatoires (hard
negatives seulement plus tard).

**Prérequis technique non résolu (à vérifier avant le premier entraînement) :** aucun
utilitaire de sampling de paires n'existe dans le codebase (confirmé par inspection de
`assembly/data/breaking_bad/base.py` — `collate_fn` standard, pas de pair sampling).
Il faudra écrire un dataset/sampler dédié qui extrait les paires positives à partir de
`graph` + `points_per_part` (en respectant le padding `max_parts`), avant le modèle.

**Vérification de la plomberie (`scripts/phase3a_pair_dataset_check.py`), 2026-06-27 —
VALIDÉE.** Run initial (`label_mode=hard`, `everyday/val`, 10 batches, 234 paires
directed) : `valid_target_col_rate=1.0000`, 0 vrai problème de sanity check (le seul
warning observé, asymétrie de masque entre i/j, est un phénomène réel de fragments de
tailles très différentes — reclassé en compteur informatif `asymmetric_pair_rate`, pas
une "issue"). `contact_row_rate=34.85%` / `dustbin_row_rate=65.15%` — quasi identique à
l'`avail_rate≈38%` mesuré en Phase 2B sur les masques `thresh*` : confirmation croisée
indépendante que le confound multi-voisins diagnostiqué en Phase 2 est bien présent
dans cette nouvelle pipeline de labels.

**Problème trouvé et corrigé : label soft trop diffus avec `label_sigma` seul.**
Premier run `label_mode=soft` (`contact_eps=0.05`, `label_sigma=0.02`, sans cap) :
`mean_matches_per_contact_row=70.53`, `mean_effective_matches_per_contact_row=38.84`
(quasi égal au compte brut → label réellement diffus, pas juste "beaucoup de voisins
mais poids concentré"). Cause : la densité de points varie énormément selon le
fragment (219 à 4027 points dans le masque observé) — un `label_sigma` fixe ne peut
pas compenser cette variation de densité. **Fix : `--label_topk`** (défaut 8) — cap le
nombre de colonnes positives aux K plus proches voisins sous `contact_eps`, *avant*
la pondération gaussienne, donc indépendant de la densité locale. Rôles découplés :
`contact_eps` décide contact vs dustbin, `label_topk` borne le nombre de colonnes
positives, `label_sigma` répartit le poids entre elles.

**Résultat sweep `--label_topk` (2026-06-27, même config, soft) :**
| topk | mean_matches | mean_effective | target_density | contact/dustbin rate |
|---|---|---|---|---|
| 8 | 7.22 | 6.18 | 2.53 | 34.85%/65.15% (inchangé) |
| 4 | 3.80 | 3.52 | 1.34 | 34.85%/65.15% (inchangé) |
| 1 (≈hard) | 1.00 | 1.00 | 0.35 | 34.86%/65.14% (inchangé) |

`contact_row_rate`/`dustbin_row_rate` parfaitement stables sur les 3 runs (confirme
que `label_topk` n'affecte que la pondération, pas la décision dustbin, comme prévu).
`target_density` chute massivement (16.3 → 2.53 → 1.34 → 0.35) — le label n'est plus
une soupe diffuse. `mean_effective_matches` à topk=8 (6.18, à la limite haute de la
fourchette visée 2-6) reflète que les 8 plus proches voisins sont souvent très
similaires en distance sur une surface de fracture lisse — pas un problème, juste un
label peu piqué là où la géométrie locale est elle-même peu discriminante.
**Décision : `--label_topk 8` retenu comme réglage par défaut pour l'entraînement**
(meilleur compromis contexte positionnel / diffusion contrôlée).

**Conclusion : la plomberie de données Phase 3A est validée.** Prochaine étape :
écrire l'encodeur partagé + matching souple (row-softmax + dustbin) + weighted Kabsch
différentiable, et la boucle d'entraînement (Phase 3A, positive pairs only).

### Modèle V0 implémenté (`assembly/models/pair_matching/soft_kabsch_matcher.py`,
`scripts/phase3a_train_pair_matcher.py`)

Architecture minimale conforme au cadrage : `PointEncoder` (MLP partagé pointwise,
LayerNorm+GELU) → `SoftCorrespondenceMatcher` (corrélation cosinus + colonne dustbin
apprise, row-softmax) → `weighted_kabsch` (différentiable, vérifié contre une pose GT
synthétique, erreur ~0.03° en float32). V0 = `L_corr` seul ; Kabsch calculé à chaque
pas pour le monitoring (`Pose@30°`, `RotErr`) mais pas inclus dans le backward avant
`--warmup_epochs` (réserve V1). Paires positives uniquement.

**Bug trouvé et corrigé : collapse vers "tout dustbin".** Premier run V0 (5 epochs,
`logit_scale` fixe `/sqrt(D)=11.3`) : `dustbin_pred_rate` converge vers 100% dès le
step ~100, `contact_pred_rate=0%`, alors que `non_dustbin_confidence` restait haut et
l'entropie proche du max — pas un problème de pondération contact/dustbin mais
d'échelle. Les descripteurs normalisés L2 donnent une similarité cosinus dans `[-1,1]`,
écrasée par `/sqrt(D)` dans `[-0.09,0.09]`, alors que le biais dustbin était un scalaire
libre sans contrainte d'échelle — le moins cher à faire grossir pour réduire la loss sur
les ~65% de lignes dustbin. **Fix : `logit_scale` appris (style CLIP, init `exp(log(10))`)**
remplace le `/sqrt(D)` fixe. Diagnostics ajoutés : `logit_scale`, `dustbin_bias`,
`match_logits_mean/std/min/max`, `dustbin_logit`, `max_match_logit_mean`,
`dustbin_minus_max_match` (signe positif partout = dustbin domine structurellement).

**Résultat V0.1 (15 epochs, `init_logit_scale=5` `init_dustbin_bias=1` — valeurs de
convergence observées du run précédent, `scalar_lr_mult=0.1` pour ne pas laisser les 2
scalaires absorber tout le gradient, `pairs_per_step=16`, features = xyz+normales+
cnn_score+dist_to_centroid, in_dim=8) — collapse réglé mais aucun apprentissage réel :**
`match_top1_acc` reste **sous** `random_top1_acc` tout le run (train epoch0: 0.70% vs
random 0.75% ; epoch14: 0.30% vs random 0.66%), `match_top8_recall` reste **au niveau
du hasard** (épart <0.5pt, dans le bruit). `RotErr` plat ~125-130°. La baisse de
`l_corr` (5.87→5.47) est entièrement explicable par le recalibrage continu de
`logit_scale`/`dustbin_bias`, pas par un signal de correspondance appris.

**Cause probable identifiée :** `points_i`/`normals_i` et `points_j`/`normals_j` sont
en coordonnées brutes, chacune dans le repère de rotation aléatoire **indépendant** du
fragment (Phase 0) — donc non comparables entre `i` et `j`. 6 des 8 dimensions
d'entrée ne portent structurellement aucun signal cross-fragment ; il ne reste que
`cnn_score`+`dist_to_centroid`, trop faible pour discriminer parmi 512 candidats.

**Ablation B — V0.2, features géométriques invariantes (`--feature_set geom_invariant`,
in_dim=5 : consistency+curvature+roughness+dist_to_centroid+cnn_score, au lieu de
xyz+normales brutes) : même conclusion négative.** 15 epochs, même protocole :
`match_top1_acc` reste sous le hasard tout le run (epoch0: 0.70% vs random 0.75% ;
epoch14: 0.41% vs random 0.66%), `match_top8_recall` oscille autour du hasard sans
tendance (écarts ±0.3-0.5pt, bruit). `RotErr` plat ~122-127°. **Ni les coordonnées
brutes (V0/V0.1) ni les descripteurs géométriques invariants faits main (V0.2) ne
donnent au matcher un signal exploitable au-dessus du hasard, sur 15 epochs
stabilisées.** Conforme à l'ordre d'ablation prévu (A négatif → B négatif → C).

### Ablation C — features internes du CNN (en cours, 2026-06-27)

**Inspection de `cnn_segmentation_model.py` (checkpoint Step15) :** `use_point_head=True`,
`point_head_feat_dim=64`, `point_head_hidden_dim=128`. `PointHead.forward` fusionne
`feat_2d` (64, projection 2D échantillonnée par point) + `feat_3d_encoder(feat_3d)` (32,
xyz+normales+geo) → MLP `Linear(96,128)→ReLU→Linear(128,64)→ReLU→Linear(64,1)`. La
cible retenue : l'activation **juste avant la dernière Linear** (dim 64) — l'embedding
fusionné 2D+3D le plus riche disponible, déjà point-aligné (pas une feature-map image
brute du U-Net, qui ne serait pas encore au niveau point).

**Modification minimale (`assembly/models/cnn_segmentation_model.py`) :**
`PointHead.forward(..., return_features=False)` retourne `(logits, point_features)` si
activé (extraction via slicing `self.mlp[:-1]`, vérifié localement équivalent au forward
complet). `CNNFracSeg.forward(batch, return_point_features=False)` propage ce flag et
ajoute `out["point_features"]` (N_sum_valid, 64) au dict de sortie — même ordre/
concatenation que `coarse_seg_pred`. Défaut `False` partout : comportement inchangé pour
tous les appelants existants (training/eval steps, autres scripts).

**Intégration dans `phase3a_pair_dataset_check.py` :** `--use_cnn_features` (off par
défaut) → `iter_positive_pairs` appelle `model(batch_gpu, return_point_features=True)`,
découpe `point_features` avec les **mêmes** `offsets` que `coarse_seg_pred`, puis
`build_pair_sample` slice `cnn_feat_i/j` avec les **mêmes** `sel_i/sel_j` et
`valid_i/valid_j` que tous les autres champs par point (points/normales/cnn_score/
geom_feats) — aucun nouvel indice, donc pas de risque de désalignement silencieux.
Sanity checks étendus (NaN, padding=0) pour `cnn_feat_i/j` quand actif.

**Check de plomberie validé (2026-06-27, `--use_cnn_features`, everyday/val) :**
`cnn_feat_dim=64` constant, `SANITY CHECK ISSUES: 0`, tous les autres chiffres
identiques aux runs précédents (`contact_row_rate=34.85%`, `valid_target_col_rate=1.0`,
etc.) — confirme que `--use_cnn_features` est bien purement additif, pas de
désalignement.

**Implémenté (`phase3a_train_pair_matcher.py`) : `--feature_set cnn_feat` (C1,
in_dim=65 = `cnn_feat`(64) + `cnn_score`) et `cnn_feat_geom` (C2, in_dim=69 =
`cnn_feat` + `geom_invariant`(4) + `cnn_score`).** `use_cnn_features` dérivé
automatiquement de `feature_set` (pas de flag séparé à garder synchronisé) —
`epoch_pairs_in_chunks` ne demande `point_features` à `iter_positive_pairs` que si
C1/C2 est sélectionné.

**Résultat C1 (`cnn_feat`+`cnn_score`, in_dim=65, 15 epochs, 2026-06-27) — premier
signal positif propre de toute la Phase 3A.** `match_top8_recall` reste
**au-dessus** de `random_top8_recall` sur les 15 epochs, sans exception, train ET val
— contrairement à A et B où l'écart oscillait de part et d'autre de zéro. Écart
train : +0.73pp (epoch0, 6.58% vs random 5.85%) → +1.38pp (epoch14, 6.48% vs random
5.10%), légèrement croissant. Val similaire (+0.81 à +2.53pp selon l'epoch, plus
bruité, n_steps~20-37). **Mais `match_top1_acc` reste sous le hasard tout le run**
(0.50%→0.35% vs random 0.75%→0.66%) — le signal CNN aide à restreindre la zone
plausible (top-8) mais pas encore à pointer précisément le bon point. `RotErr` plat
~119-126° (V0, pas de pose loss). `dustbin_pred_rate` n'a atteint que 28.6% à
l'epoch 14 (vrai taux ~65%) — recalibrage pas terminé, donc pas certain que l'écart
plafonne déjà à ce niveau modeste.

**Décision (2026-06-27) : tester C2 avant de prolonger C1**, pour rester comparable
point par point avec le protocole A/B (15 epochs) avant de dépenser du temps sur un
run plus long. Ajout de `top1_gap`/`top8_gap` (= match - random) explicitement dans
les métriques/summary_json pour ne plus avoir à les recalculer à la main.

**Résultat C2 (`cnn_feat`+`geom_invariant`+`cnn_score`, in_dim=69, 15 epochs,
2026-06-27) — cas "C2 ≈ C1", pas d'amélioration nette.** `top8_gap` moyen sur les 15
epochs : train 1.19pp (C2) vs 1.04pp (C1, +14% relatif, sous le seuil "clairement
mieux" de +2-3pp fixé en amont) ; val 1.54pp (C2) vs 1.53pp (C1, quasi identique,
écart << bruit val à n_steps~20-37). `top1_gap`, `RotErr`, `dustbin_pred_rate`
(27.2% vs 28.6% epoch14) tous comparables entre C1/C2. **Décision : garder C1 (plus
simple, in_dim=65 vs 69) — `geom_invariant` n'apporte rien une fois `cnn_feat` déjà
présent**, cohérent avec B (géométrie invariante seule = hasard).

**Ajout : `--resume_from`/`--start_epoch`** (`phase3a_train_pair_matcher.py`) pour
prolonger un run existant sans repartir de zéro (charge le state_dict du matcher
seul, pas l'optimizer ; `--start_epoch` garde la numérotation d'epoch globale
cohérente dans les logs/summary_json).

**Résultat run prolongé C1 (epochs 15→40, `--resume_from output/phase3a_matcher_c1/last.pt
--start_epoch 15 --epochs 40`, 2026-06-27) — signal confirmé réel, mais rendements
nettement décroissants.** `top8_gap` moyen monte par rapport aux 15 premières epochs :
train 1.04pp→1.61pp (+55% relatif), val 1.53pp→2.19pp (+43% relatif) — pas un plateau
immédiat, le signal continue de se renforcer. `dustbin_pred_rate` a continué de
progresser vers le vrai taux ~65% (28.6%→56.6% train, 27.7%→41.7% val à l'epoch 39) —
calibrage toujours pas totalement terminé. **Mais** doubler le nombre d'epochs (15→40)
n'a donné qu'une augmentation modeste du gap, pas une explosion ; `top1_gap` reste
négatif tout le long (~-0.3 à -0.6pp) — toujours aucun signal de pointage exact, juste
un signal de "bon voisinage" (top-8) ; `RotErr` reste plat (~118-126°) ; et
`match_top8_recall` reste dans la zone 6-9% en absolu, train et val — loin d'un
matching exploitable en pratique.

### Conclusion finale Phase 3A — décidé de clôturer ici (2026-06-27)

> La Phase 3A montre que les features internes du CNN de segmentation contiennent une
> information utile pour le matching pairwise, contrairement aux coordonnées brutes
> (ablation A) et aux descripteurs géométriques faits main (ablation B). Cette
> information reste cependant faible avec une architecture minimale (MLP partagé +
> corrélation cosinus + row-softmax+dustbin) : elle améliore légèrement le rappel
> top-8 par rapport au hasard (`top8_gap` ≈ +1 à +2pp, croissant mais à rendement
> décroissant avec plus d'entraînement), mais ne permet ni un top-1 fiable ni une
> pose relative exploitable (`RotErr` reste quasi aléatoire, ~120-126°, du début à la
> fin de tous les runs V0/C1/C2). **La limite n'est donc plus la plomberie dataset, ni
> le prior CNN fracture, ni seulement le choix des features d'entrée — elle vient de
> la capacité du matcher minimal (corrélation cosinus simple, sans interaction
> pairwise explicite) à exploiter le signal déjà présent dans `cnn_feat`.**

**Chaîne de preuve complète (ordre des ablations, toutes sur 15+ epochs, comparées
systématiquement à la baseline aléatoire `random_top1_acc`/`random_top8_recall`) :**
- A. `raw_xyz_normal` (coordonnées/normales brutes, repères de rotation indépendants
  entre fragments) → négatif, jamais au-dessus du hasard.
- B. `geom_invariant` (descripteurs géométriques invariants faits main) → négatif,
  même conclusion que A.
- C1. `cnn_feat` (embedding fusionné 2D+3D du PointHead) → **premier signal positif
  net**, `top8_gap` > 0 sur tous les runs, train et val, qui croît avec
  l'entraînement (1.04→1.61pp train sur 0-14 puis 15-39 epochs).
- C2. `cnn_feat`+`geom_invariant` → n'ajoute rien à C1 seul (gap quasi identique) —
  la géométrie invariante n'apporte aucune information complémentaire une fois le
  signal CNN présent.

**Prochaine vraie direction si reprise un jour (PAS une petite correction, une
nouvelle phase) :** architecture avec interaction pairwise explicite —
`cnn_feat_i`/`cnn_feat_j` → cross-attention légère ou message passing pairwise →
matching souple → weighted Kabsch. Plus d'epochs ou un LR différent sur l'architecture
actuelle (corrélation cosinus simple) ne suffira pas, le run prolongé l'a montré
(rendements décroissants nets dès 15→40 epochs). À ne déclencher que si une suite
expérimentale est explicitement souhaitée (ex. demande du tuteur) — sinon, le
résultat ci-dessus est suffisant et solide pour le rapport de stage tel quel.

## Phase 4 — Matcher interactif (cadrage décidé le 2026-06-27)

**Recadrage important après la clôture de la Phase 3A :** on arrête l'architecture
minimale (MLP + corrélation cosinus statique), pas le projet. La Phase 3A a déjà
établi une carte claire du problème : CNN fracture → OK (Phase 1) ; matching
géométrique/RANSAC → plafond très bas (Phase 2) ; matcher MLP+cosinus → trop faible
(Phase 3A) ; features CNN internes → premier vrai signal mais inexploitable seul
(Phase 3A C1/C2). La cause structurelle identifiée : le matcher encode `i` et `j`
**séparément** puis fait juste `desc_i @ desc_j.T` — chaque point est représenté sans
jamais "voir" l'autre fragment, alors que le matching de fragments cassés est
fondamentalement pair-dependent.

**Roadmap (3 mois, à ajuster au fil de l'eau) :**
- **4A — dustbin par point + `L_contact`** (fait, ci-dessous) : corrige une limite
  architecturale avant même d'attaquer l'attention.
- **4B — cross-attention léger** : `cnn_feat_i`/`cnn_feat_j` → self-attention sur
  chaque fragment → cross-attention `i←j`/`j←i` → matrice de similarité →
  row-softmax+dustbin. Minimal (D=128, 2 couches, 4 têtes), pas un Transformer
  massif. Critère de succès : `top8_gap` > C1-long, `top1_gap` moins négatif
  (idéalement positif), `dustbin_pred_rate` proche du vrai taux.
- **4C — pose avec weighted Kabsch** : seulement si 4B améliore le matching (sinon la
  pose restera plate comme en Phase 3A, déjà vérifié).
- **4D — paires négatives / score de compatibilité fragment-fragment** : branche
  alternative/sécurité — positives = arêtes `graph` GT, négatives = non-voisins,
  métriques AUC/AP/precision@k. Résultat solide même si la pose reste difficile :
  savoir quels fragments vont ensemble est déjà une étape majeure pour le
  réassemblage.
- Explicitement écarté pour l'instant : 80-100 epochs sur l'archi minimale,
  fine-tuning complet du CNN, assembly de graphe global, Sinkhorn lourd, gros
  Transformer — risque de consommer le temps restant sans diagnostic propre.

**4A — implémenté (2026-06-27) :** le biais dustbin global (Phase 3A) ne pouvait
exprimer qu'un taux moyen de dustbin, pas "ce point précis de `i` a-t-il un
correspondant dans `j`" — exactement le confound multi-voisins que la colonne
dustbin est censée gérer. Remplacé par `dustbin_head = MLP(desc_i)` (par point),
avec une loss auxiliaire `L_contact` (BCE entre le logit dustbin et l'existence
réelle d'un contact) : `L = L_corr + lambda_contact * L_contact` (`--lambda_contact`,
défaut 0.5). Dernière couche du head zero-init + biais = `init_dustbin_bias`, donc
comportement identique à l'ancien biais global au tout premier pas (transition
douce, pas de régression). `scalar_lr_mult` ne s'applique plus qu'à `logit_scale`
(le head n'a plus le "raccourci gratuit" d'un scalaire libre). Nouveau diagnostic
`dustbin_logit_std` (0 à l'init, doit croître si le head apprend un vrai signal par
point). Vérifié localement (forward/backward, gradients, std qui croît).

**Note de compatibilité :** ce changement modifie la structure du `state_dict` du
matcher (`dustbin_bias` scalaire → `dustbin_head` MLP) — les checkpoints Phase 3A
(`output/phase3a_matcher_c1/`, `_c1_long/`, `_c2/`) ne sont plus chargeables via
`--resume_from` avec cette architecture. Repartir de zéro pour 4A/4B.

**Bug trouvé et corrigé sur le premier run 4A (2026-06-27) : `L_contact` collapsait
vers "tout dustbin", même mécanisme que le bug original.** `dustbin_minus_max_match`
passe de -1.39 (epoch0) à **+2.44** (epoch14, train) et +3.92 (val) ; `contact_pred_rate`
chute 85.8%→6.7% (train), →0.37% (val). Cause : `contact_loss()` était une BCE **non
pondérée** — avec ~65% de lignes dustbin, l'optimiseur réduit la loss en poussant
`dustbin_logit` vers +∞ pour toutes les lignes (gagnant sur les 65% correctes, perdant
sur les 35% restantes), exactement le même raccourci d'imbalance que l'ancien
`dustbin_bias` global. **Fix : repondération de `contact_loss` par
`contact_row_weight`/`dustbin_row_weight`** (mêmes valeurs 2.0/1.0 que
`soft_correspondence_loss`, pour cohérence). Vérifié sur un cas synthétique (BCE par
ligne, ~65% déséquilibre) : sépare parfaitement les deux classes avec la
repondération.

**Second bug trouvé sur le run 4A après le fix de pondération (2026-06-27) :**
`contact_pred_rate` ne s'effondre plus à 0% (la pondération fonctionne), mais le
collapse vers dustbin persiste sous une forme différente — `dustbin_minus_max_match`
passe de -1.45 à **+1.86** (train) / +3.43 (val) sur 15 epochs, `dustbin_pred_rate`
jusqu'à 86.6% (train) / 99% (val). Cause : `dustbin_logit` (sortie de la MLP) n'avait
aucune borne liée à l'échelle des logits de matching. `L_contact` (BCE) pousse
légitimement `dustbin_logit` vers des valeurs extrêmes pour les lignes dustbin
correctement classées (comportement BCE normal) — mais ces valeurs (observées
3.7-6+, vs logits de matching ~0.3-2.6) dominent ensuite le softmax de `L_corr`, qui
compare `dustbin_logit` aux logits de matching **de la même ligne** : un point peut
être "correctement classé" par la BCE en moyenne tout en faisant systématiquement
gagner dustbin dans la comparaison softmax réelle. **Fix : borner `dustbin_logit` à
`(-scale, scale)` via `scale * tanh(MLP(desc_i))`**, en réutilisant le même
`logit_scale` que les logits de matching (au lieu d'une sortie MLP libre) — supprime
structurellement le degré de liberté incontrôlé plutôt que d'espérer un équilibre
entre les deux pertes. Vérifié localement : après 300 pas de gradient agressif,
`|dustbin_logit| <= scale` reste garanti.

**Résultat 4A après les deux fixs (2026-06-28) — ne bat pas C1.**
`dustbin_minus_max_match` reste positif en fin de run (+1.75 train, +2.28 val), mais
ce n'est plus un bug d'échelle : `dustbin_logit` et les logits de matching sont bien
sur la même plage bornée `[-scale, scale]`, vérifié. C'est que `dustbin_logit` sature
naturellement près de la borne haute (poussé par `L_contact` à être confiant sur les
~65% de lignes vraiment dustbin), alors que le meilleur candidat de match par ligne
(`max_match_logit_mean`≈2.0-2.7) reste modeste — parce que le signal de matching
sous-jacent (corrélation cosinus simple) est structurellement faible, comme établi
dans toute la Phase 3A. Comparaison directe `top8_gap` moyen sur 15 epochs : train
1.01pp (4A) vs 1.04pp (C1, quasi identique) ; val 1.32pp (4A) vs 1.53pp (C1,
**légèrement inférieur**). **4A ne dégrade pas, mais n'améliore pas non plus la
qualité du matching par rapport au biais global simple de C1.**

**Conclusion 4A : confirme, plutôt que résout, le diagnostic de départ de la Phase 4.**
Séparer "dustbin ou pas" de "quel point" (l'objectif de 4A) ne change rien si le
signal de matching lui-même (issu d'un encodeur qui traite `i` et `j` indépendamment)
reste trop faible pour produire un meilleur candidat que le seuil dustbin appris. Le
verrou n'est pas la formulation du dustbin (désormais propre et numériquement stable)
— c'est l'absence d'interaction entre fragments dans l'encodeur. **Décision : ne pas
continuer à affiner 4A (ex. tuner `lambda_contact`), passer directement à 4B
(cross-attention).**

**4B — implémenté (2026-06-28) :** `CrossAttentionBlock` (self-attention intra-fragment
puis cross-attention `i↔j`, poids partagés entre les deux directions) empilé en
`CrossAttnEncoder` (défaut 2 couches, 4 têtes), branché sur la **même**
`SoftCorrespondenceMatcher` déjà validée en 4A (dustbin borné, `logit_scale` appris) —
isole l'effet de l'interaction de tout le reste (rien d'autre ne change).
`--matcher_arch {mlp_cosine, cross_attn}` dans `phase3a_train_pair_matcher.py` ;
signature `forward` uniformisée `(feat_i, feat_j, valid_i, valid_j)` pour les deux
architectures. Vérifié localement (forward/backward, gradients dans tout l'encodeur,
padding géré via `key_padding_mask`, init identique à 4A pour `dustbin_logit`).

**Note de compatibilité :** nouvelle classe de modèle (`CrossAttnPairMatcherModel`),
checkpoints 4A/C1/C2 non chargeables ici non plus (architecture différente). Repartir
de zéro.

**Premier run 4B (2026-06-28, `--pairs_per_step 8`) — bug d'implémentation puis
instabilité d'entraînement, pas encore de verdict sur l'hypothèse.** Crash initial :
`CrossAttentionBlock._apply()` écrasait `nn.Module._apply` (utilisé en interne par
`.to(device)`) — méthode renommée en `_attend`, vérifié (`.to()` + forward/backward).

Une fois lancé, **collapse de représentation classique des transformers** : entre
l'epoch 4 et 5 (train), `match_logits_std`/`dustbin_logit_std` s'effondrent vers ~0
(tous les logits, matching et dustbin, convergent vers la même valeur ≈ `scale`) —
l'encodeur produit un descripteur quasi constant indépendamment de l'entrée. `loss`
**augmente** 5.3→7.2 au lieu de descendre ; `top8_gap` tombe à ~0, parfois négatif
(pire que le hasard) sur la majorité du run. Signature d'instabilité d'optimisation
(LR constant élevé sans warmup/écrêtage sur un transformer), pas une preuve contre
l'hypothèse d'interaction cross-fragment. **Fix : `--grad_clip`** (0=off par défaut,
comportement Phase 3A/4A inchangé ; recommandé 1.0 pour `cross_attn`).

**Prochaine étape : relancer 4B avec `--grad_clip 1.0`**, même protocole sinon (15
epochs, `--feature_set cnn_feat`, `--pairs_per_step 8`, mêmes inits), pour enfin
obtenir un verdict propre sur l'hypothèse d'interaction cross-fragment. Si
l'instabilité persiste malgré le clipping, envisager aussi un LR plus bas pour
l'encodeur (`--lr` actuel 1e-3, possiblement trop élevé pour un transformer) avant de
conclure sur l'architecture elle-même.

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

Phase 0, 1, 2 confirmées/closes. Phase 3A clôturée (2026-06-27, matcher minimal
insuffisant mais signal CNN confirmé réel — voir conclusion ci-dessus). Phase 4
ouverte : **4A fait et clos** (dustbin par point + `L_contact`, deux bugs trouvés et
corrigés, conclusion : confirme le diagnostic sans le résoudre — `top8_gap` ≈ C1,
pas d'amélioration). **4B implémenté (cross-attention, ci-dessus), prêt à lancer.**
Prochaine étape concrète : lancer l'entraînement avec `--matcher_arch cross_attn`
(repartir de zéro, pas de `--resume_from`), `--pairs_per_step` réduit (4-8), même
protocole sinon (15 epochs, `--feature_set cnn_feat`), comparer `top1_gap`/`top8_gap`
à C1/4A.
