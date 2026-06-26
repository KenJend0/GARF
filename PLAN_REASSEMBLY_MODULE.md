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

Étapes 2-3 (pas encore implémentées, dépendent du résultat de l'étape 1) :
- Critère d'inlier combiné : `distance < tau_dist ET dot(R @ n_i, n_j) < tau_normal`,
  tester `tau_normal ∈ {-0.3, -0.5, -0.7}` en commençant par le plus permissif (-0.3).
- Score soft normal-aware : `score = n_inliers × mean(clamp(-dot, 0, 1))`, ou combiné
  avec la qualité de distance (`dist_quality × normal_quality`).
- Bon signe attendu : `score_gap` diminue fortement (idéalement négatif), `Pose@30`
  passe au-dessus de 10-15%, `RotErr` nettement sous 120°. `InlierRatio`/`RansacValid`
  peuvent baisser — acceptable, on préfère moins de poses candidates mais plus fiables.

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
