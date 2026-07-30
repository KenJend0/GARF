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

**Run prolongé 4B (2026-06-28, `--resume_from output/phase3a_matcher_4b/last.pt
--start_epoch 15 --epochs 40 --seed 2026`, `--grad_clip 1.0`, sinon protocole
identique) — pas de redressement, instabilité confirmée plutôt que résolue.**
`top8_gap` train oscille autour de zéro sur les 25 epochs (premier epoch +0.004pp,
dernier epoch **-0.15pp**, plusieurs epochs négatives en cours de route) — nettement
sous C1 (1.04pp) et 4A (1.01pp), cohérent avec la moyenne déjà mesurée sur le premier
run 4B (0.19pp). `top8_gap` val part à +1.54pp (epoch15) mais devient instable et
traverse zéro à plusieurs reprises (jusqu'à **-0.83pp**), pour finir à +0.43pp
(epoch39) — contrairement à C1/4A où le gap val restait positif sur tout le run.
**`dustbin_pred_rate` empire plutôt que de se rapprocher du vrai taux (~65%)** : train
97.6%→99.3%, val atteint **100.0% à 4 reprises** (collapse total,
`contact_pred_rate=0`) avant de remonter partiellement — pire que le premier run 4B
(87-97%), pas la stabilisation espérée du grad clipping. `dustbin_minus_max_match`
continue de croître (train 4.70→5.10, pic val 5.76) et `match_logits_std` reste bas et
instable (creux jusqu'à 0.19-0.27 par endroits) — le grad clipping a empêché la
divergence de loss observée au premier run, mais n'a pas résolu le collapse de
représentation sous-jacent, juste ralenti/lissé sa manifestation. `rot_err_deg_mean`
reste plat ~125-130° (attendu, `use_kabsch=false`).

### Conclusion finale Phase 4B — clos (2026-06-28)

> Le grad clipping a corrigé l'instabilité numérique grossière du premier run (loss qui
> explosait, std qui s'effondrait à zéro net en quelques epochs), mais le verdict de
> fond sur l'hypothèse d'interaction cross-fragment reste négatif, et même légèrement
> pire qu'avant : `top8_gap` ne dépasse jamais durablement les niveaux de C1/4A,
> oscille autour de zéro (parfois négatif) au lieu de croître comme en C1-long
> (15→40 epochs, 1.04→1.61pp), et `dustbin_pred_rate` s'éloigne du vrai taux plutôt
> que de s'en approcher (collapses répétés à 100% en val). **L'architecture
> self-attention + cross-attention, telle qu'implémentée ici (2 couches, 4 têtes,
> D=128), n'apporte donc pas l'interaction pairwise utile espérée** — soit la capacité
> est mal exploitée (peu de pairs_per_step, pas assez de données par step pour
> stabiliser un transformer), soit le signal de matching point-à-point reste
> structurellement trop faible pour qu'une meilleure architecture d'encodeur seule le
> débloque, cohérent avec la chaîne de preuve Phase 2/3A déjà établie. **Décision,
> conforme à l'arbre fixé en amont : ne pas pousser plus loin 4B (pas de sweep
> LR/architecture additionnel), passer à la Phase 4D.**

## Phase 4D — Classifieur de compatibilité fragment-fragment (cadrage, 2026-06-28, fallback)

> **Statut :** cadré mais pas encore lancé. Rétrogradé en fallback (voir Phase 5A.0
> ci-dessous) : la Phase 5A sera tentée en premier. Si le pre-check 5A.0 échoue
> (trop peu d'objets à 2 fragments, ou faces trop courbes), 4D devient la prochaine
> action concrète sans modification supplémentaire — la plomberie décrite ci-dessous
> reste intacte.

**Pourquoi cette branche maintenant.** 4A et 4B ont chacun confirmé, sans résoudre, le
même diagnostic : le signal de matching point-à-point issu de `cnn_feat` (Phase 3A C1)
reste trop faible pour qu'aucune des deux pistes testées (dustbin par point + `L_contact`
en 4A, interaction cross-attention en 4B) ne le débloque significativement au-dessus du
hasard. Continuer à itérer sur le matching point-à-point (3e architecture, plus
d'epochs, LR différent) risquerait de consommer le temps restant sans nouveau
diagnostic — la Phase 4D pose une **question différente**, déjà identifiée comme
fallback solide en Phase 2 (distinction Protocole A/B) et dans le cadrage Phase 4 :
« deux fragments donnés vont-ils ensemble ? » plutôt que « quel point correspond à
quel point ? ». Une réponse positive ici reste une contribution utile au réassemblage
même si le matching pose-level fin échoue.

**Question posée :** à partir des features déjà disponibles par fragment (`cnn_feat`
agrégé, ou les descripteurs déjà calculés en Phase 2/3A), un classifieur peut-il
distinguer une paire de fragments adjacente (`graph[i,j]=True`) d'une paire non-adjacente
(`graph[i,j]=False`) — score de compatibilité, pas de pose.

**Protocole (à ne pas mélanger avec 3A/4A/4B, cf. distinction Protocole A/B déjà
établie en fin de Phase 2) :**
- Paires positives : toutes les arêtes `graph[i,j]=True` (comme tous les runs Phase
  3A/4A/4B).
- Paires négatives : fragments non-adjacents du **même objet** (pas de fragments
  d'objets différents — trop facile, ne testerait pas une vraie ambiguïté de
  réassemblage). Échantillonnage 2-3x négatifs par positif (cf. Phase 3B déjà prévu),
  pas de hard negatives dans une première itération.
- Représentation de paire : agrégation simple des features point-level déjà
  disponibles (`cnn_feat` Phase 3A C1, in_dim=64) par fragment — ex. mean/max pooling
  sur les points fracture `thresh0.3` de chaque fragment, puis concat ou différence
  des deux vecteurs agrégés, **avant** tout MLP de classification. Garder minimal,
  cohérent avec l'esprit du plan (pas de nouvelle architecture lourde avant d'avoir un
  premier signal).
- Label : binaire (`graph[i,j]`), pas besoin du label soft `topk`/`sigma` de
  3A/4A/4B (différent problème, pas de correspondance point-à-point ici).

**Métriques :** AUC, AP (average precision), precision@k (k = nombre réel de voisins
GT par fragment, déjà disponible via `graph`) — pas de `top8_gap`/`RotErr`, qui
n'ont pas de sens pour cette question. Comparer à une baseline triviale (distance
entre centroïdes de fragments, ou nombre de points fracture mutuellement proches sans
apprentissage) pour situer le niveau de difficulté avant de juger le classifieur appris.

**Fichier :** `scripts/phase4d_pair_compatibility.py` — **IMPLÉMENTÉ (2026-07-08)**.

Architecture : MLP 3 couches (input 256 → hidden 128 → 64 → 1), dropout 0.2.
Représentation de paire : mean+max pool de `point_features` (64-dim) sur les points
fracture de chaque fragment → 128-dim par fragment → concat → 256-dim pour la paire.
Négatifs : toutes les paires non-adjacentes intra-objet, sous-échantillonnées à
`NEG_RATIO=3 × n_pos` max. BCE avec `pos_weight=n_neg/n_pos` dynamique.
Évaluation : 3 conditions (gt / thresh0.3 / random) × {AUC, AP, Prec@k}.
Baseline triviale incluse : `-distance_inter_centroïdes` (AUC attendu ~0.5-0.7).

**Commandes serveur :**
```bash
# Test rapide
python scripts/phase4d_pair_compatibility.py \
    --ckpt output/cnn_step15_final_model/last.ckpt \
    --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
    --experiment cnn_step15_final_model \
    --categories everyday --epochs 20 \
    --max_batches_train 100 --max_batches_val 50 \
    --summary_json /tmp/student7/phase4d_quick.json

# Run complet
python scripts/phase4d_pair_compatibility.py \
    --ckpt output/cnn_step15_final_model/last.ckpt \
    --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
    --experiment cnn_step15_final_model \
    --categories everyday --epochs 40 \
    --summary_json /tmp/student7/phase4d_val.json
```

**Critère de succès :** AUC (thresh0.3) > baseline centroïde + marge significative
(> 0.05 pp), signalant que les features CNN encodent une compatibilité de fracture
au-delà de la proximité géométrique brute.

---

### Conclusion finale Phase 4D — POSITIF (2026-07-16)

**Run complet :** `everyday/val`, 40 epochs, mode cache (extraction CNN une fois,
entraînement MLP 1s/epoch sur features pré-calculées).

```
Baseline centroïde : AUC=0.32 (inversé — fragments proches ≠ adjacents dans les poses aléatoires)

Run 500 epochs (cache, 1s/epoch) — best checkpoint epoch 83 :
Strategy      N_pairs     AUC      AP   Prec@k
gt              94984    0.793   0.744    0.729
thresh0.3       94496    0.798   0.749    0.733
random          95071    0.627   0.505    0.669

Meilleur AUC val (thresh0.3) : 0.798 @ epoch 83 — plateau confirmé (no overfitting jusqu'à 500)
```

**Verdict : Phase 4D POSITIVE.** Le MLP appris sur features CNN agrégées distingue
les paires adjacentes des non-adjacentes avec AUC = 0.797.

Trois faits confirmés :

**(1) gt ≈ thresh0.3 (+0.4 pp).**
Le masque CNN prédit est presque aussi informatif que le masque GT pour agréger les
features. La segmentation fracture du Step 15 est suffisamment précise pour ce cas
d'usage — pas besoin d'un oracle.

**(2) thresh0.3 > random (+4.3 pp AUC, +15 pp AP).**
Les points fracture spécifiquement portent plus d'information de compatibilité que
des points aléatoires sur le même fragment. Le masque fracture CNN est discriminant.

**(3) AUC = 0.797 >> hasard (0.50).**
Les features `point_features` (64-dim, PointHead) encodent un signal de
compatibilité fragment-fragment genuinement appris — non trivial, au-delà de la
proximité géométrique brute (baseline centroïde = 0.32, inversée).

**Implications pour le rapport de stage :**
- La question "deux fragments vont-ils ensemble ?" est résoluble à AUC≈0.80 avec
  features CNN et un MLP minimal (41k params).
- Ce classifieur peut filtrer les paires incompatibles avant une estimation de pose
  coûteuse (Prec@k≈0.70 → 70% des vrais voisins retrouvés dans le top-k).
- L'approche reste partielle : on détecte la compatibilité mais pas la pose.
  Combiner 4D (compatibilité) + 3A (matching) reste une piste ouverte.

**Limites :**
- P@k = 0 dans le JSON final = artefact du mode cache (par-fragment non stocké).
  La vraie valeur (~0.70) vient du quick run (max_batches 50/100).
- Modèle encore en hausse à epoch 40 → relancer `--epochs 80` sur le cache
  (`--cache_file ...`) prend < 1 min.
- AUC plafonne probablement vers 0.81-0.82 (la difficulté intrinsèque du problème).

## Phase 5 — Depth-map fracture-face matching (piste tuteur, 2026-06-28)

**Motivation.** Les Phases 3A/4A/4B ont épuisé le matching point-à-point sur le
masque fracture global : ni les features CNN seules (3A C1/C2), ni le dustbin par
point (4A), ni la cross-attention (4B) n'ont dépassé durablement la baseline
aléatoire. La Phase 2D a identifié le verrou structural en amont : ~62% des points
fracture n'ont pas de correspondance disponible dans le voisin évalué à cause du
mélange d'interfaces multi-voisins. **La piste tuteur proposée ici change de niveau
d'abstraction :** au lieu de matcher des points individuels, on projette la *face
de fracture* de chaque fragment en une depth map 2D (après estimation d'un repère
local par PCA), puis on cherche un alignement 2D (rotation + translation) par score
de complémentarité relief. La complémentarité est la contrainte physique forte
manquante : `depth_A(u,v) ≈ -depth_B(R_θ(u,v) + t)` — bosse contre creux.

**Stratégie conditionnelle.** Phase 5 est une branche expérimentale, pas une
direction garantie. Son déclenchement est conditionné au pre-check 5A.0 :
- Pre-check OK → Phase 5A prioritaire, 4D en fallback.
- Pre-check KO → 4D directement, 5A archivée comme piste future.

### Phase 5A.0 — Pre-check (script à lancer, aucun modèle appris)

**Deux vérifications avant toute implémentation lourde :**

**V1 — Population suffisante.** Compter les objets à **exactement 2 fragments** dans
`everyday/train` et `everyday/val`. Critère indicatif : ≥100 paires positives val
pour avoir des statistiques robustes dans le rapport de stage. En-dessous, les
résultats seront trop fragiles pour être une conclusion centrale.

**V2 — Planéité des faces de fracture.** Sur les objets à 2 fragments, extraire
les points du masque GT fracture, réappliquer le `scale` (comme Phase 0, formule
`points_gt_scale = pointclouds_gt[i] * scale[i]` si on travaille en repère
assemblé, ou `pointclouds[i] * scale[i]` si repère local), puis PCA sur ces points :
```
eigenvalues = eigvalsh(cov)  # trié croissant
planarity = lambda_min / (lambda1 + lambda2 + lambda3)
```
- `planarity ≈ 0` → face quasi-plane (bon pour une projection en height field)
- `planarity ≈ 0.33` → isotrope (sphère, pas exploitable en depth map 2D)

Critère indicatif : médiane de planéité < 0.10 sur les deux fragments pour qu'une
depth map locale soit une représentation pertinente. Rapporter aussi la distribution
(percentiles 25/50/75/90), pas seulement la médiane.

**Rapporter aussi** :
- `n_2frag_train`, `n_2frag_val`, `n_pos_pairs_val` (= n_2frag_val, toujours 1
  paire positive par objet 2-fragments)
- Distribution de `n_frac_points_per_face` (nb de points fracture GT par face) —
  si trop peu de points (< 30-50), la PCA sera bruitée et la depth map trop éparse
- Proportion d'objets exclus parce qu'une des deux faces a < 10 points fracture
  (face de fracture quasi-vide, ex: fragment gros avec peu de contact)

Implémenté dans `scripts/phase5a0_precheck.py`.

**Résultat pre-check (2026-06-28, everyday/val, n=7872 objets scannés) — PASSÉ.**
```
Distribution num_parts :
  2 fragments : 3803 objets (48.3%)  ← cible 5A
  3 fragments : 1408 objets (17.9%)
  4+ fragments : le reste

V1 — Population :
  n_2frag_val = 3803 (>> seuil 100) : OK
  Exclus (face < 10 pts fracture GT) : 139 (3.6%)
  Retenus pour planéité : 3664

V2 — Planéité (GT mask, n=7467 faces) :
  mean=0.063, std=0.058
  p25=0.016, p50=0.043, p75=0.098, p90=0.150, max=0.269
  Critère médiane < 0.10 : OK (médiane=0.043)

n_frac_pts_per_face :
  p25=56, p50=340, p75=1805, p90=3052
  → forte dispersion ; filtre taille dans Phase 5A (exclure faces < 50-100 pts)
```
**Décision : Phase 5A depth-map matching PRIORITAIRE (4D en fallback).**

Notes post-precheck :
- 48% des objets val ont exactement 2 fragments — population très large, résultats
  seront statistiquement solides même avec un sous-ensemble limité du val.
- p75 planéité = 0.098 → 75% des faces individuelles passent le critère 0.10 ;
  les 25% restantes (faces courbes) peuvent être filtrées dans 5A avec un seuil.
- La faible variance entre p25 et p90 (0.016→0.150) montre une distribution
  relativement homogène — pas de bi-modalité plane/sphérique qui rendrait le
  pipeline inapplicable à une grande fraction des objets.
- p25 n_frac_pts = 56 pts : quelques faces éparses, un filtre `min_frac_points`
  (ex. 50) dans le script 5A éliminera les cas à PCA bruitée.

**Arbre de décision après le pre-check :**
```
→ Phase 5A depth-map matching PRIORITAIRE (4D en fallback)
   (les deux critères passés avec marges larges)
```

### Phase 5A — Two-fragment depth-map matching (cadrage, sujet au pre-check)

**Restriction à 2 fragments.** Supprime presque entièrement le confound
multi-voisins identifié en Phase 2D : avec 2 fragments, chaque point fracture d'un
fragment a un seul voisin possible, donc le masque fracture est déjà pair-specific
sans avoir besoin d'un clustering ni d'un oracle pair-specific. C'est la condition
minimale pour isoler la capacité du matching depth-map en elle-même, avant de
réintroduire la complexité multi-fragments.

**Pipeline Phase 5A :**
```
CNN Step 15 thresh0.3 (ou masque GT)
→ objets à 2 fragments uniquement (filtré par points_per_part)
→ réapplication du scale : raw_i = pointclouds[i] * scale[i]
→ PCA sur les points fracture → repère local (u, v, n)
→ planarity check (exclure si planarity > seuil, à fixer selon 5A.0)
→ projection en depth map 2D (rasterization en grille u×v)
→ sweep : θ ∈ [0°, 360°, pas 10°] + translation 2D (grille ou phase shift FFT)
→ score = mean_abs(depth_A(u,v) + depth_B(R_θ(u,v)+t)) sur la zone d'overlap
→ conversion de la meilleure pose 2D en pose 3D candidate (via repère PCA)
→ évaluation : RotErr, TransErr, Pose@30°/0.1, Pose@15°/0.05
```

**Trois conditions** (même logique que toutes les phases précédentes) :
- masque GT fracture → matching (oracle mask, isole la responsabilité du matching)
- masque CNN thresh0.3 → matching
- points aléatoires (même budget) → contrôle

Lecture : si GT échoue, l'hypothèse depth-map locale ne suffit pas (faces trop
courbes, résolution trop basse, ou score de complémentarité inadapté) — conclusion
valide pour le rapport. Si GT marche mais CNN échoue, problème de filtrage.

**Coût estimé** : sweep 36 rotations × translation 2D sur image L×L (ex. 64×64).
Si sliding brut : O(36 × L² × L²) par paire — trop lent pour L=128. Alternative :
corrélation de phase (FFT 2D) pour la translation à chaque rotation → O(36 × L²
log L) — raisonnable. À mesurer sur le serveur et documenter avant de lancer sur
tout le dataset.

**Fichier :** `scripts/phase5a_depthmap_matching.py` — **IMPLÉMENTÉ (2026-07-08)**.

Constantes : `RESOLUTION=64`, `N_ANGLES=36` (pas 10°), `MIN_FRAC_POINTS=50`,
`MAX_PLANARITY=0.15`, `MIN_OVERLAP_PIXELS=20`.

Fonctions clés :
- `compute_pca_frame(pts)` → centroid, u, v, n, planarity (eigvec de la plus petite valeur propre)
- `rasterize(pts, …, resolution, pixel_size)` → dmap, valid, u_min, v_min (np.add.at)
- `match_depthmaps(dmap_i, …, n_angles)` → FFT cross-corrélation par rotation + flip de normale ; score = -CC/overlap (complémentarité)
- `build_correspondences(…)` → pts_i_3d, pts_j_3d pour Kabsch (paires pixel-à-pixel)
- `process_pair(…)` → dict résultats par stratégie (gt / thresh0.3 / random)
- `kabsch(P, Q)` → R, t (convention identique Phase 2)

**À lancer sur le serveur — commande rapide :**
```bash
python scripts/phase5a_depthmap_matching.py \
    --ckpt output/cnn_step15_final_model/last.ckpt \
    --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
    --experiment cnn_step15_final_model \
    --categories everyday --split val --max_batches 50 \
    --summary_json /tmp/student7/phase5a_quick.json
```

**Critère de succès :** Pose@30°/0.1 (GT) > référence Phase 2 oracle (9.6%).

---

### Conclusion finale Phase 5A — CLOSE (2026-07-08)

**Run :** `--max_batches 50`, `everyday/val`, 103 paires 2-frags traitées.

```
Strategy          N   Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr
gt               21     82   155.02    0.5272       0.00%        0.00%    38.5
thresh0.3        16     87   159.36    0.5186       0.00%        0.00%    45.9
random            3    100   122.73    0.5113       0.00%        0.00%    42.7
```

**Verdict : négatif.** Deux symptômes distincts, deux causes distinctes.

**(1) Skip rate anormal : 80% pour gt.**
MIN_FRAC_POINTS=50 + MAX_PLANARITY=0.15 + condition OR par fragment → les seuils
s'appliquent sur DEUX fragments indépendants, la probabilité de survie est le produit
des deux taux de survie individuels. Avec p25 n_frac_pts=56 (pré-check), ~25% des
faces échouent déjà au filtre taille.

**(2) RotErr ≈ 155° — pire que le hasard (attendu ~90°).**
La rotation aléatoire uniforme donne en moyenne ~90°. Obtenir 155° ≈ 180° signifie
que le matcher trouve **systématiquement la pose inverse** (dos-à-dos) plutôt
que la pose correcte.

**Cause fondamentale : faces de fracture trop plates.**
Médiane planéité = 0.043 → le relief de la depth map est quasi-nul. Quand
`depth_i ≈ 0` et `depth_j ≈ 0`, la cross-corrélation `CC(depth_i, depth_j_rot) ≈ 0`
**quelle que soit la rotation**. Le score `-CC/overlap` est alors dominé par des
artefacts de bord : la zone où `overlap` est minimal (shift extrême) donne un
dénominateur quasi-nul qui gonfle artificiellement le score → argmax converge
vers une translation maximale et une rotation arbitraire (souvent 180°).

**Le signal de complémentarité depth-map nécessite un relief suffisant.**
Les faces de fracture du dataset Breaking Bad sont trop lisses pour cette approche.
La planéité très faible (bonne pour valider la rasterisation) est aussi une limite :
les fragments se cassent selon des plans, pas des surfaces texturées.

**Phase 5A close — Phase 5B annulée** (conditionnait à 5A réussi).

---

### Réouverture partielle (2026-07-20) — défaut identifié dans la formule de score

En préparant une présentation orale sur ce mail, remise en cause a posteriori de la
conclusion "négatif, définitif" : le score `-CC/overlap` utilisé dans `match_depthmaps()`
divise par le recouvrement au lieu de le récompenser, avec un garde-fou quasiment
inexistant (`overlap > 0.5 PIXEL`). Or `overlap` est déjà calculé, via FFT, pour
CHAQUE décalage testé — c'est la forme du contour de la zone de fracture qui coïncide
à ce décalage précis, une info de contour indépendante du relief. À faible
recouvrement, diviser par un dénominateur minuscule peut gonfler artificiellement le
score, ce qui est cohérent avec le biais observé (convergence systématique vers
~180°, cf. conclusion Phase 5A ci-dessus) — un artefact numérique plutôt qu'un vrai
optimum.

**Hypothèse non testée jusqu'ici, en particulier pertinente à 2 fragments :** les deux
faces d'une même fracture partagent exactement le même contour — cette forme seule
(sans aucun relief) pourrait suffire à fixer la rotation, même sur des faces
parfaitement plates.

**Corrigé dans `scripts/phase5a_depthmap_matching.py` (2026-07-20) :** nouveau
paramètre `--score_mode {relief, overlap_only, joint}` sur `match_depthmaps()` :
- `relief` : formule d'origine (buggée), conservée pour comparaison directe.
- `overlap_only` : score = fraction de recouvrement du contour SEULE (`overlap_frac`,
  normalisé par `min(n_valid_i, n_valid_j_rot)`), sans aucun relief — teste
  isolément l'hypothèse "le contour suffit".
- `joint` (nouveau défaut) : `score = relief_score * overlap_frac` — pénalité
  continue, sans seuil arbitraire à caler : un bon score de relief à recouvrement
  quasi nul est ramené vers 0 au lieu d'exploser par division ; un bon recouvrement
  sans complémentarité de relief ne suffit pas non plus à gagner seul.

`overlap_frac` à la meilleure hypothèse est maintenant loggé dans les résultats
(`results[strat]["overlap_frac"]`) et le résumé (`OvlpFrac` dans le tableau,
`overlap_frac_mean` dans le JSON) pour diagnostiquer sans devoir tout relancer.

**Premier run rapide (2026-07-20, `--max_batches 50 --batch_size 1`, N=4-5 paires
valides seulement) :** tendance dans le bon sens (RotErr gt : 114°→105°→83° pour
relief→overlap_only→joint) mais `Pose@30` reste à 0% partout — échantillon bien
trop petit pour conclure quoi que ce soit (même ordre de grandeur que le run
original qui avait donné 0% par accident statistique, cf. ci-dessous).

### Conclusion révisée Phase 5A — RÉOUVERTE, POSITIVE MAIS PARTIELLE (2026-07-20)

**Run à plus grande échelle** (`batch_size=1`, ~500 batches, 255 objets 2-frags vus
— 2.5x l'échantillon du run de clôture original) :

```
score_mode=relief (formule d'origine, MÊME algo que la conclusion "close" du 2026-07-08) :
Strategy    N   Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         44    211   129.29    0.3697       9.09%        6.82%    58.0    0.182
thresh0.3  36    219   134.05    0.4081       5.56%        2.78%    65.8    0.167
random     19    236   130.17    0.4706       0.00%        0.00%    43.8    0.137

score_mode=overlap_only (contour seul, sans relief) :
Strategy    N   Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         51    204   105.60    0.3795      19.61%        7.84%    80.8    0.728
thresh0.3  39    216   128.04    0.4275      10.26%        0.00%    89.3    0.702
random     27    228   145.97    0.4992       0.00%        0.00%    51.3    0.906

score_mode=joint (le fix : relief_score * overlap_frac) :
Strategy    N   Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         52    203    98.15    0.3285      21.15%       15.38%    80.5    0.666
thresh0.3  40    215   128.62    0.4381      12.50%        7.50%    74.7    0.611
random     22    233   143.72    0.4891       0.00%        0.00%    43.2    0.735
```

**Découverte n°1 — le "0%, pire que le hasard" du 2026-07-08 était un accident
statistique, pas un vrai résultat.** Avec la formule `relief` STRICTEMENT
identique (même algorithme, même bug), un échantillon 2.5x plus grand
(N=44 vs N=21 paires valides) donne `Pose@30=9.09%`, `RotErr=129°` — plus rien à
voir avec le "0%, RotErr=155°, pire que le hasard" documenté comme conclusion
définitive. Le run de clôture original reposait sur N=21 paires seulement :
échantillon trop petit, le "0%" est tombé par malchance et non par une vraie
absence de signal. **Leçon méthodologique à retenir pour la suite du stage :**
toujours vérifier la taille d'échantillon avant de clore une phase sur un résultat
à 0% ou 100% — ces valeurs extrêmes sont les plus sensibles au bruit statistique.

**Découverte n°2 — le fix de score (hypothèse du contour, cf. discussion du
2026-07-20) est confirmé, et sur un échantillon solide cette fois :**
`Pose@30` (gt) progresse proprement `relief` 9.09% → `overlap_only` 19.61% →
`joint` 21.15% ; `Pose@15/0.05` (critère strict) 6.82% → 7.84% → 15.38% ;
`RotErr` baisse 129°→106°→98°. `random` reste à 0% dans les trois modes (bon
signe : le gain vient bien du signal de fracture, pas d'un artefact de mesure
général). **`joint` (oracle GT) dépasse même la référence Phase 2 `gt_edge`
(9.6%, meilleur résultat de tout le matching géométrique classique du projet) —
plus du double.** Et ça tient aussi en condition réelle, pas seulement à
l'oracle : `thresh0.3` (masque CNN, pas GT) passe de 5.56% à 12.50%, au-dessus
de la référence Phase 2 sans connaître le masque GT.

**Mais lecture calibrée, à ne pas survendre :** `Pose@30=21.15%` (meilleur cas,
oracle GT) veut dire que **~79% des paires échouent encore**, et à un seuil plus
réaliste (`Pose@15°/0.05`) le taux de succès tombe à 15.38% — donc plus de 8
paires sur 10 ratent encore, même avec le fix. Ce n'est PAS "le réassemblage par
depth-map fonctionne" — c'est "le signal existe et est mesurable, le fix double
le meilleur résultat connu du projet sur cette tâche, mais la méthode reste très
loin d'un taux de réussite exploitable pour un réassemblage automatique fiable"
(cf. aussi la mise en garde déjà actée en Phase 6 sur l'accumulation d'erreurs à
un taux de succès partiel par paire, sur des objets à plusieurs fragments).

**Verdict : Phase 5A rouverte, résultat POSITIF MAIS PARTIEL** (pas "close
négatif" comme au 2026-07-08, pas "ça marche" non plus). À rapporter dans le
stage comme : diagnostic initial correct (le relief seul est insuffisant, faces
trop plates), mais la conclusion "donc la depth-map ne marche pas du tout" était
prématurée — un score qui exploite aussi la forme du contour (pas seulement le
relief) double le meilleur résultat de matching géométrique du projet, sans
toutefois le rendre exploitable en l'état.

### Run définitif à grande échelle (2026-07-20, `--max_batches 5000 --batch_size 1`)

**2415 objets 2-frags traités** (63% du val complet, ~9.5x l'échantillon du run
précédent) — chiffre à citer dans le rapport :

```
score_mode=relief (formule d'origine buggée) :
Strategy    N    Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         447   1968   142.96    0.4412       4.47%        2.24%    55.4    0.167
thresh0.3  339   2076   138.25    0.4437       3.83%        1.18%    60.7    0.161
random     162   2253   131.19    0.4709       0.00%        0.00%    46.7    0.096

score_mode=overlap_only (contour seul) :
Strategy    N    Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         522   1893   101.97    0.3466      22.03%        7.85%    80.5    0.738
thresh0.3  400   2015   114.98    0.3926      13.50%        5.00%    89.3    0.726
random     196   2219   131.93    0.4921       0.00%        0.00%    50.5    0.951

score_mode=joint (le fix) :
Strategy    N    Skip  RotErr°  TransErr Pose@30/0.1 Pose@15/0.05  N_corr OvlpFrac
gt         511   1904    94.95    0.3288      27.59%       13.70%    80.6    0.659
thresh0.3  399   2016   106.28    0.3745      21.05%        8.52%    82.5    0.624
random     175   2240   132.73    0.4784       0.00%        0.00%    47.9    0.764
```

**Découverte n°3 — la formule `relief` d'origine n'est pas juste moins bonne,
elle est INSTABLE.** Sur le run à N=255, `relief` donnait `Pose@30=9.09%` ; sur
ce run 9.5x plus grand (N=447), elle tombe à **4.47%** — une formule fiable ne
devrait pas autant bouger avec plus de données, elle devrait converger. C'est la
signature attendue d'un score dominé par du bruit numérique (division par un
recouvrement quasi nul), cohérent avec le bug identifié : `relief` ne mesure pas
un vrai signal stable, ses résultats sont eux-mêmes peu fiables d'un run à
l'autre. À l'inverse, `joint` **s'améliore** avec plus de données (21.15%→27.59%)
— signe d'un score qui converge vers un vrai signal plutôt que de fluctuer.

**Chiffres définitifs à retenir pour le rapport (oracle GT, `joint`) :**
`Pose@30°/0.1 = 27.59%` (contre 9.6% pour l'oracle `gt_edge` de la Phase 2 —
**quasi 3x mieux**), `Pose@15°/0.05 = 13.70%`. En condition réelle (`thresh0.3`,
pas d'oracle) : `Pose@30 = 21.05%`, toujours largement au-dessus des références
Phase 2. `random` reste à 0% partout, à cette échelle aussi — le gain est bien
spécifique au signal de fracture.

**Lecture calibrée inchangée :** même au meilleur cas (27.59%), **~72% des
paires échouent encore** à Pose@30, ~86% à Pose@15/0.05. Le fix est confirmé,
robuste, et significatif — mais la méthode reste loin d'un taux de réussite
exploitable pour un réassemblage automatique fiable.

---

### Phase 5B (si 5A marche bien) — Extension 3–5 fragments

**Toujours en attente, pas relancée.** La Phase 5A est repassée positive et
confirmée sur un échantillon large (27.59% de réussite au mieux, sur des objets
à 2 fragments seulement — le cas le plus simple). Le confond multi-voisins
(Phase 2D/2B) que la restriction à 2 fragments supprime spécifiquement
reviendrait dès 3+ fragments : pas de raison de penser que 5B ferait mieux tant
que 5A n'est pas nettement plus solide sur son propre cas simplifié.

---

## Phase 7 — Depth-map pose refinement + validation dataset réel (direction validée avec le tuteur, 2026-07-20)

**Conclusions de la présentation orale du 2026-07-20** (à partir des résultats
Phase 5A réouverte, confirmés à N=2415), notées ici dans l'ordre où elles ont
été formulées avec le tuteur, pas nécessairement l'ordre d'exécution :

**1. Recentrage du périmètre : pose entre deux fragments déjà connus comme
adjacents.** On continue sur le cadrage Phase 5A (comme le Protocole A de tout
le projet) — pas sur la détection de paires. **Le classifieur de compatibilité
(Phase 4D) est mis de côté pour plus tard, pas abandonné** — cohérent avec la
clôture NO-GO de la Phase 6.0 (2026-07-19) qui avait déjà écarté un pipeline
combiné 4D+pose pour l'instant.

**2. Conviction du tuteur : le matching par depth-map est la bonne direction,
sous-exploité, à améliorer.** Cinq pistes identifiées, aucune encore cadrée en
détail — à spécifier une par une avant implémentation (esprit diagnostic-avant-
code du reste du projet, cf. Phase 0-2) :
- **Sliding de fenêtre** : matcher des fenêtres locales plutôt que la depth map
  entière en un bloc. À préciser : fenêtre sur la depth map rasterisée ou sur
  le nuage de points avant rasterisation ? Quelle taille ? Comment agréger les
  scores de plusieurs fenêtres en une seule pose ?
- **Extrapolation si trop plat (?)** — le tuteur lui-même marque une incertitude
  sur le mécanisme. Piste à creuser : densifier/interpoler les cases `valid=False`
  de la depth map plutôt que les laisser vides, ou amplifier un relief faible
  mais réel. Risque à surveiller : fabriquer un faux signal si mal fait — donc
  diagnostic avant tout code, comme d'habitude sur ce projet.
- **Plus de résolution(s)** : `RESOLUTION=64` actuellement (grille fixe). Tester
  plus fin, et/ou une approche multi-échelle (grossier → fin) plutôt qu'une
  résolution unique.
- **Plus de features** : seule la hauteur (`depth`) est utilisée par pixel
  aujourd'hui. Ajouter d'autres canaux : normales locales, courbure, ou les
  features CNN du Step 15 (`point_features`, comme en Phase 3A/4D) — combinerait
  le signal de fracture appris avec la structure géométrique 2D.
- **Modèle appris** : remplacer le score géométrique fixe (`joint`) par un
  matcher appris sur les depth maps (ex. petit CNN 2D sur cartes multi-canal),
  dans l'esprit de la Phase 3 mais appliqué à la représentation depth-map
  plutôt qu'aux points bruts.

**3. Validation sur un dataset réel (pas seulement Breaking Bad, synthétique).**
Dataset proposé : [3D Puzzles, TU Wien](https://www.geometrie.tuwien.ac.at/ig/3dpuzzles.html).
Motivation explicite : la platitude anormale des faces de fracture Breaking Bad
(médiane planéité=0.043, Phase 5A.0) vient de la simulation de cassure — un
dataset de vrais objets physiquement cassés devrait avoir des surfaces plus
irrégulières/texturées, exactement la condition qui manquait pour que le
matching par relief fonctionne bien.

**Vérifié le 2026-07-20 (page du dataset) — deux points importants avant de s'engager :**
- **7 objets réels scannés au laser** (pierre/argile/mortier — gargouilles,
  sculptures, pièces architecturales) : Gargoyle=30 fragments, Cake=11,
  Brick=6, Venus=7, Sculpture=15, Head=12, **Forma Urbis Romae=1186**. Format
  PCD (nuage de points **+ normales**, pas de mesh) ou CDM (scan brut Minolta).
- **Aucune pose GT / label de réassemblage mentionné sur la page.** Contrairement
  à Breaking Bad, on ne pourra probablement pas calculer `RotErr`/`Pose@30`
  directement — toute l'évaluation quantitative du projet en dépend. À vérifier
  en premier (peut-être disponible via une publication associée, ou à construire
  manuellement sur un sous-ensemble de paires pour une évaluation qualitative
  limitée) avant d'investir du temps d'implémentation dessus.
- Aucun objet n'a exactement 2 fragments (le cas simple de la Phase 5A) — les
  plus petits sont à 6-7 fragments (Brick, Venus). Le confond multi-voisins
  (Phase 2D/2B) reviendrait donc immédiatement sur ce dataset, contrairement à
  Breaking Bad filtré à 2 fragments.

### Diagnostics post-réouverture (2026-07-20, avant tout code d'amélioration)

Deux diagnostics ajoutés à `phase5a_depthmap_matching.py` pour trancher entre
plusieurs causes possibles avant d'investir dans les 5 pistes d'amélioration —
esprit habituel du projet (diagnostic avant grosse implémentation).

**Diagnostic A — Stratification par planéité (`--max_planarity 0.333`, N=2415,
`joint`, GT) : l'hypothèse du tuteur ("plus de courbure aiderait") est
CONTREDITE, pas confirmée.**

```
Bin          N    RotErr°  Pose@30   Pose@15
<0.02        23     93.73   26.09%    21.74%
0.02-0.05    69     85.52   34.78%    17.39%
0.05-0.10   225     92.37   28.44%    15.11%
0.10-0.15   211    101.32   24.17%     9.95%
0.15-0.20   121     94.68   24.79%    10.74%
0.20+        59    116.45   10.17%     3.39%
```

`Pose@15` (critère strict) baisse quasiment de façon monotone en s'éloignant de
la platitude : 21.74%→17.39%→15.11%→9.95%→10.74%→**3.39%**. `random` reste à 0%
dans toutes les tranches (pas un artefact d'évaluation). **Plus une face est
courbée, moins bon est le résultat — l'inverse de ce qui était attendu.**
Explication retenue : la méthode entière (PCA + plan unique + rasterisation)
suppose une face quasi-plane ; quand la courbure augmente, cette hypothèse se
dégrade et ce qui ressemble à "plus de relief" est probablement du bruit de
projection PCA, pas un vrai signal de complémentarité exploitable. **Conclusion :
chercher des faces plus courbées ne suffira pas avec la représentation
actuelle (PCA + plan unique) — il faudrait une représentation qui gère mieux
la non-planéité** (ex. paramétrisation par self-organizing map, cf. discussion
du 2026-07-20, plutôt que la PCA classique).

**Diagnostic B — `oracle_overlap_frac` (recouvrement à la VRAIE pose GT, sans
recherche) : le vrai goulot est en amont, pas dans la recherche.**

```
Strategy    OvlpFrac (trouvé)   OracleOvlp (plafond théorique)   Ratio
gt                0.659                    0.758                 87%
thresh0.3         0.624                    0.722                 86%
random            0.764                    0.493                  —  (recherche gagne un mauvais recouvrement à une pose fausse)
```

Deux faces d'une même fracture partagent le même contour par construction —
l'overlap DEVRAIT tendre vers 1 à la vraie pose. Il plafonne à **0.758**, pas 1.0
— ~24% du plus petit contour ne correspond jamais, même dans le meilleur des
cas (bruit d'échantillonnage indépendant entre les deux faces, seuil GT/CNN
pas parfaitement symétrique). La recherche, elle, atteint déjà ~87% de ce
plafond — **améliorer l'algorithme de recherche a un potentiel limité tant que
le plafond lui-même (la correspondance entre masques) n'est pas amélioré.**

Sur `random` : la recherche trouve un BON recouvrement (0.764) à une pose
FAUSSE (RotErr≈132°) — confirme que le recouvrement seul peut être trompé par
la silhouette générale du fragment, pas seulement par la fracture spécifique.
Argument supplémentaire pour ne jamais utiliser `overlap_only` seul en
production, cohérent avec le choix de `joint` comme défaut.

### Roadmap des améliorations, priorisée par les diagnostics ci-dessus (2026-07-20)

| # | Piste | Justification (diagnostic) | Lien avec les 5 pistes du tuteur |
|---|---|---|---|
| 1 | Densifier/lisser les masques de fracture (interpolation, seuil plus cohérent entre les 2 côtés) | Diagnostic B : le plafond réel est 0.758, pas 1.0 — le levier avec le plus de marge | "Extrapolation si trop plat" |
| 2 | Stratifier aussi par `n_frac_pts` (déjà loggé, pas encore agrégé) | Vérifier si la sparsité (p25=56 pts) explique une partie du plafond à 0.758 | Prépare la piste 3 |
| 3 | Résolution adaptative à la densité + fenêtre glissante (grille plus petite pour fragments épars) | Découle de la discussion du 2026-07-20 sur la taille fixe des depth maps | "Plus de résolution(s)" + "sliding de fenêtre" |
| 4 | Représentation non-planaire pour les faces courbées (ex. SOM au lieu de PCA+plan unique) | Diagnostic A : la courbure nuit avec la représentation actuelle | Nouvelle piste, cohérente avec la conviction du tuteur mais pas dans sa liste initiale |
| 5 | Plus de features par pixel (normales, features CNN Step15) | Peu risqué, mais n'attaque pas le vrai goulot (le plafond de correspondance) | "Plus de features" |
| 6 | Modèle appris sur les depth maps | Le plus ambitieux — plafonné par le même problème amont tant qu'il n'est pas résolu | "Modèle appris" |

**Ordre recommandé : 1 et 2 d'abord** (diagnostics/améliorations peu coûteux qui
attaquent directement le goulot identifié), **puis 3-6** (plus de travail
d'implémentation, gain potentiellement plafonné par le même problème amont
sinon).

### Piste 1, tentative 1 — dilatation binaire : NÉGATIF, confirmé (2026-07-20)

**Test en 2 temps, esprit diagnostic avant implémentation :**

**Étape A — sweep de dilatation sur `oracle_overlap_frac` seul** (`--dilate_sweep`,
diagnostic pur, ne touche pas la recherche) : dilater le masque de fracture de
1 pixel fait remonter le plafond théorique `gt` de 0.757 à **0.992** — semblait
confirmer que l'écart au plafond 1.0 est surtout du bruit de discrétisation/
échantillonnage. **Mais signal d'alerte immédiat** : `random` en profite presque
autant (0.430→0.830) — l'écart discriminant `gt` vs `random` s'effondre avec la
dilatation (0.327→0.162 à d=1, →0.027 à d=4). Suspicion soulevée avant même de
tester en conditions réelles : la dilatation élargit la tolérance pour tout le
monde, elle ne récupère pas spécifiquement une vraie correspondance.

**Étape B — dilatation branchée dans le pipeline réel** (`--dilate_px 1`,
appliquée à `rasterize()` avant la recherche/le score, pas seulement au
diagnostic) — verdict : **la suspicion était fondée.**

```
                    d=0        d=1 (pipeline)
OracleOvlp (gt)     0.757  →   0.974   ↑↑ (comme prévu)
OvlpFrac trouvé     0.648  →   0.838   ↑
Pose@30 (gt)       20.87%  →  18.28%   ↓ (n'améliore pas, empire même légèrement)
Pose@15 (gt)       10.43%  →   9.68%   ↓
RotErr (gt)        105.90° → 110.72°   ↑ (pire)
```

**Le plafond de recouvrement explose, mais la métrique qui compte
(`Pose@30`/`Pose@15`) ne s'améliore pas — elle baisse légèrement.** La
dilatation binaire élargit la tolérance sans ajouter de vraie information :
elle donne à la recherche plus de positions "à peu près plausibles" à
départager, ce qui la rend moins précise plutôt que plus précise.

**Conclusion : la dilatation binaire (piste 1, tentative 1) est un pansement,
pas une solution — confirmé empiriquement, pas juste par intuition.** Rejetée
comme approche. Code conservé (`--dilate_px`, `--dilate_sweep`) pour
comparaison/diagnostic futur, mais pas comme réglage par défaut recommandé.

**Piste ouverte pour une "vraie" densification** (discussion en cours,
2026-07-20) : au lieu d'élargir aveuglément le masque existant (dilatation),
ajouter de l'information positionnelle — ex. **splat gaussien** (chaque point
contribue à plusieurs cases voisines pondéré par sa distance réelle, au lieu
d'un remplissage binaire tout-ou-rien) ou interpolation pondérée par distance
dans les trous entourés de cases valides. Contrairement à la dilatation, ces
approches utilisent la position réelle des points plutôt que de simplement
étendre un masque déjà là — reste à cadrer et tester avec le même protocole
(sweep diagnostic sur `OracleOvlp` d'abord, avec vérification systématique
que `random` n'en profite pas autant que `gt`, AVANT de brancher dans le
pipeline réel comme pour la dilatation).

### Piste 1, tentative 2 — splat gaussien : NÉGATIF aussi, confirmé (2026-07-20)

**Motivation.** Après le rejet de la dilatation binaire, hypothèse que le
problème vient de la nature "tout-ou-rien" de la dilatation (elle étend le
masque sans utiliser la position réelle des points). Le **splat gaussien**
floute le canal "nombre de points par case" avec un noyau gaussien
(`scipy.ndimage.gaussian_filter`) au lieu d'une dilatation binaire — flou
correct d'une moyenne pondérée (numérateur et dénominateur floutés
séparément, pas la moyenne déjà calculée).

**Implémenté dans `phase5a_depthmap_matching.py`** : `points_to_valid_mask()`
(helper partagé entre `rasterize()` et `oracle_overlap_frac()`, refactorisée
pour prendre `frac_j` — points bruts — au lieu d'un `valid_j` précalculé, afin
que i et j soient traités de façon strictement symétrique). `--gaussian_sigma_px`
(pipeline réel) + `--gaussian_sweep` (diagnostic) + table `SWEEP SPLAT GAUSSIEN`.

**Diagnostic isolé (sweep sur `OracleOvlp` seul) — plus encourageant que la
dilatation à faible sigma :**
```
σ=0.0 : gt=0.757  random=0.442  →  écart = 0.315
σ=0.5 : gt=0.948  random=0.629  →  écart = 0.319   (maintenu, léger mieux)
σ=1.0 : gt=0.989  random=0.765  →  écart = 0.224   (érosion commence)
σ=2.0 : gt=0.881  random=0.840  →  écart = 0.041   (quasi effondré, comme la dilatation)
```
Bon signe à σ=0.5 : le plafond monte nettement (0.757→0.948) sans éroder
l'écart gt-random. Au-delà de σ=1.0, même défaut que la dilatation.

**Test en conditions réelles (`--gaussian_sigma_px 0.5`, dans le vrai
pipeline) — verdict : NÉGATIF, comme la dilatation :**
```
                  σ=0 (référence)   σ=0.5
OvlpFrac trouvé        0.655         0.839   ↑
Pose@30 (gt)          20.87%        19.08%   ↓
Pose@15 (gt)          10.43%         9.92%   ↓
RotErr (gt)          105.90°       107.57°   ↑ (pire)
```
Quasiment identique au résultat négatif de la dilatation (`d=1` : 18.28%/9.68%).

**Explication (discussion avec l'utilisateur, 2026-07-20) : un bon
recouvrement est une condition nécessaire mais pas suffisante.** Il dit "ces
deux formes ont à peu près la même taille et position", pas "elles s'emboîtent
précisément ICI et pas ailleurs" — analogie du puzzle : flouter une pièce fait
disparaître les petites encoches (le détail fin qui distingue le bon angle des
autres), donc deux formes floutées se recouvrent bien à plein d'angles
différents, pas seulement au bon. **Les deux sources possibles de précision
fine (le relief, déjà faible car les surfaces sont trop planes ; le contour
précis) sont affaiblies simultanément par le lissage** — d'où le double rejet.

**Conclusion : piste 1 (densification) rejetée dans ses deux versions
testées.** Le diagnostic isolé (`OracleOvlp` gt vs random à la vraie pose) ne
suffit pas à prédire la performance réelle de la recherche — il ne teste
jamais si le lissage garde la vraie pose plus précise que les poses fausses
très proches, qui est la vraie question que la recherche doit résoudre.

---

### Outil de visualisation diagnostique (2026-07-20)

Après deux résultats négatifs sur les métriques agrégées, besoin de regarder
concrètement des paires plutôt que rester sur des moyennes qui cachent le
mécanisme d'échec. **`scripts/phase5a_visualize_pair.py`** — réutilise
directement les fonctions du pipeline réel (aucune réimplémentation), génère
par paire une figure PNG à 8 panneaux : nuages de points complets (1-2),
points de fracture surlignés (3-4), depth maps 2D (5-6), overlay 3D de la pose
trouvée (7) vs la vraie pose GT (8). `--filter success/fail` cible
spécifiquement des réussites ou des échecs.

**Bug trouvé et corrigé le jour même :** `bp_pairs` contient des tuples
`(b, p)`, pas `(k, p)`. Le code initial faisait `(k0, p0), (k1, p1) = bp_pairs`,
assignant `b` (toujours 0 en `batch_size=1`) à `k0`/`k1` au lieu du vrai indice
de fragment dans `pc_per_k`/`gt_per_k`/`score_per_k` — **`raw_i` et `raw_j`
pointaient donc sur LE MÊME fragment** pour toute paire. Explique un premier
run à 0 succès trouvés sur tout le split (`--filter success`, alors que le
pipeline principal mesure ~27% de réussite sur `gt`) : comparer un fragment à
lui-même ne peut jamais matcher la vraie pose relative. Fix : `k` = position
dans `enumerate(bp_pairs)`, pas `b`. Toutes les figures générées avant ce fix
sont invalides et ont été régénérées.

---

### Découverte majeure — hypothèse du maillon faible, puis root cause identifiée (2026-07-20)

**Inspection visuelle de l'objet#21** (après le fix du bug ci-dessus) : sur
une paire à 2 fragments, fragment i (le reste de l'objet) a **131 points de
fracture** (depth map quasi vide, 124/4096 pixels valides), fragment j (le
petit bout cassé) en a **2832** (depth map bien remplie, 1334/4096) — ratio
~21x sur la MÊME fracture, alors que les deux fragments sont échantillonnés
avec le même budget total de points (5000 chacun).

**Explication physique :** échantillonnage `uniform` (`BreakingBadUniform`,
utilisé par le CNN Step15) — budget de points FIXE par fragment, indépendant
de sa taille. Un grand fragment a sa fracture diluée dans une immense surface
totale (peu de points y tombent) ; un petit fragment a sa fracture qui occupe
quasiment toute sa surface (beaucoup de points y tombent).

**Test de l'hypothèse — stratification par `MIN(n_i, n_j)` (le côté le plus
pauvre), pas la moyenne** (qui cachait le déséquilibre : (131+2832)/2≈1481,
tranche "dense" alors que la paire échoue). Ajouté `frac_pts_min_stratification()`
dans `phase5a_depthmap_matching.py`. **Résultat (N=2415, `joint`) : confirmation
PARTIELLE, pas dominante :**
```
Bin (gt)     Pose@30   Pose@15
50-100        28.57%     9.52%
100-200       26.75%    12.72%
200-500       28.43%    15.69%
500-1000      25.00%    18.75%
```
`Pose@30` reste plat (~26-28%), mais `Pose@15` (précision fine) double presque
(9.52%→18.75%) — et pour `thresh0.3` (condition réelle), `Pose@30` progresse
nettement sur la partie fiable (14.81%→25.26%). Le maillon faible affecte
réellement la précision, mais n'est pas, à lui seul, LE facteur dominant sur
la plage testée (données `gt` manquantes au-delà de 1000 points).

**Root cause identifiée : GARF a déjà la solution, le CNN ne peut juste pas
l'utiliser.** Vérifié dans le code : `BreakingBadWeighted`
(`assembly/data/breaking_bad/weighted.py`) répartit le budget total de points
**proportionnellement à l'aire réelle de chaque fragment** — exactement la
parade cherchée, et c'est le sampling **par défaut de GARF**
(`configs/data/breaking_bad.yaml`). Le CNN Step15 force `sample_method: uniform`
(`configs/experiment/cnn_ablation_shared.yaml`) parce que son architecture de
backprojection bilinéaire exige un nombre de points constant par fragment —
contrainte technique du CNN, pas une limitation du dataset/repo.

**Test rendu possible sans toucher au CNN :** `fracture_surface_gt` est un
label géométrique du dataset (confirmé dans
`assembly/models/cnn_segmentation_model.py` — le CNN ne fait que le relire,
aucun calcul dérivé), donc la stratégie `gt` peut être testée directement en
`sample_method=weighted`, sans jamais charger le CNN.

**Implémenté : `scripts/phase5a_weighted_gt_check.py`.** Charge le datamodule
via `load_config_and_model(..., model_type="garf")` (force déjà
`sample_method=weighted` + `batch_size=1`, mécanisme existant réutilisé tel
quel). `extract_gt_variable()` découpe `pointclouds`/`fracture_surface_gt` par
les offsets réels de `points_per_part` (PAS `extract_gt_for_valid_frags`
existante, qui suppose un reshape `(B,P,num_pts)` invalide en weighted —
tailles de fragments variables). Vérifié : formule de `scale` identique entre
`BreakingBadUniform`/`BreakingBadWeighted` (même sémantique, même alignement
des slots) — pas un bug de ce côté.

**Résultat définitif (N=107 valides sur 2545 objets scannés, comparable à la
référence uniform N=511) — POSITIF, confirmé à densité égale, pas juste en
moyenne globale :**
```
Bin           uniform Pose@30   weighted Pose@30   ratio
100-200          26.75%             55.56%          ~2.1x
200-500          28.43%             47.50%          ~1.7x
500-1000         25.00%             28.57%          ~1.1x

Global : uniform N=511  RotErr=94.95°  Pose@30=27.59%  Pose@15=13.70%
         weighted N=107 RotErr=69.67°  Pose@30=48.60%  Pose@15=29.91%
```
**`Pose@30` quasi doublé, `Pose@15` plus que doublé — à densité de points
comparable, pas par simple déplacement de la distribution.** C'est le
résultat le plus solide de la Phase 7 à ce jour.

**Caveat à garder** : le taux de skip `no_overlap_3d` reste élevé en weighted
(~90% des paires ayant passé les filtres planéité/points) — le sampling
pondéré améliore beaucoup la QUALITÉ des poses trouvées, mais moins de paires
produisent une pose du tout (rendement plus faible). Compromis à creuser, mais
qui n'enlève rien au résultat principal.

**Limite connue, pas encore résolue : ça ne marche que pour `gt`.** Pour
`thresh0.3` (condition réelle), le CNN reste contraint à `uniform` — il
faudrait un pipeline en deux temps (CNN sur échantillon uniforme pour la
prédiction, puis ré-échantillonnage dense/pondéré de la zone détectée
spécifiquement pour le matching) — pas encore implémenté.

### Correction de la sémantique `num_points_to_sample` (2026-07-20)

**Hypothèse posée avant de tester (discipline habituelle du projet) :**
`num_points_to_sample` n'a pas le même sens en `uniform` et en `weighted` —
en `uniform` c'est un budget **par fragment** (5000/fragment, donc 10000 au
total sur un objet à 2 fragments) ; en `weighted` c'est un budget **par
objet entier**, réparti par aire. Laisser la valeur de config (5000) en
`weighted` divise donc par ~2 la densité totale de points par rapport à
`uniform`, sans le vouloir — suspecté d'expliquer le taux élevé de
`too_few_points`/`no_overlap_3d` du premier test `weighted` (N=107).

**Ajouté à `phase5a_weighted_gt_check.py` : `--num_points_to_sample`**
(override de `cfg.data.num_points_to_sample` après `load_config_and_model`,
avant `instantiate`).

**Résultat (`--num_points_to_sample 10000`, N=2545 scanné) — hypothèse
confirmée pour le RENDEMENT, avec une nuance sur la qualité agrégée :**
```
                  5k (par objet)    10k (par objet)   uniform (référence)
N_ok                   107               426                511
too_few_points         45%               20%                 —
no_overlap_3d (rel.)   ~90%              ~75%                 —
RotErr                69.67°            86.55°             94.95°
Pose@30               48.60%            39.20%             27.59%
Pose@15               29.91%            27.23%             13.70%
```
`N_ok` quadruple (107→426), `too_few_points` divisé par ~2 — confirme le
diagnostic sur le rendement. La qualité agrégée baisse par rapport au run à
N=107 (`RotErr`/`Pose@30` moins bons), **mais ce n'est probablement pas une
vraie dégradation** : avec N=426 vs 107, le mélange des tranches change
(beaucoup plus de paires dans `200-500`, tranche moins performante,
auparavant sous-représentée) — même piège de confusion qu'on a appris à
éviter toute la journée. **Dans tous les cas, largement au-dessus de la
référence `uniform`** (39.20% vs 27.59%, 27.23% vs 13.70%).

**Indice (faible N, à confirmer) en faveur de la résolution adaptative :**
la tranche `1000+` tombe à **0%** (N=9). Hypothèse : à haute densité de
points, la grille fixe `RESOLUTION=64` sur-agrège plusieurs points très
différents dans la même case (perte de détail par écrasement) — symétrique
au problème de sparsité (trop peu de points → trop de cases vides). Si
confirmé à plus grande échelle, ça validerait la piste résolution
adaptative dans les deux sens (pas assez ET trop de points sont mauvais
pour une grille fixe), pas seulement comme réponse à la sparsité.

### Discussion de direction (2026-07-20, avec l'utilisateur) — architecture en deux étages

**Décision d'architecture pour la suite du pipeline, actée avec l'utilisateur :**
ne pas utiliser le matching point-à-point dès le départ (les fragments ne
sont pas alignés dans l'espace 3D au début — exactement pourquoi Phase
3A/4A/4B avaient échoué, aucune bonne initialisation). Le garder pour un
**raffinement final**, après une première estimation grossière par
depth-map matching. Schéma retenu :
```
1. Depth-map matching (avec résolution adaptative) → estimation grossière de pose
2. [à calibrer] Raffinement point-à-point (ICP ou équivalent) → pose précise
```

**Un test à faire plus tard (pas maintenant, mais retenu comme prochaine
étape après stabilisation du depth-map matching) : le bassin de convergence
du raffinement point-à-point.** Prendre des paires GT, perturber la vraie
pose de 10°/20°/30°/45°, mesurer jusqu'à quel angle un raffinement simple
(ICP ou équivalent) reconverge encore vers la bonne réponse. Objectif :
savoir quel niveau de précision viser en sortie de l'étape 1 (est-ce que
"~30°" suffit, ou faut-il viser plus précis avant de passer la main au
point-à-point) — sans ce calibrage, on ne sait pas quel est le vrai objectif
de qualité pour le depth-map matching.

**Ordre de travail retenu :** stabiliser/améliorer le depth-map matching
(résolution adaptative en priorité) jusqu'à un résultat jugé satisfaisant,
**puis seulement** faire le test de bassin de convergence pour calibrer
l'objectif exact, **puis** construire le raffinement point-à-point.

### Audit complet des rejets, avant tout fix (2026-07-20)

**Recadrage méthodologique de l'utilisateur, avant d'imaginer un fix :**
comprendre PRÉCISÉMENT pourquoi les paires sont rejetées, pas juste
supposer. Argument de fond : deux fragments d'une même cassure ont
forcément une vraie surface de fracture des deux côtés — aucune paire
n'est structurellement impossible ; un rejet vient d'un choix de seuil de
notre pipeline, pas d'une absence physique de signal. Objectif premier
avant tout : maximiser le nombre de paires traitées (élimine les rejets
artificiels), ce qui est aussi un préalable pour faire confiance aux
métriques de qualité elles-mêmes (calculées uniquement sur les paires
survivantes, potentiellement biaisées si ce sous-ensemble n'est pas
représentatif).

**Implémenté : `scripts/phase5a_skip_audit.py`.** Contrairement aux
scripts précédents, ne s'arrête JAMAIS aux seuils officiels
(`MIN_FRAC_POINTS=50`, `max_planarity=0.15`, `MIN_OVERLAP_PIXELS=20`) — pousse
chaque paire aussi loin que possible (plancher numérique pur `ABS_MIN_POINTS=5`),
enregistre à la fois si chaque seuil AURAIT rejeté la paire (`would_skip_*`)
et le résultat réel obtenu malgré tout.

**Résultat (N=1487 paires auditées, `num_points_to_sample=10000`, `joint`) :**

**Funnel réel (seuils ignorés) :**
```
unusable_too_few         4  ( 0.3%)   -- vraiment dégénéré, négligeable
no_correspondence      1094  (73.6%)   -- LE vrai goulot
pose_computed           389  (26.2%)   -- dont 134 (34.4%) réussissent Pose@30
```
**Taux de succès réel sur l'ensemble des paires : ~9% (134/1487)** — bien
plus bas que les chiffres agrégés précédents (`Pose@30`≈40-48%), qui étaient
calculés uniquement sur le sous-ensemble survivant aux filtres, pas sur
l'ensemble des paires réelles.

**Coût de chaque seuil officiel (paires rejetées qui auraient quand même
réussi si on ne les avait pas rejetées) :**
```
too_few_points (50 pts)   : 227 rejetées → 0%    auraient réussi  → seuil JUSTIFIÉ
sparse_dmap (20 px)       :  38 rejetées → 0%    auraient réussi  → seuil JUSTIFIÉ
too_curved (planarity 0.15): 161 rejetées → 16.8% auraient réussi  → seuil TROP STRICT
```
**Seul `too_curved` est un vrai artefact** — `too_few_points` et
`sparse_dmap` sont validés comme non-gaspilleurs (ils ne rejettent
quasiment que des cas qui n'auraient de toute façon rien donné).

**Distributions par résultat final (percentiles) — où est la vraie
frontière :**
```
                    n_pix (après rasterisation)   planarity moyenne   best_overlap_frac
Pose@30 succès              médiane 370                médiane 0.11        médiane 0.25
Pose@30 échec               médiane 368 (≈ succès!)    médiane 0.11 (≈ succès!) médiane 0.20
no_correspondence           médiane  88 (4x moins)     médiane 0.03 (bien + plat) médiane 0.12
```

**Deux enseignements distincts et importants :**
1. **Ce qui distingue "on obtient une pose" de "on n'obtient rien"** (`no_correspondence`
   vs `pose_computed`) : la **densité** (n_pix ~4x plus bas) et le **relief**
   (planéité 3x plus plat). C'est exactement la cible pour la résolution
   adaptative : les paires qui échouent totalement remplissent la grille
   64×64 à ~2% (88/4096), celles qui réussissent à ~9% (370/4096) — une
   grille dimensionnée à la densité réelle du fragment devrait faire
   franchir ce seuil à beaucoup plus de paires.
2. **Ce qui distingue "pose juste" de "pose fausse"** (une fois qu'on EN a
   une) n'est PAS la densité ni la planéité (quasi identiques entre succès
   et échec) — **c'est `best_overlap_frac`, qui discrimine proprement à
   tous les percentiles** (0.25 vs 0.20 vs 0.12). C'est un signal de
   confiance exploitable même sans connaître la vérité terrain (utile pour
   un futur filtre de confiance en condition réelle `thresh0.3`).

**Implication directe pour la résolution adaptative :** ne pas viser une
grille plus fine ou plus grossière au hasard — viser un dimensionnement qui
fait passer le taux de remplissage des paires actuellement `no_correspondence`
(~2%) vers celui des paires qui réussissent (~9%), par exemple en
dimensionnant la grille pour un nombre de points par case cible plutôt
qu'un nombre de cases fixe.

**Prochaine action concrète, dans l'ordre (diagnostic avant grosse implémentation) :**
1. **Assouplir/retirer le seuil `too_curved` (planarity > 0.15)** — gain
   rapide et déjà quantifié par l'audit (16.8% des paires rejetées
   auraient réussi), contrairement à `too_few_points`/`sparse_dmap` qui
   sont validés comme légitimes. Remplacer le couperet dur par une
   pénalité continue (cohérent avec la conclusion déjà tirée du diagnostic
   planéité : pénaliser en douceur plutôt que couper).
2. **Résolution adaptative sur les depth maps** — piste prioritaire,
   maintenant avec une cible chiffrée précise grâce à l'audit : faire
   passer le taux de remplissage des paires `no_correspondence` (~2%,
   88/4096 px) vers celui des paires qui réussissent (~9%, 370/4096 px) —
   dimensionner la grille pour une densité de points par case cible,
   plutôt qu'un nombre de cases fixe (`RESOLUTION=64` aujourd'hui).
3. Exploiter `best_overlap_frac` comme signal de confiance (discrimine
   proprement succès/échec/no_correspondence) — utile notamment pour un
   futur filtre de confiance en condition réelle (`thresh0.3`, sans GT).
4. Réfléchir/cadrer le pipeline en deux temps pour rendre le gain `weighted`
   exploitable en condition réelle (`thresh0.3`), pas seulement à l'oracle GT.
5. Test de bassin de convergence du raffinement point-à-point (reporté,
   après stabilisation du depth-map matching — voir discussion ci-dessus).
6. Dataset réel TU Wien (reporté, moins prioritaire vu ces résultats) :
   télécharger un objet simple (Brick ou Venus), vérifier l'existence d'une
   pose GT exploitable, mesurer la planéité, tester le CNN Step 15 en
   zero-shot.

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
pas d'amélioration). **4B fait et clos** (cross-attention, run initial + run prolongé
avec `--grad_clip 1.0`, conclusion : ne dépasse pas C1/4A, instabilité/collapse
dustbin pas résolu par le clipping — voir conclusion ci-dessus). **4D cadré (2026-06-28, fallback, ci-dessus), aucun code écrit.** **5A.0 PASSÉ (2026-06-28, everyday/val, n=3803 objets 2-frags, médiane planéité=0.043).**
**Phase 5A RÉOUVERTE — POSITIF MAIS PARTIEL, CONFIRMÉ À GRANDE ÉCHELLE
(révisé 2026-07-20, run initial du 2026-07-08 invalidé pour cause d'échantillon
trop petit N=21).** Bug trouvé dans le score (`-CC/overlap` divisait par le
recouvrement au lieu de le récompenser) ; fix `--score_mode joint` (relief ×
recouvrement du contour). Run définitif (N=2415 objets 2-frags, 63% du val,
N=447-522 paires valides) : `Pose@30` (gt) **4.47%→22.03%→27.59%** en
relief→overlap_only→joint — dépasse la référence Phase 2 `gt_edge` (9.6%),
quasi 3x mieux. `relief` s'est révélée INSTABLE (9.09% sur N=44 → 4.47% sur
N=447, signature d'un score bruité), `joint` au contraire s'améliore avec plus
de données (21.15%→27.59%), signe d'un vrai signal qui converge. Mais reste
partiel : ~72% des paires échouent encore à Pose@30, ~86% à Pose@15/0.05. Pas
une solution, un signal réel et robuste qui triple quasiment le meilleur
résultat géométrique du projet. Phase 5B toujours en attente (pas relancée, 5A
trop partielle pour justifier l'extension à 3+ fragments pour l'instant).

**Phase 4D CLOSE — POSITIF (2026-07-16).** AUC=0.798 (thresh0.3), AP=0.749,
P@k=0.733 — best @ epoch 83/500. thresh0.3 > random (+17pp AUC à convergence).

## Phase 6 — Compatibility-guided pose refinement (cadrage, 2026-07-16)

**Pourquoi.** Le diagnostic est maintenant net : segmentation OK (Step 15),
compatibilité de paire OK (4D, AUC≈0.80), pose KO (3A/4A/4B/5A tous négatifs).
Décision explicite du tuteur et de l'utilisateur : ne plus retenter une nouvelle
variante du matching point-à-point (3e architecture, plus d'epochs...) — pivoter vers un
**raffinement de pose en deux étages** (identifier les bonnes paires via 4D, puis
raffiner une pose candidate localement) plutôt que ré-estimer la pose from scratch.

**Risque identifié et acté avant de coder quoi que ce soit :** un P@k=0.733 sur 4D
veut dire ~27% des vrais voisins manqués si on prend une décision dure top-k — sur
un objet à 8-10 fragments, ça peut suffire à casser tout l'assemblage global (erreur
de voisinage → mauvaise pose → contamination du reste). **4D doit donc être utilisé
comme filtre souple à haut recall (candidate generator), jamais comme décision finale
des voisins.**

**Sous-phases, dans cet ordre strict (pas de refinement avant d'avoir la table
Phase 6.0) :**
- **Phase 6.0 — Recall@k sweep de 4D** (implémenté ci-dessous) : mesure si 4D peut
  fournir une shortlist top-k/2k/3k/5k à haut recall, PAR FRAGMENT (pas sur les
  paires mélangées comme le Prec@k de la 4D), stratifié par nombre de fragments de
  l'objet (2 / 3-5 / 6-10 / 11+ — les objets à 2 fragments sont non-informatifs,
  rapportés à part, jamais utilisés pour la décision).
- **Phase 6A — Refinement oracle** (pas encore codé) : sur les vraies paires GT
  uniquement, pose GT perturbée (5°/15°/30°/60° rotation + bruit translation),
  objectif local (Chamfer symétrique fracture + opposition de normales + pénalité
  de pénétration/overlap) → est-ce qu'on reconverge vers la pose GT ? Question :
  "un objectif local peut-il améliorer une pose déjà proche ?", indépendamment de
  4D — sert de borne haute avant de brancher quoi que ce soit dessus.
- **Phase 6B — Pipeline réel** (seulement si 6.0 ET 6A passent) : 4D comme shortlist
  large (pas top-k strict) → refinement local sur les candidats → score géométrique
  + cohérence globale pour la décision finale. Explicitement PAS "4D choisit les
  voisins puis on assemble greedily" (risque d'accumulation d'erreurs déjà identifié).

**Critères de décision Phase 6.0 (objets 3+ fragments uniquement, stratégie
thresh0.3 = condition réelle, gt = référence oracle) :**
```
Go  6A  si Recall@3k >= 90% OU Recall@5k >= 95%,
        ET avg_kept_ratio@5k < ~0.8 (sinon le "haut recall" est trivial —
        garder quasi tout le monde donne recall≈100% sans être une shortlist utile)
No-go   si Recall@5k < 90%, ou si top-5k dégénère vers quasi tous les fragments
```

**Implémenté (2026-07-16) : `scripts/phase6_0_recall_sweep.py`.** Réutilise
directement l'infrastructure 4D (`aggregate_fragment`, `PairCompatibilityMLP`,
CNN Step 15 figé) — pas de ré-entraînement. Nécessite un checkpoint MLP sauvegardé :
ajout de `--model_out` à `phase4d_pair_compatibility.py` (le script 4D original ne
sauvegardait aucun poids, seulement un JSON de résultats — corrigé). Pour chaque
fragment i d'un objet, classe tous les autres fragments du même objet par score MLP
symétrique décroissant (même score que le Prec@k 4D, calculé une fois par paire non
ordonnée), mesure si les n_pos vrais voisins GT sont dans le top n_pos/2·n_pos/
3·n_pos/5·n_pos. Sortie : table par bucket de complexité (Recall@k/2k/3k/5k,
kept_ratio@5k, n_frags) + verdict go/no-go automatique par stratégie.

### Conclusion Phase 6.0 — NO-GO (2026-07-16)

**Run :** `everyday/val`, `mlp_ckpt` best_epoch=83 (AUC=0.798), stratégies
`thresh0.3` et `gt`.

```
Stratégie thresh0.3 :
Group           Recall@k  Recall@2k  Recall@3k  Recall@5k  KeptRatio@5k  N_frags
2                  100.0%     100.0%     100.0%     100.0%        100.0%     7426
3-5                 85.0%      98.4%      99.7%     100.0%         99.1%     8926
6-10                58.3%      86.7%      96.2%      99.5%         97.4%     6659
11+                 45.8%      70.6%      83.8%      94.5%         82.9%     9497
All multi (3+)      63.1%      84.8%      92.8%      97.8%         92.5%    25082

Stratégie gt : quasi identique (63e décimale près) sur tous les buckets.
```

**Lecture (pas le verdict brut du script, qui confond deux effets différents) :**

Le `kept_ratio@5k` proche de 100% sur 3-5/6-10 est un **artefact du multiplicateur
×5**, pas une dégénérescence du classifieur : dès que l'objet a peu de fragments,
`5×n_pos` (5× le degré GT) dépasse `n_other`, donc "garder le top-5k" revient
mécaniquement à garder presque tout l'objet, quelle que soit la qualité du score —
ces buckets ne sont pas informatifs pour juger 4D comme filtre.

**Le seul bucket réellement discriminant est 11+ fragments — celui où un filtre
serait le plus utile** (objets complexes, exactement la préoccupation soulevée dès
la Phase 1 : "le recall peut s'effondrer sur les objets complexes"). Résultat :
`Recall@3k=83.8%` (sous le seuil 90%), `Recall@5k=94.5%` (juste sous 95%),
`KeptRatio@5k=82.9%` (il faut garder ~83% des candidats pour atteindre ce recall —
filtrage réel mais modeste). `gt≈thresh0.3` confirme une nouvelle fois que le CNN
n'est pas en cause — c'est le MLP de compatibilité qui plafonne sur les objets
complexes.

**Verdict (critère strict acté avant le run) : NO-GO.** 83.8%/94.5% sur 11+ ne
passe pas les seuils 90%/95%. 4D reste un résultat de stage solide en tant que tel
(compatibilité de paire, AUC≈0.80) mais **ne peut pas servir de shortlist à haut
recall pour guider un refinement de pose sur les objets complexes** — décision
prise pour éviter d'empiler les erreurs (voir mise en garde du tuteur, Phase 6
cadrage). **Phase 6.0 close. Phase 6A/6B non lancées, branche fermée.**

## Prochaine action concrète (mise à jour 2026-07-20, post-présentation tuteur)

**Périmé, ne pas suivre :** le paragraphe précédent (2026-07-16) disait "toutes
les phases closes, rédaction du rapport" — dépassé depuis par la réouverture de
la Phase 5A (2026-07-20) et la nouvelle direction validée avec le tuteur le
même jour (voir **Phase 7** ci-dessus, section complète).

**État réel au 2026-07-20 (fin de journée) :**
- Phase 4D (compatibilité fragment-fragment, AUC≈0.80) : close, positive, mise
  de côté pour plus tard (pas abandonnée) — le tuteur a explicitement recentré
  sur la pose entre paires déjà connues comme adjacentes.
- Phase 5A (depth-map matching) : rouverte, positive mais partielle. **Deux
  pistes de densification rejetées** (dilatation binaire, splat gaussien —
  toutes deux améliorent le recouvrement mesuré sans améliorer `Pose@30` en
  pratique). **Root cause du vrai goulot identifiée et corrigée pour `gt` :**
  le CNN force un échantillonnage `uniform` (budget de points fixe par
  fragment) alors que GARF utilise déjà par défaut un échantillonnage
  `weighted` (proportionnel à l'aire réelle) — en testant `weighted` sur la
  stratégie `gt` (qui n'a pas besoin du CNN), `Pose@30` passe de 27.59% à
  **48.60%** et `Pose@15` de 13.70% à **29.91%**, confirmé à densité de points
  égale (pas un artefact de moyenne). **C'est le résultat le plus solide de
  toute la Phase 7.** Reste à étendre ce gain à `thresh0.3` (condition réelle
  avec CNN, pipeline en deux temps pas encore implémenté).
- Phase 6.0 (recall@k du classifieur 4D) : close NO-GO — non concernée par la
  réouverture 5A (échantillon déjà large à l'époque, pas un problème de
  taille d'échantillon comme 5A).

**Mise à jour (`num_points_to_sample` corrigé, 5000→10000 sur objets à 2
fragments) :** confirme le diagnostic sur le rendement (`N_ok` ×4, 107→426 ;
`too_few_points` divisé par ~2) sans perdre le gain vs `uniform`
(`Pose@30`=39.20% vs 27.59%, `Pose@15`=27.23% vs 13.70%). Indice faible-N
(`1000+`=0%, N=9) suggérant que `RESOLUTION=64` fixe devient elle-même
limitante à haute densité — pas seulement à la sparsité.

**Architecture retenue avec l'utilisateur (2026-07-20) : pipeline en deux
étages, pas un seul matching point-à-point direct.** Depth-map matching
d'abord (estimation grossière), point-à-point (ICP ou équivalent) en
raffinement final seulement — le point-à-point direct avait déjà échoué en
Phase 3A/4A/4B faute de bonne initialisation. Le calibrage exact de l'angle
"assez bon" pour passer la main (test de bassin de convergence) est reporté
après stabilisation du depth-map matching, pas avant.

**Prochaine action concrète (détail complet dans la section Phase 7, sous-section
"Correction de la sémantique num_points_to_sample" + "Discussion de direction") :**
1. **Résolution adaptative sur les depth maps** — piste prioritaire retenue.
   Deux aspects liés à traiter ensemble : densité de points par case (partiel
   avec le fix `num_points_to_sample`, indice `1000+`=0% à creuser) et taille
   de grille elle-même (`RESOLUTION=64` fixe → variable selon densité/taille).
2. Cadrer un pipeline en deux temps pour étendre le gain du sampling pondéré
   à `thresh0.3` (condition réelle) sans casser la contrainte `uniform` du CNN.
3. Test de bassin de convergence du raffinement point-à-point — reporté après
   stabilisation du depth-map matching (calibrera l'objectif de précision
   exact à viser en sortie de l'étape 1, ex. "30° suffit" ou pas).
4. Dataset réel TU Wien 3D Puzzles — reporté, moins prioritaire vu ces
   résultats (poses GT à vérifier, planéité à mesurer, CNN Step 15 en
   zero-shot).

## Piste 3 — Résolution adaptative (2026-07-21, en cours)

**Diagnostic ajouté (pas encore de résultat, pas encore branché dans le pipeline
réel) : `--resolution_sweep`** dans `phase5a_depthmap_matching.py`, même principe
que `--dilate_sweep`/`--gaussian_sweep` (tous deux rejetés le 2026-07-20) — pour
chaque paire, calcule `oracle_overlap_frac` à plusieurs résolutions de grille
(défaut `24 32 48 64 96 128`, `RESOLUTION=64` actuel étant fixe pour tous), en
recalculant `pixel_size` et `u_min/v_min` à chaque résolution (ils en dépendent).
Ne touche pas au pipeline réel de recherche/scoring — diagnostic pur avant
implémentation, discipline établie par les deux tentatives de densification
précédentes.

**Critère de lecture (identique à dilatation/splat) :** si `OracleOvlp` sur `gt`
remonte nettement plus vite que sur `random` à une résolution donnée, la
résolution actuelle est mal calibrée et une résolution adaptative (liée à la
densité de points, cf. cible chiffrée de l'audit des rejets : fill-rate ~2%
paires en échec vs ~9% paires en succès) a de bonnes chances d'aider
réellement. Si `gt` et `random` bougent pareil, c'est un gain mécanique à
rejeter, comme pour la dilatation.

**Prochaine étape :** lancer le sweep sur le serveur, lire le résultat, puis
seulement si positif implémenter une résolution adaptative par paire (liée à
`min(n_i, n_j)`, cf. hypothèse du maillon faible objet#21) dans `rasterize()`/
`oracle_overlap_frac()`/`match_depthmaps()`.

**Résultat du sweep OracleOvlp (2026-07-21, N=2415, `joint`) :**
```
Strategy        R=24    R=32    R=48    R=64    R=96   R=128
gt             0.989   0.971   0.882   0.758   0.544   0.404
thresh0.3      0.939   0.915   0.831   0.722   0.530   0.394
random         0.803   0.732   0.601   0.493   0.344   0.248
```
Résolution plus grossière (R plus petit) fait mécaniquement remonter
`OracleOvlp` pour TOUT LE MONDE, `gt` comme `random` — même défaut identifié
que la dilatation/le splat gaussien : une case plus grosse est plus facile à
faire coïncider, indépendamment d'une vraie correspondance. Rapport gt/random
progresse légèrement avec R (1.23 à R=24 → 1.63 à R=128) mais c'est un
effet-recouvrement pur, pas la métrique qui compte.

**Critique de l'utilisateur (2026-07-21, juste) : ce sweep est biaisé sur deux
points** — (1) il ne mesure que `OracleOvlp` (recouvrement de footprint), pas
`Pose@30` réel, exactement l'écart qui a fait échouer la dilatation/le splat
(bon recouvrement ≠ bonne pose) ; (2) c'est une moyenne globale sur toutes les
paires, alors que l'hypothèse à tester est que la résolution optimale dépend
de la densité de points (côté le plus pauvre, cf. objet#21) — un chiffre
unique masque cette dépendance.

**Fix implémenté (2026-07-21) : `--pose_resolution_sweep` + `run_match_at_resolution()`.**
Nouvelle fonction factorisée qui exécute le pipeline COMPLET (rasterize →
match_depthmaps → Kabsch) à une résolution donnée (utilisée pour le sweep, le
run principal reste inchangé pour ne pas risquer de casser le pipeline validé).
Pour chaque résolution candidate testée (coût contrôlé : restreint par défaut
à la stratégie `gt` via `--pose_resolution_sweep_strategies`, oracle-first
comme toujours), mesure `Pose@30`/`Pose@15`/`RotErr` réels ET `min(n_i,n_j)`,
stratifiés après-coup par tranche de densité (`density_pose_stratification()`,
même tranches que `frac_pts_min_stratification`). Nouvelle table de sortie
**SWEEP DENSITÉ x RÉSOLUTION**.

**Résultat (2026-07-21, N=2415, `gt`, R=24/32/48/64/96/128) — hypothèse
INFIRMÉE dans le sens attendu, mais un signal réel dans l'autre sens.** Par
tranche de densité : les tranches DENSES (200-500, 500-1000) sont quasi
plates quelle que soit R (Pose@30 ~28-30%, N stable) — la résolution ne les
affecte pas, assez de points pour remplir la grille à toute résolution
testée. Les tranches ÉPARSES (50-100, 100-200) montent en Pose@30 avec R
(100-200 : 23.68%→35.86% de R=24 à R=128) MAIS perdent énormément de paires
en route (50-100 : N=97→10). **Métrique qui compte réellement — succès nets
sur la population totale (N×Pose@30 / 2415)** : R=24 → 6.25%, R=32 → 6.09%,
R=48 → 6.00%, R=64 (actuel) → 5.84%, R=96 → 5.47%, R=128 → 4.64% — **décroît
continûment avec R**. Donc la résolution actuelle (64) n'est pas optimale,
mais dans le sens inverse de l'hypothèse "adaptative fin/grossier selon
densité" : une grille plus GROSSIÈRE partout donne plus de succès nets, pas
une résolution variable par paire. Gain modeste (+7% relatif R=24 vs R=64),
loin du saut ciblé par l'audit (fill-rate 2%→9%).

**Investigation demandée par l'utilisateur (2026-07-21) : POURQUOI l'attrition
grandit avec R, pas seulement CONSTATER le compromis.** Ajout du tracking des
deux raisons d'échec possibles de `run_match_at_resolution` (`sparse_dmap` :
pas assez de cases occupées après rasterisation ; `no_overlap_3d` :
alignement trouvé mais <3 correspondances 3D reconstruites après rotation/
arrondi pixel), stratifié par densité, dans une nouvelle table **ATTRITION
PAR RÉSOLUTION x DENSITÉ**. Hypothèse de travail avant le run : `no_overlap_3d`
devrait grandir avec R (pas `sparse_dmap`) car `n_angles=36` (pas angulaire
fixe, 10°/pas) ne s'affine pas avec la grille — l'erreur de quantification
angulaire (jusqu'à 5° d'écart, déplacement tangentiel ≈ r·sin(5°) pour un
point à distance r du centre) devient proportionnellement plus grosse en
pixels quand le pixel physique rétrécit, ce qui ferait rater la case de
correspondance après arrondi.

**Résultat (2026-07-21, N=2415, `gt` + `random`) — hypothèse CONFIRMÉE pour le
mécanisme, et surtout : la piste résolution est un VRAI signal, pas un piège
type dilatation.** `no_overlap_3d` grandit bien avec R pour `gt` (430→669 de
R=24 à R=128), `sparse_dmap` reste à 0 sur toute la plage (les points GT sont
assez répartis spatialement pour toujours passer le plancher de 20 cases,
même à R=24) — confirme le mécanisme du plancher mathématique à 3
correspondances minimum pour Kabsch, expliqué par l'utilisateur : avec peu de
cases remplies des deux côtés, même la meilleure position possible a de
fortes chances de ne jamais faire coïncider 3 cases, indépendamment de la
qualité de la recherche.

**Test décisif demandé par l'utilisateur — `random` dans CE sweep précis
(pas l'ancien sweep OracleOvlp isolé) : `Pose@30(random)` reste à ~0% à
TOUTES les résolutions** (46/0%, 104/0.96% — 1 seul cas sur 104, bruit de
tirage —, 160/0%, 175/0%, 182/0%, 166/0%), alors que son nombre de tentatives
qui passent le plancher des 3 correspondances augmente lui aussi avec R
grossier (comme `gt`). **Différence structurelle avec la dilatation/le
splat** : ces derniers fabriquaient de la tolérance (marquaient "remplies"
des cases sans vrais points, `random` en profitait presque autant que `gt`) ;
grossir la résolution ne fabrique rien, ça regroupe seulement de vrais
points en paquets plus gros — donc `random` obtient plus de TENTATIVES mais
aucun gain de VALIDITÉ (il n'y a aucune vraie relation géométrique entre deux
nuages aléatoires, quelle que soit la grille). **Conclusion validée : le gain
de succès nets R=24 vs R=64 (6.25% vs 5.84%, calcul précédent) est un vrai
signal.**

**Suite demandée par l'utilisateur (2026-07-21) : isoler la confusion
résolution/pas-angulaire avant de conclure sur l'optimum.** Remarque
pertinente : `n_angles=36` (10°/pas) est resté FIXE pendant tout le sweep
précédent — à résolution fine, la même imprécision angulaire se traduit par
un déplacement en pixels plus grand (proportionnel), ce qui peut expliquer
une partie de l'attrition `no_overlap_3d` à haute résolution sans que ce
soit un vrai défaut de la finesse de grille elle-même.

**Implémenté (2026-07-21) : `--pose_angle_sweep` + refactor `n_angles`
explicite.** `run_match_at_resolution()` prend maintenant `n_angles` en
paramètre explicite (plus lu depuis `args.n_angles`), et une nouvelle
fonction `build_resolution_angle_combos(args)` construit la liste des
combinaisons (résolution, n_angles) à tester : chaque R de
`--pose_resolution_sweep` avec le `n_angles` global (comportement identique
au run précédent), PLUS des combinaisons explicites `"R:NA"` via
`--pose_angle_sweep` (ex. `128:72 128:144`) pour tester une résolution fine
avec une recherche angulaire plus fine, isolément. Tables de sortie mises à
jour (clé `R,n_angles` au lieu de `R` seul).

## Recadrage stratégique (2026-07-21) : étape A (éligibilité) vs étape B (précision)

**Le tuteur/l'utilisateur a clarifié la structure du problème en deux
étapes distinctes, à ne pas mélanger :**
- **Étape A (maintenant)** : maximiser l'ÉLIGIBILITÉ — combien de paires
  produisent une pose (n'importe laquelle), indépendamment de sa justesse.
  Plafond réaliste ~99.7% (100% − `unusable_too_few` de l'audit des rejets).
- **Étape B (après le test de bassin de convergence, pas encore fait)** :
  maximiser Pose@X° pour le X° que le raffinement point-à-point pourra
  vraiment digérer comme point de départ (pas encore connu — 20°, 30°, 45°
  selon ce que le bassin de convergence tolère).

**Conséquence : tout ce qui portait sur `Pose@30` (stratification par
résolution, garde-fou `random`, comparaison RotErr appariée fin/grossier)
était en fait de l'étape B, fait un peu tôt.** Ces résultats restent valides
et utiles pour plus tard, mais ne doivent pas guider les décisions
d'éligibilité maintenant — le vrai signal d'étape A déjà disponible dans les
données du 2026-07-21 est la colonne `N` (nombre de paires atteignant Kabsch)
du sweep densité x résolution, pas `Pose@30`.

**Idée de l'utilisateur, reformulée comme architecture cible : résolution
ADAPTATIVE en cascade**, pas une résolution fixe ni un choix a priori par
densité (l'hypothèse "fin pour dense / grossier pour épars" avait déjà été
infirmée le même jour). Principe : pour chaque paire, essayer la résolution
la plus fine d'abord (précision maximale) ; si `no_overlap_3d` (échec),
retomber sur une résolution plus grossière ; répéter jusqu'à succès ou
épuisement des niveaux. En l'absence de GT en conditions réelles
(`thresh0.3`), la décision "garder cette pose" s'appuiera sur
`best_overlap_frac` (déjà identifié dans l'audit comme un signal de
confiance discriminant sans GT) plutôt que sur `Pose@30`.

**Test intermédiaire décidé avant de coder la cascade (bon marché, réutilise
les données déjà collectées) : plafond d'éligibilité par UNION.** Question
posée : sur les résolutions déjà testées, quelle fraction de paires réussit
à AU MOINS UNE résolution (pas une résolution fixe) ? Si ce plafond approche
~99.7% avec peu de niveaux de repli, la cascade est justifiée avant même
d'écrire sa logique de décision. Implémenté : tracking `cascade_union_downto_res{R}`
par paire (dans l'ordre décroissant de résolution, comme une cascade réelle
tenterait), nouvelle table **PLAFOND D'ÉLIGIBILITÉ PAR CASCADE (UNION)**.
Résultat pas encore lancé — prochaine action : relancer avec
`--pose_resolution_sweep` étendu vers le bas (ex. `12 16 20 24 32 48 64 96 128`)
et lire cette nouvelle table.

**Reporté après ce test (étape B, pas urgent) :** comparaison RotErr appariée
par paire (finer vs coarser, parmi les succès aux deux résolutions) — ne sert
qu'à décider QUELLE pose garder quand plusieurs résolutions réussissent en
même temps, pas si la cascade vaut la peine d'être construite.

**Résultat du plafond d'union (2026-07-21, `phase5a_depthmap_matching.py`,
R=12..128, gt, N_total=1015) : 68.7% (697/1015) — plafonne loin des ~99.7%
visés, avec des rendements décroissants (+9.5pp par niveau au début, +1.2pp
en fin de cascade).** Mais l'utilisateur a fait remarquer un point essentiel :
ces 1015 paires sont DÉJÀ celles qui passent les filtres `too_few_points`/
`too_curved` — sur les 2415 paires GT totales, **1400 (58%) sont exclues
avant même d'atteindre ce calcul**, un filtrage bien plus massif que le
plafond des 31% restants. Retour au principe fondateur du projet (déjà
énoncé en ouvrant l'audit des rejets) : deux fragments de la même cassure ont
FORCÉMENT une vraie surface de fracture des deux côtés — aucune paire n'est
structurellement impossible, un rejet reflète un choix de pipeline, pas une
absence physique de signal.

**Clarification (2026-07-21) : les étapes 1 et 2 du plan en 3 étapes sont en
fait DÉJÀ mesurées par `phase5a_skip_audit.py`, pas besoin de les refaire.**
Ce script ne s'arrête jamais à `too_few_points`/`too_curved` (seul
`ABS_MIN_POINTS=5`, plancher numérique pur, arrête vraiment) ET tourne déjà
en `sample_method=weighted` avec `num_points_to_sample=10000` (forcé via
`model_type="garf"`). Son dernier résultat (N=1487) reflète donc déjà le
régime corrigé : `too_few_points` légitime (0% récupérable), `too_curved` un
vrai artefact mais modeste (16.8% récupérable). **Ce qui manque réellement :**
ce script tournait à résolution FIXE (64) — jamais testé avec la cascade de
résolution développée le même jour dans `phase5a_depthmap_matching.py`.

**L'utilisateur a aussi noté : le verdict "légitime" sur `too_few_points`
mérite d'être requestionné, pas juste accepté** — il a été mesuré SOUS le
régime `weighted`/10000 déjà amélioré, donc c'est le meilleur chiffre déjà
obtenu, mais le déséquilibre extrême de l'objet#21 (131 vs 2832 points)
montre qu'une partie de ce qui tombe sous 50 points n'est pas intrinsèquement
trop petite — c'est mal servie par l'échantillonnage. Ce point rejoint
directement l'étape 2 (déjà appliquée dans ce script), pas une 4e étape
séparée.

**Implémenté (2026-07-22) : fusion des 3 axes en une seule mesure dans
`phase5a_skip_audit.py`.** `process_pair_audit()` réutilise maintenant
`run_match_at_resolution()` (de `phase5a_depthmap_matching.py`, pas de
réimplémentation) en cascade : `--resolution_sweep` (défaut
`128 96 64 48 32 24 20 16 12`, plus fine d'abord) est essayé jusqu'à la
PREMIÈRE résolution qui produit ≥3 correspondances — reproduit directement
l'architecture cible (fin d'abord pour la précision, repli sur grossier
seulement si échec), pas juste un test d'union après coup. Combiné avec le
`weighted`/`no_points_to_sample` et l'absence de hard-stop déjà en place,
cette seule mesure donne le plafond d'éligibilité réel une fois les trois
corrections appliquées ENSEMBLE. Nouvelles sorties : `resolution_used`/
`n_resolutions_tried` par paire, table **CASCADE DE RÉSOLUTION** (répartition
des résolutions retenues, distribution du nombre de niveaux essayés).
Retiré : `would_skip_sparse_dmap`/`n_pix_i`/`n_pix_j` (n'ont plus de sens
sous cascade multi-résolution — un pair "sparse" à une résolution peut
réussir à une autre).

**Résultat (2026-07-22, N=1487, `num_points_to_sample=10000`, cascade
R=128..12) — gros progrès sur l'éligibilité, ET une découverte majeure sur
`too_curved`.** `no_correspondence` passe de **73.6% (hier, résolution fixe)
à 48.1%** (715/1487) — `pose_computed` (éligibilité réelle) quasiment double,
de ~26% à **51.6%**. La cascade seule (combinée au `weighted`/10000 déjà en
place) comble une bonne partie de l'écart, sans qu'il ait fallu toucher aux
seuils `too_few_points`/`too_curved`. `n_resolutions_tried` : médiane=8 (sur 9
niveaux) — la plupart des paires résolues ont besoin de plusieurs niveaux de
repli, la cascade n'est pas un luxe cosmétique.

**`too_curved` : le coût réel est BEAUCOUP plus grand qu'estimé le
2026-07-20.** Avec la cascade : **90.4%** des 157 paires que ce seuil aurait
rejetées produisent quand même une pose (contre les 16.8% de Pose@30
estimés hier SANS cascade — la cascade rattrape presque toute l'éligibilité
perdue par la courbure, même si seulement 12.7% de ces paires atteignent
Pose@30). `too_few_points` reste confirmé légitime (1.7% récupérable,
inchangé). **Décision (2026-07-22, utilisateur) : supprimer complètement le
seuil `too_curved`, pas seulement le relâcher.** Implémenté dans
`phase5a_depthmap_matching.py` : le bloc `if plan_i > args.max_planarity ...:
skip too_curved` retiré de `process_pair()`, `--max_planarity` et
`DEFAULT_MAX_PLANARITY` supprimés (plus de raison d'être) ; `plan_i`/`plan_j`
restent calculés pour la stratification diagnostique
(`planarity_stratification`), juste plus utilisés comme seuil de rejet.

**Sur la sparsité restante (48.1% `no_correspondence`) : `n_frac_pts_min`
discrimine très fort (médiane=64 pour `no_correspondence` vs 257-335 pour les
paires qui produisent une pose), `planarity` NE discrimine PAS dans le sens
attendu (`no_correspondence` est en fait plus PLAT, pas plus courbé — 0.02 vs
0.07-0.10) — confirme que ce n'est pas une histoire de courbure mais de pure
quantité de points.** **Décision de l'utilisateur (2026-07-22) : ne PAS
augmenter `num_points_to_sample` comme solution — plutôt identifier
précisément les tranches de densité concernées, puis explorer un zoom +
rééchantillonnage LOCAL ciblé sur la zone de fracture identifiée (pas
augmenter le budget global de points sur tout le fragment), ou une
représentation non-planaire type SOM (discutée, à reprendre plus tard).**
Implémenté : `density_outcome_stratification()` dans `phase5a_skip_audit.py`
— répartit `unusable_too_few`/`no_correspondence`/`pose_computed`/`Pose@30`
par tranche fine de `n_frac_pts_min` (`<50` à `1000+`), nouvelle table
**STRATIFICATION FINE PAR DENSITÉ**. Objectif : trouver le plancher de
densité exact en dessous duquel la méthode échoue structurellement, avant de
choisir/dimensionner une solution de zoom-resampling. Résultat pas encore
lancé — prochaine action : relancer `phase5a_skip_audit.py` (cascade +
`max_planarity` déjà sans effet ici puisque jamais un hard-stop) et lire
cette nouvelle table.

**Résultat (2026-07-22, N=1487, cascade R=128..12) : stratification fine
confirme une courbe dose-réponse nette et exploitable.** `NoCorr` chute de
97.0% (`<50` pts) à ~10-12% (`300+` pts), la plus grosse baisse se jouant
entre 100 et 300 points ; au-delà de 300 ça plafonne (résidu probablement
géométrique, pas de densité). **Population concernée : 771/1487 paires
(52%) sous 150 points**, avec des taux d'échec 97%/84.6%/74.7%/42.5% — cible
énorme pour un fix de densité ciblé. `too_curved` confirmé à 89.4%
récupérable (cohérent avec le run précédent) → **décision : suppression
complète du seuil (pas juste relâché)**, implémentée dans
`phase5a_depthmap_matching.py` (`--max_planarity`/`DEFAULT_MAX_PLANARITY`
retirés, `plan_i`/`plan_j` restent calculés pour la stratification
diagnostique uniquement).

## Piste "zoom + rééchantillonnage local" (2026-07-22)

**Idée de l'utilisateur, étendue en architecture de pipeline complète** :
plutôt qu'augmenter `num_points_to_sample` globalement (déjà écarté par
l'utilisateur — gaspille le budget sur tout le fragment), zoomer
spatialement sur la zone de fracture déjà repérée (GT ici ; CNN `thresh0.3`
plus tard, ça referme l'item resté ouvert du "pipeline en deux temps") et y
rééchantillonner densément, EN PLUS de l'échantillonnage normal.

**Faisabilité vérifiée avant de coder (inspection de
`assembly/data/breaking_bad/base.py`/`weighted.py`)** : chaque fragment est
stocké comme un MAILLAGE complet (`data["meshes"]`, conservé en val/test,
supprimé en train pour la mémoire — pas un problème, tout Phase 5A/7 tourne
sur val). `shared_faces` (indices de faces fracture) existe côté dataset
mais n'est PAS conservé dans le dict retourné par `get_data()` — seul
`fracture_surface_gt` (labels par POINT) et `pointclouds_gt` (repère
maillage, avant recentrage/rotation/rescale) survivent jusqu'au batch.
Solution : localiser la zone fracture par PROXIMITÉ SPATIALE
(`trimesh.proximity.closest_point`, donne le triangle le plus proche de
chaque point fracture déjà connu) plutôt que par indice de face stocké —
cette méthode fonctionne IDENTIQUEMENT pour GT (points exacts) et pour
`thresh0.3` (points prédits par le CNN) plus tard, un seul mécanisme pour
les deux cas.

**Implémenté : `scripts/phase5a_zoom_resample_check.py`.** Pour chaque
paire : (1) localise la zone fracture sur le maillage à partir des points
fracture déjà échantillonnés (`trimesh.proximity.closest_point`), étend d'un
anneau d'adjacence (`mesh.face_adjacency`, `--expand_rings`, défaut 1) pour
couvrir un peu plus que les seules faces déjà touchées par l'échantillon
épars ; (2) tire `--extra_budget` (défaut 500) nouveaux points RESTREINTS à
cette zone via `trimesh.sample.sample_surface(mesh, face_weight=...)` (déjà
utilisé ailleurs dans le codebase pour l'échantillonnage `weighted`, pas de
réimplémentation) ; (3) ramène ces nouveaux points du repère maillage/GT au
repère "input" (celui de `raw_i` dans les autres scripts) via l'inverse de
la formule de reconstruction Phase 0 : `q = (pts_gt - translation) @
R(quat)` ; (4) concatène aux points fracture existants, relance la MÊME
cascade de résolution (`run_match_at_resolution`, pas de réimplémentation)
que `phase5a_skip_audit.py`, sur les MÊMES paires en conditions baseline vs
zoomed. Comparaison globale + par tranche de densité baseline (mêmes bornes
que `density_outcome_stratification`), pour voir si le zoom aide
spécifiquement les tranches basses sans diluer les tranches déjà bonnes.

**Résultat (2026-07-22, N=1483, `extra_budget=500`, `expand_rings=1`) —
gain massif en éligibilité, coût réel en précision.** `no_correspondence`
47.9%→**22.3%**, `pose_computed` 52.1%→**77.7%** (+25.6 points) — gain net
sur l'objectif actuel (étape A). Mais `Pose@30` global BAISSE : 13.6%→10.8%.
Par tranche : `<50` et `75-100` s'améliorent aussi en précision (2.2%→3.5%,
10.1%→11.3%), mais `50-75`, `100-150`, `150-200`, `200-300`, `300-500`
baissent tous en `Pose@30`, y compris des tranches qui n'avaient presque pas
besoin d'aide en éligibilité (`500-1000` : `NoCorr` empire même,
8.4%→12.6%). Hypothèse de travail : `--expand_rings 1` déborde légèrement
au-delà de la vraie limite de fracture, les points ajoutés sur cette bordure
élargie décalent le repère PCA (u/v/normale) même quand il était déjà bon.

**Décision (2026-07-22, utilisateur) : accepter tel quel pour l'étape A**
(éligibilité = objectif actuel), la baisse de précision est une question
d'étape B à recalibrer une fois le seuil du raffinement point-à-point connu
— pas la peine d'optimiser maintenant. Utilisateur a demandé un test pour
vérifier si la baisse est un effet de composition (les paires récupérées
sont juste plus dures intrinsèquement) ou une vraie dégradation (le zoom
abîme aussi les paires qui marchaient déjà). **Implémenté :** nouvelle
section **COMPOSITION vs DÉGRADATION** dans
`phase5a_zoom_resample_check.py` — compare `Pose@30` sur le sous-ensemble
FIXE des paires déjà éligibles en baseline (avant/après zoom) séparément des
paires nouvellement récupérées par le zoom. Si le premier chiffre reste
stable, effet de composition pur ; s'il baisse aussi, vraie dégradation.
Ajout aussi `--csv_out` (dump par paire, permet de recalculer d'autres
croisements sans relancer).

**Résultats successifs (2026-07-22, N=1483 à chaque fois) — confirment une
vraie dégradation (pas juste de la composition), et `expand_rings=0`
résout l'essentiel :**
- `expand_rings=1, budget=500` : `Pose@30` global 13.6%→10.8% (baisse),
  déjà-éligibles 27.7%→18.1% (baisse forte) → vraie dégradation confirmée.
- `expand_rings=0, budget=500` : `Pose@30` global 13.9%→**16.3%** (hausse
  nette !), déjà-éligibles 26.5%→24.1% (baisse légère seulement),
  nouvellement récupérées Pose@30=19.8% (vs 9.6% avec rings=1) → confirme
  l'hypothèse, l'expansion d'anneau débordait de la vraie limite de fracture.
- `expand_rings=0, budget=200` : PIRE que budget=500 sur les 3 indicateurs
  globaux (`NoCorr` 37.4% vs 31.4%, `pose_computed` 62.6% vs 68.6%,
  `Pose@30` 14.9% vs 16.3%) — réduire le budget n'aide pas, contre-intuitif
  mais net sur les chiffres agrégés (plus fiables que les sous-tranches).

**Meilleure config trouvée jusqu'ici : `--expand_rings 0 --extra_budget 500`**
(gain net sur éligibilité ET précision simultanément, pas un compromis).

**Sweep étendu (2026-07-22, N=1483, budgets 200/300/500/800/1200, même passe) :
tendance monotone et contre-intuitive.** Plus le budget augmente, MIEUX ça se
passe sur les deux axes à la fois — `NoCorr` 37.2%→24.8%, `pose_computed`
62.8%→75.2%, ET la dégradation sur les paires déjà éligibles en baseline
**diminue** avec plus de points (-4.7pp à budget=200 jusqu'à seulement
-1.6pp à budget=1200), au lieu d'empirer. Pas de plafond trouvé à 1200 ;
`NoCorr` reste loin du ~0.3% théorique. Explication : avec `ring=0` (zone
correctement délimitée), plus de points dans la zone CORRECTE stabilise le
repère PCA plutôt que de le diluer — le problème initial venait bien de la
délimitation de zone (rings=1), pas du volume de points.

## Recadrage stratégique (2026-07-22) : arrêter de peaufiner GT, généraliser à thresh0.3

**Question de l'utilisateur, avant de pousser le sweep plus loin (1500-3000) :
"est-ce que ce qu'on fait est cohérent ?"** Réponse convenue : oui
méthodologiquement (diagnostic avant action, contrôles pour isoler les
confusions, sweeps en une passe plutôt que tâtonnage), mais le rendement
marginal de peaufiner `extra_budget` sur l'oracle GT (qui ne sera jamais
utilisé tel quel en production) est faible comparé à la vraie question
ouverte : est-ce que la recette (`ring=0` + budget généreux) se transfère à
`thresh0.3` (condition réelle, jamais testée). **Décision : rediriger vers
`thresh0.3` maintenant plutôt que de continuer à raffiner GT.**

**Deuxième point soulevé par l'utilisateur, décisif pour la suite : le CNN
n'a jamais été entraîné/évalué sur des points zoomés — est-ce un problème ?**
Réponse établie par l'architecture en deux temps déjà actée : NON
directement — le CNN ne voit JAMAIS les points zoomés, il tourne seulement
sur son échantillonnage `uniform` standard pour LOCALISER (masque
`thresh0.3`), le zoom se fait ensuite sur le maillage, sans repasser par le
CNN. **La vraie question ouverte : le masque `thresh0.3` (faux positifs/
négatifs que le GT n'a jamais) est-il assez propre spatialement pour bien
localiser la zone ?** La Phase 2D avait déjà montré que le clustering
spatial de `thresh0.3` est quasi identique à `gt` (signe encourageant), mais
jamais testé pour CE mécanisme précis.

**Implémenté (2026-07-22) : `scripts/phase5a_zoom_resample_thresh03_check.py`.**
Contrainte technique résolue : le CNN exige `sample_method=uniform`, mais
`BreakingBadUniform.transform()` ne conserve PAS `data["meshes"]`
(contrairement à `BreakingBadWeighted`, vérifié par inspection du code) —
besoin du maillage pour zoomer. Solution : DEUX datasets sur le même split,
en lockstep par position (`cnn_dataset` via DataLoader batch_size=1
shuffle=False pour l'inférence CNN ; `mesh_dataset` indexé directement
`dataset[idx]`, UNIQUEMENT pour `data["meshes"]` — le maillage est identique
entre les deux instances, seule la façon d'échantillonner des points dessus
change), avec `assert` sur le nom d'objet à chaque itération pour vérifier
l'alignement. Graines du zoom = points prédits fracture par le CNN
(`coarse_seg_pred > threshold`), en repère maillage (`pointclouds_gt` du
`cnn_dataset`, aligné index-à-index avec les prédictions via
`extract_fragment_list` appelé deux fois avec les mêmes offsets). Repère de
sortie des points zoomés : quaternion/translation du `cnn_dataset` (pas du
`mesh_dataset` — chaque `transform()` tire sa propre rotation aléatoire).
Reporting factorisé : `report_zoom_sweep()` extrait de
`phase5a_zoom_resample_check.py` et réutilisé tel quel (même structure de
résultats).

**Résultat (2026-07-22, N=1475, `expand_rings=0`, budgets 500/1200) — ÉCHEC
NET, confirme la crainte de l'utilisateur, pas un compromis.** Contrairement
au GT, le zoom EMPIRE tout sur `thresh0.3` : `no_correspondence` monte
(58.0%→68.4% à budget=500, →66.2% à budget=1200 — pire, pas mieux),
`pose_computed` baisse (42.0%→31.6%/33.8%), et surtout `Pose@30` s'effondre
(6.9%→0.5%/0.8%). Les paires déjà éligibles en baseline (N=620,
`Pose@30`=16.5%) s'effondrent à ~1% une fois zoomées — le zoom détruit des
poses qui marchaient déjà, pas juste "n'aide pas". Par tranche de densité,
`no_correspondence` empire dans presque tous les bins (`150-200` :
23.2%→61.1%, `200-300` : 17.1%→40.8%), inverse exact du comportement GT.

**Diagnostic (confirmé sans ambiguïté par sample_method) : le CNN tourne
bien en `sample_method=uniform`** (vérifié dans
`configs/experiment/cnn_ablation_shared.yaml:26`, hérité par
`cnn_step15_final_model.yaml` — "required: bilinear backprojection assumes
uniform num_pts", `num_points_to_sample=5000`) — pas un bug de config, le
problème est ailleurs.

**Cause probable identifiée : le masque `thresh0.3` a des faux positifs
dispersés que le GT n'a jamais (par construction), et le script utilisait
TOUS les points prédits-positifs comme graines de zoom, sans filtrage
spatial.** Contrairement au GT (graines toujours concentrées au même
endroit), des graines dispersées font localiser plusieurs zones à la fois
sur le maillage, corrompant le repère PCA même pour les paires qui
marchaient déjà en baseline.

**Fix implémenté (2026-07-22) : filtrage par clustering avant zoom,
réutilise la Phase 2D.** `dominant_cluster_mask()` dans
`phase5a_zoom_resample_thresh03_check.py` appelle directement
`cluster_points()` de `scripts/phase2d_interface_clustering_diagnostic.py`
(composantes connexes par proximité, `eps=0.02`/`min_cluster_size=10` —
mêmes valeurs que le compromis trouvé en Phase 2D, `EdgeCoverage`/
`BestCorrPrec` plafonnaient déjà à `eps=0.02`) et ne garde que le cluster
spatial DOMINANT parmi les points prédits-positifs comme graines de zoom.
Le masque `baseline` (comparaison au pipeline réel actuel) reste
INCHANGÉ — seul le choix des graines de zoom est purifié, pour isoler
précisément si c'est bien la dispersion des graines qui posait problème.

**Résultat du run avec clustering (2026-07-22, N=1469) : quasi identique au
run brut précédent (`no_correspondence` toujours pire que baseline,
`Pose@30` toujours effondré, `base_eligible` toujours ~13%→~1%).**
L'utilisateur a réagi ("y'a un pb dans le code c'est impossible") — à
raison : si le clustering ne change presque rien, ce n'est pas un problème
de bruit dispersé (le clustering l'aurait résolu), c'est un bug de
plomberie. Exactement le test diagnostique qu'on avait posé avant de
lancer.

**Bug trouvé par inspection du code (pas par essai-erreur) : mismatch de
repère entre les deux datasets.** `BreakingBadUniform.transform()`
(`uniform.py:49`, dataset CNN) applique une rotation aléatoire
supplémentaire à TOUT L'OBJET assemblé (`rotate_whole_part`) avant la
rotation par fragment — `BreakingBadWeighted.transform()` (dataset mesh)
**n'applique jamais cette étape** (littéralement commentée dans son code).
Donc `pointclouds_gt` du dataset CNN est dans un repère tourné par
`init_rot` (aléatoire, différent par objet) relativement au maillage
récupéré depuis le dataset weighted — chercher "quelle face du maillage
est la plus proche de ce point fracture" sans corriger ça compare des
points dans deux repères différents, tombant sur des faces essentiellement
aléatoires. Explique tout : dégradation uniforme (pas seulement les paires
bruitées), et pourquoi le clustering ne change rien (filtrer des points
déjà mal placés dans l'espace ne les remet pas au bon endroit).

**Fix implémenté (2026-07-22), vérifié mathématiquement par inspection de
`rotate_whole_part()`** (`assembly/data/transform.py:109-146` :
`rotated = rot_mat @ canonical`, `quat_gt` stocké = `rot_mat.T`, donc
`canonical = R(init_rot) @ rotated`) : lit `batch["init_rot"]` (déjà
renvoyé par le dataset, juste jamais utilisé), calcule `R_init`, et corrige
dans les deux sens — graines ramenées au repère canonique du maillage
avant la recherche de proximité (`gtgt @ R_init.T`), nouveaux points
zoomés reconvertis vers le repère "tourné" avant `to_input_frame`
(`new_pts @ R_init`).

**Résultat après fix (2026-07-22, N=1463, `expand_rings=0`, budgets 500/1200)
— CONFIRME que c'était bien un bug, pas un problème de qualité du masque
CNN, et valide le transfert à `thresh0.3`.** Profil maintenant cohérent
avec le GT : `no_correspondence` baisse (57.3%→44.8%/40.4% selon budget,
au lieu de monter à 68-70% avant le fix), `pose_computed` monte
(42.7%→55.2%/59.6%), `Pose@30` global reste quasi stable (7.3%→6.5%/6.4%,
pas d'effondrement à 0.5% comme avant). Paires déjà éligibles en baseline
(N=625, 17.1%→13.0%, -4.2pp) : dégradation modeste, du même ordre de
grandeur que ce qu'on observait en GT à réglages comparables — plausible,
plus l'effondrement à ~1% d'avant le fix. Par tranche de densité, presque
tout s'améliore (`<50` : `NoCorr` 98.0%→76.8%/68.5%, `75-100` :
72.3%→50.8%/44.6%), sans effondrement de `Pose@30` par tranche.

**Conclusion : le zoom + rééchantillonnage local se transfère à
`thresh0.3`** — gain net sur l'éligibilité (étape A, l'objectif actuel),
coût modeste et cohérent sur la précision (étape B, à recalibrer plus tard
via le test de bassin de convergence). Contrairement au GT, pas de gain net
sur `Pose@30` global (reste stable plutôt que d'augmenter) — cohérent avec
le fait que le masque `thresh0.3` a un peu plus de bruit résiduel que le
GT, même après filtrage par clustering.

**Sweep de budget étendu (2026-07-22, N=1468, budgets 500/1200/2000/3000) :
éligibilité continue de progresser, rendements décroissants mais pas de
plafond net.** `no_correspondence` : 59.0%(base)→47.0%→41.6%→39.6%→**37.7%**.
`Pose@30` reste stable autour de 6% à tous les budgets (pas de tendance
claire). **Le vrai goulot pour l'éligibilité totale : la tranche `<50`**,
qui représente **462/1468 paires (31.5% de TOUT le dataset)** et reste à
64.3% de `NoCorr` même à budget=3000 (contre 98.7% en baseline — progrès
réel mais très incomplet). Cette seule tranche pèse plus que tout le reste
de l'écart vers les ~99.7% visés.

**Décision utilisateur (2026-07-22) : ne pas continuer à augmenter le
budget** (rendements décroissants, ne résout pas le vrai problème de la
tranche `<50`) — investiguer plutôt POURQUOI même un budget modeste (200
points) ne suffit pas à créer 3 correspondances dans cette tranche.
Hypothèse proposée : `MIN_CLUSTER_SIZE=10` (calé sur la Phase 2D, masques
GT toujours bien peuplés) pourrait être trop strict pour des masques
`thresh0.3` déjà épars — un fragment où le CNN prédit seulement 15-20
points pourrait perdre la moitié de ses graines au clustering, précisément
là où on en a le plus besoin.

**Implémenté (2026-07-22) : instrumentation du tracking des graines +
filtre de tranche ciblé.** `phase5a_zoom_resample_thresh03_check.py` :
`--max_n_min_base` (ne traite que les paires sous ce seuil de densité
baseline, économise le calcul sur les paires déjà bien servies) ;
`row["n_seed_raw_i/j"]`/`row["n_seed_clustered_i/j"]` trace le nombre de
graines avant et après `dominant_cluster_mask()`, nouveau résumé console
**GRAINES AVANT/APRÈS CLUSTERING** (percentiles du côté le plus pauvre,
+ compteur de paires réduites à 0 graine par le clustering).
`phase5a_zoom_resample_check.py::report_zoom_sweep()` : export CSV
généralisé pour passer automatiquement tout champ additionnel présent dans
`rows` (pas de couplage explicite entre les deux scripts).

**Résultat (2026-07-22, N=462, tranche `<50` uniquement) — surprise : les
graines ne sont PAS rares, le problème est ailleurs.** Médiane 32 points
bruts (avant clustering, largement au-dessus du minimum de 3 pour Kabsch),
20 après clustering (0 paire réduite à 0 graine). Et pourtant, `NoCorr`
reste à 98.5% en baseline et **78.6% même avec 500 points de zoom** —
largement pire que ce que la densité seule prédirait. Ça pointe vers un
problème de LOCALISATION, pas de quantité : le cluster dominant retenu est
peut-être spatialement cohérent en lui-même sans être sur la vraie
fracture (le clustering garantit "les points se regroupent", pas "ils sont
au bon endroit").

**Implémenté (2026-07-22) : comparaison GT sur les MÊMES objets.** Pour
trancher entre "ces objets sont difficiles pour tout le monde" (fracture
physiquement petite/ambiguë) et "c'est spécifiquement le CNN qui les
localise mal", ajout dans `phase5a_zoom_resample_thresh03_check.py` d'un
calcul de la cascade sur le masque GT (`fracture_surface_gt`, déjà présent
dans le batch `uniform`, pas besoin du `mesh_dataset`) pour les MÊMES
paires que celles sélectionnées par le CNN (`<50` points prédits). Nouvelle
table **GT vs CNN SUR LES MÊMES OBJETS**.

**Résultat (2026-07-22, N=473, tranche `<50`) — tranche l'ambiguïté
CNN vs difficulté structurelle, et RECADRE le plafond d'éligibilité
réaliste.**
```
                     CNN (thresh0.3)   GT (oracle)
no_correspondence          98.5%          93.0%
pose_computed                1.5%          6.6%
Pose@30                      0.6%          3.2%
```
Le GT fait mieux que le CNN (93.0% vs 98.5% de `NoCorr`, ~5x plus de
`Pose@30`) — un vrai déficit de localisation du CNN existe, réel mais
modeste. **Mais l'essentiel du problème n'est pas le CNN : même avec la
vérité terrain parfaite, 93% de ces objets échouent quand même.** Cette
tranche (31-32% de tout le dataset, N=462-473 sur ~1470-1490) est donc une
population **structurellement difficile pour la méthode elle-même**
(depth-map matching + correspondances 3D pour Kabsch) — probablement des
interfaces de fracture physiquement minuscules par rapport à la taille des
fragments — indépendamment de qui fournit le masque (CNN ou GT) et
indépendamment du budget de zoom (déjà testé jusqu'à 3000 points sans
percée).

## Décision stratégique (2026-07-22) : plafond d'éligibilité réaliste réévalué à ~69%

**Le plafond de ~99.7% (100% − `unusable_too_few`, ~0.3%) supposé depuis
l'audit des rejets était trop optimiste** — il ne tenait pas compte de
cette population de ~31% qui semble hors de portée de la méthode actuelle,
pas seulement mal desservie par le pipeline. **Nouveau plafond réaliste
pour la méthode depth-map + zoom + cascade : ~69%** (100% moins les ~31%
structurellement durs). D'après le sweep de budget étendu (2026-07-22,
budget=3000, N=1468 sur tout le dataset) : `pose_computed`=62.3% globalement
— soit déjà ~91% du plafond réaliste de 69% (62.3/68.5), quasiment atteint.

**Décision (utilisateur, 2026-07-22) :**
1. **Continuer avec la méthode actuelle (depth-map + zoom + cascade de
   résolution) pour les ~69% de paires "faciles"** — plafond presque déjà
   atteint, pas de nouvelle piste de tuning nécessaire dans l'immédiat sur
   cette portion.
2. **Chercher une méthode fondamentalement différente pour les ~31%
   restants** (interfaces de fracture minuscules) — reporté, pas encore
   scopé. Le depth-map matching (quel que soit le masque, quel que soit le
   budget de zoom) ne semble structurellement pas adapté à ces cas.

**Correction méthodologique demandée par l'utilisateur (2026-07-22) : ne pas
tâtonner un paramètre à la fois en relançant tout le run à chaque fois —
construire un vrai sweep dans la même passe**, comme déjà fait pour la
résolution. **Implémenté : `--extra_budget_sweep`** (liste de budgets,
remplace `--extra_budget` si fourni) — le budget le plus grand du sweep est
tiré UNE FOIS par fragment (`zoom_resample` appelé une seule fois à
`max(budgets)`), les budgets plus petits prennent un SOUS-ENSEMBLE de ce même
tirage (statistiquement valide : un sous-ensemble d'un tirage i.i.d. est un
tirage i.i.d. valide de cette taille, pas besoin de retirer à chaque fois).
Toutes les tables (comparaison globale, par tranche de densité, composition
vs dégradation) sont maintenant imprimées par budget testé, plus `--csv_out`
mis à jour (colonnes `zoom_b{k}_*` par budget). Résultat pas encore lancé —
prochaine action : `--extra_budget_sweep 200 500 800 1200` avec
`--expand_rings 0` pour trouver le vrai optimum en une seule passe.

## Phase 6A — Test de bassin de convergence du raffinement point-à-point (2026-07-22)

**Cadrage, suite à la décision du même jour** (plafond d'éligibilité ~69%
presque atteint pour la méthode actuelle, tranche `<50` mise de côté) : plutôt
que de continuer à pousser l'éligibilité (rendements décroissants), passer au
test de bassin de convergence — question orthogonale, ne dépend pas d'avoir
"fini" l'étape A. Toute la journée on a optimisé `Pose@30°/0.1` sans savoir si
c'est le bon seuil pour l'architecture en deux temps retenue (depth-map
matching grossier → raffinement point-à-point) ; ce test le mesure
directement.

**Protocole (oracle GT, indépendant du depth-map matching lui-même) :** part
de la VRAIE pose relative (formule Phase 0), la perturbe d'une rotation
d'amplitude θ connue (axe aléatoire) + bruit de translation gaussien, lance
un ICP point-à-point (nearest-neighbor + Kabsch itéré, avec rognage des
correspondances les plus éloignées à chaque itération — "trimmed ICP",
nécessaire car les deux nuages de points de fracture ne se correspondent
jamais parfaitement un-à-un), et mesure si ça reconverge près de la vraie
pose. Taux de succès par amplitude θ testée (10°/20°/30°/45°/60°/90°) = le
bassin de convergence.

**Implémenté : `scripts/phase6a_convergence_basin_check.py`.** Restreint à
`n_frac_pts_min >= 50` (exclut la tranche mise de côté le même jour — ce test
caractérise le raffinement pour la population "facile", pas pour les cas
structurellement durs). Réutilise `kabsch`/`rot_err_deg`/`trans_err`/
`quat_wxyz_to_rotmat` de `phase5a_depthmap_matching.py` et
`extract_gt_variable` de `phase5a_weighted_gt_check.py` — aucune
réimplémentation. Stratégie GT uniquement, aucun CNN chargé (même discipline
oracle-first que tout Phase 7) : caractérise le raffinement indépendamment de
la qualité du masque qui l'alimentera en pratique. Seuil de succès volontairement
strict (`--success_rot_thresh 5.0`, `--success_trans_thresh 0.02`) — on veut
vérifier une VRAIE reconvergence vers la pose exacte, pas juste "un peu mieux
qu'au départ".

**Résultat (2026-07-22, N=1252 essais/perturbation, ICP vanille) —
bassin de convergence ÉTROIT, dès les petites perturbations.**
```
Perturbation   Succès(5°/0.02)   RotErr moy   RotErr méd
   10°              27.9%           20.93°        7.97°
   20°              20.2%           34.26°       13.51°
   30°              15.5%           50.42°       20.72°
   45°               8.8%           71.85°       49.02°
   60°               6.3%           88.65°       84.17°
   90°               1.8%          114.48°      124.71°
```
Écart moyenne/médiane énorme à 10° (20.93° vs 7.97°) — signature bimodale
classique de mauvais minimum local en ICP (soit ça reconverge près de la
vraie pose, soit ça diverge complètement). Recalcul à seuils plus
permissifs (directement depuis le CSV, sans relancer) : même à 20°/0.05, le
succès à 10° de perturbation ne dépasse pas 54.7%, et continue de chuter
(30°→34.9%, 60°→14.1%, 90°→5.5%). **L'hypothèse "Pose@30 était trop
strict" est infirmée** — le vrai seuil est plus proche de 10-20° que de 30°,
et même à 10° la reconvergence n'est fiable qu'une fois sur deux au mieux.

**Diagnostic : probablement le glissement dans le plan de contact**, le
même problème géométrique qui avait fait échouer le scoring RANSAC en
Phase 2 ("une pose fausse qui glisse/tourne dans le plan de contact peut
accumuler autant ou plus d'inliers apparents qu'une pose correcte") —
surfaces de fracture quasi-planes (planéité médiane 0.043, Phase 5A.0),
donc un ICP point-à-point classique n'a pas de vraie force de rappel
contre un glissement tangentiel.

**Décision utilisateur (2026-07-22) : tester un ICP avec pénalité
d'orientation avant de conclure.** Important : PAS un "point-to-plane" au
sens strict (minimiser la distance selon la normale) — cette variante
serait en fait PIRE ici, elle n'offre aucune résistance au glissement
tangentiel, c'est sa faiblesse connue sur surface plate. **Implémenté :
`trimmed_icp_normals()`** dans `phase6a_convergence_basin_check.py` — à
chaque itération, après avoir trouvé les plus proches voisins, rejette les
correspondances dont les normales (après rotation courante) ne sont PAS à
peu près opposées (`dot(R@n_i, n_j) < --normal_dot_thresh`, défaut -0.3)
AVANT de refaire Kabsch sur les survivantes — même principe que le scoring
`count_times_quality_and_normal` qui avait fait la différence en Phase 2C
(Pose@30 5%→9.6% à l'époque). Nouveau CLI `--refiner {icp, icp_normals}`.
**Résultat (2026-07-22, N=1248/perturbation, `icp_normals`) — gain net et
substantiel, confirme l'hypothèse du glissement tangentiel.**
```
Perturbation   Succès(vanille)   Succès(normales)   Gain
   10°              27.9%             48.4%          x1.7
   20°              20.2%             43.5%          x2.2
   30°              15.5%             40.9%          x2.6
   45°               8.8%             34.3%          x3.9
   60°               6.3%             28.6%          x4.5
   90°               1.8%             12.3%          x6.8
```
L'écart moyenne/médiane se resserre nettement (30° : médiane=5.34° contre
20.72° en ICP vanille — la médiane est maintenant juste à la frontière du
seuil strict, plus 3-4x au-dessus comme avant) : la pénalité d'orientation
élimine bien le glissement dans le plan de contact, pas un effet marginal.

**Implication pratique pour le seuil utile étage 1 → étage 2 :** le bassin
reste solide jusqu'à 45-60° (34.3%/28.6% de succès), pas seulement 10-20°
comme le suggérait l'ICP vanille — beaucoup plus compatible avec ce que le
depth-map matching produit réellement (les poses `Pose@30` réussies
avaient déjà des erreurs de rotation dans cette gamme). L'écart entre
"précision produite par l'étage 1" et "précision requise par l'étage 2"
est donc bien moins large que redouté après le premier test avec l'ICP
vanille.

**Prochaine étape possible (pas encore actée) :** chaîner concrètement les
deux étages — prendre les poses produites par `phase5a_depthmap_matching.py`
(cascade + zoom) telles quelles comme initialisation, lancer
`trimmed_icp_normals()` dessus, et mesurer le taux de succès final réel du
pipeline complet (pas juste en perturbation contrôlée) sur les ~69% de
paires "faciles".

## Phase 6B — Pipeline complet bout-en-bout (2026-07-22)

**Implémenté : `scripts/phase6b_pipeline_check.py`.** Chaîne réellement les
deux étages validés le même jour, sur les VRAIES poses produites par
l'étage 1 (pas une perturbation contrôlée comme en Phase 6A) :
- Étage 1 : masque GT + zoom (`ring=0`, `--extra_budget` fixe, défaut 1200 —
  meilleur compromis trouvé dans le sweep GT du jour) → `run_cascade()` →
  pose `(R_est, t_est)` si `pose_computed`.
- Étage 2 : `trimmed_icp_normals()` initialisé par `(R_est, t_est)`, sur les
  VRAIS points GT non zoomés (le zoom servait à l'éligibilité de l'étage 1,
  l'étage 2 a déjà assez de vrais points).
- Compare `Pose@30`/`Pose@15` étage-1-seul vs étage-1+étage-2, plus le taux
  de succès strict (5°/0.02) du bassin de convergence Phase 6A.

**Changement nécessaire pour ça : `R_est`/`t_est` exposés.**
`run_match_at_resolution()` (`phase5a_depthmap_matching.py`) et
`run_cascade()` (`phase5a_zoom_resample_check.py`) calculaient déjà la pose
estimée en interne mais la jetaient (ne retournaient que `rot_err`/
`trans_err` déjà comparés au GT) — ajout non-cassant (nouvelles clés dans
les dicts de retour, les appelants existants les ignorent simplement).
Restreint à `n_frac_pts_min >= 50` (population "facile", cohérent avec la
décision du jour). Réutilise `zoom_resample`/`to_input_frame`/`run_cascade`,
`trimmed_icp_normals`, `extract_gt_variable`,
`quat_wxyz_to_rotmat`/`rot_err_deg`/`trans_err` — aucune réimplémentation.
Résultat pas encore lancé — prochaine action : lancer sur le serveur et
lire la table **COMPARAISON ÉTAGE 1 SEUL vs ÉTAGE 1 + ÉTAGE 2**.

**Résultat (2026-07-22, N=1245 paires "faciles" tentées, budget=1200) —
le pipeline bout-en-bout fonctionne, gain réel et substantiel.**
```
n_2frag_seen=1487, skip densité<50=242, zoom échoué=0, échec étage1=240
n_pairs_stage2=1005 (éligibilité étage 1 = 1005/1245 = 80.7%)

Parmi les 1005 paires ayant atteint l'étage 2 :
                        étage 1 seul   étage 1+2   gain
Pose@30°/0.1                24.3%        38.7%     x1.6
Pose@15°/0.05                14.3%        32.8%     x2.3
Succès strict (5°/0.02)        —          22.6%       —
```
Rapporté à TOUTE la population "facile" (1245 paires, dénominateur commun) :
étage 1 seul ≈ 19.6% de `Pose@30` sur l'ensemble ; étage 1+2 ≈ **31.3%**
(`Pose@30`) et ≈ **18.2%** (succès strict 5°/0.02) — le pipeline complet
fait quasiment le double de poses correctes que le depth-map matching seul.

**Conclusion de la journée (Phase 7) :** les deux étages, chacun validé
séparément (zoom+cascade pour l'éligibilité, `icp_normals` pour le
raffinement), se combinent bien en pratique sur les VRAIES poses produites
par l'étage 1 — pas seulement en test contrôlé (Phase 6A). Architecture en
deux temps confirmée comme la bonne direction pour la suite du projet.

## Plan étape par étape (2026-07-22, fin de journée) — discussion de direction

Discussion avec l'utilisateur sur la suite : accord explicite pour NE PAS
enchaîner directement sur un niveau supérieur (ex. réintégrer la Phase 4D
pour l'ordre d'assemblage global) tant que la pose par paire elle-même reste
fragile — priorité à consolider/comprendre le pipeline actuel d'abord.
Questions soulevées par l'utilisateur, avec réponses données :
- **SOM (représentation non-planaire)** : justification pas encore établie.
  Le 2026-07-20 on avait déjà trouvé que "plus de courbure" DÉGRADE le
  matching sous la méthode actuelle (pas confirmé le contraire) — cohérent
  avec un problème de représentation sur faces courbées, MAIS jamais vérifié
  si la tranche difficile `<50` (31% du dataset) est plus courbée que la
  moyenne, ni si Breaking Bad a assez de courbure en général (planéité
  médiane 0.043) pour que ça vaille le coup. À vérifier avant d'investir.
- **Modèle appris pour recomputer rotation+translation à partir des deux
  depth maps** : idée jugée prometteuse, différente de l'échec de la Phase 3
  (matcher point-à-point appris) — une depth map est une grille régulière de
  taille fixe (terrain CNN-friendly), contrairement à un nuage de points
  non-ordonné. Le pipeline validé aujourd'hui génère déjà des paires
  depth-map + vraie transformation à grande échelle, exactement les données
  d'entraînement nécessaires. Chantier plus lourd, à scoper après avoir
  cerné les limites du pipeline actuel (pas en parallèle).
- **Plafond `oracle_overlap_frac`≈0.7** : pas un invariant fixe -- le sweep de
  résolution du jour montre qu'il varie énormément avec la résolution (0.989
  à R=24, 0.404 à R=128). Reste à déterminer si le résidu (à résolution
  fixe) est du bruit d'échantillonnage ou une vraie non-coïncidence
  géométrique du maillage Breaking Bad.

**Plan convenu, dans cet ordre :**
1. **Chaîner le pipeline complet (étage 1 + étage 2) sur `thresh0.3`**
   (condition réelle, CNN) — même principe que
   `phase5a_zoom_resample_thresh03_check.py` (deux datasets en lockstep,
   graines filtrées par clustering, correction `init_rot`) combiné à
   `phase6b_pipeline_check.py` (chaînage étage 1 → étage 2). **Point d'arrêt
   strict : si le résultat est faible, NE PAS monter d'un niveau** — rester
   ici et creuser.
2. **En parallèle (bon marché, n'attend pas le point 1) : deux diagnostics
   ciblés** — (a) la tranche `<50` corrèle-t-elle avec la courbure/planéité
   (répond à la question SOM) ; (b) le plafond `oracle_overlap_frac` :
   bruit d'échantillonnage ou vraie non-coïncidence géométrique (test
   distance maillage-à-maillage à la vraie pose GT, indépendant de tout
   échantillonnage de points).
3. **Visualisations des cas de succès/échec** sur le résultat du point 1
   (adapter `phase5a_visualize_pair.py` au pipeline chaîné `thresh0.3`) —
   transformer l'observation qualitative en hypothèses testables.
4. **Décision sur la suite une fois 1+2+3 en main** : scoper la SOM si 2a le
   justifie ; recadrer les attentes d'overlap selon 2b ; discuter à ce
   moment-là si on scope le modèle appris de recalage depth-map.

**Prochaine action immédiate : point 1** (chaînage `thresh0.3`).

**Implémenté : `scripts/phase6b_pipeline_thresh03_check.py`.** Combine
l'intégralité de la plomberie déjà validée : deux datasets en lockstep +
correction `init_rot` + filtrage des graines par clustering
(`phase5a_zoom_resample_thresh03_check.py`) pour l'étage 1 (zoom + cascade),
puis `trimmed_icp_normals` (`phase6a_convergence_basin_check.py`) initialisé
par `(R_est, t_est)` pour l'étage 2, sur les VRAIS points prédits par le CNN
(masque `thresh0.3` brut, pas les points de zoom synthétiques). Restreint à
`n_frac_pts_min >= 50` (population "facile"). Aucune réimplémentation.
Résultat pas encore lancé — prochaine action : lancer sur le serveur et
lire la table **COMPARAISON ÉTAGE 1 SEUL vs ÉTAGE 1 + ÉTAGE 2** en condition
réelle. Rappel du point d'arrêt convenu : si le résultat est faible, ne pas
enchaîner sur autre chose, revenir au point 2 (diagnostics courbure/overlap)
et point 3 (visualisations) du plan.

**Résultat (2026-07-22, N=1487 vus, N=716 atteignant l'étage 2) — point
d'arrêt atteint, en dessous du seuil des 20% que l'utilisateur avait fixé.**
```
n_2frag_seen=1487, echec etage1=292, atteint etage2=716
Eligibilite etage1 (population >=50) = 716/1008 = 71.0%
                        etage 1 seul   etage 1+2   gain
Pose@30                     10.75%       24.02%     x2.2
Pose@15                      5.17%       19.69%     x3.8
Succes strict (5/0.02)         --        14.11%       --
```
Rapporté à TOUTES les 1487 paires vues (y compris la tranche `<50` mise de
côté) : **éligibilité globale = 716/1487 = 48.2%**, `Pose@30` global
(pipeline complet) ≈ **11.6%**. Le raffinement aide encore plus qu'en GT en
relatif (×2.2/×3.8 contre ×1.6/×2.3), mais le chiffre absolu reste modeste
et sous le seuil des 20% fixé par l'utilisateur comme repère de décision.

**Constat de l'utilisateur, confirmé : l'éligibilité reste le vrai
problème.** Écart notable CNN vs GT sur la MÊME tranche de densité `≥50` :
éligibilité étage 1 = 71.0% en CNN contre 80.7% en GT (2026-07-22, run
précédent) — un écart qui n'est PAS expliqué par la tranche `<50` déjà mise
de côté (celle-ci est la même dans les deux cas). Clarification importante :
les diagnostics 2a (courbure) et 2b (overlap maillage) ne portent PAS sur
cet écart précis — 2a porte sur la population déjà exclue, 2b sur le
plafond de précision, pas l'éligibilité.

**Hypothèse plus directement pertinente, décidée avec l'utilisateur avant
2a/2b : le masque `baseline` (utilisé directement par la cascade de
l'étage 1) n'est JAMAIS filtré par clustering, contrairement aux graines du
zoom** — des faux positifs dispersés du CNN pourraient polluer le calcul du
repère PCA et la recherche de correspondance de la cascade elle-même, pas
seulement la localisation du zoom.

**Implémenté : `--cluster_baseline`** dans
`phase6b_pipeline_thresh03_check.py` — applique le même filtre de cluster
dominant (`dominant_cluster_mask`, déjà utilisé pour les graines de zoom)
au masque baseline envoyé à la cascade. Le filtre de densité `<50` reste
basé sur le compte BRUT (avant clustering), pour comparer sur exactement
les mêmes paires qu'avec l'option désactivée. Résultat pas encore lancé —
prochaine action : relancer avec `--cluster_baseline` et comparer
l'éligibilité étage 1 (cible : se rapprocher de 80.7%, le niveau GT).
Ensuite seulement : points 2a/2b du plan.

**Résultat (2026-07-22, N=721 avec `--cluster_baseline`) — hypothèse
RÉFUTÉE.** Éligibilité quasiment inchangée (721/1008 ≈ 71.5% contre 71.0%
sans clustering du baseline — dans le bruit), et la précision **se
dégrade** (`Pose@30` étage1+2 : 24.0%→17.1%, `Pose@15` : 19.7%→13.5%,
succès strict : 14.1%→9.3%). Le clustering du masque baseline retire
autant de vrais points utiles que de bruit dispersé, sans gain net — donc
ce n'est PAS la dispersion spatiale des faux positifs qui explique l'écart
CNN (71%) vs GT (80.7%) sur la même tranche `≥50`.

**Diagnostic suivant, décidé avec l'utilisateur : précision/rappel du
masque CNN vs GT (qualité intrinsèque, pas dispersion spatiale).**
Implémenté directement dans `phase6b_pipeline_thresh03_check.py` (pas de
nouveau script) : calcule `precision`/`recall` du masque `thresh0.3` par
rapport au label `fracture_surface_gt` (déjà dans le batch `uniform`, tous
deux alignés index-à-index), pour CHAQUE paire (succès ET échec étage 1,
pas seulement celles atteignant l'étage 2 comme le reste du script).
Nouvelle table **PRÉCISION/RAPPEL DU MASQUE CNN vs GT (succès vs échec
étage 1)** — compare les moyennes de `precision_min`/`recall_min` (le côté
le plus pauvre de la paire) entre le groupe qui réussit l'étage 1 et celui
qui échoue.

**Résultat (2026-07-22, N=716 réussit / N=293 échoue) — hypothèse
CONFIRMÉE, referme l'investigation sur l'écart d'éligibilité.**
```
                     N     Precision(min)   Recall(min)
Étage 1 RÉUSSIT    716          83.9%          93.1%
Étage 1 ÉCHOUE     293          62.1%          81.3%
```
Écart net et cohérent sur les deux métriques (-21.8pp précision, -11.8pp
rappel). Confirme que l'écart d'éligibilité CNN (71%) vs GT (80.7%) vient
de la QUALITÉ INTRINSÈQUE de la segmentation sur ces paires précises — pas
de sa dispersion spatiale (déjà écartée par `--cluster_baseline`), pas du
pipeline de matching en aval (clustering/zoom/résolution déjà validés).
Pas catastrophique non plus (81.3% de rappel en échec, pas 20%) — une
dégradation modérée mais suffisante pour faire basculer la cascade.
**Implication : cet écart ne se corrige pas en retouchant le pipeline de
matching — il faudrait améliorer la segmentation du CNN elle-même sur ces
cas, ou accepter la limite.** Investigation refermée, retour au plan :
points 2a/2b (courbure vs tranche difficile, overlap maillage-à-maillage).

**Implémenté : point 2a — `scripts/phase7_curvature_vs_hardbin_check.py`.**
Un seul dataset (CNN, `uniform`) -- pas besoin du maillage, juste des points
et la planéité PCA (`compute_pca_frame`, réutilisée telle quelle). Pour
chaque paire, calcule `n_min` (côté le plus pauvre du masque `thresh0.3`) et
la planéité moyenne/max des deux fragments, compare la distribution entre
la tranche difficile (`<50`) et le reste. Résultat pas encore lancé.

**Implémenté : point 2b — `scripts/phase7_overlap_ceiling_check.py`.**
PAS de rasterisation en grille (élimine toute dépendance à une résolution,
contrairement au sweep du 2026-07-21) — transforme le côté le plus PAUVRE
dans le repère du côté le plus RICHE via la VRAIE pose GT, mesure la
distance au plus proche voisin (`cKDTree`) vers l'autre côté, fraction de
points à moins de `--contact_eps` (0.05, même tolérance que Phase 0/2B) =
overlap continu sans artefact de grille. Stratégie GT uniquement,
oracle-first. Résultat pas encore lancé — si ce chiffre est proche de 1.0,
le plafond ~0.7 mesuré par grille était un pur artefact de discrétisation ;
s'il plafonne aussi nettement en dessous, c'est du bruit d'échantillonnage
ou une vraie non-coïncidence géométrique du dataset (distinction à faire
par un test de sensibilité à la densité, en suivi).

**Prochaine action : lancer les deux diagnostics 2a et 2b sur le serveur.**

**Résultat 2b (2026-07-22, N=1483) — CONCLUSIF, clarifie tout : le plafond
~0.7 était un pur artefact de grille, pas un problème réel.**
`overlap_frac_continuous` moyenne = **99.99%**, médiane = **100%**. Sans
discrétisation en grille, les deux faces de la fracture coïncident quasiment
parfaitement à la vraie pose — exactement ce qu'on attendait théoriquement.
Ni bruit d'échantillonnage ni non-coïncidence géométrique du dataset : la
perte venait de `build_correspondences()` qui exige une coïncidence de case
de grille EXACTE après rotation/arrondi, perdant du signal qui existe
pourtant réellement dans les données. **Piste d'amélioration concrète pour
plus tard : remplacer la correspondance par case de grille par un
appariement plus-proche-voisin direct sur les points** — récupérerait
probablement des correspondances valides aujourd'hui perdues par
arrondi/discrétisation, indépendamment de la résolution choisie.

**Résultat 2a (2026-07-22, N=1475, N=488 dur / N=987 facile) — infirme
l'hypothèse SOM, dans le sens INVERSE de ce qui était attendu.**
```
                    planarity_mean (médiane)
Tranche difficile        0.023
Reste du dataset         0.087
```
La tranche difficile est PLUS PLATE que le reste, pas plus courbée.
Nuance méthodologique probable : avec très peu de points (définition même
de la tranche `<50`), la planéité PCA est biaisée statistiquement vers
"plat" (un petit échantillon a moins de chances de révéler une vraie
courbure) — possiblement pas un vrai signal géométrique, un artefact de la
sparsité elle-même. Dans tous les cas : **aucun argument pour investir dans
la SOM** sur cette population — la difficulté de la tranche `<50` est liée
à la sparsité, pas à la courbure.

**Conclusion des points 2a/2b :** SOM écartée (pas justifiée par les
données) ; piste overlap/correspondance (nearest-neighbor au lieu de
coïncidence de grille) identifiée comme concrète et actionnable pour une
prochaine itération. Prochaine étape à discuter avec l'utilisateur : point
3 (visualisations) ou directement prototyper le fix de correspondance.

**Décision utilisateur : prototyper le fix de correspondance maintenant.**
Implémenté : `build_correspondences_nn()` dans `phase5a_depthmap_matching.py`
— reproduit exactement la même transformation 2D (rotation + translation
trouvée par `match_depthmaps`, inchangée) que `build_correspondences`
original, mais résout la correspondance par **plus-proche-voisin continu**
(`cKDTree` dans le plan (u,v) aligné, tolérance physique absolue
`contact_eps`, même convention que Phase 0/2B) au lieu d'une coïncidence de
case de grille après arrondi. Le flip de normale n'affecte que
l'interprétation de profondeur PENDANT la recherche, pas les coordonnées
(u,v) utilisées ici — pas de traitement spécial nécessaire, les points 3D
appariés sont réels dans les deux cas.

Câblé dans `run_match_at_resolution()` via `getattr(args, "correspondence_mode",
"grid")` — rétrocompatible : tous les scripts existants (qui n'exposent pas
cette option) gardent le comportement "grid" inchangé. Nouveaux flags
`--correspondence_mode {grid,nn}` / `--contact_eps` ajoutés à
`phase5a_depthmap_matching.py`. **Testable directement via l'infrastructure
`--pose_resolution_sweep` déjà existante** (passe par `run_match_at_resolution`
avec `args` inchangé) — pas besoin de toucher au chemin principal
`process_pair` ni d'écrire un nouveau script. Résultat pas encore lancé —
prochaine action : comparer `--correspondence_mode grid` vs `nn` sur le
même sweep de résolution, stratégie `gt` (oracle-first).

**Résultat (2026-07-23), `--pose_resolution_sweep 64 --pose_resolution_sweep_strategies gt`,
`grid` vs `nn`, mêmes paires (`gt.density_resolution_sweep.64_na36`) :**
```
                              grid        nn
n (paires éligibles à R=64)   708        1195
Pose@30 (parmi éligibles)   25.57%      16.99%
cascade_union_eligibility   54.80%      92.49%
```
**Éligibilité : gain massif, +487 paires (54.8%→92.5%), quasiment le
plafond continu ~100% trouvé au diagnostic 2b — à une SEULE résolution
fixe, sans zoom ni clustering ni cascade.** Dépasse déjà, seul, ce que
zoom+cascade+clustering combinés obtenaient (~70-80%) tout au long de la
journée.

Le `Pose@30` par survivant baisse (effet de composition attendu, déjà vu
avec le zoom/budget : les ~487 paires nouvellement récupérées sont celles
que `grid` ratait *complètement* par arrondi, plus dures en moyenne) —
mais en nombre ABSOLU de poses correctes : `grid`≈181, `nn`≈203, soit
**+12% net**, pas une simple dilution. Vérifié bin par bin
(`n_frac_pts_min`) : `nn` augmente N dans TOUTES les tranches (75→348,
297→436, 313→385, 23→26) avec une baisse de précision par tranche modeste,
pas un effondrement.

**Décision utilisateur : combiner `nn` avec le pipeline zoom+cascade déjà
validé (Phase 6B), pour voir si les gains se cumulent.** Comme
`run_cascade()`/`run_match_at_resolution()` transmettent déjà l'objet
`args` complet, aucun changement de plomberie nécessaire — seuls
`--correspondence_mode {grid,nn}` et `--contact_eps` ont été ajoutés en
CLI passthrough à `phase6b_pipeline_check.py` (GT) et
`phase6b_pipeline_thresh03_check.py` (CNN/thresh0.3), défaut `grid`
(rétrocompatible). Prochaine action : relancer `phase6b_pipeline_check.py`
(GT, oracle-first) avec `--correspondence_mode nn`, comparer à la
baseline `grid` du 2026-07-22 (éligibilité 80.7%, Pose@30 stage1+2
38.7%), puis seulement si concluant, la version thresh0.3.

**Résultat (2026-07-23), pipeline complet GT (zoom+cascade+nn), même
population "facile" (1245 paires) que la baseline `grid` du 2026-07-22 :**
```
n_2frag_seen=1487, skip densité<50=242, zoom échoué=0, échec étage1=11
n_pairs_stage2=1234 (éligibilité étage 1 = 1234/1245 = 99.1%)

Parmi les 1234 paires ayant atteint l'étage 2 :
                        étage 1 seul   étage 1+2
Pose@30°/0.1                21.5%        37.9%
Pose@15°/0.05                13.9%        32.6%
Succès strict (5°/0.02)        —          23.4%
```
Rapporté à TOUTE la population facile (1245, dénominateur commun avec la
baseline grid) : Pose@30 étage1+2 ≈ **37.6%** (vs 31.3% en grid), succès
strict ≈ **23.2%** (vs 18.2% en grid).

**Le fix referme quasiment tout l'écart d'éligibilité (240 échecs étage 1
→ 11) SANS diluer la précision par paire éligible** (38.7%→37.9%
Pose@30, 32.8%→32.6% Pose@15 -- quasi stable, contrairement au test isolé
à R=64 où la précision par survivant chutait nettement). Le zoom+cascade
absorbe apparemment la difficulté des paires nouvellement récupérées.
Gain net sur toute la ligne : éligibilité ET précision absolue montent
ensemble, pas de compromis. `nn` devient le mode par défaut recommandé
pour la suite. Prochaine action : version thresh0.3 (CNN, condition
réelle) avec `--correspondence_mode nn`, pour confirmer que ça tient hors
oracle.

**Résultat (2026-07-23), pipeline complet thresh0.3/CNN (zoom+cascade+nn),
même population (1487 paires vues) que la baseline `grid` du 2026-07-22 :**
```
n_2frag_seen=1487, échec étage1=71, atteint étage2=931

Parmi les 931 paires ayant atteint l'étage 2 :
                        étage 1 seul   étage 1+2
Pose@30°/0.1                 8.6%        23.5%
Pose@15°/0.05                 4.0%        18.0%
Succès strict (5°/0.02)        —          10.8%

Précision/rappel masque CNN vs GT :
                     N     Precision(min)   Recall(min)
Étage 1 RÉUSSIT     931          79.6%          90.1%
Étage 1 ÉCHOUE       71          63.0%          84.1%
```
Rapporté à TOUTES les 1487 paires (dénominateur commun avec la baseline
grid) :
```
                     grid (2026-07-22)   nn (2026-07-23)
Éligibilité               48.2%              62.6%
Pose@30 global             11.6%              14.7%
Succès strict global       ≈6.8%              ≈6.8%  (inchangé)
```
**Le fix aide nettement moins qu'en GT (99.1% d'éligibilité) -- confirme
encore la conclusion du 2026-07-22 : l'écart d'éligibilité CNN vs GT vient
de la qualité intrinsèque du masque CNN, pas du mécanisme de
correspondance.** L'écart précision/rappel réussite-vs-échec est un peu
moins marqué qu'avant (21.8pp→16.6pp de précision, 11.8pp→6.0pp de
rappel) -- `nn` récupère une partie des cas à masque légèrement dégradé,
mais pas les pires. Le succès strict global reste identique (≈6.8%) : les
~215 paires nouvellement récupérées gagnent du Pose@30 (tolérance large)
mais pas du succès strict -- cohérent avec l'effet de composition observé
partout ailleurs (paires plus dures en moyenne).

**Toujours sous le seuil des 20% fixé comme point d'arrêt (2026-07-22) :
Pose@30 global = 14.7%.** Le fix de correspondance a fait ce qu'il
pouvait sur ce levier -- la limite restante est la qualité de
segmentation du CNN elle-même, un levier différent (amélioration/retrain
du CNN, pas du pipeline de matching). Prochaine décision à prendre avec
l'utilisateur : visualisations succès/échec (point 3 du plan) ou pivot
vers l'amélioration du CNN.

## Step 16 CNN — boundary loss + spatial coherence loss (2026-07-24 → 2026-07-28, TERMINÉ, résultat net-négatif)

Suite directe du pivot évoqué ci-dessus : le diagnostic géométrique du
2026-07-23 (`phase7_isolated_fp_prevalence_check.py`, 7490 fragments) a
confirmé que la limite restante sur `Pose@30 global = 14.7%` (vs 37.6% en
GT) vient de la qualité intrinsèque du masque CNN Step 15, pas du pipeline
de matching (trois tentatives de fix côté pipeline — `dominant_cluster_mask`,
`compute_pca_frame_robust`, `remove_tiny_clusters_mask` — toutes net-négatives
ou neutres). Décomposition des faux positifs en deux causes de poids très
inégal : imprécision de frontière (85.3% du volume de FP) et amas isolés
(14.7% du volume, mais 56.1% des fragments touchés — assez pour biaiser le
repère PCA du matching même à faible volume). Détail complet du diagnostic
et du raisonnement dans `AVANCEES_POST_PRESENTATION.md` section 10.

**Implémentation :**
- `assembly/models/cnn_segmentation_model.py` : `forward()` expose
  `frag_sizes`/`points_xyz_flat` (nécessaire pour du kNN/clustering par
  fragment dans `criteria()`) ; deux nouveaux termes de loss ajoutés à
  `criteria()` — `boundary_bce_loss` (GT-based, kNN k=5, même définition que
  la métrique Boundary F1 de `analyze_errors.py`) et `coherence_bce_loss`
  (prédiction-based, clustering par composantes connexes `eps=0.02`/
  `min_cluster_size=10`, mêmes paramètres que le diagnostic ci-dessus, ne
  pénalise que les points isolés ET GT-négatifs).
- `configs/experiment/cnn_step16_boundary_coherence.yaml` : fine-tuning
  DEPUIS Step 15 (`pretrained_ckpt`), `coherence_warmup_epochs=2`, poids
  `boundary_weight = coherence_weight = 0.2`, 30 epochs.
- `scripts/test_step16_losses.py` : smoke test synthétique des quatre
  nouvelles fonctions (formes, cas limites).

**Trois OOM en cascade au lancement, tous corrigés (cf. commits
`381059f`/`520b314`/`76cf183`/`09b28f3`/`8b5698c`) :**
1. `compute_boundary_mask` en `torch.cdist` GPU par fragment (matrice O(n²))
   → remplacé par `sklearn.neighbors.NearestNeighbors` CPU (même algo que
   `analyze_errors.py`, plus sûr, kd_tree O(n log n)).
2. Bug préexistant (Step 9, PAS causé par Step 16) dans `_knn_indices_batched`
   (`hybrid_geometry_features.py`) : matrice dense `(K, N, N)` pour TOUS les
   fragments du batch en un seul appel — jamais un problème pour Step 15
   (batches "chanceux"), explosif ici (jusqu'à 15 Go pour un seul appel) sur
   les GPU 7.6 Go du labo → chunké sur K, résultat vérifié identique
   numériquement à la version non-chunkée.
3. `batch_size=32` restait trop gros pour l'activation memory du
   forward/backward du U-Net sur ce GPU (Quadro RTX 4000, 7.6 Go) même après
   les deux fix précédents — descendu par paliers 32→16→8→4 (le pic mémoire
   dépend de la composition du batch : nombre de fragments par objet, jusqu'à
   `max_parts=20`, donc un batch malchanceux peut survenir n'importe quand
   pendant le run, pas seulement au démarrage) avec `accumulate_grad_batches`
   compensatoire (8) pour garder le même batch effectif (32) que Step 15.

**Résultat final (30/30 epochs, confirmé identique au checkpoint
intermédiaire epoch ~21 — pas un artefact de sous-entraînement) :**

| | Step 15 | Step 16 |
|---|---|---|
| F1 pooled | 96.7% | 96.27% — quasi flat |
| Boundary F1 | 83.2% | 84.10% — légèrement mieux |
| Amas isolés (% volume FP) | 14.7% | **0.4%** — chute massive |
| Fragments touchés par amas isolés | 56.1% | **8.2%** — chute massive |
| Éligibilité étage 1 | 62.6% (931/1487) | 70.3% (1045/1487) — mieux |
| **Pose@30 global, absolu** | **219 paires (14.7%)** | **171 paires (11.5%)** — **-22%** |
| **Succès strict global, absolu** | **101 paires (6.8%)** | **82 paires (5.5%)** — **-19%** |

**Conclusion : résultat net-négatif sur le vrai test (pipeline de matching
en aval), malgré un succès quasi total du chantier 2 sur sa propre métrique
diagnostique** (amas isolés : 56.1%→8.2% des fragments touchés). L'hypothèse
de départ — que les amas isolés de faux positifs biaisent le repère PCA du
matching et cassent la cascade — est **infirmée empiriquement** : les
supprimer via la coherence loss n'améliore pas le matching, et semble même
coûter en couverture utile du masque (le nombre absolu de poses réussies
baisse malgré une éligibilité étage 1 en hausse). Hypothèse pour expliquer
l'écart : la coherence loss, en pénalisant tout point prédit positif isolé
ET GT-négatif, supprime probablement aussi des points de bord "bruyants
mais utiles" qui contribuaient (même de façon imparfaite) à la stabilité du
repère PCA/depth-map matching — la frontière entre "amas isolé nuisible" et
"point de bord légitime mais isolé" n'est pas aussi nette dans la réalité
du matching que dans le diagnostic F1/segmentation qui a motivé ce chantier.

**Décision (2026-07-28, avec l'utilisateur) : Step 15 reste le modèle CNN
final retenu.** Pas de nouvelle itération sur ce chantier (ablation
boundary-seul vs coherence-seul envisagée pour isoler le coupable, poids
réduits envisagés aussi, écartés faute de gain attendu clair vs le coût
d'un nouveau run) — documenté comme résultat négatif informatif, même statut
que Step 12 (overlap channels) dans `AVANCEES_POST_PRESENTATION.md`.
L'écart CNN vs GT sur le pipeline de matching (14.7% vs 37.6% de Pose@30)
reste donc ouvert ; la piste "amélioration du CNN par la loss" est
maintenant explorée et écartée pour ces deux angles précis (frontière,
cohérence spatiale) — un futur chantier devra soit cibler une cause
différente, soit revenir au pipeline de matching lui-même avec un œil neuf.
