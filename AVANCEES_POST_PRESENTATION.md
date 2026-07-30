# Avancées post-présentation tuteur — GARF CNN Fracture Segmentation

> Période couverte : après Step 11 (F1=86.9%)
> Checkpoint de référence : `output/cnn_step11_hard_sampling/last.ckpt`

---

## 1. Point de départ — Step 11

| Métrique (fragment-level, val) | Valeur |
|---|---|
| Mean F1 | 83.2% ± 21.1% |
| Mean Precision | 83.2% |
| Mean Recall | 86.0% |
| Median F1 | 90.2% |
| Params | 521 K |

**Architecture Step 11 :** U-Net (depth=4) + normals + Gaussian splatting + bilinear projection + global context + feature fusion (max) + geometric features (curvature, roughness, consistency) + Focal Loss + WeightedRandomSampler.

**Problème identifié visuellement :** plusieurs points 3D projetant dans le même pixel reçoivent la même prédiction 2D (backprojection naïve). La projection est une compression avec perte — le modèle prédit une *zone*, pas une *surface*.

---

## 2. Phase 1 — Métriques géométriques (implémentées)

### Motivation

Le F1 global ne mesure pas la qualité des frontières. Deux modèles peuvent avoir le même F1 global mais des frontières très différentes.

### Nouvelles métriques ajoutées dans `scripts/analyze_errors.py`

| Métrique | Définition |
|---|---|
| **Boundary F1** | F1 calculé uniquement sur les points frontière (kNN k=5 : point fracture avec au moins 1 voisin intact, ou vice versa) |
| **Hausdorff distance** | `max(max_nn_dist(pred→gt), max_nn_dist(gt→pred))` — pénalise les outliers géométriques |
| **Chamfer distance** | `mean_nn_dist(pred→gt) + mean_nn_dist(gt→pred)` — erreur géométrique moyenne |

### Baseline Step 11 (métriques géométriques, 5386 fragments)

| Métrique | Mean | Median |
|---|---|---|
| Global F1 | 83.2% | 90.2% |
| **Boundary F1** | **68.2%** | **71.6%** |
| Hausdorff distance | 0.389 | 0.326 |
| Chamfer distance | 0.047 | 0.016 |
| Pearson corr(F1, Boundary F1) | 0.881 | — |

**Constat clé :** gap de **~15 points** entre Global F1 (83.2%) et Boundary F1 (68.2%). Le modèle prédit bien l'intérieur des surfaces mais est nettement moins précis aux frontières.

---

## 3. Analyse de corrélation occupancy ↔ erreur

### Hypothèse initiale (à corriger)

> Les erreurs sont concentrées dans les pixels à forte occupancy (beaucoup de points superposés → conflit de prédiction).

### Ce que les données montrent

Analyse conduite sur **3 modèles indépendants** (Step 9, 10, 11) :

| Modèle | corr(occupancy, erreur) |
|---|---|
| Step 9 | −0.031 |
| Step 10 | −0.032 |
| Step 11 | −0.016 |

**La corrélation est négative.** Les pixels à forte occupancy ont un *taux d'erreur plus faible*.

Détail Step 11 par bin d'occupancy :

| Points/pixel | % du total | Taux d'erreur |
|---|---|---|
| 1 | 30.6% | **9.9%** ← le plus élevé |
| 2 | 12.5% | 9.6% |
| 3 | 8.1% | 8.9% |
| 4-5 | 3.6% | **7.1%** ← minimum |
| 6-10 | 1.1% | 10.6% |
| 11+ | 0.1% | 1.1% |

### Interprétation corrigée

Les pixels à haute occupancy correspondent au **cœur des surfaces** (fracture ou intact) — beaucoup de points du même type convergent → consensus facile → faible erreur.

Les pixels à faible occupancy (occ=1) sont les **points isolés à la frontière** — un seul point projette, sans contexte local 2D robuste. Le modèle doit décider sans support spatial suffisant.

> **Reformulation du problème :**
> Ce n'est pas l'overlap qui crée les erreurs. C'est le **manque de contexte 2D sur les points isolés ou de frontière**. À faible occupancy, le point ne peut être distingué que grâce à ses features 3D propres : normale locale, courbure, roughness, position relative.

### Conséquence sur le plan

- **Phase 2 (channels overlap)** : impact probablement limité — les erreurs ne sont pas là où la densité est élevée. Reste une expérience de validation de l'hypothèse.
- **Phase 3 (PointHead)** : devient la priorité réelle — combine contexte 2D + features 3D pour chaque point individuellement, exactement là où le CNN manque d'information.
- **Phase 4 (high-res adaptative)** : doit cibler les zones d'**incertitude** (proba ∈ [0.4, 0.6] + faible occupancy), pas les zones denses.

---

## 4. Phase 2 — Channels overlap (implémentés, Step 12 terminé — résultat négatif)

### Architecture

4 channels ajoutés à la projection 2D (après les channels existants, pour compatibilité fine-tuning) :

| Channel | Calcul | Intérêt |
|---|---|---|
| `count_norm` | `cnt / cnt.max()` normalisé par vue | Densité de points par pixel |
| `depth_min` | profondeur du point le plus proche | — |
| `depth_max` | profondeur du point le plus loin | — |
| `depth_spread` | `depth_max − depth_min` | > 0 → points à profondeurs différentes dans le même pixel |

**`in_ch` : 8 → 12** (Step 11 → Step 12)

### Mécanisme de fine-tuning (sans réentraînement de zéro)

Problème : le premier Conv2d du U-Net change de shape `(16, 8, 3, 3)` → `(16, 12, 3, 3)`.

Solution implémentée — hook Lightning `on_load_checkpoint` + `on_fit_start` :
- **Chargement initial** (`on_fit_start`) : lit le checkpoint Step 11, copie les 8 colonnes existantes du premier Conv2d, initialise les 4 nouvelles colonnes à 0
- **Reprise d'entraînement** (`on_load_checkpoint`) : détecte le chargement depuis un checkpoint Step 12, skip le chargement Step 11 → pas d'écrasement des poids appris

```
Scénario A — lancement initial Step 12 :
  on_load_checkpoint : NON appelé
  on_fit_start       : charge Step 11, pad 4 colonnes à 0 ✓

Scénario B — reprise Step 12 interrompue :
  on_load_checkpoint : fixe shape si nécessaire, marque _pretrained_loaded=True
  on_fit_start       : SKIP (poids Step 12 préservés) ✓
```

### Config Step 12

```yaml
lr: 1e-4        # réduit pour fine-tuning
max_epochs: 30  # suffisant depuis Step 11 déjà convergé
loss: Focal + Dice (même que Step 11 — isoler l'effet des channels)
weighted_sampler: true
```

### Params

| Composant | Delta |
|---|---|
| Premier Conv2d (8→12 in_ch) | +576 |
| **Total Step 12** | **~522 K** |

### Résultat Step 12 — entraîné (30 epochs, checkpoint final `last-v1.ckpt`)

| Métrique (fragment-level, val "everyday", 5386 fragments) | Step 11 (baseline) | **Step 12** |
|---|---|---|
| Mean F1 | 83.2% ± 21.1% | **71.4% ± 31.1%** |
| Median F1 | 90.2% | **84.9%** |
| Mean Precision | 83.2% | 81.5% |
| Mean Recall | 86.0% | 69.3% |
| Boundary F1 | 68.2% | **55.8%** |
| Hausdorff distance | 0.389 | 0.372 |
| Chamfer distance | 0.047 | 0.049 |
| Fragments F1 > 0.9 | — | 1888/5386 (35.0%) |
| Fragments F1 < 0.5 | — | 1024/5386 (19.0%) |
| Pearson corr(occupancy, erreur) | −0.016 | −0.020 |
| Meilleur seuil (F1 global pooled) | — | 0.40 → F1=88.5% |

**Constat : régression nette par rapport à Step 11**, sur le F1 fragment-level *et* sur le Boundary F1 (-12.4 points). Les channels d'overlap (`count_norm`, `depth_min/max`, `depth_spread`) n'apportent pas de signal utile au fine-tuning et semblent même perturber le réseau pré-entraîné — cohérent avec l'analyse de corrélation occupancy↔erreur (section 3) qui prédisait déjà un impact limité. **Conclusion : abandonner cette piste**, le PointHead (Step 13) reste la bonne direction.

---

## 5. Phase 3 — PointHead MLP (implémenté, Step 13 terminé)

### Problème structurel adressé

La backprojection 2D→3D actuelle : chaque point reçoit la prédiction du pixel dans lequel il projette. Deux points dans le même pixel ont une prédiction identique (ou quasi-identique via bilinéaire).

**Avec le PointHead :** pour chaque point individuellement :

```
Feature map U-Net (16ch, 128×128)
         ↓ Conv 1×1
Feature map projetée (64ch, 128×128)
         ↓ bilinear sampling au point i
f_2D_i ∈ R^64

Features 3D du point i :
  xyz_norm (3) + normales (3) + curvature + roughness + consistency (3)
         ↓ 3D encoder MLP
f_3D_emb_i ∈ R^32

concat(LayerNorm(f_2D_i), f_3D_emb_i) ∈ R^96
         ↓ MLP(96→128→64→1)
proba_i
```

### Pourquoi la branche 3D dédiée

Risque : avec concat(f_2D=64, f_3D=9), les features 3D représentent 12% de l'entrée → peuvent être "noyées" par les features 2D, surtout si le CNN seul suffit pour les cas faciles.

Solution : **3D encoder MLP séparé** qui projette les 9 features 3D vers 32 dims avant concaténation. Les features 3D ont autant de "capacité" que les features 2D dans la fusion.

```
f_3D (9) → Linear(9→32) → ReLU → LayerNorm(32)  ─┐
                                                    ├─ Linear(96→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
f_2D (64) → LayerNorm(64) ─────────────────────────┘
```

### Pourquoi ça devrait marcher sur les points à faible occupancy

À occupancy=1, un point isolé a une feature 2D qui ne reflète que le contexte voisin (le pixel est entouré de peu de points). Le CNN ne peut pas distinguer un point fracture d'un point intact dans cette zone.

Avec le PointHead, ce point dispose en plus de sa normale locale, de sa courbure, et de sa roughness — des attributs 3D qui caractérisent géométriquement si un point est sur une surface fracturée (rugueuse, courbure forte, normale différente des voisins) ou intacte.

### LR différentiel

```
CNN backbone (pré-entraîné) : lr = 5e-5
PointHead (nouveau)          : lr = 1e-4  (× 2.0)
```

### Params Step 13

| Composant | Params |
|---|---|
| feat_proj Conv2d(16→64, 1×1) | 1 024 |
| PointHead (encoder + MLP) | 21 249 |
| **Delta Step 13** | **+22 273** |
| **Total Step 13** | **~544 K** |

Comparaison : PTv3 GARF = 12 727 K — **Step 13 est 23× plus léger**.

### Résultat Step 13 — entraîné (30 epochs, depuis Step 12)

Checkpoint final : `output/cnn_step13_point_head/last-v3.ckpt` (= `epoch-29.ckpt`, confirmé complet).

| Métrique (fragment-level, val "everyday", 5386 fragments) | Step 11 (baseline) | Step 12 | **Step 13** |
|---|---|---|---|
| Mean F1 | 83.2% ± 21.1% | 71.4% ± 31.1% | **89.3% ± 19.8%** |
| Median F1 | 90.2% | 84.9% | **96.2%** |
| Mean Precision | 83.2% | 81.5% | **93.1%** |
| Mean Recall | 86.0% | 69.3% | **88.4%** |
| **Boundary F1** | 68.2% | 55.8% | **82.0%** |
| Hausdorff distance | 0.389 | 0.372 | **0.249** |
| Chamfer distance | 0.047 | 0.049 | **0.032** |
| Fragments F1 > 0.9 | — | 1888/5386 (35.0%) | **4082/5386 (75.8%)** |
| Fragments F1 < 0.5 | — | 1024/5386 (19.0%) | **293/5386 (5.4%)** |
| Pearson corr(F1, Boundary F1) | 0.881 | 0.956 | 0.900 |
| Pearson corr(occupancy, erreur) | −0.016 | −0.020 | **−0.006** |
| Meilleur seuil (F1 global pooled) | — | 0.40 → F1=88.5% | 0.45 → **F1=94.6%** |

**Objectif initial largement atteint : Boundary F1 = 82.0%, bien au-dessus du seuil cible de 72%** (+13.8 points vs Step 11, +26.2 points vs Step 12). Le Hausdorff et le Chamfer chutent fortement (-36% et -32% vs baseline) : le PointHead corrige bien les erreurs géométriques aux frontières, comme prédit en section 3. La corrélation occupancy↔erreur reste quasi nulle, confirmant que ce n'est plus la densité de points qui pilote l'erreur résiduelle.

**Diagnostic de généralisation (everyday vs artifact)** — c'est ce constat qui motive Step 14 :

| Catégorie (val) | F1 | Precision | Recall |
|---|---|---|---|
| everyday | 92.3% | — | — |
| artifact | 82.3% | 78.8% | ~92% |

**Constat clé :** le recall transfère bien d'une catégorie à l'autre (~92%), mais la précision s'effondre sur `artifact` (78.8%). Interprétation : le U-Net apprend des raccourcis visuels liés à la distribution `everyday` (formes/textures de projection spécifiques) qui ne généralisent pas à des géométries différentes → plus de faux positifs sur `artifact`.

---

## 6. Infrastructure technique

### Fonctions ajoutées

| Fichier | Fonction | Usage |
|---|---|---|
| `projection_mapping_utils.py` | `sample_features_at_points()` | Bilinear sampling de feature maps (K,V,C,H,W) → liste (N_k, C) par fragment |
| `projection_3d_to_2d.py` | `use_overlap_channels=True` | Calcul de count/depth_min/depth_max/depth_spread via `scatter_reduce_` |
| `cnn_segmentation_model.py` | `on_load_checkpoint()` | Fix automatique du mismatch de shape du premier Conv2d |
| `cnn_segmentation_model.py` | `on_fit_start()` | Chargement partiel des poids depuis checkpoint précédent |
| `analyze_errors.py` | `compute_geometric_metrics()` | Boundary F1, Hausdorff, Chamfer par fragment |
| `analyze_errors.py` | `compute_occupancy_correlation()` | Corrélation occupancy ↔ erreur + bar chart |
| `collect_cnn_results.py` | Steps 12 et 13 dans STEPS | Intégration dans le tableau d'ablation |

### Tableau d'ablation complet (val, fragment-level)

| Step | Description | F1 | Params |
|---|---|---|---|
| Step 1 | Baseline | 68.1% | ~150 K |
| Step 7 | U-Net | 77.1% | ~350 K |
| Step 8b | FeatFuse(max) | 81.9% | ~380 K |
| Step 9 | GeoFeatures | 85.7% | 521 K |
| Step 10 | Tversky+DistCentroid | 65.6% | 521 K |
| **Step 11** | **FocalLoss+HardSampling** | **88.4%** | **521 K** |
| Step 12 | +OverlapChannels | 71.4% (régression, abandonné) | ~522 K |
| Step 13 | +PointHead MLP | 89.3% (mean frag.) / 94.6% (pooled @ seuil 0.45) | ~544 K |
| Step 14 | +RandomRotate+InstanceNorm (généralisation) | 92.6% (everyday) / 88.4% (artifact) — succès partiel | ~544 K |
| **Step 15** | **Modèle final, from scratch** | **96.7% (everyday) / 91.5% (artifact, zero-shot)** — meilleur de la série | ~543 K |
| PTv3 GARF | référence | ~90.9% | 12 727 K |

*Step 12/13 : F1 fragment-level (analyse `analyze_errors.py`), pas la même méthodologie exacte que les steps 1-11 (val/coarse_seg_f1 loggé par Lightning) puisque leurs `metrics.csv` de validation n'ont pas été conservés — recalculé directement depuis les checkpoints finaux.*

---

## 7. Comparaison rigoureuse CNN (Step 15) vs GARF-mini (PTv3)

### Pourquoi une nouvelle comparaison

Les premières comparaisons qualitatives (4 objets, doc initial) utilisaient un pipeline d'inférence GARF maison, simplifié et pas totalement aligné avec son protocole d'entraînement. En préparant un mail de synthèse pour le tuteur, deux bugs réels ont été trouvés et corrigés dans `scripts/analyze_errors.py` / `assembly/models/projection_mapping_utils.py` :

1. **`extract_fragment_list` supposait une taille égale par fragment** (`num_pts = N_total // max_parts`), valide uniquement pour `sample_method=uniform` (CNN). Avec `sample_method=weighted` (GARF, tailles de fragment variables), ce reshape découpait les points n'importe où, jetant la majorité des points réels du diagnostic. → ajout d'un chemin de découpe par offsets réels (`points_per_part`) pour le cas non-uniforme.
2. **GARF/FracSeg n'est pas robuste au batching multi-objets** (`batch_size>1`) — perd ~10 points de F1 par rapport à `batch_size=1` (la config officielle `eval_frac_seg.yaml` le fixe d'ailleurs explicitement à 1). Cause probable : bookkeeping d'offsets/graphe par-objet non conçu pour mélanger plusieurs objets.

Après ces corrections, les chiffres GARF ont été **validés en croisant deux méthodes indépendantes** : notre script `analyze_errors.py --model_type garf` (Mean F1 fragment-level, Boundary F1, Hausdorff, Chamfer) et le chemin officiel `eval_segmentation.py` + `trainer.validate()` (F1 pooled au seuil natif 0.5, calculé par le `validation_step` de PTv3 lui-même, sur l'intégralité du val set).

### Résultat final (val set complet, everyday in-domain + artifact zero-shot pour les deux modèles)

| Métrique | CNN Step 15 (everyday) | CNN Step 15 (artifact, zero-shot) | GARF-mini (everyday) | GARF-mini (artifact, zero-shot) |
|---|---|---|---|---|
| Params | 544 K | 544 K | 12 725 K | 12 725 K |
| Mean F1 fragment-level | 86.8% | 76.7% | 84.8% | 81.3% |
| Boundary F1 | 80.0% | 71.8% | **86.4%** | **85.1%** |
| Hausdorff distance | 0.251 | 0.505 | 0.337 | 0.439 |
| Chamfer distance | 0.050 | 0.131 | 0.096 | 0.128 |
| **F1 pooled (seuil natif 0.5, validé Lightning)** | **94.4%** | **88.8%** | 88.3% | 83.7% |

**Au seuil natif 0.5, le CNN devance GARF-mini sur le F1 pooled dans les deux catégories** (94.4% vs 88.3% everyday ; 88.8% vs 83.7% artifact), avec 23× moins de paramètres. **GARF garde l'avantage sur le Boundary F1**, surtout en zero-shot (85.1% vs 71.8%) — ses features de point-transformer semblent mieux généraliser la géométrie des frontières sous changement de domaine.

### Réserve méthodologique importante — budget de points par fragment

Le CNN échantillonne **5000 points par fragment** (`sample_method=uniform`), alors que GARF-mini échantillonne **5000 points au total pour l'objet entier**, répartis entre fragments selon leur aire (`sample_method=weighted`). Ce n'est pas un choix arbitraire de leur part : PTv3 traite tous les fragments d'un objet **conjointement** (nécessaire pour la tâche de réassemblage, qui doit comparer les fragments entre eux), donc son coût de calcul global force un budget de points partagé. Le CNN, lui, traite chaque fragment **indépendamment** via sa projection 2D — son coût ne dépend pas du nombre de fragments, donc il peut se permettre un budget généreux et fixe par fragment.

8 visualisations qualitatives (`output/viz_step15/*.html`, GT/CNN/GARF, seuil 0.5) confirment l'effet :

| Objet | Fragments | Pts/fragment (GARF) | F1 CNN | F1 GARF |
|---|---|---|---|---|
| BeerBottle (everyday) | 2 | 2500 | 0.978 | 0.927 |
| BeerBottle (everyday) | 15 | 333 | 0.972 | **0.654** |
| Bottle (everyday) | 15 | 333 | 0.982 | **0.678** |
| Bottle (everyday) | 12 | 417 | 0.977 | **0.615** |
| artifact #1 | 2 | 2500 | 0.956 | 0.794 |
| artifact #2 (zero-shot) | 11 | 454 | 0.835 | **0.894** ← GARF gagne |
| artifact #3 | 2 | 2500 | 0.977 | 0.565 |
| artifact #4 | 3 | 1667 | 0.809 | 0.862 |

Le F1 de GARF s'effondre presque proportionnellement au nombre de fragments sur `everyday` (0.93→0.65 entre 2 et 15 fragments), cohérent avec une limite de densité de points plutôt qu'une limite architecturale pure. **Mais** sur deux objets `artifact` (zero-shot pour les deux modèles), GARF bat le CNN malgré le même désavantage de points — la robustesse de PTv3 sous changement de domaine compte aussi.

**Conclusion à retenir** : une partie de l'avantage du CNN vient de son architecture qui permet un échantillonnage plus dense par fragment — un avantage pratique réel (pas un artefact de comparaison déloyale), mais qui doit être nommé explicitement plutôt que présenté comme une supériorité architecturale pure et simple.

---

## 8. Phase 4 — Step 14 : généralisation inter-domaines (terminé — succès partiel)

### Motivation

Step 13 montre un gap de généralisation entre catégories : F1=92.3% sur `everyday`, F1=82.3% sur `artifact`, recall stable (~92%) mais precision qui s'effondre (78.8% sur `artifact`). Le CNN apprend des raccourcis visuels spécifiques à `everyday` plutôt que des features géométriques universelles.

### Deux changements implémentés

1. **Random rotation SO(3)** (`Project3DTo2D.random_rotate`) — rotation aléatoire des points (et normales) appliquée **uniquement quand `self.training=True`**, avant projection. L'évaluation utilise toujours les 3 vues fixes habituelles, sans bruit. But : empêcher le CNN de mémoriser "une fracture vue de face ressemble à X" et le forcer à apprendre des features indépendantes de l'orientation.

2. **Instance Normalization** (`norm_type="instance"` dans `_ConvBnRelu` / `UNetBackbone`) — remplace BatchNorm dans le U-Net. `affine=True` : le réseau garde des paramètres γ/β appris, mais BatchNorm encodait implicitement les statistiques globales de la distribution d'entraînement (`everyday`) ; InstanceNorm normalise chaque image indépendamment → agnostique au domaine.

### Config Step 14

```yaml
defaults: cnn_ablation_shared + step13 architecture (PointHead, geo_features, overlap_channels)
random_rotate: true
norm_type: instance
optimizer.lr: 5e-5
max_epochs: 20
pretrained_ckpt: output/cnn_step13_point_head/last-v3.ckpt   # epoch 29/30, confirmé complet
categories: [everyday, artifact]   # entraînement sur les deux catégories
```

**Point d'attention résolu pendant le lancement** : la config Step 14 oubliait initialement `use_overlap_channels: true` (hérité de Step 12/13, `in_ch=12`). Sans ce flag le modèle se reconstruit avec `in_ch=8` et tronque silencieusement les 4 channels d'overlap appris (`count_norm`, `depth_min`, `depth_max`, `depth_spread`) en chargeant le checkpoint Step 13. Corrigé avant le lancement définitif.

Objectif : artifact F1 > 90%, everyday F1 stable.

### Résultat Step 14 — entraîné (20 epochs, `batch_size=2`, `limit_train_batches=3000`)

Checkpoint final : `output/cnn_step14_generalization/last.ckpt`. Évalué séparément par catégorie (`analyze_errors.py --categories everyday|artifact --geometric --sweep_threshold`).

| Métrique | everyday (Step 13) | **everyday (Step 14)** | artifact (Step 13)* | **artifact (Step 14)** |
|---|---|---|---|---|
| F1 pooled (meilleur seuil) | 92.3% (seuil ?) | **92.6%** (seuil 0.45) | 82.3% | **88.4%** (seuil 0.55) |
| Mean F1 fragment-level | 89.3% | 87.5% | — | 80.0% |
| Median F1 | 96.2% | 93.7% | — | 89.2% |
| Boundary F1 | 82.0% | **72.5%** | — | 68.4% |
| Fragments F1 > 0.9 | 75.8% | 65.9% | — | 47.0% |

*\* chiffres Step 13/artifact issus du diagnostic initial (commentaire du yaml Step 14), pas recalculés avec la même méthodologie exacte que les autres colonnes — comparaison indicative.*

**Bilan : succès partiel.**
- ✅ **Le gap de généralisation se réduit de moitié** : 10.0 points (92.3% − 82.3%) → 4.2 points (92.6% − 88.4%). Le F1 pooled sur `artifact` progresse de +6.1 points.
- ❌ **L'objectif "artifact F1 > 90%" n'est pas atteint** (88.4%).
- ❌ **"everyday F1 stable" n'est que partiellement vrai** : le F1 pooled reste stable (+0.3 pt) mais le **Boundary F1 régresse de 9.5 points** (82.0% → 72.5%) et le Mean F1 fragment-level recule légèrement (-1.8 pt). La rotation aléatoire + InstanceNorm forcent des features plus génériques au prix d'un peu de précision aux frontières sur le domaine déjà bien maîtrisé — compromis attendu mais à quantifier.

**Pistes si on veut pousser plus loin** : augmenter le nombre d'epochs (20 peut être insuffisant pour que le réseau finisse de réapprendre avec le bruit de rotation), tester une rotation moins agressive (cap angulaire au lieu de SO(3) complet), ou rééquilibrer le sampler pour suréchantillonner `artifact` davantage (actuellement `weight_hard_examples` pondère par ratio de fracture, pas par catégorie).

---

## 9. Step 15 — Modèle final, entraîné de zéro (résultat retenu)

### Motivation

Step 14 fine-tune séquentiellement depuis Step 13, qui fine-tune depuis Step 12, depuis Step 9... Cette chaîne peut accumuler un biais (chaque étape n'explore qu'un voisinage proche de l'optimum précédent). Hypothèse : un entraînement complet from-scratch, combinant directement tous les composants validés, pourrait atteindre un meilleur optimum global — quitte à coûter plus de temps de calcul (pas de warm-start).

### Architecture (config `cnn_step15_final_model.yaml`)

Identique à Step 14 (U-Net depth=4, normals, splatting, bilinear, contexte global, feature fusion max, geo features, Focal Loss, hard sampling, PointHead, `random_rotate`, `norm_type=instance`), **sauf** :
- `use_overlap_channels: false` — exclu volontairement (régression confirmée en Step 12, voir section 4)
- **Pas de `pretrained_ckpt`** — poids initialisés aléatoirement
- Entraîné sur `everyday` **seul** (comme Steps 1-13, pas le mix de Step 14)
- 100 epochs, `batch_size=4` (limité par la mémoire GPU partagée du labo)

### Optimisations nécessaires pour rendre l'entraînement from-scratch faisable

Le premier essai (epoch 0) projetait ~20 jours pour 150 epochs. Deux correctifs ont réduit ça à ~2 jours (100 epochs) :
1. **`HybridGeometryFeatures.forward_single`** forçait un calcul kNN+PCA (O(N²), N=5000) sur **CPU** par fragment, en boucle séquentielle Python — déplacé sur GPU (`assembly/models/hybrid_geometry_features.py`), où le même calcul est natif et bien plus rapide. C'était le vrai goulot (le GPU était à 0% d'utilisation pendant cette phase).
2. **Rotation SO(3) vectorisée** : un seul appel `torch.linalg.qr` batché sur tous les fragments du batch au lieu d'un appel par fragment (`assembly/models/projection_3d_to_2d.py`).

Gain mesuré : débit ×2.5 (3.5 → 8.8 objets/s), confirmé par `nvidia-smi` passant de 0% à 100% d'utilisation GPU pendant l'entraînement (preuve que le calcul est bien passé sur GPU).

### Résultat — évalué in-domain (`everyday`) et zero-shot (`artifact`, jamais vu à l'entraînement)

| Métrique | Step 13 (everyday) | Step 14 (everyday) | Step 14 (artifact, vu à l'entraînement) | **Step 15 (everyday, in-domain)** | **Step 15 (artifact, zero-shot)** |
|---|---|---|---|---|---|
| Mean F1 fragment-level | 89.3% | 87.5% | 80.0% | **93.1%** | **82.1%** |
| Boundary F1 | 82.0% | 72.5% | 68.4% | **83.2%** | **75.5%** |
| F1 pooled (meilleur seuil) | 94.6% | 92.6% | 88.4% | **96.7%** (seuil 0.50) | **91.5%** (seuil 0.60) |
| Hausdorff distance | 0.249 | 0.322 | 0.500 | **0.171** | 0.476 |
| Chamfer distance | 0.032 | 0.041 | 0.095 | **0.019** | 0.109 |

**Résultat clé : Step 15 domine sur les deux fronts, malgré n'avoir jamais vu `artifact`.**
- **In-domain (`everyday`)** : meilleur résultat de toute la série — Mean F1 93.1% (vs 89.3% Step 13), Boundary F1 83.2% (vs 82.0%), F1 pooled 96.7% (vs 94.6%).
- **Zero-shot (`artifact`)** : F1 pooled 91.5%, **supérieur** à Step 14 (88.4%) qui avait pourtant entraîné explicitement sur `artifact`. Le gap par rapport à `everyday` reste (96.7% → 91.5%, soit 5.2 points), mais c'est mieux que tout ce qu'on a obtenu en entraînant sur les deux catégories.

**Interprétation** : la chaîne de fine-tuning séquentiel (Step 9→…→14) semble avoir accumulé une sous-optimalité — chaque étape ne réoptimise que localement autour du checkpoint précédent. Un entraînement complet en une fois, avec la même recette finale (`random_rotate` + `instance_norm` + PointHead), atteint un meilleur optimum global, et la généralisation zero-shot tient mieux que le mix explicite de catégories. **Step 15 est le candidat naturel comme modèle final pour la présentation.**

---

## 10. Prochaines étapes

### Bilan de la phase segmentation (terminée)
- [x] Step 14 (généralisation, 20 epochs) — succès partiel, gap réduit de moitié
- [x] Step 15 (modèle final from-scratch, 100 epochs) — meilleur résultat de toute la série, retenu comme modèle final
- [x] Comparaison rigoureuse vs GARF-mini (cf. section 7) — CNN devant sur F1 pooled (94.4%/88.8% vs 88.3%/83.7%), GARF devant sur Boundary F1 ; réserve méthodologique sur le budget de points identifiée et documentée

**La phase d'ablation CNN pour la segmentation de fracture est considérée terminée.** Step 15 est le modèle final retenu.

### Nouvelle direction — le CNN comme fracture prior pour le réassemblage

Suite directe de l'ablation, déjà bien engagée (plan détaillé, tenu à jour séparément dans `PLAN_REASSEMBLY_MODULE.md` — ce qui suit n'en est qu'un résumé). Idée centrale : GARF (`assembly/models/denoiser/`) ne fait pas de matching pair-à-pair explicite — il régresse une pose SE(3) globale par fragment via flow matching + attention globale sur tous les points. Utiliser le filtre CNN pour ne garder que les points de fracture, en amont d'un module de matching pair-à-pair, est donc un changement de paradigme vers la registration classique par correspondances, pas un simple remplacement de backbone.

**Phase 0 (convention de pose) — CONFIRMÉE.** Reconstruction de l'objet assemblé à partir des poses stockées (`quaternions`/`translations`/`scale`), résidu ~1e-8 sur everyday et artifact val. Formule retenue : `R_ij = R_j⁻¹ R_i`, `t_ij = R_j⁻¹(t_i − t_j)`, valable après réapplication du facteur `scale` par fragment.

**Phase 1 (Recall@K du filtre CNN) — CONFIRMÉE, avec un résultat inattendu.** Un budget fixe par fragment (top-K ou top-percent) échoue (recall 13-73%, ne suit pas la quantité réelle de surface de fracture qui croît avec le nombre de voisins). En revanche, un **filtrage par seuil de probabilité** (0.2/0.3/0.5) réussit largement : recall 84-98%, stable même sur 11+ fragments. **Le CNN Step 15 est validé comme prior exploitable** — c'est le résultat clé de cette phase.

**Phase 2 (baseline géométrique RANSAC/Kabsch, 5 sous-phases 2A-2E) — CLOSE, conclusion forte.** Confirmé à chaque étape (précision de correspondance, structure spatiale des interfaces, clustering) : **le CNN n'est jamais le facteur limitant** — ses masques se comportent presque comme l'oracle GT. Le vrai plafond est le matcher géométrique lui-même : même dans les conditions les plus favorables testées (oracle pair-specific + scoring normal-aware), `Pose@30°` ne dépasse pas **~9.6%**. Cause identifiée : un fragment touchant plusieurs voisins a tous ses points de fracture mélangés dans un seul masque — le problème est une **séparation d'interface par paire**, pas juste un matching point-à-point, et aucun raffinement de descripteur/scoring géométrique ne lève cette limite. **Ce résultat démontre empiriquement la nécessité d'un module appris (Phase 3), pas juste une optimisation possible.**

**Phase 3 (matcher appris) — cadrage décidé, implémentation en cours.** Doit corriger deux causes indépendantes identifiées en Phase 2 (descripteurs faits-main trop faibles ET scoring RANSAC structurellement biaisé), pas juste l'une des deux. Architecture cible : encodeur léger partagé (PointNet/EdgeConv) → corrélation cross-fragment → correspondance souple (row-softmax + dustbin, **pas** Sinkhorn — les surfaces de fracture sont bruitées/partielles, pas un bipartite équilibré) → Kabsch pondéré différentiable → pose relative. Scope Phase 3A : paires positives uniquement (`graph[i,j]=True`), question du "registration" pure. Script de vérification des données (`scripts/phase3a_pair_dataset_check.py`) écrit ; prochaine action : le lancer sur le serveur et valider le format des paires avant d'écrire l'encodeur.

**Hors scope sauf si le temps le permet** : pipeline d'assemblage global complet, cohérence de cycle, paires négatives (Phase 3B), ou une version "GARF-lite" avec attention globale sur l'ensemble réduit (resterait le paradigme lourd de GARF, juste avec moins de points — pas une alternative légère).

### Pistes secondaires sur le CNN seul (si on y revient)

- **Résolution adaptative**, critère basé sur l'analyse occupancy (proba ∈ [0.4, 0.6] + faible occupancy + forte courbure) — gain marginal probable après Step 15, pas prioritaire.
- Reproduire Step 15 avec une seed différente pour vérifier la stabilité du résultat (un seul run pour l'instant).

### Deux chantiers de fine-tuning identifiés (2026-07-23, motivés par le matching géométrique — Phase 7 de `PLAN_REASSEMBLY_MODULE.md`)

Le pipeline de depth-map matching en aval (voir `PLAN_REASSEMBLY_MODULE.md`, Phase 7) a révélé un écart marqué entre GT et CNN (`thresh0.3`) : éligibilité étage 1 = 99.1% en GT contre 62.6% en CNN, Pose@30 global = 37.6% contre 14.7% — sur les MÊMES paires, avec le MÊME pipeline de matching. Trois tentatives de mitigation côté pipeline (`dominant_cluster_mask`, `compute_pca_frame_robust`, `remove_tiny_clusters_mask`) ont toutes échoué (net-négatif ou neutre) : l'écart n'est pas réparable en aval, il vient de la qualité du masque CNN lui-même. Diagnostic quantitatif sur tout le split val (`phase7_isolated_fp_prevalence_check.py`, 7490 fragments) : deux causes distinctes, de poids très inégal.

1. **Précision de frontière** (85.3% du volume de faux positifs). Rejoint directement la piste "Step 16 — Boundary Loss" déjà envisagée mais jamais lancée :
   ```python
   Loss = 0.4 × Focal + 0.4 × Dice + 0.2 × Boundary
   # Boundary : BCE uniquement sur les points frontière (kNN k=5)
   ```
   Le diagnostic géométrique (angle complètement différent, motivé par le matching plutôt que par le F1) confirme indépendamment que c'est le bon chantier.

2. **Amas isolés de faux positifs** (14.7% du volume de FP seulement, mais 56.1% des fragments en ont au moins un, et leur précision est nettement plus basse : 60.4% pour le bruit isolé / 72.7% pour les clusters secondaires, contre 88.2% pour le cluster principal). C'est un angle mort des métriques de segmentation classiques (F1, Boundary F1 global) : un amas de 10 points sur 5000 ne bouge quasiment pas le F1 fragment-level, mais suffit à biaiser le repère PCA du matching géométrique en aval et faire échouer toute la cascade. Deux masques au même F1 peuvent donc avoir un comportement radicalement différent pour le réassemblage, selon que leurs erreurs sont concentrées en frontière (chantier 1, inoffensif pour le matching) ou dispersées en amas isolés (chantier 2, ce qui casse spécifiquement le matching). Piste concrète : un terme de loss de cohérence spatiale (pénaliser les composantes connexes prédites positives loin du reste de la fracture), en complément de la boundary loss — pas redondant, cible une population de points différente.

**Ces deux chantiers ne sont pas interchangeables** : chacun cible une population de points distincte (frontière du cluster principal vs amas séparés), identifiée par un diagnostic géométrique que les métriques de segmentation seules n'auraient pas révélé.

### Step 16 — boundary loss + spatial coherence loss (2026-07-28, TERMINÉ — résultat net-négatif, Step 15 reste le modèle final)

Implémentation : `assembly/models/cnn_segmentation_model.py` (`compute_boundary_mask`/`boundary_bce_loss`, `compute_isolated_fp_mask`/`coherence_bce_loss`), config `configs/experiment/cnn_step16_boundary_coherence.yaml`, fine-tuning depuis Step 15 sur 30 epochs (`batch_size=4`, `accumulate_grad_batches=8` — cf. note OOM ci-dessous).

- **`forward()`** expose désormais `frag_sizes` (liste de K tailles) et `points_xyz_flat` ((N_sum, 3), même ordre K-fragments que `coarse_seg_pred`/`coarse_seg_gt`) — nécessaire pour que `criteria()` fasse du kNN/clustering *par fragment* sans mélanger les fragments. API des autres appelants inchangée.
- **Boundary loss** : port direct de `_boundary_mask()` (`scripts/analyze_errors.py`, kNN k=5 sklearn CPU — pas torch/GPU, cf. note OOM) — même définition que la métrique Boundary F1 d'évaluation. GT-based, `no_grad`, stable dès l'epoch 0.
- **Spatial coherence loss** : clustering par composantes connexes (`eps=0.02`, `min_cluster_size=10`, mêmes paramètres que `phase7_isolated_fp_prevalence_check.py`) sur les points *prédits* positifs, par fragment, `no_grad`. Seuls les points isolés ET GT-négatifs sont pénalisés.
- Poids `boundary_weight = coherence_weight = 0.2`, `coherence_warmup_epochs=2`, fine-tuning depuis Step 15 (pas from-scratch).
- **OOM en cascade pendant le lancement** (documenté dans les commits `381059f`/`520b314`/`76cf183`/`09b28f3`/`8b5698c`) : (1) `compute_boundary_mask` en `torch.cdist` GPU faisait une matrice O(n²) par fragment → déplacé sur CPU/sklearn ; (2) bug préexistant (Step 9, pas causé par Step 16) dans `_knn_indices_batched` (`hybrid_geometry_features.py`) — matrice dense `(K, N, N)` pour TOUS les fragments du batch en un seul appel, jamais un problème avec les batches "chanceux" de Step 15 mais explosif sur un GPU de 7.6 Go dès que K est grand → chunké sur K ; (3) même avec ces deux fix, `batch_size=32` restait trop gros pour l'activation memory du forward/backward du U-Net sur ce GPU (Quadro RTX 4000, 7.6 Go) — descendu par paliers (32→16→8→4) avec `accumulate_grad_batches` compensatoire pour garder le même batch effectif (32) que Step 15.

**Résultat (checkpoint final, 30/30 epochs, confirmé identique au checkpoint intermédiaire epoch ~21 — pas un artefact de sous-entraînement) :**

| | Step 15 | Step 16 |
|---|---|---|
| F1 pooled (meilleur seuil) | 96.7% | 96.27% (t=0.40) — quasi flat |
| Boundary F1 | 83.2% | 84.10% — légèrement mieux |
| Amas isolés (% volume FP) | 14.7% | **0.4%** — chute massive |
| Fragments touchés par amas isolés | 56.1% | **8.2%** (441/5386) — chute massive |
| Éligibilité étage 1 (pipeline matching) | 62.6% (931/1487) | 70.3% (1045/1487) — mieux |
| Pose@30 parmi étage-2 | 23.5% | 16.4% — pire |
| **Pose@30 global, en absolu** | **14.7% (219 paires)** | **11.5% (171 paires)** — **-48 paires (-22%)** |
| **Succès strict global, en absolu** | **6.8% (101 paires)** | **5.5% (82 paires)** — **-19 paires (-19%)** |

**Conclusion : résultat net-négatif sur le vrai test (pipeline de matching), malgré un succès quasi total sur sa propre métrique diagnostique.** La coherence loss élimine presque totalement les amas isolés (56.1%→8.2% des fragments touchés, 14.7%→0.4% du volume de FP) — chantier 2 réussi sur le papier. Mais ça ne se traduit PAS par un meilleur matching : au contraire, le nombre absolu de paires correctement posées baisse (-22% Pose@30, -19% succès strict), malgré une éligibilité étage 1 en hausse. **L'hypothèse de départ (les amas isolés biaisent le repère PCA du matching) est infirmée empiriquement** : les supprimer n'aide pas, et semble même coûter en couverture utile du masque — probablement parce que la coherence loss, en pénalisant tout point prédit positif isolé et GT-négatif, supprime aussi des points de bord légitimes qui contribuaient (même bruyamment) à la robustesse du repère PCA/depth-map matching.

**Décision (2026-07-28, avec l'utilisateur) : Step 15 reste le modèle final retenu.** Pas de nouvelle itération sur ce chantier (ablation boundary-seul vs coherence-seul, ou poids réduits, envisagés mais écartés faute de gain attendu clair) — documenté comme résultat négatif informatif, même statut que Step 12 (overlap channels).

**CORRECTION (2026-07-30) :** le chiffre "amas isolés 56.1%→8.2%" ci-dessus était mesuré sur le mauvais checkpoint (`last.ckpt` = `epoch-0.ckpt`, pas le run complet — trois redémarrages OOM ont versionné les checkpoints Lightning, le vrai fichier final est `last-v1.ckpt` = `epoch-29.ckpt`, même piège que Step 12/13). Revérifié sur le bon fichier : les amas isolés ont en réalité **empiré** (55.8%→61.8% des fragments, 14.0%→21.1% du volume de FP), et la précision globale du masque s'est effondrée dans toutes les catégories (cluster principal 88.5%→83.3%, secondaires 73.1%→59.5%, bruit isolé 60.3%→41.1%) — une dégradation généralisée, pas ciblée aux frontières. La régression du matching en aval (confirmée réelle, chiffres cohérents sur le bon checkpoint) n'a donc PAS l'explication qu'on lui donnait ("coherence loss réussit, boundary loss est le seul coupable") — voir `PLAN_REASSEMBLY_MODULE.md` pour le détail complet et la décision de pivot qui en découle (modèle appris sur les depth maps, en cours de cadrage).

---

## 11. Messages clés pour la présentation

1. **Le F1 global ne suffit pas.** Boundary F1 = 68.2% révèle une faiblesse réelle non visible dans le F1 global.

2. **L'hypothèse d'overlap était partiellement fausse.** L'analyse sur 3 modèles indépendants montre que les erreurs sont concentrées dans les zones *peu denses*, pas les zones denses.

3. **Le vrai problème est le manque de contexte 2D aux frontières.** Les points isolés projettent dans des pixels sans support local — le CNN seul ne peut pas les décider correctement.

4. **Le PointHead adresse directement ce problème.** Pour chaque point, il combine le contexte 2D (feature map samplée) avec les features 3D propres du point (normale, courbure, roughness). C'est exactement l'information manquante.

5. **Efficacité paramétrique maintenue.** Step 13 = 544 K params vs PTv3 = 12 727 K. L'architecture légère reste l'objectif.

6. **L'hypothèse overlap était bien fausse — confirmé empiriquement.** Step 12 régresse (-12.4 points de Boundary F1 vs Step 11) : ajouter des channels de densité/profondeur ne corrige pas les erreurs aux frontières. Step 13 (PointHead), lui, gagne +13.8 points de Boundary F1 — la bonne hypothèse était la bonne solution.

7. **Le PointHead seul ne suffit pas à généraliser.** Step 13 atteint 92.3% sur `everyday` mais chute à 82.3% sur `artifact` (precision 78.8%) — le CNN encode des raccourcis visuels propres à la distribution d'entraînement. Step 14 (rotation aléatoire + InstanceNorm) cible directement ce biais de domaine.

8. **La généralisation a un coût, pas un free lunch — du moins en fine-tuning séquentiel.** Step 14 réduit de moitié le gap everyday↔artifact (+6.1 pts de F1 sur artifact) mais le Boundary F1 sur everyday régresse de 9.5 points : forcer des features domain-invariant via rotation aléatoire + InstanceNorm dégrade un peu la précision aux frontières sur le domaine déjà bien appris, quand on part d'un checkpoint déjà fine-tuné plusieurs fois.

9. **Mais entraîné de zéro, ce coût disparaît — et le résultat dépasse tout le reste.** Step 15 (même recette finale, from-scratch, sans warm-start, `everyday` seul) atteint Mean F1=93.1% / Boundary F1=83.2% in-domain — **meilleur que tous les steps précédents**, y compris Step 13. Et en zero-shot sur `artifact` (jamais vu), il atteint F1 pooled=91.5%, **supérieur** à Step 14 qui avait pourtant vu `artifact` à l'entraînement. La chaîne de fine-tuning séquentiel accumulait une sous-optimalité ; repartir de zéro avec la recette complète donne un meilleur optimum global ET une meilleure généralisation.

10. **Le modèle final (Step 15) dépasse la référence PTv3/GARF-mini sur le F1 pooled** (94.4%/88.8% vs 88.3%/83.7%, validé sur le val set complet via le protocole d'évaluation natif de PTv3) en in-domain et en zero-shot, avec 23× moins de paramètres — mais GARF garde l'avantage sur le Boundary F1, en partie parce que le CNN bénéficie d'un budget de points par fragment plus dense que GARF (contrainte architecturale de ce dernier, pas un choix arbitraire — cf. section 7).

11. **Le prior CNN est validé pour le réassemblage, mais le matching géométrique classique ne suffit pas.** En testant le CNN comme filtre en amont d'un module de matching pair-à-pair (alternative légère à l'attention globale de GARF), le CNN n'est jamais le facteur limitant (recall, précision de correspondance, structure spatiale — confirmé à chaque étape). Mais le matcher géométrique (descripteurs faits-main + RANSAC) plafonne empiriquement à ~9.6% de poses correctes même dans les conditions les plus favorables — la limite vient de la séparation d'interface par paire, pas de la segmentation. Un module de matching appris (Phase 3, en cours) est donc une nécessité démontrée, pas une simple optimisation.
