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

## 7. Analyse qualitative — CNN vs GARF (visualisations 3D)

### Profil de précision/recall

Observations sur les 4 objets comparés dans la présentation HTML :

| Objet | CNN F1 | CNN P | CNN R | GARF F1 | GARF P | GARF R |
|---|---|---|---|---|---|---|
| Plate — 2 fragments | 0.798 | 0.71 | 0.90 | **0.935** | **0.95** | 0.92 |
| Bowl — 3 fragments | 0.845 | 0.81 | 0.88 | **0.900** | **0.99** | 0.82 |
| Plate — 9 fragments ★ | **0.902** | 0.85 | 0.96 | 0.673 | **0.99** | 0.51 |
| Mug — 5 fragments | 0.305 | 0.89 | 0.19 | **0.946** | **0.98** | 0.92 |

### Pattern structurel

**GARF : ultra-précis, recall variable**
- Precision systématiquement ≥ 0.95 — très peu de faux positifs
- Recall effondré sur les objets à beaucoup de fragments (Plate 9 frags : R=0.51)
- Cause : GARF partage le budget de 5000 pts entre tous les fragments → 5000/9 ≈ 555 pts/fragment → résolution insuffisante pour détecter les lignes de fracture fines

**CNN : recall fort, precision perfectible**
- Recall maintenu même sur objets complexes (Plate 9 frags : R=0.96)
- Cause : CNN alloue 5000 pts par fragment indépendamment du nombre de fragments
- Faiblesse : faux positifs sur zones géométriquement ambiguës (bords à forte courbure, transitions intacte/fracture)

### Implication pour Step 13

L'avantage CNN sur les objets complexes vient du sampling uniforme par fragment. La faiblesse est la **précision** — trop de FP aux frontières géométriques ambiguës.

Le PointHead (features 3D : normales, courbure, roughness) devrait permettre au modèle d'être plus sélectif : distinguer un bord à forte courbure *intact* d'une vraie fracture, ce que la projection 2D seule ne peut pas faire.

**Objectif Step 13 : rapprocher la precision CNN de celle de GARF (0.95+) tout en conservant le recall fort.**

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

### Court terme
- [x] Résultats Step 14 (20 epochs, terminé le 2026-06-18) — gap de généralisation réduit de moitié (10.0→4.2 pts) mais objectif artifact F1>90% non atteint (88.4%), et Boundary F1 everyday régresse (-9.5 pts)
- [x] Step 15 — modèle final entraîné de zéro (100 epochs, terminé le 2026-06-18) — meilleur résultat in-domain ET zero-shot de toute la série, retenu comme candidat final
- [ ] Décider si Step 15 est le résultat final à présenter, ou si on tente encore une itération (cf. pistes ci-dessous)

### Pistes restantes (optionnel, si temps disponible)
- Boundary F1 zero-shot artifact (75.5%) reste en retrait par rapport à l'in-domain everyday (83.2%) — gap résiduel de 7.7 points. Pousser plus loin nécessiterait probablement de voir au moins quelques exemples `artifact` (même peu) plutôt que du pur zero-shot.
- Reproduire Step 15 avec seed différente pour vérifier la stabilité du résultat (un seul run pour l'instant).

### Step 16 — Boundary Loss (si on veut encore réduire le gap aux frontières)
```python
Loss = 0.4 × Focal + 0.4 × Dice + 0.2 × Boundary
# Boundary : BCE uniquement sur les points frontière (kNN k=5)
```

### Phase 6 — Résolution adaptative (optionnel, gain marginal probable après Step 15)
Critère de raffinement **révisé** (basé sur l'analyse occupancy) :
```
Pixels à raffiner :
  - proba ∈ [0.4, 0.6]  (incertitude élevée)
  - faible occupancy     (manque de support 2D)
  - forte courbure locale
```
*Ne plus cibler les pixels à forte occupancy — ils sont déjà bien prédits.*

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

10. **Le modèle final (Step 15) dépasse même la référence PTv3/GARF sur le F1 pooled** (96.7% vs ~90.9%) en in-domain, avec 23× moins de paramètres (543K vs 12 727K) — confirmation que l'architecture légère spécialisée tient la comparaison avec un encodeur généraliste bien plus lourd, au moins sur cette tâche.
