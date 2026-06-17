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

## 4. Phase 2 — Channels overlap (implémentés, Step 12 en cours)

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

---

## 5. Phase 3 — PointHead MLP (implémenté, Step 13 prêt)

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
| Step 12 | +OverlapChannels | en cours | ~522 K |
| Step 13 | +PointHead MLP | à lancer | ~544 K |
| PTv3 GARF | référence | ~90.9% | 12 727 K |

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

## 8. Prochaines étapes

### Court terme (en cours)
- [ ] Résultats Step 12 (30 epochs) — valider si Boundary F1 > 70%
- [ ] Lancer Step 13 après Step 12 terminé

### Selon résultats Step 12
- **Si Boundary F1 stagne** → confirme que l'overlap n'est pas le problème → Step 13 (PointHead) est la vraie solution
- **Si Boundary F1 monte significativement** → depth_spread aide sur les pixels frontière mixtes → Step 13 consolide le gain

### Step 15 — Boundary Loss (si Step 13 insuffisant)
```python
Loss = 0.4 × Focal + 0.4 × Dice + 0.2 × Boundary
# Boundary : BCE uniquement sur les points frontière (kNN k=5)
```

### Phase 4 — Résolution adaptative (si Boundary F1 reste < 72% après Step 13)
Critère de raffinement **révisé** (basé sur l'analyse occupancy) :
```
Pixels à raffiner :
  - proba ∈ [0.4, 0.6]  (incertitude élevée)
  - faible occupancy     (manque de support 2D)
  - forte courbure locale
```
*Ne plus cibler les pixels à forte occupancy — ils sont déjà bien prédits.*

---

## 9. Messages clés pour la présentation

1. **Le F1 global ne suffit pas.** Boundary F1 = 68.2% révèle une faiblesse réelle non visible dans le F1 global.

2. **L'hypothèse d'overlap était partiellement fausse.** L'analyse sur 3 modèles indépendants montre que les erreurs sont concentrées dans les zones *peu denses*, pas les zones denses.

3. **Le vrai problème est le manque de contexte 2D aux frontières.** Les points isolés projettent dans des pixels sans support local — le CNN seul ne peut pas les décider correctement.

4. **Le PointHead adresse directement ce problème.** Pour chaque point, il combine le contexte 2D (feature map samplée) avec les features 3D propres du point (normale, courbure, roughness). C'est exactement l'information manquante.

5. **Efficacité paramétrique maintenue.** Step 13 = 544 K params vs PTv3 = 12 727 K. L'architecture légère reste l'objectif.
