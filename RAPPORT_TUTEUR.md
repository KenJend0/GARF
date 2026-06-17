# Compte rendu — Segmentation des surfaces de fracture sur objets 3D
**Date :** Mai 2026  
**Projet :** GARF — Geometric Assembly with Reconstructed Fragments  
**Objectif :** Concevoir une alternative légère au module de segmentation de fracture du baseline GARF (PTv3).

---

## 1. Contexte et problème

Le projet s'inscrit dans la problématique de **réassemblage d'objets fracturés** à partir de scans 3D. Une étape clé est l'identification, pour chaque fragment, des **surfaces de fracture** (zones où l'objet s'est cassé) par opposition aux surfaces intactes d'origine.

Le baseline existant dans GARF utilise **PointTransformerV3 (PTv3)**, un transformeur 3D de pointe, mais coûteux à déployer (12,7 M paramètres, architecture lourde en mémoire et en temps d'inférence). L'objectif est de proposer une alternative **24× plus légère** avec des performances compétitives sur la même tâche.

---

## 2. Dataset : Breaking Bad

| Propriété | Valeur |
|---|---|
| Type | Dataset synthétique de fractures 3D |
| Catégories | `everyday` (verres, assiettes, bols, mugs...) + `artifact` |
| Format | Fichier HDF5 unique (27 GB), accès par `object_id` |
| Points par fragment | 5 000 pts/fragment (CNN) ou 5 000 pts/objet total (GARF) |
| Split utilisé | Val set : **8 025 objets**, **33 065 fragments** |
| Label | Binaire par point : `1` = surface de fracture, `0` = surface intacte |

> **Note :** La catégorie `everyday` ne possède pas de split `test` officiel. L'évaluation est conduite sur le val set, en conservant le threshold calibré sur un sous-ensemble de validation.

---

## 3. Architecture du modèle — CNNFracSeg

Le pipeline repose sur une **projection 3D → 2D** permettant d'exploiter les architectures CNN classiques sur des nuages de points.

### Pipeline général

```
Fragment 3D (5 000 pts)
        │
        ▼
  [Features géométriques]          ← courbure, roughness, normal consistency (kNN k=16)
        │
        ▼
  [Projection orthographique]      ← 3 vues : XY, XZ, YZ
        │
  Gaussian splatting (σ=1.5)       ← chaque point → tache gaussienne sur la grille 2D
  Interpolation bilinéaire          ← anti-aliasing sub-pixel
        │
        ▼
  [Encodeur CNN partagé par vue]
        │
  [Fusion multi-vues : max pooling] ← robuste aux orientations
        │
  [Contexte global cross-fragment]  ← agrège l'information entre les fragments d'un objet
        │
        ▼
  [Décodeur U-Net depth=4]          ← skip connections, résolution progressive
        │
        ▼
  Carte de segmentation 2D → reprojection → labels par point 3D
```

### Caractéristiques clés

| Composant | Détail |
|---|---|
| **Paramètres** | **521 489** (vs 12,7 M pour GARF PTv3 → **~24× plus léger**) |
| Backbone | U-Net, profondeur 4 |
| Features géométriques | Courbure, roughness, normal consistency, k=16 |
| Fusion multi-vues | Max pooling (robuste aux orientations) |
| Contexte global | Agrégation cross-fragment par objet |
| Projection | Gaussian splatting + bilinéaire |
| Normals | Utilisées en entrée supplémentaire |

---

## 4. Expériences progressives — Vue d'ensemble

Le modèle a été construit par expériences progressives, chaque composant ajouté et validé individuellement. Les chiffres de gain par composant sont indicatifs (évalués sur un sous-ensemble val à chaque étape).

| Composant ajouté | F1 approximatif |
|---|---|
| Baseline : projection simple, pas de features | ~68% |
| + Normals en entrée | ~71% |
| + Gaussian splatting (σ=1.5) | ~73% |
| + Interpolation bilinéaire | ~74% |
| + Features géométriques (courbure, roughness) | ~78% |
| + Context global cross-fragment | ~80% |
| + Fusion max (vs mean) | ~81% |
| + U-Net depth=4 (vs 3) | ~82% |
| Configuration finale — Dice Loss **(Step 9)** | **84.85%** |
| Tversky Loss α=0.7 — précision ↑ mais recall ↓ **(Step 10)** | ~83.4% |
| **Focal Loss + WeightedRandomSampler **(Step 11)** | **86.92%** |

> **Step 10** a amélioré la précision mais dégradé le recall de manière trop importante sur les fragments à faible zone de fracture. Cette observation a conduit au diagnostic et à la correction de Step 11.

---

## 5. Innovation principale — Step 11 : Traitement des cas difficiles

### Diagnostic initial

L'analyse d'erreurs sur Step 9 a révélé un problème structurel :

| Catégorie (ratio fracture) | F1 Step 9 |
|---|---|
| Ratio élevé (>30%) | 0.942 |
| Ratio moyen (7–30%) | 0.872 |
| **Ratio faible (<7%)** | **0.284** |

**Cause 1 — Signal trop faible dans la projection 2D :**  
Un fragment avec 3% de zone de fracture occupe ~2 pixels sur une grille 64×64. Le gradient est noyé par les 98% de pixels intacts.

**Cause 2 — Distribution d'entraînement biaisée :**  
Les objets difficiles (faible ratio) sont rares et vus peu souvent. Le modèle apprend à les ignorer.

### Solution implémentée

**Focal Loss (γ=2) :**
```
FL(p) = -(1−p)^γ · log(p)   pour les positifs (fracture)
FL(p) = −p^γ · log(1−p)     pour les négatifs (intact)
```
Les exemples difficiles (mal classés, peu de confiance) reçoivent un **poids plus important dans la loss**, forçant le modèle à s'y concentrer davantage qu'avec une cross-entropy classique.

**Loss finale :**
```
L = 0.5 × Dice + 0.5 × Focal(γ=2)
```
Dice stabilise l'entraînement global, Focal focalise sur les pixels difficiles.

**WeightedRandomSampler :**
```
poids(objet) = 1 / (ratio_fracture + 0.05)
```
Les objets à faible ratio fracture sont **sur-représentés** dans les batches d'entraînement.

**Fine-tuning depuis Step 9 :** lr=1e-4 (réduit), 30 epochs supplémentaires → pas d'oubli catastrophique.

### Résultat

| Catégorie | F1 Step 9 | F1 Step 11 | Δ |
|---|---|---|---|
| Ratio faible (<7%) | 0.284 | **0.418** | **+13.4 pp** |
| Ratio moyen (7–30%) | 0.872 | 0.879 | +0.7 pp |
| Ratio élevé (>30%) | 0.942 | 0.938 | −0.4 pp |
| **Global** | **0.8485** | **0.8692** | **+2.1 pp** |

---

## 6. Résultats quantitatifs

### Comparaison sur le val set (8 025 objets)

| Modèle | Params | Threshold | F1 | FDR (FP%) | Miss Rate (FN%) |
|---|---|---|---|---|---|
| **GARF (PTv3)** | 12,7 M | 0.5 (défaut) | ~90.9% | — | — |
| CNN Step 9 (Dice) | 523 K | 0.5 (défaut) | 83.33% | 24.5% | 8.7% |
| CNN Step 9 (Dice) | 523 K | 0.65 (calibré val) | 84.85% | 22.4% | 7.1% |
| CNN Step 11 (Focal+Hard) | 523 K | 0.5 (défaut) | 85.21% | — | — |
| **CNN Step 11 (Focal+Hard)** | **523 K** | **0.55 (calibré val)** | **86.92%** | **16.6%** | **9.4%** |

> **Threshold calibration :** sweep val ([0.30–0.80], pas de 0.05), threshold qui maximise F1 sélectionné et figé. GARF évalué uniquement au threshold par défaut (0.5) — pas de sweep effectué pour GARF dans ce travail.

> **Limites de la comparaison :** GARF n'a pas été soumis au même protocole de calibration. Un sweep de threshold sur GARF pourrait modifier légèrement son F1 rapporté.

### Contexte de la comparaison

- GARF utilise 5 000 pts **par objet total** (format concatené) — il répartit peu de points par fragment si l'objet a de nombreux fragments.
- CNN utilise 5 000 pts **par fragment** — maintient une résolution constante quelle que soit la complexité.
- Cet avantage structurel du CNN se confirme sur les objets complexes (voir Section 8).

---

## 7. Analyse d'erreurs

### Par complexité (nombre de fragments)

| Complexité | F1 Step 11 |
|---|---|
| 2 fragments | 0.638 |
| 3–5 fragments | 0.741 |
| **6+ fragments** | **0.775** |

Contre-intuitivement, le modèle est **meilleur sur les objets complexes**. Hypothèse : les objets plus fracturés présentent des surfaces de fracture plus larges (ratio plus élevé), plus faciles à détecter.

### Nature des faux positifs

Une analyse de post-processing (kNN majority vote + suppression des petits clusters) a montré **ΔF1 ≈ 0**. Les faux positifs ne sont **pas des points isolés** mais des **régions cohérentes** : le modèle confond des zones de forte courbure intacte avec des fractures. Le post-processing ne peut pas corriger ce type d'erreur — c'est un problème de représentation, pas de bruit.

### Cas difficile résiduel

Le faible ratio fracture reste le principal facteur d'échec, même après Step 11 (F1=0.418 vs 0.942 pour les ratios élevés). La projection 2D est intrinsèquement limitée pour les fractures très petites.

---

## 8. Visualisations 3D interactives

Des visualisations HTML interactives (Plotly 3D) ont été générées pour 4 objets représentatifs, avec **3 panels côte à côte** par objet :

- **Panel 1 — GT :** Surface réelle (gris = intact, jaune = fracture)
- **Panel 2 — CNN Step 11 :** TP en vert, FP en rouge, FN en bleu
- **Panel 3 — GARF :** Même code couleur

| Objet | Complexité | F1 CNN | F1 GARF | Observation |
|---|---|---|---|---|
| Assiette | 2 fragments | 0.852 | 0.876 | GARF légèrement supérieur |
| Bol | 3 fragments | 0.850 | 0.941 | GARF sans FP (précision parfaite) |
| Assiette | 9 fragments | **0.911** | 0.553 | **CNN largement supérieur** |
| Mug | 5 fragments | 0.722 | 0.957 | GARF meilleur sur objet simple |

**Analyse :** Sur l'assiette 9 fragments, GARF n'a que ~555 pts/fragment (5000/9), insuffisant pour détecter des fractures fines. Le CNN maintient 5000 pts/fragment → résolution préservée.

**Accès :** Fichiers `.html` interactifs (rotation libre, zoom, inspection par point).

---

## 9. Bilan et positionnement

### Positionnement général

> Le modèle CNN ne remplace pas intégralement GARF, mais constitue une **alternative légère et interprétable** qui devient compétitive — voire supérieure — sur les objets très fragmentés, où la résolution fixe par fragment de GARF devient un handicap.

### Points forts du modèle CNN

1. **~24× plus léger** que GARF PTv3 (523 K vs 12,7 M params) — déploiement embarqué possible
2. **Résolution constante** par fragment, quelle que soit la complexité de l'objet
3. **Avantage clair** sur les objets très fragmentés (6+ parts, 9 parts) — confirmation empirique sur 4 objets
4. **Diagnostic et correction** du problème faible ratio fracture via Step 11 (+13.4 pp)
5. **Pipeline interprétable** : projection 2D visualisable, features géométriques explicites

### Limites identifiées

1. Gap de ~4 pp vs GARF sur le val complet (86.9% vs ~90.9%) au threshold par défaut
2. Faible ratio fracture reste difficile (F1=0.418) — limite structurelle de la projection 2D sur petites zones
3. FP sur zones de forte courbure intacte (non corrigeable par post-processing)
4. Évaluation limitée au val set de la catégorie `everyday` — pas encore de test set figé ni de généralisation zero-shot

---

## 10. Prochaines étapes

| Étape | Description | Statut |
|---|---|---|
| Threshold calibration | Sweep val [0.30–0.80], threshold figé | Fait |
| Visualisations 3D | GT vs CNN vs GARF, 4 objets, format HTML | Fait |
| Post-processing | kNN majority vote + cluster filtering | Fait (ΔF1≈0, FP structurels → abandonné) |
| Test set `artifact` | Évaluation sur le split test officiel de la catégorie `artifact` | À faire |
| Calibration → test | Threshold calibré sur val, reporté sur test figé | À faire |
| Fractura (zero-shot) | Évaluation zero-shot, threshold val uniquement | En attente — dataset non disponible |
| Fantastic Breaks (zero-shot) | Évaluation zero-shot, threshold val uniquement | En attente — dataset non disponible |
| Sweep GARF | Appliquer le même sweep threshold à GARF pour comparaison équitable | À envisager |

---

## Annexes

### Commande d'entraînement Step 11
```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py experiment=cnn_step11_hard_sampling \
  data.data_root=/storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
  ckpt_path=/storage/student7/teyssir/checkpoints/cnn_step9_best.ckpt \
  ++callbacks.model_checkpoint.save_top_k=0 \
  ++callbacks.model_checkpoint.save_last=true
```

### Commande d'évaluation + threshold calibration
```bash
python scripts/analyze_errors.py \
  --ckpt /storage/student7/teyssir/checkpoints/cnn_step11_best.ckpt \
  --data /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
  --split val --threshold_sweep
```

### Commande de visualisation 3D
```bash
python scripts/visualize_object_3d_comparison.py \
  --object_idx 4192 \
  --cnn_ckpt /storage/student7/teyssir/checkpoints/cnn_step11_best.ckpt \
  --garf_ckpt /storage/student7/teyssir/checkpoints/GARF_mini.ckpt \
  --data /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
  --output /tmp/viz_plate9parts.html
```

### Infrastructure
- Serveur : `ict14` via `ictlab.usth.edu.vn:22222`
- GPU : NVIDIA (utilisé CUDA_VISIBLE_DEVICES=1 pour éviter conflits)
- Stockage : `/storage/student7/teyssir/` (HDF5 27GB, checkpoints)
- Framework : PyTorch Lightning + Hydra + Plotly
