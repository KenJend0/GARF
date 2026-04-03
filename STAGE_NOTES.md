# Notes de stage — Amélioration de GARF

## Contexte

GARF est un modèle de reassembly 3D de fragments (fractures) basé sur le **flow matching sur SE(3)**.
Pipeline en 2 phases :
1. **FracSeg** (pré-entraînement) — segmentation fracture vs surface originale via PointTransformerV3
2. **DenoiserFlowMatching** — prédiction des poses (rotation + translation) par flow matching

**Sanity checks initiaux** :
- Les masques de segmentation produisent des contours de fracture plausibles
- Le pipeline d'alignement nécessite un tuning des paramètres pour les entrées bruitées

---

## Plan d'action global

| # | Amélioration | Statut |
|---|---|---|
| 1 | Segmentation hybride (features géométriques explicites) | 🔄 En cours |
| 2 | Pairwise matching adaptatif (multi-hypothèses) | ⏳ À faire |
| 3 | Graphe global + pruning intelligent | ⏳ À faire |
| 4 | ICP différentiable | ⏳ À faire |
| 5 | Priors adaptatifs | ⏳ À faire |
| 6 | Généralisation multi-domaines (LoRA fine-tuning) | ⏳ À faire |

---

## Étape 1 — Segmentation hybride : ajout de features géométriques

### Problème identifié

GARF fait une segmentation **implicite** : PTv3 apprend à distinguer fracture/original
uniquement à partir des coordonnées xyz et des normales.
Il manque des **descripteurs géométriques explicites** qui capturent directement :
- la rugosité locale (fractures = surfaces irrégulières)
- la courbure (fractures = courbures brusques, discontinuités)
- la cohérence des normales dans le voisinage

### Solution : modèle hybride

Ajouter des features géométriques calculées analytiquement **en entrée de PTv3**.

```
ACTUEL :
  feat = [xyz(3) | normales(3)] = 6D → PTv3 → segmentation

FUTUR :
  feat = [xyz(3) | normales(3) | surface_variation(1) | mean_curvature(1) | normal_consistency(1)]
       = 9D → PTv3 → segmentation
```

### Features géométriques choisies (Niveau 1 — simple et robuste)

Calculées par **PCA locale** sur k=16 voisins pour chaque point :

| Feature | Dimension | Calcul | Intuition |
|---|---|---|---|
| `surface_variation` | 1D | λ_min / (λ0+λ1+λ2) de la covariance locale | ≈0 sur plan, ≈1 sur fracture rugueuse |
| `mean_curvature` | 1D | (λ_max - λ_min) / 2 | forte aux bords/discontinuités |
| `normal_consistency` | 1D | 1 - mean(dot(n_i, n_voisins)) | ≈0 sur surface lisse, élevé sur fracture |

**Total : 3 features** — léger, pas de dépendances supplémentaires (numpy + sklearn).

> **Niveau 2 (optionnel)** : FPFH 33D via `open3d` — plus discriminant mais ~10s/objet de calcul.

### Fichiers à modifier

#### 1. `assembly/data/breaking_bad/uniform.py`

**Ajouter la méthode `_compute_geom_features()`** :

```python
def _compute_geom_features(self, pointcloud, normals, k=16):
    """
    Calcule 3 features géométriques par PCA locale sur k voisins.
    
    Args:
        pointcloud : (N, 3)
        normals    : (N, 3)
        k          : nb de voisins
    Returns:
        geom_feat  : (N, 3) — [surface_variation, mean_curvature, normal_consistency]
    """
    from sklearn.neighbors import KDTree
    tree = KDTree(pointcloud)
    _, idx = tree.query(pointcloud, k=k+1)
    idx = idx[:, 1:]  # retire le point lui-même

    N = pointcloud.shape[0]
    surface_variation  = np.zeros(N)
    mean_curvature     = np.zeros(N)
    normal_consistency = np.zeros(N)

    for i in range(N):
        neighbors = pointcloud[idx[i]]            # (k, 3)
        centered  = neighbors - neighbors.mean(0)
        cov       = centered.T @ centered / k
        eigvals   = np.linalg.eigvalsh(cov)       # λ0 ≤ λ1 ≤ λ2

        s = eigvals.sum()
        surface_variation[i] = eigvals[0] / s if s > 1e-8 else 0.0
        mean_curvature[i]    = (eigvals[2] - eigvals[0]) / 2.0

        neighbor_normals       = normals[idx[i]]  # (k, 3)
        dot_products           = (neighbor_normals * normals[i]).sum(-1)
        normal_consistency[i]  = 1.0 - dot_products.mean()

    return np.stack([surface_variation, mean_curvature, normal_consistency], axis=-1)
```

**Modifier `sample_points()`** pour retourner `geom_feat` :

```python
# Dans sample_points(), après avoir récupéré pointclouds_gt et pointclouds_normals_gt :
geom_features_gt = [
    self._compute_geom_features(pc, n)
    for pc, n in zip(pointclouds_gt, pointclouds_normals_gt)
]
return pointclouds_gt, pointclouds_normals_gt, fracture_surface, geom_features_gt
```

**Modifier `transform()`** pour :
1. Appliquer le **même shuffle** (variable `order`) à `geom_feat`
2. Padder et stocker dans `data_dict["geom_feat"]` : forme `(P, N, 3)`

```python
# Dans la boucle for part_idx in range(num_parts) de transform() :
geom_feat[part_idx] = geom_feat[part_idx][order]  # même permutation que le pointcloud

# À la fin de transform(), après les autres _pad_data :
geom_feat = self._pad_data(np.stack(geom_feat, axis=0))  # (P, N, 3)

# Dans le dict retourné :
"geom_feat": geom_feat,
```

#### 2. `assembly/models/pretraining/frac_seg.py`

**Modifier `forward()`** — concaténer `geom_feat` dans le `feat` de PTv3 :

```python
# Récupère et aplatit les features géométriques
geom_feat = batch["geom_feat"]             # (B, N, 3)  ou (B*P, N, 3) selon le format
geom_feat_flat = geom_feat.view(-1, 3)    # (B*N, 3)

# AVANT :
"feat": torch.cat([part_pcds, part_normals], dim=-1),  # (N_total, 6)

# APRÈS :
"feat": torch.cat([part_pcds, part_normals, geom_feat_flat], dim=-1),  # (N_total, 9)
```

#### 3. Config YAML — `configs/model/frac_seg.yaml`

```yaml
# Changer in_channels de PTv3 : 6 → 9
encoder:
  in_channels: 9  # xyz(3) + normales(3) + geom_feat(3)
```

### Pourquoi cette approche (Option A — dans PTv3) ?

| Option | Pour | Contre |
|---|---|---|
| **A : dans PTv3** ✅ | PTv3 apprend à combiner géo + deep dès la 1ère couche | Phase 1 à re-entraîner |
| B : après PTv3 (head seulement) | Pas besoin de re-entraîner PTv3 | PTv3 ne "voit" jamais les features géo |
| C : A + B | Maximum d'info | Plus complexe |

### Chaîne de dépendances complète

```
dataset.sample_points()
  └─ _compute_geom_features(pc, normals) → geom_feat (N, 3)

dataset.transform()
  └─ applique shuffle(order) à geom_feat → cohérence
  └─ pad → data_dict["geom_feat"] (P, N, 3)

FracSeg.forward()
  └─ feat = [xyz | normals | geom_feat] → PTv3 (in_channels=9)
  └─ → segmentation améliorée

(optionnel, plus tard) DenoiserBase._extract_features()
  └─ latent["geom_feat"] = data_dict["geom_feat"]
  └─ DenoiserTransformer._gen_cond()
       └─ concat_emb = [..., latent["geom_feat"]]
       └─ shape_embedding Linear : in_dim + 3 supplémentaires
```

### Métriques à surveiller

Comparer avant/après sur le split **val** de FracSeg :

| Métrique | Baseline (GARF) | Objectif |
|---|---|---|
| `val/coarse_seg_f1` | à mesurer | +2-5 pts |
| `val/coarse_seg_recall` | à mesurer | améliorer (ne pas rater de fractures) |
| `val/coarse_seg_precision` | à mesurer | maintenir |

---

## Cloud computing — Options pour exécuter GARF

> Contrainte locale : mémoire et GPU limités.
> Contrainte du projet : `torch==2.8.0`, `spconv-cu126`, `flash-attn`, `pytorch3d` → Linux only, CUDA 12.6

### Comparatif des options

| Option | GPU | RAM | Stockage | Coût | Facilité |
|---|---|---|---|---|---|
| **Google Colab Pro** | A100 (40GB) | 52GB | Drive (15GB gratuit) | ~12€/mois | ⭐⭐⭐⭐⭐ |
| **Kaggle Notebooks** | P100/T4 (16GB) | 30GB | 20GB | Gratuit (30h/sem GPU) | ⭐⭐⭐⭐ |
| **Vast.ai** | RTX 4090 (24GB) | 64GB | flexible | ~0.40€/h | ⭐⭐⭐ |
| **Runpod** | A100/H100 | 80GB+ | 50GB pod | ~1.5€/h | ⭐⭐⭐ |
| **Lambda Labs** | A100 (40GB) | 200GB | 1TB | ~1.1€/h | ⭐⭐⭐ |

### Recommandation : Google Colab Pro + Drive

Raison : le plus simple à mettre en place, pas de config serveur, notebook Jupyter directement.

**Setup recommandé** :

```python
# Dans un notebook Colab :

# 1. Monte Google Drive (pour les données et checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone le repo
!git clone <ton-repo> /content/GARF
%cd /content/GARF

# 3. Installe les dépendances avec uv (le projet utilise uv)
!pip install uv
!uv sync --extra post

# 4. Vérifie CUDA
import torch
print(torch.cuda.is_available(), torch.version.cuda)

# 5. Lance l'entraînement
!python train.py
```

**Stocker les données HDF5 sur Drive** :
```
Mon Drive/
  GARF/
    data/
      breaking_bad.h5    ← dataset
    checkpoints/
      frac_seg.ckpt      ← checkpoint FracSeg
      garf.ckpt          ← checkpoint principal
```

> ⚠️ Le projet requiert `python===3.12.3` et `spconv-cu126` → vérifie la version CUDA de Colab
> (`!nvidia-smi`). Si CUDA < 12.6, utiliser Runpod/Vast.ai avec une image `CUDA 12.6`.

### Alternative rapide : Kaggle (gratuit)

30h de GPU/semaine, sans carte bancaire. Adapté pour les expériences courtes (validation des features géométriques).
Limites : 16GB GPU (P100), pas de persistance entre sessions.

---

*Dernière mise à jour : 2026-04-03*
