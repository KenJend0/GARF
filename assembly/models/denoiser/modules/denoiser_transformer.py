"""
assembly/models/denoiser/modules/denoiser_transformer.py
==========================================================
Architecture du DenoiserTransformer — le cœur du réseau de GARF.

=== Rôle ===
Prend en entrée :
  - les poses bruitées x_t = [translation(3) | quaternion(4)] pour chaque fragment
  - les features 3D des points (sorties du PointTransformerV3 gelé)
  - le timestep σ (niveau de bruit)
  - un indicateur "fragment de référence" (anchor)

Prédit en sortie :
  - le champ vectoriel v = [Δtrans(3) | axis-angle(3)] par fragment (6D total)

=== Architecture détaillée ===

  Pour chaque point du nuage de points :
    ┌─ features PTv3 (latent["feat"])            ─────────────────────┐
    ├─ coordonnées xyz rotées selon pose courante → NeRF embedding    │ → shape_emb
    ├─ normales rotées selon pose courante → NeRF embedding           │
    └─ scale du fragment → NeRF embedding                             ─┘

  Pour chaque fragment :
    ┌─ pose bruitée [t, q] → NeRF embedding → Linear → x_emb
    └─ embedding "est-ce le fragment anchor ?" → ajouté à x_emb

  data_emb = x_emb + shape_emb (broadcast x_emb sur tous les points du fragment)

  Transformer à 2 niveaux d'attention :
    - Self-attention INTRA-fragment : les points d'un même fragment se parlent
                                      → capture la géométrie locale
    - Global attention INTER-fragments : tous les fragments (et leurs points) se voient
                                          → capture les relations entre fragments

  Mean pooling des points → un vecteur par fragment

  MLP_trans → Δtranslation (3D)
  MLP_rot   → axis-angle (3D)

=== Pourquoi les coordonnées sont rotées par la pose courante ? ===
  En appliquant la rotation courante aux coordonnées, le modèle voit la géométrie
  dans le référentiel "actuel" du fragment (pas le référentiel global).
  Cela rend le réseau équivariant aux rotations de manière implicite :
  si la pose change, les coordonnées changent en conséquence.

=== NeRF Embeddings ===
  Les NeRF positional encodings transforment x en [x, sin(x), cos(x), sin(2x), cos(2x), ...]
  Cela permet au réseau de représenter des signaux à haute fréquence spatiale
  (important pour discriminer des poses proches).

Adapté de puzzlefusion++ (https://github.com/eric-zqwang/puzzlefusion-plusplus)
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_scatter
import pytorch3d.transforms as p3dt

from assembly.models.utils import PositionalEncoding, EmbedderNerf
from .attention import EncoderLayer


class DenoiserTransformer(nn.Module):
    """
    Transformer pour prédire le champ vectoriel de flow matching SE(3).

    Args:
        in_dim          : dimension des features d'entrée (PTv3 output dim)
        out_dim         : dimension de sortie (6 = 3 trans + 3 rot)
        embed_dim       : dimension interne du transformer (ex: 256)
        num_layers      : nombre de couches transformer
        num_heads       : nombre de têtes d'attention
        dropout_rate    : taux de dropout
        trans_out_dim   : dimension de sortie de la tête translation (= 3)
        rot_out_dim     : dimension de sortie de la tête rotation (= 3, axis-angle)
        use_flash_attn  : utilise FlashAttention (plus rapide) si True, SDPA sinon
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        trans_out_dim: int,
        rot_out_dim: int,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.trans_out_dim = trans_out_dim
        self.rot_out_dim = rot_out_dim
        self.use_flash_attn = use_flash_attn

        # Embedding pour distinguer le fragment de référence (anchor) des autres
        # weight[0] = embedding "fragment normal"
        # weight[1] = embedding "fragment de référence"
        self.ref_part_emb = nn.Embedding(2, self.embed_dim)

        self.activation = nn.SiLU()

        # num_embeds_ada_norm = 6 * embed_dim car AdaNorm utilise 6 paramètres par couche
        # (scale et shift pour avant et après attention + feedforward)
        num_embeds_ada_norm = 6 * self.embed_dim

        # Couches transformer avec Adaptive Layer Normalization (conditionnées par le timestep σ)
        # AdaNorm permet au transformer d'adapter son comportement selon le niveau de bruit
        self.transformer_layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim=self.embed_dim,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.embed_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    activation_fn="geglu",         # GEGLU : meilleur que ReLU en transformers
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        # === NeRF Positional Encodings ===
        # Transforme les entrées scalaires en embeddings haute-fréquence
        # Format : [x, sin(x), cos(x), sin(2x), cos(2x), ..., sin(2^(k-1)*x), cos(2^(k-1)*x)]
        # Pour multires=10 et input_dims=7 : out_dim = 7 + 7*2*10 = 147

        multires = 10  # nombre de fréquences log-espacées

        # Embedding pour la POSE [trans(3) | quat(4)] = 7D → 147D
        embed_kwargs = {
            "include_input": True,    # garde x en plus des sin/cos
            "input_dims": 7,          # 7 entrées
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,     # fréquences log-espacées (capturent mieux le multiscale)
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)

        # Embedding pour les COORDONNÉES xyz [3D] → 63D
        embed_pos_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)

        # Embedding pour les NORMALES [3D] → 63D
        embed_normal_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_normal = EmbedderNerf(**embed_normal_kwargs)
        self.normal_embedding = lambda x, eo=embedder_normal: eo.embed(x)

        # Embedding pour le SCALE [1D] → 21D
        embed_scale_kwargs = {
            "include_input": True,
            "input_dims": 1,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_scale = EmbedderNerf(**embed_scale_kwargs)
        self.scale_embedding = lambda x, eo=embedder_scale: eo.embed(x)

        # Projection vers embed_dim : concatène features PTv3 + scale_emb + pos_emb + normal_emb
        # in_dim + 21 + 63 + 63 = in_dim + 147
        self.shape_embedding = nn.Linear(
            self.in_dim
            + embedder_scale.out_dim   # 21
            + embedder_pos.out_dim     # 63
            + embedder_normal.out_dim, # 63
            self.embed_dim,
        )

        # Projection de la pose embeddée vers embed_dim
        self.param_fc = nn.Linear(embedder_obj.out_dim, self.embed_dim)  # 147 → embed_dim

        # Têtes de prédiction : embed_dim → 3D pour translation et rotation
        # Architecture : 3 couches avec SiLU (smooth ReLU) et réduction progressive
        self.mlp_out_trans = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, self.trans_out_dim),  # → 3
        )

        self.mlp_out_rot = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, self.rot_out_dim),   # → 3 (axis-angle)
        )

        # NOTE : le code ci-dessous (graph prediction) a été tenté et commenté.
        # L'idée était d'ajouter une tête pour prédire le graphe de connectivité
        # entre fragments en même temps que les poses. Cela n'a pas amélioré les résultats.
        # C'est une piste d'amélioration possible (ton point #3 du plan).
        # self.mlp_out_graph = nn.Sequential(...)
        # self.graph_param = nn.Parameter(...)

    def _gen_cond(
        self,
        x,      # (valid_P, 7) — poses bruitées [trans | quat]
        latent, # dict Point de PTv3 : {'feat', 'coord', 'batch', 'normal'}
        scale,  # (valid_P, 1) — facteur d'échelle par fragment
    ):
        """
        Génère les embeddings conditionnels (shape + pose) pour chaque point.

        Deux types d'embeddings sont créés :
          1. x_emb     : embedding de la POSE du fragment (un par fragment)
          2. shape_emb : embedding de la GÉOMÉTRIE du point (un par point)

        La clé est que les coordonnées et normales sont ROTÉES par la pose courante
        avant d'être embeddées. Cela permet au réseau de raisonner dans le référentiel
        local de chaque fragment.

        Args:
            x      : poses bruitées (valid_P, 7)
            latent : features PTv3 des points
            scale  : échelle de chaque fragment (valid_P, 1)

        Returns:
            (x_emb, shape_emb) — deux embeddings de dimension embed_dim
            x_emb     : (valid_P, embed_dim)   — un par fragment
            shape_emb : (n_points, embed_dim)  — un par point
        """
        trans = x[..., :3]  # (valid_P, 3) — translation courante
        rot = x[..., 3:]    # (valid_P, 4) — quaternion courant

        # Broadcast de la rotation de chaque fragment vers TOUS ses points
        # latent["batch"] = [0, 0, 0, ..., 1, 1, ..., N] indice du fragment pour chaque point
        trans_broadcasted = trans[latent["batch"]]  # (n_points, 3)
        rot_broadcasted = rot[latent["batch"]]      # (n_points, 4)

        # Rotation 6D (pour regularisation SO(3)) — utilisée dans gen_cond mais pas embeddée directement
        rot_6d = p3dt.matrix_to_rotation_6d(p3dt.quaternion_to_matrix(rot))

        # === Embedding des coordonnées ===
        # Rote les coordonnées par la rotation COURANTE du fragment
        # → le réseau voit la géométrie dans le référentiel local du fragment
        xyz = latent["coord"]                               # (n_points, 3) — coordonnées mondiales
        xyz = p3dt.quaternion_apply(rot_broadcasted, xyz)  # (n_points, 3) — coord. dans ref. local
        xyz_pos_emb = self.pos_embedding(xyz)              # (n_points, 63)

        # === Embedding des normales ===
        # Même chose pour les normales (vecteurs directionnels, pas de translation)
        normal = latent["normal"]
        normal = p3dt.quaternion_apply(rot_broadcasted, normal)  # (n_points, 3)
        normal_emb = self.normal_embedding(normal)               # (n_points, 63)

        # === Embedding de l'échelle ===
        scale_emb = self.scale_embedding(scale)         # (valid_P, 21)
        scale_emb = scale_emb[latent["batch"]]          # (n_points, 21) — broadcast par fragment

        # === Embedding de la géométrie (shape_emb) ===
        # Concatène : features PTv3 + embedding xyz + embedding normales + embedding scale
        concat_emb = torch.cat(
            (latent["feat"], xyz_pos_emb, normal_emb, scale_emb), dim=-1
        )  # (n_points, in_dim + 63 + 63 + 21)
        shape_emb = self.shape_embedding(concat_emb)  # (n_points, embed_dim)

        # === Embedding de la pose (x_emb) ===
        # La pose [trans(3) | quat(4)] = 7D → NeRF embedding (147D) → Linear (embed_dim)
        x_emb = self.param_fc(self.param_embedding(x))  # (valid_P, embed_dim)

        return x_emb, shape_emb

    def _out(self, data_emb):
        """
        Applique les têtes de prédiction sur les embeddings finaux des fragments.

        Args:
            data_emb : (valid_P, embed_dim) — embedding moyen par fragment (après mean pool)

        Returns:
            (valid_P, 6) — [Δtrans(3) | axis-angle(3)] — champ vectoriel prédit
        """
        trans = self.mlp_out_trans(data_emb)  # (valid_P, 3) — composante translation
        rots = self.mlp_out_rot(data_emb)     # (valid_P, 3) — axis-angle de la rotation
        return torch.cat([trans, rots], dim=-1)

    def _add_ref_part_emb(
        self,
        x_emb,    # (valid_P, embed_dim) — embeddings de pose
        ref_part, # (valid_P,) booléen — True si c'est le fragment de référence
    ):
        """
        Ajoute un embedding spécial pour identifier le fragment de référence (anchor).

        Le fragment de référence est celui dont la pose est FIXÉE et connu du modèle.
        En lui donnant un embedding différent, le transformer peut "savoir" qu'il
        n'a pas à prédire sa pose et peut l'utiliser comme référence pour les autres.

        weight[0] → "je suis un fragment normal, ma pose est à prédire"
        weight[1] → "je suis le fragment anchor, ma pose est la vérité"
        """
        valid_P = x_emb.shape[0]

        # Par défaut, tous les fragments reçoivent l'embedding "normal"
        ref_part_emb = self.ref_part_emb.weight[0].repeat(valid_P, 1)  # (valid_P, embed_dim)

        # Le fragment de référence reçoit l'embedding "anchor"
        ref_part_emb[ref_part.to(torch.bool)] = self.ref_part_emb.weight[1]

        x_emb = x_emb + ref_part_emb  # addition résiduelle
        return x_emb

    def _gen_mask(self, B, N, L, part_valids):
        """
        [Non utilisé dans le code FlashAttention, mais garde pour référence]
        Génère les masques d'attention pour la version dense (sans FlashAttention).

        self_mask : blocs diagonaux → chaque fragment ne voit que ses propres points
        gen_mask  : masque de validité → ignore les fragments paddés
        """
        # Masque bloc-diagonal : chaque bloc de L points parle à son propre bloc
        self_block = torch.ones(L, L, device=part_valids.device)
        self_mask = torch.block_diag(*([self_block] * N))   # (N*L, N*L)
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # (B, N*L, N*L)
        self_mask = self_mask.to(torch.bool)

        # Masque de validité : ignore les points des fragments paddés
        gen_mask = part_valids.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)
        gen_mask = gen_mask.to(torch.bool)

        return self_mask, gen_mask

    def calc_graph_mask(
        self,
        graph: torch.Tensor,           # (B, P, P) — matrice d'adjacence entre fragments
        points_per_part: torch.Tensor, # (B, P) — nb de points par fragment
        max_seq_len: torch.Tensor,
    ):
        """
        [Non utilisé actuellement — préparé pour une future intégration du graphe]

        Convertit la matrice d'adjacence entre fragments en un masque d'attention
        au niveau des POINTS : deux points peuvent s'attendre si leurs fragments
        sont connectés dans le graphe.

        Cela permettrait une attention "guidée par le graphe" :
          seuls les fragments adjacents (qui se touchent) se voient dans l'attention.
          → moins de bruit, plus de cohérence topologique (ton point #3 du plan)

        Returns:
            (graph_mask, valid_mask) — masques addictifs (-inf pour bloquer l'attention)
        """
        B, P = points_per_part.shape

        # Calcule les indices de début et fin de chaque fragment dans la séquence de points
        cum_points = torch.cumsum(points_per_part, dim=1)   # (B, P) — fin de chaque fragment
        start_indices = cum_points - points_per_part         # (B, P) — début de chaque fragment
        end_indices = cum_points                             # (B, P) — fin de chaque fragment

        # Génère un masque part_to_points[b, p, n] = True si le point n appartient au fragment p
        point_indices = torch.arange(max_seq_len, device=points_per_part.device)  # (max_seq_len,)
        part_to_points = (
            point_indices.unsqueeze(0).unsqueeze(0) >= start_indices.unsqueeze(2)
        ) & (
            point_indices.unsqueeze(0).unsqueeze(0) < end_indices.unsqueeze(2)
        )  # (B, P, max_seq_len)

        # Pour chaque point, trouve son fragment parent
        part_for_points = torch.argmax(part_to_points.float(), dim=1)  # (B, max_seq_len)

        # Récupère quels fragments sont connectés au fragment de chaque point
        connected_parts = graph.gather(
            1, part_for_points.unsqueeze(-1).expand(B, max_seq_len, P)
        )  # (B, max_seq_len, P) — pour chaque point : liste des fragments connectés

        # Convertit les connections de fragments en connections de points
        connected_points = torch.bmm(connected_parts.float(), part_to_points.float())
        valid_points_mask = part_to_points.any(dim=1).float()
        connected_points = (
            connected_points
            * valid_points_mask.unsqueeze(1)
            * valid_points_mask.unsqueeze(2)
        )

        # Masque additif : 0 pour les paires connectées, -inf pour les autres
        graph_mask = connected_points - 1
        graph_mask[graph_mask < 0] = -torch.inf

        # Masque de validité : -inf pour les points de padding
        valid_mask = valid_points_mask.unsqueeze(1) * valid_points_mask.unsqueeze(2)
        valid_mask = valid_mask - 1
        valid_mask[valid_mask < 0] = -torch.inf

        return graph_mask, valid_mask

    def forward(
        self,
        x,           # (valid_P, 7) — poses bruitées [trans | quat]
        timesteps,   # (valid_P,) — niveau de bruit σ pour chaque fragment
        latent,      # dict Point PTv3 : features 3D de tous les points
        part_valids, # (B, P) booléen — masque des fragments non-paddés
        scale,       # (valid_P, 1) — échelle de chaque fragment
        ref_part,    # (valid_P,) booléen — fragment de référence
    ):
        """
        Forward pass avec FlashAttention (implémentation varlen / packed).

        FlashAttention "varlen" traite les séquences de longueurs variables
        sans padding → plus efficace en mémoire et en calcul.

        Si use_flash_attn=False, délègue à forward_sdpa() qui utilise
        le SDPA de PyTorch (plus compatible mais légèrement moins rapide).

        Pipeline :
          1. Génère les embeddings de pose (x_emb) et de géométrie (shape_emb)
          2. Ajoute l'embedding anchor au fragment de référence
          3. Broadcast x_emb sur tous les points du fragment : data_emb = x_emb + shape_emb
          4. Pour chaque couche transformer :
             a. Self-attention INTRA-fragment (Flash varlen) : points d'un même fragment
             b. Global attention INTER-fragments (Flash varlen) : tous les points
          5. Mean pool : n_points → valid_P (un vecteur par fragment)
          6. Têtes MLP → prédiction [Δtrans | axis-angle]
        """
        if not self.use_flash_attn:
            return self.forward_sdpa(x, timesteps, latent, part_valids, scale, ref_part)

        # === Étape 1 : Génération des embeddings ===
        x_emb, shape_emb = self._gen_cond(x, latent, scale)
        # (valid_P, embed_dim), (n_points, embed_dim)

        # Ajoute l'embedding "anchor" au fragment de référence
        x_emb = self._add_ref_part_emb(x_emb, ref_part)  # (valid_P, embed_dim)

        # Broadcast x_emb : chaque point reçoit l'embedding de son fragment
        x_emb = x_emb[latent["batch"]]  # (n_points, embed_dim)

        # Embedding total = pose + géométrie (addition résiduelle)
        data_emb = x_emb + shape_emb  # (n_points, embed_dim)

        # === Étape 2 : Préparation des longueurs de séquence ===
        # FlashAttention varlen attend des offsets cumulatifs (cu_seqlens) pour savoir
        # où commence et finit chaque séquence dans le batch "packed"

        # Longueurs de séquence pour la self-attention INTRA-fragment
        # = nb de points par fragment
        self_attn_seqlen = torch.bincount(latent["batch"])  # (valid_P,) — nb points par fragment
        self_attn_max_seqlen = self_attn_seqlen.max()       # nb max de points dans un fragment

        # Offsets cumulatifs : [0, n_pts_frag0, n_pts_frag0+n_pts_frag1, ...]
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        # Longueurs de séquence pour la global attention INTER-fragments
        # = nb total de points par OBJET (somme sur tous les fragments de l'objet)
        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen  # remplissage (padding → 0)
        global_attn_seqlen = points_per_part.sum(1)      # (B,) — nb total de points par objet
        global_attn_max_seqlen = global_attn_seqlen.max()

        global_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(global_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        # === Étape 3 : Transformer layers ===
        for i, layer in enumerate(self.transformer_layers):
            data_emb = layer(
                hidden_states=data_emb,                         # (n_points, embed_dim)
                timestep=timesteps,                             # (valid_P,) — conditioning σ
                batch=latent["batch"],                          # (n_points,) — fragment de chaque point
                # Paramètres pour FlashAttention varlen (self-attention intra-fragment)
                self_attn_seqlens=self_attn_seqlen,
                self_attn_cu_seqlens=self_attn_cu_seqlens,
                self_attn_max_seqlen=self_attn_max_seqlen,
                # Paramètres pour FlashAttention varlen (global attention inter-fragments)
                global_attn_seqlens=global_attn_seqlen,
                global_attn_cu_seqlens=global_attn_cu_seqlens,
                global_attn_max_seqlen=global_attn_max_seqlen,
            )

        # === Étape 4 : Mean Pooling ===
        # Réduit les n_points features → 1 feature par fragment (mean pool)
        # segment_csr fait une moyenne groupée efficacement
        data_emb = torch_scatter.segment_csr(
            data_emb,
            self_attn_cu_seqlens.long(),
            reduce="mean",
        )  # (valid_P, embed_dim) — un embedding par fragment

        # === Étape 5 : Têtes de prédiction ===
        out_trans_rots = self._out(data_emb)  # (valid_P, 6) — [Δtrans(3) | axis-angle(3)]

        return {
            "pred": out_trans_rots,   # champ vectoriel prédit (valid_P, 6)
            "graph_pred": None,       # pas de prédiction de graphe pour l'instant
        }

    def forward_sdpa(
        self,
        x,           # (valid_P, 7)
        timesteps,   # (valid_P,)
        latent,
        part_valids, # (B, P)
        scale,       # (valid_P, 1)
        ref_part,    # (valid_P,)
    ):
        """
        Forward pass avec PyTorch Scaled Dot-Product Attention (SDPA).

        Alternative à FlashAttention pour une meilleure compatibilité matérielle.
        Utilise un padding + masque booléen au lieu du format varlen.

        Optimisation "pad-once" :
          On padde les séquences UNE SEULE FOIS au format (B, S, D) avant les couches,
          et on dépade UNE SEULE FOIS après. Évite de re-padder à chaque couche.

        Les masques d'attention sont calculés une fois et réutilisés par toutes les couches.
        """
        # Génère les embeddings (même logique que forward())
        x_emb, shape_emb = self._gen_cond(x, latent, scale)
        x_emb = self._add_ref_part_emb(x_emb, ref_part)
        x_emb = x_emb[latent["batch"]]
        data_emb = x_emb + shape_emb  # (n_points, embed_dim)

        batch_idx = latent["batch"]  # (n_points,) — indice de fragment pour chaque point
        device = data_emb.device

        # Longueurs de séquences
        self_attn_seqlen = torch.bincount(batch_idx)  # (valid_P,)
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen
        global_attn_seqlen = points_per_part.sum(1)  # (B,) — nb points total par objet
        global_max_seqlen = global_attn_seqlen.max().item()
        B = part_valids.shape[0]

        # === Padding unique vers format (B, S, D) ===
        # S = longueur max de séquence parmi tous les objets du batch
        seq_ranges = torch.arange(global_max_seqlen, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = seq_ranges < global_attn_seqlen.unsqueeze(1)  # (B, S) — True = point réel

        # Crée le tenseur paddé (avec zéros pour les positions invalides)
        padded_data = torch.zeros(
            (B, global_max_seqlen, self.embed_dim), device=device, dtype=data_emb.dtype
        )
        padded_data[valid_mask] = data_emb  # place les vraies features aux bonnes positions

        # Padde aussi les indices de fragments (pour construire les masques)
        padded_batch = torch.zeros(
            (B, global_max_seqlen), device=device, dtype=torch.long
        )
        padded_batch[valid_mask] = batch_idx

        # === Masques d'attention (calculés une seule fois) ===

        # Masque self-attention INTRA-fragment :
        # deux positions s'attendent ssi elles appartiennent au même fragment
        # padded_batch[b, i] == padded_batch[b, j] ⟺ même fragment
        self_attn_mask = (padded_batch.unsqueeze(2) == padded_batch.unsqueeze(1))  # (B, S, S)
        # Restrict aux positions valides uniquement (pas de padding → padding)
        self_attn_mask = self_attn_mask & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        self_attn_mask = self_attn_mask.unsqueeze(1)  # (B, 1, S, S) — broadcast sur les têtes

        # Masque global attention INTER-fragments :
        # toutes les positions valides d'un même objet peuvent s'attendre
        global_attn_mask = (valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)).unsqueeze(1)
        # (B, 1, S, S)

        # === Transformer layers (format paddé) ===
        for layer in self.transformer_layers:
            padded_data = layer.forward_sdpa(
                hidden_states=padded_data,          # (B, S, embed_dim)
                timestep=timesteps,                 # (valid_P,) — σ par fragment
                batch=padded_batch,                 # (B, S) — indice fragment par point
                self_attn_mask=self_attn_mask,      # (B, 1, S, S) — masque intra-fragment
                global_attn_mask=global_attn_mask,  # (B, 1, S, S) — masque inter-fragments
            )

        # === Dépaddage (une seule fois) ===
        # Récupère seulement les features des vraies positions (retire le padding)
        data_emb = padded_data[valid_mask]  # (n_points, embed_dim)

        # Mean pooling par fragment
        data_emb = torch_scatter.segment_csr(
            data_emb,
            self_attn_cu_seqlens.long(),
            reduce="mean",
        )  # (valid_P, embed_dim)

        out_trans_rots = self._out(data_emb)  # (valid_P, 6)

        return {
            "pred": out_trans_rots,
            "graph_pred": None,
        }
