"""
assembly/models/denoiser/denoiser_flow_matching.py
====================================================
PHASE 2 : Entraînement principal — Flow Matching sur SE(3).

C'est le cœur de GARF : apprendre à prédire les poses (rotation + translation)
de chaque fragment pour les assembler correctement.

=== Rappel du concept Flow Matching ===
Au lieu d'apprendre à enlever du bruit (comme DDPM), on apprend un
CHAMP VECTORIEL v* qui donne la "direction" pour aller du bruit vers le GT.

  Entraînement (forward pass) :
    1. On part des poses GT : x_0 = [trans_GT | quat_GT]
    2. On tire un bruit aléatoire : x_1 = [bruit_gaussien | rotation_aléatoire]
    3. On interpole : x_t = (1-σ) * x_0 + σ * x_1  (σ = niveau de bruit)
    4. Le modèle prédit : v = x_1 - x_0  (le champ vectoriel)
    5. Loss : MSE entre v prédit et v cible

  Inference (reverse pass) :
    1. On part de x_1 = bruit pur (poses aléatoires)
    2. On répète 20 fois :
       a. Le modèle prédit v
       b. x_{t-1} = x_t + Δσ * v  (intégration Euler, Δσ < 0)
    3. À la fin : x_0 ≈ poses assemblées

=== Fragment de référence (ref_part) ===
Un fragment est fixé à sa pose GT (le "anchor"). Tous les autres
fragments sont assemblés par rapport à lui. Son gradient est annulé
pendant l'entraînement (sa pose n'est pas à prédire).

Adapté de puzzlefusion++ (https://github.com/eric-zqwang/puzzlefusion-plusplus)
"""

import torch
import torch.nn.functional as F

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

from .denoiser_base import DenoiserBase
from .modules.scheduler import SE3FlowMatchEulerDiscreteScheduler


class DenoiserFlowMatching(DenoiserBase):
    """
    Modèle de flow matching pour le reassembly de fragments 3D.

    Hérite de DenoiserBase qui gère :
      - Le chargement du feature extractor (FracSeg pré-entraîné, gelé)
      - Le training/validation/test loop (Lightning)
      - Les métriques d'évaluation (RMSE rotation/translation, Part Accuracy, Chamfer Distance)

    Args:
        noise_scheduler     : scheduler pour l'entraînement (1000 steps)
        val_noise_scheduler : scheduler pour l'inference (20 steps par défaut)
        **kwargs            : autres args passés à DenoiserBase
                              (feature_extractor, denoiser, optimizer, etc.)
    """

    def __init__(
        self,
        noise_scheduler: SE3FlowMatchEulerDiscreteScheduler,
        val_noise_scheduler: SE3FlowMatchEulerDiscreteScheduler,
        **kwargs,
    ):
        super().__init__(
            noise_scheduler=noise_scheduler,
            val_noise_scheduler=val_noise_scheduler,
            **kwargs,
        )
        self.noise_scheduler = noise_scheduler
        self.val_noise_scheduler = val_noise_scheduler

    def get_sigmas(self, timesteps: torch.Tensor, ndim: int, dtype: torch.dtype):
        """
        Récupère les valeurs σ correspondant aux timesteps donnés.

        Args:
            timesteps : timesteps discrets pour lesquels on veut σ (valid_P,)
            ndim      : nombre de dimensions du tenseur cible (pour le broadcast)
            dtype     : type de données

        Returns:
            sigmas : (valid_P, 1, ..., 1) — σ avec les bonnes dimensions pour le broadcast
        """
        sigmas = self.noise_scheduler.sigmas
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()  # (valid_P,)
        # Ajoute des dimensions pour broadcaster sur (valid_P, 7) par exemple
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, data_dict: dict):
        """
        Forward pass d'entraînement (flow matching).

        Pipeline :
          1. Récupère les poses GT de tous les fragments valides
          2. Tire un timestep aléatoire σ pour chaque objet du batch
          3. Génère le bruit : translation gaussienne + rotation aléatoire SO(3)
          4. Interpole : x_t = scale_noise(x_GT, σ, x_bruit)
          5. Fixe le fragment de référence à sa pose GT (son champ = 0)
          6. Extrait les features 3D via le feature extractor gelé (FracSeg)
          7. Passe x_t + features dans le DenoiserTransformer → champ prédit v
          8. Applique le preconditioning (formule de dénoising) pour obtenir x_0 prédit
          9. Retourne prédictions et GT pour la loss

        Notations :
          B        = batch size
          P        = nb max de fragments (padding)
          valid_P  = nb de fragments réellement présents (sum over batch)
          7        = [translation(3) | quaternion_scalar_first(4)]
          6        = [vec_field_trans(3) | vec_field_rot_axis_angle(3)]

        Args:
            data_dict : dict du batch contenant :
              - 'points_per_part' (B, P) : nb de points par fragment (0 si absent)
              - 'translations'    (B, P, 3) : translations GT
              - 'quaternions'     (B, P, 4) : quaternions GT (scalar-first)
              - 'ref_part'        (B, P) : masque booléen du fragment de référence
              - 'scale'           (B, P, 1) : facteur d'échelle de chaque fragment
              - 'pointclouds'     : nuages de points (pour le feature extractor)
              - etc.

        Returns:
            output_dict avec :
              - 'model_pred'       : champ vectoriel prédit (valid_P, 6)
              - 'model_pred_trans' : translation reconstruite (valid_P, 3)
              - 'model_pred_rots'  : quaternion reconstruit (valid_P, 4)
              - 'target'           : champ vectoriel cible (valid_P, 6)
              - 'gt_trans'         : translation GT (valid_P, 3)
              - 'gt_rots'          : quaternion GT (valid_P, 4)
              - 'weighting'        : pondération de loss par niveau de bruit (valid_P,)
        """
        B, P = data_dict["points_per_part"].shape
        part_valids = data_dict["points_per_part"] != 0  # (B, P) masque des fragments valides

        # Récupère les poses GT pour les fragments valides uniquement (enlève le padding)
        gt_trans = data_dict["translations"][part_valids]   # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]     # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        # === Échantillonnage des timesteps ===
        # On tire UN timestep par objet (pas par fragment), pour que tous les fragments
        # d'un même objet soient bruités au même niveau σ
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",  # distribution uniforme sur [0,1]
            batch_size=B,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        # Convertit u ∈ [0,1] en indices discrets dans le schedule
        indices = (u * self.noise_scheduler.config.get("num_train_timesteps")).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(
            device=gt_trans_and_rots.device
        )  # (B,) — un timestep par objet

        # Répète le même timestep pour tous les fragments de chaque objet
        timesteps = timesteps.repeat(P, 1).T   # (B, P) — même t pour tous les fragments d'un objet
        timesteps = timesteps[part_valids]     # (valid_P,) — filtre le padding

        # === Génération du bruit ===
        # σ correspondant aux timesteps (pour le weighting)
        sigmas = self.get_sigmas(
            timesteps, ndim=gt_trans_and_rots.ndim, dtype=gt_trans_and_rots.dtype
        ).to(self.device)  # (valid_P, 1, ...) pour le broadcast

        # Bruit pour les translations : gaussien N(0, I)
        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)  # (valid_P, 7)

        # Bruit pour les rotations : rotations aléatoires uniformes sur SO(3)
        # scipy.Rotation.random() → distribution uniforme sur SO(3)
        # Convention scalar-first : [w, x, y, z]
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0)).as_quat()).float().to(self.device)
        )[..., [3, 0, 1, 2]]  # réordonne [x,y,z,w] → [w,x,y,z]
        noise[..., 3:] = noise_rots  # remplace la partie rotation du bruit

        # === Interpolation (processus forward de flow matching) ===
        # noisy = (1-σ) * GT + σ * bruit
        # gt_vec_field = bruit - GT (la cible à prédire)
        noisy_trans_and_rots, gt_vec_field = self.noise_scheduler.scale_noise(
            sample=gt_trans_and_rots,   # x_0 = GT
            timestep=timesteps,
            noise=noise,                # x_1 = bruit
        )  # → (valid_P, 7), (valid_P, 6)

        # === Fragment de référence ===
        # Le fragment de référence est fixé à sa pose GT :
        #   - Son champ vectoriel cible = 0 (rien à corriger)
        #   - Sa pose bruitée est remplacée par la pose GT exacte
        # Cela donne au modèle un "ancrage" pour aligner les autres fragments
        gt_vec_field[data_dict["ref_part"][part_valids]] = 0.0
        noisy_trans_and_rots[data_dict["ref_part"][part_valids]] = gt_trans_and_rots[
            data_dict["ref_part"][part_valids]
        ]

        # === Extraction de features (FracSeg gelé) ===
        # Le feature extractor (PointTransformerV3 pré-entraîné sur la segmentation)
        # encode chaque point 3D en un vecteur de features.
        # Ces features capturent la géométrie locale (fracture vs surface originale).
        latent = self._extract_features(data_dict)  # dict point avec 'feat', 'coord', 'batch', 'normal'

        # === DenoiserTransformer ===
        # Prend x_t (poses bruitées) + features 3D + timesteps
        # Renvoie le champ vectoriel prédit v ∈ R^6 par fragment
        denoiser_out = self.denoiser(
            x=noisy_trans_and_rots,                    # (valid_P, 7) — poses bruitées
            timesteps=timesteps,                       # (valid_P,)   — niveau de bruit
            latent=latent,                             # features 3D des points
            part_valids=part_valids,                   # (B, P)       — masque
            scale=data_dict["scale"][part_valids],     # (valid_P, 1) — taille des fragments
            ref_part=data_dict["ref_part"][part_valids],  # (valid_P,) — anchor
        )
        model_pred = denoiser_out["pred"]  # (valid_P, 6) — champ vectoriel prédit

        # Pondération de la loss selon le niveau de bruit σ
        # (les timesteps proches de 0 ou de 1 ont moins d'importance)
        weighting = compute_loss_weighting_for_sd3("none", sigmas)

        # === Preconditioning ===
        # Le modèle prédit le champ vectoriel v, mais on veut aussi la pose finale x_0.
        # On "déplie" la prédiction pour obtenir x_0 :
        #   x_0_trans = x_t_trans - σ * v_trans  (cf. formule de flow matching)
        model_pred_trans = (
            model_pred[..., :3] * (-sigmas) + noisy_trans_and_rots[..., :3]
        )  # (valid_P, 3)

        # Pour les rotations, le champ est un axis-angle → on l'exponentie et compose :
        #   R_0 = exp(-σ * ω) @ R_t
        model_pred_rots = transforms.matrix_to_quaternion(
            transforms.axis_angle_to_matrix(-sigmas * model_pred[..., 3:])
            @ transforms.quaternion_to_matrix(noisy_trans_and_rots[..., 3:])
        )  # (valid_P, 4)

        output_dict = {
            "model_pred": model_pred,          # champ prédit (valid_P, 6)
            "model_pred_trans": model_pred_trans,  # translation reconstruite (valid_P, 3)
            "model_pred_rots": model_pred_rots,    # quaternion reconstruit (valid_P, 4)
            "target": gt_vec_field,            # champ cible (valid_P, 6)
            "gt_trans": gt_trans,              # translation GT (valid_P, 3)
            "gt_rots": gt_rots,                # quaternion GT (valid_P, 4)
            "weighting": weighting,            # pondération (valid_P, 1)
        }

        return output_dict

    def _loss(self, data_dict: dict, output_dict: dict):
        """
        Calcule les losses d'entraînement.

        4 losses calculées, mais seule vec_mse_loss est comptée pour le gradient :
          1. vec_mse_loss  : MSE sur le champ vectoriel (loss principale)
                            → force le modèle à bien prédire la direction du flow
          2. trans_mse_loss: MSE sur les translations reconstruites (supervision auxiliaire)
          3. rot_mse_loss  : MSE sur les quaternions reconstruits (supervision auxiliaire)
          4. rot_rmse      : RMSE angulaire en degrés (métrique interprétable)

        Pourquoi le weighting ?
          Les timesteps à σ ≈ 0.5 (bruit intermédiaire) sont les plus informatifs.
          À σ ≈ 0 ou σ ≈ 1, la tâche est triviale → on pondère moins ces exemples.

        Args:
            data_dict   : batch original (non utilisé directement ici)
            output_dict : sortie de forward()

        Returns:
            (loss_dict, counted_losses) où :
            - loss_dict     : dict de toutes les losses/métriques
            - counted_losses : set des losses à inclure dans le gradient total
        """
        model_pred = output_dict["model_pred"]       # (valid_P, 6)
        model_pred_trans = output_dict["model_pred_trans"]  # (valid_P, 3)
        model_pred_rots = output_dict["model_pred_rots"]    # (valid_P, 4)
        target = output_dict["target"]               # (valid_P, 6) — champ vectoriel GT
        gt_trans = output_dict["gt_trans"]           # (valid_P, 3)
        gt_rots = output_dict["gt_rots"]             # (valid_P, 4)
        weighting = output_dict["weighting"]         # (valid_P, 1) — pondération par σ

        # Loss principale : MSE sur le champ vectoriel pondéré par σ
        vec_mse_loss = F.mse_loss(model_pred, target, reduction="none")
        vec_mse_loss = (vec_mse_loss * weighting).mean()

        # Loss auxiliaire : MSE sur la translation reconstruite
        trans_mse_loss = F.mse_loss(model_pred_trans, gt_trans, reduction="none")
        trans_mse_loss = (trans_mse_loss * weighting).mean()

        # Loss auxiliaire : MSE sur le quaternion reconstruit
        rot_mse_loss = F.mse_loss(model_pred_rots, gt_rots, reduction="none")
        rot_mse_loss = (rot_mse_loss * weighting).mean()

        # Métrique : RMSE angulaire en degrés
        # La "distance" entre deux quaternions q1 et q2 = 2 * arccos(|q1 · q2|)
        # ici on omet le facteur 2 et on prend arccos(q1 · q2)
        cos_theta = torch.sum(model_pred_rots * gt_rots, dim=-1)  # produit scalaire quaternion
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)            # évite arccos(±∞)
        rot_rmse = torch.acos(cos_theta)                          # angle en radians
        rot_rmse = torch.rad2deg(rot_rmse)                        # → degrés
        rot_rmse = torch.sqrt(rot_rmse.pow(2).mean())             # RMSE

        return {
            "vec_mse_loss": vec_mse_loss,
            "trans_mse_loss": trans_mse_loss,
            "rot_mse_loss": rot_mse_loss,
            "rot_rmse": rot_rmse,
        }, set(["vec_mse_loss"])  # seule vec_mse_loss entre dans le gradient total
