"""
assembly/models/denoiser/modules/scheduler.py
===============================================
Schedulers de bruit pour le flow matching et la diffusion sur SE(3).

SE(3) = groupe des transformations rigides 3D = translations (R³) + rotations (SO(3))

Pourquoi un scheduler spécial pour SE(3) ?
  Les espaces de translations (R³) et de rotations (SO(3)) sont DIFFÉRENTS :
  - R³ est un espace euclidien plat → interpolation linéaire OK
  - SO(3) est une variété courbe → il faut interpoler via axis-angle (géodésique)

Ce fichier contient 3 schedulers :
  1. SE3FlowMatchEulerDiscreteScheduler  ← UTILISÉ DANS GARF (flow matching)
  2. SE3PiecewiseScheduler               ← variante diffusion DDPM (non utilisé par défaut)
  3. SE3DDPMScheduler                    ← variante diffusion DDPM standard (non utilisé par défaut)

=== CONCEPT CLÉ : Flow Matching ===
Contrairement à la diffusion qui apprend à "débruiter",
le flow matching apprend un CHAMP VECTORIEL qui guide les poses de x_noise vers x_GT.

  Entraînement :
    x_t = (1 - σ) * x_0 + σ * x_noise   (interpolation à temps t)
    cible = x_noise - x_0                (le champ vectoriel à prédire)

  Inference :
    On part de x_noise et on fait des petits pas dans la direction prédite
    x_{t-1} = x_t + Δσ * champ_prédit   (intégration d'Euler)

  Pour les rotations, on remplace l'interpolation linéaire par une interpolation
  sur la variété SO(3) via la représentation axis-angle.
"""

import math
from typing import Union, Tuple, Literal
from dataclasses import dataclass

import torch
from diffusers import SchedulerMixin, ConfigMixin, DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config
from pytorch3d import transforms as p3dt


@dataclass
class SE3FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Sortie de la méthode step() du scheduler.

    prev_sample : pose du timestep précédent (t-1), forme (batch_size, 7)
                  7 = 3 (translation) + 4 (quaternion scalar-first [w,x,y,z])
    """
    prev_sample: torch.FloatTensor


class SE3FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler Flow Matching Euler pour SE(3).

    Principe du flow matching :
      - σ(t) : niveau de bruit au temps t, varie de σ_max (=1, bruit pur) à σ_min (≈0, signal pur)
      - x_t = (1-σ) * x_0 + σ * x_noise  pour les translations
      - x_t = exp(σ * axis_angle_noise) @ x_0  pour les rotations (géodésique sur SO(3))

    Args:
        num_train_timesteps : nombre de timesteps discrets pour l'entraînement (ex: 1000)
        stochastic_paths    : ajoute du bruit stochastique pendant les steps (optionnel)
        sigma_schedule      : forme de la courbe σ(t) :
                              - 'linear'              : σ croît linéairement
                              - 'piecewise-linear'    : lent au début (0→0.1), rapide ensuite
                              - 'piecewise-quadratic' : quadratique par morceaux
                              - 'exponential'         : σ = exp(-5*(1-t))
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        stochastic_paths: bool = False,
        stochastic_level: float = 0.1,
        min_stochastic_epsilon: float = 0.01,
        sigma_schedule: Literal[
            "linear", "piecewise-linear", "piecewise-quadratic", "exponential",
        ] = "linear",
    ):
        super().__init__()

        # Génère les timesteps de num_train_timesteps → 1 (ordre décroissant pour le dénoising)
        timesteps = torch.flip(
            torch.linspace(1, num_train_timesteps, num_train_timesteps), dims=[0]
        )
        # Calcule σ(t) pour chaque timestep selon le schedule choisi
        sigmas = torch.tensor(
            [self._sigma_schedule(t, num_train_timesteps) for t in timesteps],
            dtype=torch.float32,
        )
        # Les "timesteps" exposés sont en fait σ * num_timesteps (pratique pour l'indexation)
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()  # ≈ 0 (quasi signal pur)
        self.sigma_max = self.sigmas[0].item()   # = 1 (bruit pur)

        self._step_index = None
        self._begin_index = None

    def _calc_stochastic_epsilon(self, sigma: torch.FloatTensor):
        """
        Calcule l'amplitude du bruit stochastique ajouté pendant les steps.
        Vaut 0 quand σ=0 ou σ=1, maximal à σ=0.5.
        """
        return torch.sqrt(
            self.config.get("stochastic_level") ** 2 * sigma * (1 - sigma)
            + self.config.get("min_stochastic_epsilon")
        )

    def _sigma_schedule(self, t: torch.FloatTensor, num_timesteps: int = 1000):
        """
        Définit σ(t) ∈ [0, 1] en fonction du timestep t.

        t est rescalé dans [0, 1000] pour faciliter les comparaisons.

        Piecewise-linear : σ monte lentement jusqu'à 0.1 (70% du temps),
                           puis rapidement jusqu'à 1.0 (30% restants).
                           → Le modèle passe plus de temps sur des poses quasi-correctes.
        """
        t = t * 1000 / num_timesteps  # normalise t dans [0, 1000]
        if self.config.get("sigma_schedule") == "linear":
            return t / 1000                     # 0 → 1 linéaire

        elif self.config.get("sigma_schedule") == "piecewise-linear":
            if t <= 700:
                return t / 700 * 0.1            # 0 → 0.1 sur les 70% premiers
            else:
                return 0.1 + (t - 700) / 300 * 0.9  # 0.1 → 1.0 sur les 30% restants

        elif self.config.get("sigma_schedule") == "piecewise-quadratic":
            if t <= 700:
                return 0.1 * (t / 700) ** 2
            else:
                return 0.1 + 0.9 * ((t - 700) / 300) ** 2

        elif self.config.get("sigma_schedule") == "exponential":
            return math.exp(-5 * (1 - t / 1000))   # démarre proche de 0, monte exponentiellement

        else:
            raise ValueError(f"Invalid sigma schedule: {self.config.get('sigma_schedule')}")

    @property
    def step_index(self):
        """Index du timestep courant (incrémenté à chaque appel de step())."""
        return self._step_index

    @property
    def begin_index(self):
        """Index de départ du dénoising (utile pour démarrer en milieu de schedule)."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Fixe le timestep de départ pour l'inference."""
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        """Convertit un σ en timestep discret."""
        return sigma * self.config.get("num_train_timesteps")

    def set_timesteps(self, num_inference_steps: int = 50, sigmas: torch.FloatTensor = None):
        """
        Configure les timesteps pour l'INFERENCE (peut être moins que pendant l'entraînement).

        En inference, on utilise souvent 20 steps au lieu de 1000 :
        le modèle doit faire des "grands pas" pour aller du bruit au signal.

        Args:
            num_inference_steps : nb de steps d'inference (ex: 20)
            sigmas              : si fourni, utilise directement ces valeurs σ
        """
        if sigmas is None:
            # Génère un nouveau schedule avec num_inference_steps points
            timesteps = torch.flip(
                torch.linspace(1, num_inference_steps, num_inference_steps), dims=[0]
            )
            sigmas = torch.tensor(
                [self._sigma_schedule(t, num_inference_steps) for t in timesteps],
                dtype=torch.float32,
            )
        else:
            num_inference_steps = len(sigmas)

        self.timesteps = sigmas * self.config.get("num_train_timesteps")
        # Ajoute un 0 final (σ=0 = signal pur = pose finale)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def _scale_noise_for_translation(
        self,
        x_0_trans: torch.FloatTensor,  # (B, 3) — translation GT
        sigma: torch.FloatTensor,       # (B,)   — niveau de bruit
        x_1_trans: torch.FloatTensor,  # (B, 3) — bruit gaussien pur
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Processus forward pour les TRANSLATIONS (flow matching).

        Formule d'interpolation linéaire :
          x_t = (1 - σ) * x_0 + σ * x_1
          où x_0 = GT, x_1 = bruit, σ ∈ [0,1]

        Le champ vectoriel cible (ce que le modèle apprend à prédire) :
          v* = x_1 - x_0  (direction du bruit vers le GT, normalisée)

        Note : le modèle prédit v*, et pendant l'inference on intègre :
          x_{t-1} = x_t + Δσ * v_prédit

        Returns:
            (x_t_trans, trans_vec_field)
            - x_t_trans      : pose bruitée à temps t [B, 3]
            - trans_vec_field : champ vectoriel cible [B, 3]
        """
        sigma = sigma.unsqueeze(-1)  # (B,) → (B, 1) pour le broadcast avec (B, 3)
        # Interpolation linéaire entre GT (x_0) et bruit (x_1)
        x_t_trans = (1 - sigma) * x_0_trans + sigma * x_1_trans

        if self.config.get("stochastic_paths"):
            # Ajoute un bruit brownien pour rendre les trajectoires stochastiques
            x_t_trans += torch.randn_like(x_t_trans) * self._calc_stochastic_epsilon(sigma)

        # Le champ vectoriel = différence entre bruit et GT (à intégrer pour aller de x_0 → x_1)
        trans_vec_field = x_1_trans - x_0_trans
        return (x_t_trans, trans_vec_field)

    def _scale_noise_for_rotation(
        self,
        x_0_rot: torch.FloatTensor,  # (B, 4) — rotation GT (quaternion scalar-first)
        sigma: torch.FloatTensor,     # (B,)   — niveau de bruit
        x_1_rot: torch.FloatTensor,  # (B, 4) — rotation bruit (quaternion aléatoire)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Processus forward pour les ROTATIONS sur SO(3) (flow matching géodésique).

        Contrairement aux translations, on NE PAS faire d'interpolation linéaire de quaternions
        (ça sortirait de SO(3)). On utilise la représentation axis-angle :

          R_t = exp(σ * ω) @ R_0
          où ω = log(R_1)  est l'axis-angle de la rotation de bruit R_1

        Cela garantit que R_t reste une matrice de rotation valide à tout instant.

        Le champ vectoriel cible pour les rotations = ω = axis-angle du bruit
        (le modèle prédit un axis-angle, appliqué via exp() pendant l'inference)

        Returns:
            (x_t_rot, rot_vec_field)
            - x_t_rot      : quaternion bruité à temps t [B, 4]
            - rot_vec_field : axis-angle de la rotation de bruit [B, 3]
        """
        sigma = sigma.unsqueeze(-1)  # (B,) → (B, 1)

        # Convertit les quaternions en matrices de rotation 3x3
        x_0_rot_mat = p3dt.quaternion_to_matrix(x_0_rot)
        x_1_rot_mat = p3dt.quaternion_to_matrix(x_1_rot)

        # Calcule le champ vectoriel = axis-angle de la rotation de bruit R_1
        # log(R_1) = représentation axis-angle de R_1, vecteur ∈ R³
        rot_vec_field = p3dt.matrix_to_axis_angle(x_1_rot_mat)

        # Interpolation géodésique : R_t = exp(σ * ω) @ R_0
        # exp(σ * ω) = rotation d'angle σ*||ω|| autour de l'axe ω/||ω||
        x_t_rot_mat = p3dt.axis_angle_to_matrix(sigma * rot_vec_field) @ x_0_rot_mat

        if self.config.get("stochastic_paths"):
            # Bruit stochastique appliqué via composition de rotations
            epsilon_t = self._calc_stochastic_epsilon(sigma)
            x_t_rot_mat = x_t_rot_mat @ p3dt.axis_angle_to_matrix(
                epsilon_t * torch.randn_like(rot_vec_field)
            )

        # Reconvertit la matrice de rotation en quaternion scalar-first
        x_t_rot = p3dt.matrix_to_quaternion(x_t_rot_mat)
        return (x_t_rot, rot_vec_field)

    def scale_noise(
        self,
        sample: torch.FloatTensor,    # (B, 7) — pose GT [trans(3) + quat(4)]
        timestep: torch.FloatTensor,  # (B,)   — timesteps
        noise: torch.FloatTensor,     # (B, 7) — bruit [trans_noise(3) + rot_noise(4)]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Applique le processus forward (ajout de bruit) sur une pose SE(3).

        Traite séparément les translations et les rotations, puis concatène.

        Returns:
            (x_t, vec_field) où :
            - x_t       : pose bruitée (B, 7) = [trans_bruitée(3) + quat_bruité(4)]
            - vec_field : champ vectoriel cible (B, 6) = [vec_trans(3) + vec_rot(3)]
                          Note : 6D car le champ rotatif est un axis-angle (3D), pas un quaternion
        """
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = self.timesteps.to(sample.device)
        timestep = timestep.to(sample.device)

        # Trouve l'index σ correspondant au timestep demandé
        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        sigma = sigmas[step_indices].flatten()  # (B,) — σ pour chaque élément du batch

        # Applique séparément aux translations et aux rotations
        x_t_trans, trans_vec_field = self._scale_noise_for_translation(
            sample[..., :3],   # translations GT
            sigma,
            noise[..., :3],    # bruit gaussien pour les translations
        )
        x_t_rots, rot_vec_field = self._scale_noise_for_rotation(
            sample[..., 3:],   # quaternions GT
            sigma,
            noise[..., 3:],    # quaternions aléatoires comme bruit
        )

        # Concatène : x_t = [trans_bruitée | quat_bruité]
        # vec_field = [vec_trans(3) | vec_rot(3)] → 6D total
        return torch.cat([x_t_trans, x_t_rots], dim=-1), torch.cat(
            [trans_vec_field, rot_vec_field], dim=-1
        )

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        """Trouve l'index d'un timestep dans le schedule (gère les doublons)."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        # Prend le 2ème index si plusieurs matches (évite de sauter un σ en milieu de schedule)
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        """Initialise l'index de step pour l'inference."""
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def _step_for_translation(
        self,
        vec_field: torch.FloatTensor,    # (B, 3) — champ vectoriel prédit (translations)
        delta_sigma: torch.FloatTensor,  # (B,)   — Δσ = σ_{t-1} - σ_t (négatif !)
        sample: torch.FloatTensor,       # (B, 3) — pose actuelle
    ) -> torch.FloatTensor:
        """
        Step d'Euler pour les translations.

        Intégration d'Euler : x_{t-1} = x_t + Δσ * v
        Comme Δσ < 0 (σ décroît), on avance vers le signal (GT) en soustrayant du bruit.
        """
        prev_sample = sample + delta_sigma * vec_field
        if self.config.get("stochastic_paths"):
            # Bruit stochastique (Langevin dynamics) : √(-Δσ) car Δσ < 0
            prev_sample += (
                self.config.get("stochastic_level")
                * torch.sqrt(-delta_sigma)
                * torch.randn_like(vec_field)
            )
        return prev_sample

    def _step_for_rotation(
        self,
        vec_field: torch.FloatTensor,    # (B, 3) — axis-angle prédit (champ rotatif)
        delta_sigma: torch.FloatTensor,  # (B,)   — Δσ (négatif)
        sample: torch.FloatTensor,       # (B, 4) — quaternion actuel
    ) -> torch.FloatTensor:
        """
        Step d'Euler pour les rotations sur SO(3).

        Intégration géodésique : R_{t-1} = exp(Δσ * ω) @ R_t
        Comme Δσ < 0, exp(Δσ * ω) est une rotation dans la direction INVERSE du bruit,
        ce qui ramène progressivement la rotation vers la GT.

        Returns:
            Quaternion scalar-first de la rotation mise à jour [B, 4]
        """
        # Convertit le step en matrice de rotation et compose avec la rotation actuelle
        prev_sample = (
            p3dt.axis_angle_to_matrix(delta_sigma * vec_field)
            @ p3dt.quaternion_to_matrix(sample)
        )
        if self.config.get("stochastic_paths"):
            z = (
                self.config.get("stochastic_level")
                * torch.sqrt(-delta_sigma)
                * torch.randn_like(vec_field)
            )
            prev_sample = prev_sample @ p3dt.axis_angle_to_matrix(z)

        prev_sample = p3dt.matrix_to_quaternion(prev_sample)
        return prev_sample

    def step(
        self,
        model_output: torch.FloatTensor,       # (B, 6) — sortie du denoiser [Δtrans | axis-angle]
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,             # (B, 7) — pose actuelle [trans | quat]
    ) -> SE3FlowMatchEulerDiscreteSchedulerOutput:
        """
        Effectue UN step d'intégration d'Euler pour l'inference.

        À chaque step :
          1. Calcule Δσ = σ_{t-1} - σ_t  (négatif : on va de plus de bruit vers moins)
          2. Met à jour la translation : x_{t-1} = x_t + Δσ * v_trans
          3. Met à jour la rotation    : R_{t-1} = exp(Δσ * ω) @ R_t
          4. Incrémente le compteur de steps

        Args:
            model_output : champ vectoriel prédit par le modèle (B, 6)
                           [:3] = translation field, [3:] = rotation axis-angle
            timestep     : timestep courant
            sample       : pose actuelle (B, 7)

        Returns:
            SE3FlowMatchEulerDiscreteSchedulerOutput avec prev_sample = pose mise à jour (B, 7)
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        # Assure la précision float32 pour éviter des erreurs numériques
        sample = sample.to(torch.float32)

        # σ_t et σ_{t-1} depuis la table des sigmas
        sigma = self.sigmas[self.step_index]           # σ actuel
        sigma_next = self.sigmas[self.step_index + 1]  # σ suivant (plus petit)
        delta_sigma = sigma_next - sigma               # Δσ < 0

        # Step Euler pour translations et rotations séparément
        prev_sample_trans = self._step_for_translation(
            model_output[..., :3],   # champ vectoriel de translation
            delta_sigma,
            sample[..., :3],         # translation actuelle
        )
        prev_sample_rot = self._step_for_rotation(
            model_output[..., 3:],   # axis-angle du champ rotatif
            delta_sigma,
            sample[..., 3:],         # quaternion actuel
        )

        # Concatène translation et rotation mise à jour
        prev_sample = torch.cat([prev_sample_trans, prev_sample_rot], dim=-1)
        prev_sample = prev_sample.to(model_output.dtype)  # revient au dtype d'origine (fp16)

        # Avance au timestep suivant
        self._step_index += 1

        return SE3FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


"""
=====================================================================
Ce qui suit est adapté de puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus

Les classes SE3PiecewiseScheduler et SE3DDPMScheduler sont des variantes
basées sur la DIFFUSION DDPM classique (non utilisées par défaut dans GARF).

Différence principale avec le Flow Matching :
  DDPM prédit le BRUIT ajouté à chaque step et doit l'enlever progressivement
  Flow Matching prédit le CHAMP VECTORIEL qui pointe vers la solution

DDPM sur SO(3) nécessite de travailler dans l'espace de Lie algebra (axis-angle = log(R))
pour interpoler les rotations, puis d'utiliser exp() pour revenir dans SO(3).
=====================================================================
"""


def betas_for_alpha_bar(
    num_diffusion_timesteps=1000,
    max_beta=0.999,
    alpha_transform_type="piece_wise",
):
    """
    Génère un schedule de betas pour la diffusion DDPM.

    alpha_bar(t) = produit cumulatif de (1 - beta_t)
    → contrôle la quantité de signal restante au temps t

    Trois formes supportées :
      - cosine     : schedule cosinus standard (Ho et al. 2020 amélioré)
      - exp        : exponentiel
      - piece_wise : quadratique par morceaux (utilisé ici)
                     resemble au piecewise-linear du flow matching
    """
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    elif alpha_transform_type == "piece_wise":
        def alpha_bar_fn(t):
            t = t * 1000
            if t <= 700:
                # Quadratique lent : alpha_bar va de 1 à 0.9 sur les premiers 70%
                return 1 - 0.1 * (t / 700) ** 2
            else:
                # Quadratique rapide : alpha_bar va de 0.9 à 0 sur les derniers 30%
                return 0.9 * (1 - ((t - 700) / 300) ** 2)
    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class PiecewiseScheduler(DDPMScheduler):
    """DDPM Scheduler avec un schedule de betas quadratique par morceaux."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.betas = betas_for_alpha_bar(alpha_transform_type="piece_wise")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


class SE3PiecewiseScheduler(PiecewiseScheduler):
    """
    DDPM Scheduler adapté pour SE(3) avec schedule piecewise.

    Surcharge add_noise et step pour gérer les rotations SO(3) via log/exp maps.
    Utilise so3_log_map pour passer dans l'algèbre de Lie (espace tangent),
    fait les calculs DDPM classiques, puis so3_exp_map pour revenir dans SO(3).
    """
    def add_noise(
        self,
        original_samples: torch.Tensor,  # (B, 7) — pose GT [trans | quat]
        noise: torch.Tensor,              # (B, 6) — bruit [trans_noise(3) | rot_noise(3)]
        timesteps: torch.Tensor,
    ):
        """
        Processus forward DDPM : x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise

        Pour les rotations : on travaille dans l'espace axis-angle (log de la rotation)
          log_rot = so3_log_map(R)  ∈ R³
          noisy_log_rot = sqrt(α) * log_rot + sqrt(1-α) * rot_noise
          R_noisy = so3_exp_map(noisy_log_rot)
        """
        translations = original_samples[:, :3]
        quaternions = original_samples[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # (B, 3) — dans l'espace de Lie algebra

        trans_noise = noise[:, :3]
        rot_noise = noise[:, 3:]

        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # DDPM classique sur translations
        noisy_translations = sqrt_alpha_prod * translations + sqrt_one_minus_alpha_prod * trans_noise
        # DDPM dans l'espace de Lie pour les rotations
        noisy_log_rot = sqrt_alpha_prod * log_rot + sqrt_one_minus_alpha_prod * rot_noise
        # Retour dans SO(3) via exp map
        noisy_rot_matrics = p3dt.so3_exp_map(noisy_log_rot)
        noisy_quaternions = p3dt.matrix_to_quaternion(noisy_rot_matrics)

        noisy_samples = torch.cat([noisy_translations, noisy_quaternions], dim=1)
        return noisy_samples

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        """
        Step de dénoising DDPM (processus inverse).

        Formule DDPM inverse :
          x_{t-1} = (sqrt(alpha_{t-1}) * beta_t / (1-alpha_t)) * x_0_pred
                  + (sqrt(alpha_t) * (1-alpha_{t-1}) / (1-alpha_t)) * x_t
                  + variance * epsilon

        Pour les rotations : calculs dans l'espace log, puis conversion finale.
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # Calcul des coefficients alpha/beta
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        translations = sample[:, :3]
        quaternions = sample[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # espace de Lie

        if self.config.prediction_type == "epsilon":
            # Le modèle prédit le bruit → on déduit x_0
            pred_trans = (translations - beta_prod_t**0.5 * model_output[:, :3]) / alpha_prod_t**0.5
            pred_log_rot = (log_rot - beta_prod_t**0.5 * model_output[:, 3:6]) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            # Le modèle prédit x_0 directement
            pred_trans = model_output[:, :3]
            pred_log_rot = model_output[:, 3:6]
        elif self.config.prediction_type == "v_prediction":
            # Prédiction "velocity" (mélange bruit et signal)
            pred_trans = (alpha_prod_t**0.5) * translations - (beta_prod_t**0.5) * model_output[:, :3]
            pred_log_rot = (alpha_prod_t**0.5) * log_rot - (beta_prod_t**0.5) * model_output[:, 3:6]
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`")

        # Coefficients pour la moyenne de x_{t-1}
        pred_trans_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_trans_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        pred_log_rot_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_log_rot_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # Moyenne de la distribution p(x_{t-1} | x_t)
        pred_prev_trans = pred_trans_coeff * pred_trans + current_trans_coeff * translations
        pred_prev_log_rot = pred_log_rot_coeff * pred_log_rot + current_log_rot_coeff * log_rot

        # Ajout du bruit stochastique (sauf au dernier step t=0)
        variance = torch.zeros_like(model_output)
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_trans = pred_prev_trans + variance[:, :3]
        pred_prev_log_rot = pred_prev_log_rot + variance[:, 3:6]

        # Retour dans SO(3) via exp map
        pred_prev_rot_matrices = p3dt.so3_exp_map(pred_prev_log_rot)
        pred_prev_quaternions = p3dt.matrix_to_quaternion(pred_prev_rot_matrices)
        pred_prev_sample = torch.cat([pred_prev_trans, pred_prev_quaternions], dim=1)

        if not return_dict:
            return (pred_prev_sample,)
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample)


class SE3DDPMScheduler(DDPMScheduler):
    """
    DDPM Scheduler adapté pour SE(3) avec schedule cosinus standard.
    Identique à SE3PiecewiseScheduler mais avec le schedule de betas par défaut de DDPM.
    """
    def add_noise(self, original_samples, noise, timesteps):
        """Même logique que SE3PiecewiseScheduler.add_noise (log map pour SO(3))."""
        translations = original_samples[:, :3]
        quaternions = original_samples[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)

        trans_noise = noise[:, :3]
        rot_noise = noise[:, 3:]

        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_translations = sqrt_alpha_prod * translations + sqrt_one_minus_alpha_prod * trans_noise
        noisy_log_rot = sqrt_alpha_prod * log_rot + sqrt_one_minus_alpha_prod * rot_noise
        noisy_rot_matrics = p3dt.so3_exp_map(noisy_log_rot)
        noisy_quaternions = p3dt.matrix_to_quaternion(noisy_rot_matrics)

        noisy_samples = torch.cat([noisy_translations, noisy_quaternions], dim=1)
        return noisy_samples

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        """Même logique que SE3PiecewiseScheduler.step."""
        t = timestep
        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        translations = sample[:, :3]
        quaternions = sample[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)

        if self.config.prediction_type == "epsilon":
            pred_trans = (translations - beta_prod_t**0.5 * model_output[:, :3]) / alpha_prod_t**0.5
            pred_log_rot = (log_rot - beta_prod_t**0.5 * model_output[:, 3:6]) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_trans = model_output[:, :3]
            pred_log_rot = model_output[:, 3:6]
        elif self.config.prediction_type == "v_prediction":
            pred_trans = (alpha_prod_t**0.5) * translations - (beta_prod_t**0.5) * model_output[:, :3]
            pred_log_rot = (alpha_prod_t**0.5) * log_rot - (beta_prod_t**0.5) * model_output[:, 3:6]
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`")

        pred_trans_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_trans_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        pred_log_rot_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_log_rot_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_trans = pred_trans_coeff * pred_trans + current_trans_coeff * translations
        pred_prev_log_rot = pred_log_rot_coeff * pred_log_rot + current_log_rot_coeff * log_rot

        variance = torch.zeros_like(model_output)
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_trans = pred_prev_trans + variance[:, :3]
        pred_prev_log_rot = pred_prev_log_rot + variance[:, 3:6]

        pred_prev_rot_matrices = p3dt.so3_exp_map(pred_prev_log_rot)
        pred_prev_quaternions = p3dt.matrix_to_quaternion(pred_prev_rot_matrices)
        pred_prev_sample = torch.cat([pred_prev_trans, pred_prev_quaternions], dim=1)

        if not return_dict:
            return (pred_prev_sample,)
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample)


if __name__ == "__main__":
    # Test rapide : affiche les sigmas du scheduler linéaire
    scheduler = SE3FlowMatchEulerDiscreteScheduler()
    print(scheduler.sigmas)
