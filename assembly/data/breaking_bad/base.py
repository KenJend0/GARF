"""
assembly/data/breaking_bad/base.py
===================================
Classe de base du dataset "Breaking Bad".

Breaking Bad est un dataset synthétique de fragments 3D :
  - Des objets 3D (vases, bols, etc.) sont "cassés" numériquement en morceaux
  - Chaque morceau est un fragment avec une pose GT (translation + rotation)
  - L'objectif du modèle : retrouver ces poses GT à partir des nuages de points

Structure du fichier HDF5 (data_root) :
  data_split/
    everyday/train/  → liste des noms d'objets du split train
    artifact/test/   → liste des noms d'objets du split test
  <nom_objet>/
    pieces/
      piece_0/vertices, faces, shared_faces
      piece_1/...
    pieces_names        → noms lisibles des pièces
    removal_masks       → masques pour simuler pièces manquantes
    removal_order       → ordre de retrait des pièces
    redundant_pieces    → pièces "parasites" pour simuler des fragments non-appartenant

Sous-classes concrètes :
  - BreakingBadUniform  : échantillonnage uniforme des points sur le mesh
  - BreakingBadWeighted : échantillonnage pondéré par surface (plus réaliste)
"""

from typing import Literal, List

import trimesh
import logging
import h5py
import numpy as np
from torch.utils.data import Dataset, default_collate

# Palette de couleurs pour la visualisation des fragments (1 couleur par fragment)
COLORS = [
    [254, 138, 24],  # orange
    [201, 26, 9],    # rouge
    [35, 120, 65],   # vert foncé
    [0, 85, 191],    # bleu
    [242, 112, 94],  # saumon
    [252, 151, 172], # rose
    [75, 159, 74],   # vert clair
    [0, 143, 155],   # cyan
    [245, 205, 47],  # jaune
    [67, 84, 163],   # indigo
    [179, 215, 209], # turquoise clair
    [199, 210, 60],  # jaune-vert
    [255, 128, 13],  # orange vif
]


class BreakingBadBase(Dataset):
    """
    Dataset de base pour le reassembly de fragments 3D (Breaking Bad).

    Paramètres clés :
      split             : 'train', 'val' ou 'test'
      data_root         : chemin vers le fichier HDF5
      category          : 'everyday' (objets courants), 'artifact' (archéo), 'all'
      min_parts         : nombre minimum de fragments par objet (filtre)
      max_parts         : nombre maximum de fragments (sert aussi de padding)
      num_points_to_sample : points échantillonnés par fragment
      num_removal       : nb de fragments à retirer (simulation pièces manquantes)
      num_redundancy    : nb de fragments parasites à ajouter (simulation)
      random_anchor     : si True, choisit le fragment référence aléatoirement
    """

    COLORS = COLORS

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_root: str = "data",
        category: Literal["everyday", "artifact", "other", "all"] = "everyday",
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 8192,
        min_points_per_part: int = 20,
        num_removal: int = 0,
        num_redundancy: int = 0,
        multi_ref: bool = False,
        mesh_sample_strategy: Literal["uniform", "poisson"] = "uniform",
        random_anchor: bool = False,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.category = category
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.num_removal = num_removal
        self.num_redundancy = num_redundancy
        self.multi_ref = multi_ref
        self.mesh_sample_strategy = mesh_sample_strategy
        self.random_anchor = random_anchor
        self.data_list = self.get_data_list()

        print("Using mesh sample strategy:", self.mesh_sample_strategy)
        trimesh.util.log.setLevel(logging.ERROR)  # silence les logs verbeux de trimesh
        # Contrainte : on ne peut pas simuler simultanément des pièces manquantes ET parasites
        assert not (self.num_removal and self.num_redundancy), (
            "Cannot enable removal and redundancy at the same time"
        )

    def get_data_list(self) -> List[str]:
        """
        Lit le fichier HDF5 et retourne la liste des objets valides pour ce split.

        Filtre les objets selon min_parts / max_parts, en tenant compte
        du nombre de removals/redundancies souhaités.

        Returns:
            Liste de strings (clés HDF5) comme ['<category>/<obj_name>', ...]
        """
        h5_file = h5py.File(self.data_root, "r")
        if self.category == "all":
            # Combine les deux catégories disponibles
            everyday_objs = list(h5_file["data_split"]["everyday"][self.split])
            artifact_objs = list(h5_file["data_split"]["artifact"][self.split])
            data_list = everyday_objs + artifact_objs
        else:
            data_list = list(h5_file["data_split"][self.category][self.split])

        # Décode les bytes HDF5 → strings Python
        data_list = [d.decode("utf-8") for d in data_list]

        filtered_data_list = []
        for item in data_list:
            try:
                num_parts = len(h5_file[item]["pieces"].keys())
                # Règle de filtrage :
                #   - Si removal : après retrait, il doit rester au moins min_parts fragments
                #   - Si redundancy : après ajout, il ne doit pas dépasser max_parts
                if (
                    self.min_parts + self.num_removal
                    <= num_parts
                    <= self.max_parts - self.num_redundancy
                    and num_parts > self.num_redundancy
                ):
                    filtered_data_list.append(item)
            except:
                continue

        h5_file.close()
        return filtered_data_list

    def get_meshes(self, name: str) -> List[trimesh.Trimesh]:
        """
        Charge les meshes bruts d'un objet pour la visualisation.

        Note : cette méthode n'applique PAS la normalisation d'échelle.
        Utilisée pour la visualisation, pas pour l'entraînement.

        Returns:
            Liste de dicts {'vertices', 'faces', 'color'} par fragment
        """
        h5_file = h5py.File(self.data_root, "r")
        pieces = h5_file[name]["pieces"].keys()

        meshes = [
            {
                "vertices": np.array(h5_file[name]["pieces"][piece]["vertices"][:]),
                "faces": np.array(h5_file[name]["pieces"][piece]["faces"][:]),
                "color": self.COLORS[idx % len(self.COLORS)],
            }
            for idx, piece in enumerate(pieces)
        ]
        h5_file.close()
        return meshes

    def get_data(self, index: int):
        """
        Charge un exemple complet (tous les fragments d'un objet) depuis le HDF5.

        Pipeline :
          1. Charge les meshes des fragments
          2. Normalise l'échelle (max extent → 1.0)
          3. Applique removal/redundancy si configuré
          4. Échantillonne les nuages de points + normales + labels fracture
          5. Retourne un dict prêt pour transform()

        Normalisation d'échelle :
          On divise tous les meshes par la plus grande étendue (bounding box max).
          → Tous les objets font "environ 1 unité" de taille.
          → Évite que le modèle apprenne des biais de taille absolue.

        Returns:
            dict avec les clés :
              'index', 'name', 'num_parts',
              'pointclouds_gt'        : nuages de points des fragments
              'pointclouds_normals_gt': normales des fragments
              'fracture_surface_gt'   : labels binaires fracture/original par point
              'graph'                 : matrice d'adjacence (qui touche qui)
              'removal_pieces'        : noms des pièces retirées
              'redundant_pieces'      : noms des pièces parasites ajoutées
              'mesh_scale'            : facteur d'échelle appliqué (pour dénormaliser)
              'meshes'                : objets trimesh (uniquement pour val/test)
        """
        name = self.data_list[index]

        h5_file = h5py.File(self.data_root, "r")
        pieces = h5_file[name]["pieces"].keys()
        pieces_names = h5_file[name]["pieces_names"][:]
        pieces_names = [name.decode("utf-8") for name in pieces_names]
        num_parts = len(pieces)

        # Charge tous les meshes des fragments
        meshes = [
            trimesh.Trimesh(
                vertices=np.array(h5_file[name]["pieces"][piece]["vertices"][:]),
                faces=np.array(h5_file[name]["pieces"][piece]["faces"][:]),
            )
            for piece in pieces
        ]

        # Normalisation d'échelle : trouve la plus grande dimension parmi tous les fragments
        meshes_max_scale = 1.0
        for i in range(num_parts):
            extents = meshes[i].extents  # [dx, dy, dz] de la bounding box
            meshes_max_scale = max(meshes_max_scale, max(extents))
        # Divise tous les meshes par cette échelle → objet normalisé dans [0, 1]
        meshes = [mesh.apply_scale(1.0 / meshes_max_scale) for mesh in meshes]

        # shared_faces[i] contient les indices de faces du fragment i qui sont
        # sur la surface de fracture (faces partagées avec d'autres fragments)
        # Si pas défini dans le HDF5, on met -1 pour tout (pas de fracture connue)
        shared_faces = [
            (
                np.array(h5_file[name]["pieces"][piece]["shared_faces"][:])
                if "shared_faces" in h5_file[name]["pieces"][piece]
                else -np.ones(len(meshes[idx].faces), dtype=np.int64)
            )
            for idx, piece in enumerate(pieces)
        ]

        # Construit le graphe de connectivité entre fragments (qui touche qui)
        graph = self.get_graph(shared_faces=shared_faces)

        # --- Simulation de pièces MANQUANTES (removal) ---
        # Simule le cas réel où on n'a pas tous les fragments
        removal_pieces = []
        if self.num_removal > 0:
            num_parts -= self.num_removal
            # removal_masks[i] = masque booléen → quels fragments garder si on retire i pièces
            removal_mask = h5_file[name]["removal_masks"][self.num_removal - 1]
            meshes = np.array(meshes)[removal_mask]
            shared_faces = np.array(shared_faces, dtype="object")[removal_mask]

            # Enregistre quelles pièces ont été retirées (pour les métriques)
            removal_order = np.array(h5_file[name]["removal_order"][: self.num_removal])
            removal_pieces = list(
                np.array(
                    [name.decode("utf-8") for name in h5_file[name]["pieces_names"]]
                )[removal_order]
            )
            assert len(meshes) == num_parts
            assert len(shared_faces) == num_parts

        # --- Simulation de pièces PARASITES (redundancy) ---
        # Simule le cas réel où on a des fragments qui n'appartiennent pas à l'objet
        redundant_pieces = []
        if self.num_redundancy > 0:
            assert num_parts > self.num_redundancy, (
                "num of parts should greater than num of redundancy"
            )
            # redundant_pieces : fragments pris d'un AUTRE objet, ajoutés comme distracteurs
            redundant_pieces = h5_file[name]["redundant_pieces"][: self.num_redundancy]
            redundant_pieces = [
                f"{p[0].decode('utf-8')}/pieces/{p[1].decode('utf-8')}"
                for p in redundant_pieces
            ]
            redundant_meshes = [
                trimesh.Trimesh(
                    vertices=np.array(h5_file[p]["vertices"][:]),
                    faces=np.array(h5_file[p]["faces"][:]),
                )
                for p in redundant_pieces
            ]
            # Les fragments parasites n'ont pas de shared_faces (pas de voisins)
            redundant_shared_faces = [
                -np.ones(len(mesh.faces), dtype=np.int64) for mesh in redundant_meshes
            ]

            # Ajoute les fragments parasites à la fin de la liste
            # Note : le graphe n'est pas mis à jour car ces fragments ne sont connectés à rien
            meshes.extend(redundant_meshes)
            shared_faces.extend(redundant_shared_faces)
            num_parts += self.num_redundancy

        h5_file.close()

        # Échantillonne les nuages de points depuis les meshes (implémenté dans les sous-classes)
        pointclouds_gt, pointclouds_normals_gt, fracture_surface_gt = (
            self.sample_points(
                meshes=meshes,
                shared_faces=shared_faces,
            )
        )

        data = {
            "index": index,
            "name": name,
            "num_parts": num_parts,
            "pointclouds_gt": pointclouds_gt,          # nuages de points [P, N, 3]
            "pointclouds_normals_gt": pointclouds_normals_gt,  # normales [P, N, 3]
            "fracture_surface_gt": fracture_surface_gt, # labels fracture [P, N] (0/1)
            "graph": graph,                             # adjacence [max_parts, max_parts]
            "removal_pieces": ",".join(removal_pieces),
            "redundant_pieces": ",".join(redundant_pieces),
            "pieces": ",".join(pieces_names),
            "mesh_scale": meshes_max_scale,             # facteur de normalisation
            "meshes": meshes,                           # objets trimesh (pour visualisation)
        }

        # En train, les meshes ne sont pas nécessaires et prennent trop de mémoire
        if self.split == "train":
            del data["meshes"]

        return data

    def transform(self, data: dict):
        """Applique les augmentations (rotations, poses GT...). Défini dans les sous-classes."""
        raise NotImplementedError

    def sample_points(
        self,
        meshes: List[trimesh.Trimesh],
        shared_faces: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Échantillonne des points + normales + labels fracture sur les meshes.
        Défini dans les sous-classes (uniform ou weighted).
        """
        raise NotImplementedError

    def get_graph(
        self,
        shared_faces: List[np.ndarray],
    ) -> np.ndarray:
        """
        Construit la matrice d'adjacence entre fragments.

        Deux fragments i et j sont connectés si le fragment i a des faces
        "shared" avec j (i.e. des faces qui touchent j dans l'objet original).

        Returns:
            np.ndarray [max_parts, max_parts] booléen
            graph[i, j] = True si les fragments i et j se touchent
        """
        num_parts = len(shared_faces)
        # Initialise à zéro, taille max_parts × max_parts (padding pour le batch)
        graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)

        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                # shared_faces[i] contient les indices des fragments voisins du fragment i
                if j in shared_faces[i]:
                    graph[i, j] = graph[j, i] = 1  # connexion symétrique

        return graph.astype(bool)

    def _pad_data(self, input_data: np.ndarray):
        """
        Padde les données à la forme [max_parts, ...] avec des zéros.

        Pourquoi ? PyTorch nécessite des tenseurs de taille fixe dans un batch.
        On padde jusqu'à max_parts, et on utilisera un masque (part_valids) pour
        ignorer les parties paddées lors des calculs.

        Args:
            input_data : tableau de forme [P, ...] où P <= max_parts

        Returns:
            tableau paddé de forme [max_parts, ...]
        """
        d = np.array(input_data)
        pad_shape = (self.max_parts,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: d.shape[0]] = d  # copie les vraies données au début
        return pad_data

    def __getitem__(self, index):
        """Retourne un exemple transformé (DataLoader appelle cette méthode)."""
        data = self.get_data(index)
        data = self.transform(data)
        return data

    def visualize(self, index: int):
        """Visualisation d'un exemple. Défini dans les sous-classes."""
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def export_hdf5(self, output_path: str):
        """
        Exporte le dataset prétraité dans un nouveau fichier HDF5.
        Utile pour pré-calculer les nuages de points une fois et réutiliser.
        """
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor

        f = h5py.File(output_path, "w")
        # Métadonnées du dataset
        f.attrs["dataset"] = self.__class__.__name__
        f.attrs["split"] = self.split
        f.attrs["data_root"] = self.data_root
        f.attrs["category"] = self.category
        f.attrs["min_parts"] = self.min_parts
        f.attrs["max_parts"] = self.max_parts
        f.attrs["num_points_to_sample"] = self.num_points_to_sample
        f.attrs["min_points_per_part"] = self.min_points_per_part
        f.attrs["num_samples"] = len(self)

        # Parallélise le prétraitement sur 48 workers
        pool = ProcessPoolExecutor(max_workers=48)
        for data in tqdm(
            pool.map(self.get_data, range(len(self))),
            desc="Exporting data",
            total=len(self),
        ):
            name = data["name"]
            group = f.create_group(name)
            group.create_dataset("num_parts", data=data["num_parts"])
            group.create_dataset("pointclouds_gt", data=data["pointclouds_gt"])
            group.create_dataset("pointclouds_normals_gt", data=data["pointclouds_normals_gt"])
            group.create_dataset("fracture_surface_gt", data=data["fracture_surface_gt"])
            group.create_dataset("graph", data=data["graph"])

        f.close()

    @staticmethod
    def collate_fn(batch):
        """
        Fonction de collation personnalisée pour le DataLoader.

        Problème : les meshes trimesh ne sont pas collatables avec default_collate
        (ce sont des objets Python, pas des tenseurs).
        Solution : on les met dans une liste simple, et on collate le reste normalement.
        """
        collated_batch = {}
        for key in batch[0].keys():
            if key == "meshes":
                # Les meshes restent une liste de listes (pas de tenseur)
                collated_batch[key] = [item[key] for item in batch]
            else:
                # Tous les autres champs → tenseurs PyTorch via default_collate
                collated_batch[key] = default_collate([item[key] for item in batch])

        return collated_batch
