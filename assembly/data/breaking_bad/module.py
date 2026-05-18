from typing import Literal, Optional, List, Dict

import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from . import BreakingBadUniform, BreakingBadWeighted


class BreakingBadDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data",
        categories: List[str] = ["everyday"],
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 1000,
        min_points_per_part: int = 20,
        sample_method: Literal["uniform", "weighted"] = "uniform",
        batch_size: int = 32,
        num_workers: int = 16,
        num_removal: int = 0,
        num_redundancy: int = 0,
        multi_ref: bool = False,
        mesh_sample_strategy: Literal["uniform", "poisson"] = "poisson",
        random_anchor: bool = False,
        additional_data_root: Optional[Dict[str, str]] = None,
        weight_hard_examples: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.categories = categories
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.sample_method = sample_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Please be noted that num_removal and num_redundancy are only used in the testing phase
        self.num_removal = num_removal
        self.num_redundancy = num_redundancy

        # Please be noted that multi_ref is only used in the training phase
        self.multi_ref = multi_ref

        self.mesh_sample_strategy = mesh_sample_strategy
        self.random_anchor = random_anchor

        print("Using mesh sample strategy:", self.mesh_sample_strategy)

        # If breaking_bad_other_data_root is provided
        self.additional_data_root = additional_data_root
        # Suréchantillonnage des objets difficiles (faible ratio fracture)
        # weight_hard_examples=True active le WeightedRandomSampler sur train
        self.weight_hard_examples = weight_hard_examples

        if self.sample_method == "uniform":
            self.dataset_cls = BreakingBadUniform
        elif self.sample_method == "weighted":
            self.dataset_cls = BreakingBadWeighted
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage):
        if stage == "fit":

            self.train_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="train",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        multi_ref=self.multi_ref,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                    )
                    for category in self.categories
                ]
            )

            self.val_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="val",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                    )
                    for category in self.categories
                ]
            )

        if stage == "test" or stage == "predict":
            self.test_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="test",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        num_removal=self.num_removal,
                        num_redundancy=self.num_redundancy,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                    )
                    for category in self.categories
                ]
            )

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.weight_hard_examples:
            # Calcule les ratios fracture de tous les datasets concaténés
            all_ratios = []
            for ds in self.train_dataset.datasets:
                print("Computing fracture ratios for weighted sampling...")
                all_ratios.extend(ds.compute_fracture_ratios())
            all_ratios = np.array(all_ratios, dtype=np.float32)
            # Poids inversement proportionnel au ratio + epsilon pour éviter /0
            # Les objets difficiles (ratio~0) reçoivent un poids élevé
            weights = 1.0 / (all_ratios + 0.05)
            sampler = WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=len(all_ratios),
                replacement=True,
            )
            shuffle = False   # incompatible avec sampler
            print(f"WeightedRandomSampler: {len(all_ratios)} objects, "
                  f"ratio range [{all_ratios.min():.3f}, {all_ratios.max():.3f}]")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            sampler=sampler,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def test_dataloader(self):
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=BreakingBadWeighted.collate_fn
        )
