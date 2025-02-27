import json
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
import torch
import numpy as np
import h5py
import pandas as pd

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule
from monai import transforms
from kornia.augmentation import RandomAffine3D, RandomDepthicalFlip3D, RandomHorizontalFlip3D, RandomVerticalFlip3D, CenterCrop3D

import torch.nn.functional as F


SITE_MAP = {11025: 0., 11026: 1., 11027: 2.}


class TrainingAugments(torch.nn.Module):
    def __init__(self, p_affine: float = 0.5, p_flip: float = 0.5,
                 scale: Tuple[float, float] = (0.95, 1.05),
                 shift: Tuple[float, float, float] = (0.02, 0.02, 0.02),
                 degrees: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()

        random_affine = RandomAffine3D(degrees=degrees, scale=scale, translate=shift, p=p_affine)
        flip_vertical = RandomVerticalFlip3D(p=p_flip)
        flip_horizontal = RandomHorizontalFlip3D(p=p_flip)
        flip_depth = RandomDepthicalFlip3D(p=p_flip)
        self.transforms = torch.nn.Sequential(flip_vertical,
                                              flip_horizontal,
                                              flip_depth,
                                              random_affine)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        return self.rescale_intensity(self.transforms(x))


    def rescale_intensity(self, x: torch.Tensor):
        assert x.dim() == 5
        x_min = x.flatten(2,-1).min(-1)[0].view(*x.shape[:2], 1, 1, 1)
        x_max = x.flatten(2,-1).max(-1)[0].view(*x.shape[:2], 1, 1, 1)
        return (x-x_min)/(x_max-x_min)


# class ValidationAugments(torch.nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()
    
#     @torch.no_grad()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.dim() == 5
#         return self.rescale_intensity(x)

#     def rescale_intensity(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.dim() == 5
#         x_min = x.flatten(2,-1).min(-1)[0].view(*x.shape[:2], 1, 1, 1)
#         x_max = x.flatten(2,-1).max(-1)[0].view(*x.shape[:2], 1, 1, 1)
#         return (x-x_min)/(x_max-x_min)


def standardize_age(age: torch.Tensor) -> torch.Tensor:
    return (age - 63.709) / 7.519


def one_hot_encode_site(site: int | float) -> torch.Tensor:
    site_tensor = torch.zeros(3)
    site_tensor[int(site)] = 1.0
    return site_tensor


def get_context_vector(age_std: torch.Tensor, sex: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
    age_std = age_std.unsqueeze(0) if age_std.dim() == 0 else age_std
    sex = sex.unsqueeze(0) if sex.dim() == 0 else sex
    assert age_std.dim() == sex.dim() == site.dim() == 1
    return torch.cat([age_std, sex, site])


def get_context_vector_batch(age_std: torch.Tensor, sex: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
    """
    Generates context vectors for a batch of samples.

    Args:
        age_std (torch.Tensor): Standardized ages, shape (B,)
        sex (torch.Tensor): Sex values, shape (B,)
        site (torch.Tensor): One-hot encoded site values, shape (B, 3)

    Returns:
        torch.Tensor: Context vectors with shape (B, 1, D), where D is the concatenated dimension
    """
    # Ensure each tensor has shape (B, 1) or (B, 3) for sites
    age_std = age_std.unsqueeze(1)  # Shape: (B, 1)
    sex = sex.unsqueeze(1)          # Shape: (B, 1)
    # site is already (B, 3)

    # Concatenate along the feature dimension
    context = torch.cat([age_std, sex, site], dim=1)  # Shape: (B, 5)

    # Add a sequence dimension as required by the model
    context = context.unsqueeze(1)  # Shape: (B, 1, 5)

    return context


class UKBDataset(Dataset):
    def __init__(self, path: str, ids: List[Union[int, str]], transform: Optional[Callable] = None,
                 has_bin_age: bool=False, has_orig_eid: bool=False) -> None:
        super().__init__()
        # Force ids to str
        self.ids = list(map(str, ids))
        self.file = h5py.File(path, mode="r")
        self.transform = transform
        self.has_bin_age = has_bin_age
        self.has_orig_eid = has_orig_eid

    def close_file(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        mri = self.file[id]["mri"][:]  # Assuming mri is a NumPy array
        tabular = self.file[id]["tabular"][:]

        sample = {
            "eid": torch.as_tensor(int(id), dtype=torch.long),
            "mri": torch.as_tensor(mri[np.newaxis], dtype=torch.float32),  # Adding channel dimension
            "age": torch.as_tensor(tabular[1], dtype=torch.float32),
            "sex": torch.as_tensor(tabular[2], dtype=torch.float32),
            "site": torch.as_tensor(SITE_MAP.get(tabular[3], -1.0), dtype=torch.long)
        }

        if self.transform:
            sample['mri'] = self.transform(sample['mri'])

        if self.has_bin_age:
            sample["age_bin"] = torch.as_tensor(tabular[4], dtype=torch.float32)
        
        if self.has_orig_eid:
            sample["orig_eid"] = torch.as_tensor(int(tabular[-1]), dtype=torch.long)

        age_std = standardize_age(sample["age"])
        sex = sample["sex"]

        # one-hot encoding for site
        site = one_hot_encode_site(int(sample["site"]))

        context_vector = get_context_vector(age_std, sex, site)

        sample["age_std"] = torch.as_tensor(age_std, dtype=torch.float32).squeeze()
        sample["context"] = context_vector.unsqueeze(0)

        return sample

    def __del__(self):
        self.close_file()


class UKBDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        batch_size: int,
        num_workers: int = 4,
        model_type: str = "autoencoder",
        crop_images: bool = True,
        p_flip: float = 0.0,
        p_shift: float = 0.1,
        p_affine: float = 0.1,
        p_contrast: float = 0.1,
        has_bin_age: bool = False,
        has_orig_eid: bool = False,
        **kwargs: Any,):

        super().__init__()
        self.data_dir = Path(data_dir)
        self.id_path = Path(split_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.has_bin_age = has_bin_age
        self.has_orig_eid = has_orig_eid

        self.p_flip = p_flip
        self.p_shift = p_shift
        self.p_affine = p_affine
        self.p_contrast = p_contrast

        self.train_ids, self.val_ids, self.test_ids = self._load_ids_from_file()

        self.roi_start = [20, 20, 8]
        self.roi_end = [116, 116, 120]

        self.pad_tuples = [(20, 12), (20, 12), (8, 8)]

        self.crop_images = crop_images

        self.train_transforms = self._get_transforms(train=True)
        self.val_transforms = self._get_transforms(train=False)

        self.gpu_train_augments = TrainingAugments(p_affine=p_affine, p_flip=p_flip) if model_type == "predictmeta" else None
        # self.gpu_val_augments = ValidationAugments() if model_type == "predictmeta" else None

        self.gpu_train_augments = TrainingAugments(p_affine=p_affine, p_flip=p_flip, scale=(0.98, 1.02), shift=(0.02, 0.02, 0.02)) if model_type == "predictmeta_soft" else self.gpu_train_augments
        # self.gpu_val_augments = ValidationAugments() if model_type == "predictmeta_soft" else self.gpu_val_augments

        self.gpu_val_augments = None

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.model_type in ("predictmeta", "predictmeta_soft"):
            batch["mri"] = self.gpu_train_augments(batch["mri"]) if self.trainer.training else batch["mri"]
            return batch
        else:
            return super().on_after_batch_transfer(batch, dataloader_idx)

    def _load_ids_from_file(self) -> Tuple[List, List, List]:
        with open(self.id_path, "r") as file:
            splits = json.load(file)
        return splits["train"], splits["val"], splits["test"]

    def _get_transforms(self, train: bool) -> Callable:
        if train:
            if self.model_type == "autoencoder":
                transform = transforms.Compose(
                    [
                        transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                        transforms.RandAffine(
                            translate_range=(1, 1, 1),
                            scale_range=(-0.02, 0.02),
                            prob=self.p_affine,
                        ),
                        transforms.RandShiftIntensity(offsets=0.05, prob=self.p_shift),
                        transforms.RandAdjustContrast(gamma=(0.97, 1.03), prob=self.p_contrast),
                        transforms.ThresholdIntensity(threshold=1.0, above=False, cval=1.0),
                        transforms.ThresholdIntensity(threshold=0.0, above=True, cval=0.0),
                    ]
                )
            elif self.model_type == "diffusion":
                transform = transforms.Compose(
                    [
                        transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                        transforms.RandAffine(
                            translate_range=(1, 1, 1),
                            scale_range=(-0.02, 0.02),
                            prob=self.p_affine,
                        ),
                        transforms.RandShiftIntensity(offsets=0.05, prob=self.p_shift),
                        transforms.RandAdjustContrast(gamma=(0.97, 1.03), prob=self.p_contrast),
                        transforms.ThresholdIntensity(threshold=1.0, above=False, cval=1.0),
                        transforms.ThresholdIntensity(threshold=0.0, above=True, cval=0.0),
                    ]
                )
            elif self.model_type == "predictmeta":
                transform = transforms.Compose(
                    [
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                        transforms.RandShiftIntensity(offsets=0.07, prob=self.p_shift),
                        transforms.RandAdjustContrast(gamma=(0.95, 1.05), prob=self.p_contrast),
                    ]
                )
            elif self.model_type == "predictmeta_soft":
                transform = transforms.Compose(
                    [
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                        transforms.RandShiftIntensity(offsets=0.05, prob=self.p_shift),
                        transforms.RandAdjustContrast(gamma=(0.97, 1.03), prob=self.p_contrast),
                    ]
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
        else:
            if self.model_type in ("predictmeta", "predictmeta_soft"):
                transform = transforms.Compose(
                    [
                        transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                        transforms.SpatialCrop(roi_start=self.roi_start, roi_end=self.roi_end) if self.crop_images else transforms.Identity(),
                    ]
                )
        return transform

    def get_all_dataloaders_with_val_transforms(self):
        """First get train, val and test datasets with val_transforms.
        Then create dataloaders for each and return them."""
        train_dataset = UKBDataset(
            path=str(self.data_dir),
            ids=self.train_ids,
            transform=self.val_transforms,
            has_bin_age=self.has_bin_age,
            has_orig_eid=self.has_orig_eid
        )
        val_dataset = UKBDataset(
            path=str(self.data_dir),
            ids=self.val_ids,
            transform=self.val_transforms,
            has_bin_age=self.has_bin_age,
            has_orig_eid=self.has_orig_eid
        )
        test_dataset = UKBDataset(
            path=str(self.data_dir),
            ids=self.test_ids,
            transform=self.val_transforms,
            has_bin_age=self.has_bin_age,
            has_orig_eid=self.has_orig_eid
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader, val_loader, test_loader

    def setup(self, stage: Optional[str] = None):
        # Assign train/val/test datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_dataset = UKBDataset(
                path=str(self.data_dir),
                ids=self.train_ids,
                transform=self.train_transforms,
                has_bin_age=self.has_bin_age,
                has_orig_eid=self.has_orig_eid
            )
            self.val_dataset = UKBDataset(
                path=str(self.data_dir),
                ids=self.val_ids,
                transform=self.val_transforms,
                has_bin_age=self.has_bin_age,
                has_orig_eid=self.has_orig_eid
            )
        if stage == "test" or stage is None:
            self.test_dataset = UKBDataset(
                path=str(self.data_dir),
                ids=self.test_ids,
                transform=self.val_transforms,
                has_bin_age=self.has_bin_age,
                has_orig_eid=self.has_orig_eid
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
