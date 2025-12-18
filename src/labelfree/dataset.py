"""Dataset classes for microscopy image loading and patch extraction."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class MicroscopyPatchDataset(Dataset):
    """Dataset that loads paired transmitted light / fluorescence patches.

    Uses random cropping - one random patch per image pair per access.
    This provides more variety across epochs compared to fixed tiling.

    Args:
        image_pairs: List of (transmitted_light_path, fluorescence_path) tuples
        patch_size: (Z, Y, X) patch dimensions
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        image_pairs: List[Tuple[Path, Path]],
        patch_size: Tuple[int, int, int],
        augment: bool = False,
    ):
        self.image_pairs = image_pairs
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tl_path, fl_path = self.image_pairs[idx]
        pz, py, px = self.patch_size

        # Load images
        tl_image = tifffile.imread(tl_path).astype(np.float32)
        fl_image = tifffile.imread(fl_path).astype(np.float32)

        if tl_image.ndim == 2:
            tl_image = tl_image[np.newaxis, ...]
            fl_image = fl_image[np.newaxis, ...]

        z_dim, y_dim, x_dim = tl_image.shape

        # Random crop location
        z_start = random.randint(0, max(0, z_dim - pz))
        y_start = random.randint(0, max(0, y_dim - py))
        x_start = random.randint(0, max(0, x_dim - px))

        # Extract patches
        tl_patch = tl_image[
            z_start : z_start + pz, y_start : y_start + py, x_start : x_start + px
        ]
        fl_patch = fl_image[
            z_start : z_start + pz, y_start : y_start + py, x_start : x_start + px
        ]

        # Normalize transmitted light: per-image standardization
        tl_max = tl_patch.max()
        if tl_max > 0:
            tl_patch = tl_patch / tl_max
        tl_mean = tl_patch.mean()
        tl_std = tl_patch.std()
        if tl_std > 0:
            tl_patch = (tl_patch - tl_mean) / tl_std

        # Normalize fluorescence: 0-1 scaling
        fl_max = fl_patch.max()
        if fl_max > 0:
            fl_patch = fl_patch / fl_max
        fl_patch = np.clip(fl_patch, 0, 1)

        # Data augmentation
        if self.augment:
            if random.random() < 0.5:
                tl_patch = np.flip(tl_patch, axis=2).copy()
                fl_patch = np.flip(fl_patch, axis=2).copy()
            if random.random() < 0.5:
                tl_patch = np.flip(tl_patch, axis=1).copy()
                fl_patch = np.flip(fl_patch, axis=1).copy()

        # Add channel dimension: (Z, Y, X) -> (C, Z, Y, X)
        tl_patch = tl_patch[np.newaxis, ...]
        fl_patch = fl_patch[np.newaxis, ...]

        return torch.from_numpy(tl_patch), torch.from_numpy(fl_patch)


def find_image_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all paired c1/c2 TIFF files in the data directory.

    Expects directory structure:
        data_dir/
            position1/
                *c1.tif  (fluorescence - target)
                *c2.tif  (phase contrast - input)
            position2/
                ...

    Args:
        data_dir: Path to root data directory

    Returns:
        List of (phase_contrast_path, fluorescence_path) tuples
    """
    pairs = []

    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue

        c1_files = sorted(folder.glob("*c1.tif"))
        for c1_path in c1_files:
            c2_path = c1_path.parent / c1_path.name.replace("c1.tif", "c2.tif")
            if c2_path.exists():
                pairs.append((c2_path, c1_path))  # c2=phase contrast (input), c1=fluorescence (target)

    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Split image pairs into train and validation sets.

    Args:
        pairs: List of image pairs
        validation_fraction: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * validation_fraction)
    return shuffled[split_idx:], shuffled[:split_idx]
