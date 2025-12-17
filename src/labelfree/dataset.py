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

    Args:
        image_pairs: List of (transmitted_light_path, fluorescence_path) tuples
        patch_size: (Z, Y, X) patch dimensions
        augment: Whether to apply data augmentation
        leak_empty_fraction: Fraction of empty patches to include
    """

    def __init__(
        self,
        image_pairs: List[Tuple[Path, Path]],
        patch_size: Tuple[int, int, int],
        augment: bool = False,
        leak_empty_fraction: float = 0.05,
    ):
        self.image_pairs = image_pairs
        self.patch_size = patch_size
        self.augment = augment
        self.leak_empty_fraction = leak_empty_fraction

        self.patches: List[Tuple[int, int, int, int]] = []
        self._index_patches()

    def _index_patches(self) -> None:
        """Index all valid patch locations across all image pairs."""
        pz, py, px = self.patch_size

        for pair_idx, (tl_path, fl_path) in enumerate(self.image_pairs):
            tl_image = tifffile.imread(tl_path)
            if tl_image.ndim == 2:
                tl_image = tl_image[np.newaxis, ...]

            z_dim, y_dim, x_dim = tl_image.shape

            patch_idx = 0
            for x_start in range(0, x_dim - px + 1, px):
                for y_start in range(0, y_dim - py + 1, py):
                    z_offset_max = max(0, z_dim % pz)
                    z_offset = patch_idx % (z_offset_max + 1)
                    patch_idx += 1

                    for z_start in range(z_offset, z_dim - pz + 1, pz):
                        self.patches.append((pair_idx, z_start, y_start, x_start))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, z_start, y_start, x_start = self.patches[idx]
        tl_path, fl_path = self.image_pairs[pair_idx]
        pz, py, px = self.patch_size

        # Load images
        tl_image = tifffile.imread(tl_path).astype(np.float32)
        fl_image = tifffile.imread(fl_path).astype(np.float32)

        if tl_image.ndim == 2:
            tl_image = tl_image[np.newaxis, ...]
            fl_image = fl_image[np.newaxis, ...]

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
                *c1.tif  (transmitted light)
                *c2.tif  (fluorescence)
            position2/
                ...

    Args:
        data_dir: Path to root data directory

    Returns:
        List of (transmitted_light_path, fluorescence_path) tuples
    """
    pairs = []

    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue

        c1_files = sorted(folder.glob("*c1.tif"))
        for c1_path in c1_files:
            c2_path = c1_path.parent / c1_path.name.replace("c1.tif", "c2.tif")
            if c2_path.exists():
                pairs.append((c1_path, c2_path))

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
