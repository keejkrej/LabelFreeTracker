"""Prediction/inference script for LabelFree model."""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch

from labelfree.model import LabelFreeUNet

# Setup logger
logger = logging.getLogger(__name__)

Z_DIVISIBLE_BY = 8
Z_MAX_SIZE = 32


def predict_direct(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run prediction on a single image.

    Args:
        model: Trained model
        image: Input image of shape (Z, Y, X)
        device: Torch device

    Returns:
        Predicted image of shape (Z, Y, X)
    """
    # Add batch and channel dimensions: (Z, Y, X) -> (1, 1, Z, Y, X)
    input_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Remove batch and channel: (1, 1, Z, Y, X) -> (Z, Y, X)
    return output.cpu().numpy()[0, 0]


def predict_in_parts(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    z_size: int = Z_MAX_SIZE,
) -> np.ndarray:
    """Predict on large images by splitting into parts.

    Args:
        image: Input image of shape (Z, Y, X)
        model: Trained model
        device: Torch device
        z_size: Maximum Z size per chunk

    Returns:
        Predicted image of shape (Z, Y, X)
    """
    if image.shape[0] <= z_size:
        # Pad to divisible size
        desired_z = int(math.ceil(image.shape[0] / Z_DIVISIBLE_BY) * Z_DIVISIBLE_BY)
        if desired_z == 0:
            desired_z = Z_DIVISIBLE_BY

        if image.shape[0] < desired_z:
            larger_image = np.zeros(
                (desired_z, image.shape[1], image.shape[2]), dtype=image.dtype
            )
            larger_image[: image.shape[0]] = image
            result = predict_direct(model, larger_image, device)
            return result[: image.shape[0]]

        return predict_direct(model, image, device)

    # Split into overlapping parts
    output_image = np.empty_like(image)
    z_starts = list(range(0, image.shape[0], z_size - 4))
    z_starts.reverse()

    if len(z_starts) > 0 and z_starts[0] + z_size > image.shape[0]:
        z_starts[0] = image.shape[0] - z_size

    for z_start in z_starts:
        z_end = min(z_start + z_size, image.shape[0])
        chunk = image[z_start:z_end]

        # Pad if needed
        if chunk.shape[0] < z_size:
            padded = np.zeros((z_size, chunk.shape[1], chunk.shape[2]), dtype=chunk.dtype)
            padded[: chunk.shape[0]] = chunk
            result = predict_direct(model, padded, device)
            output_image[z_start:z_end] = result[: chunk.shape[0]]
        else:
            output_image[z_start:z_end] = predict_direct(model, chunk, device)

    return output_image


def load_model(
    model_path: Path,
    device: torch.device,
    z_patch_size: Optional[int] = None,
) -> tuple[torch.nn.Module, int]:
    """Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pt) or directory with settings.json
        device: Torch device
        z_patch_size: Override Z patch size

    Returns:
        Tuple of (model, z_patch_size)
    """
    # Try to load settings.json
    if model_path.is_dir():
        settings_path = model_path / "settings.json"
        checkpoint_path = model_path / "final_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_path / "checkpoints" / "best_model.pt"
    else:
        settings_path = model_path.parent / "settings.json"
        checkpoint_path = model_path

    z_size = z_patch_size or 16
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        if "patch_size_xyz" in settings:
            z_size = settings["patch_size_xyz"][2]
            print(f"Loaded Z patch size from settings: {z_size}")

    if z_patch_size is not None:
        z_size = z_patch_size
        print(f"Using provided Z patch size: {z_size}")

    # Create and load model
    model = LabelFreeUNet(in_channels=1, out_channels=1, z_patch_size=z_size)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, z_size


def predict(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    z_patch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> None:
    """Run prediction on an input image.

    Args:
        input_path: Path to input TIFF image
        model_path: Path to model checkpoint
        output_path: Path to save output TIFF
        z_patch_size: Override Z patch size
        device: Torch device (auto-detected if None)
        verbose: Enable verbose logging
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if verbose:
        print(f"Using device: {device}")

    # Load model
    logger.debug(f"Loading model from {model_path}...")
    model, z_size = load_model(model_path, device, z_patch_size)
    logger.info(f"Loaded model with Z patch size: {z_size}")
    
    if verbose:
        print(f"Loading model from {model_path}...")

    # Load image
    logger.debug(f"Loading image from {input_path}...")
    image = tifffile.imread(input_path).astype(np.float32)

    if image.ndim == 2:
        image = image[np.newaxis, ...]
        logger.info("Reshaped 2D image to 3D")
        if verbose:
            print(f"Reshaped 2D image to 3D: {image.shape}")
    elif image.ndim != 3:
        raise ValueError("Image must be 2D or 3D")

    logger.info(f"Image shape: {image.shape}")
    
    if verbose:
        print(f"Image shape: {image.shape}")

    # Preprocess
    logger.debug("Preprocessing image...")
    image_max = image.max()
    if image_max > 0:
        logger.debug(f"Normalizing image by max value: {image_max}")
        image = image / image_max

    # Per-image standardization
    image_mean = image.mean()
    image_std = image.std()
    if image_std > 0:
        logger.debug(f"Standardizing image: mean={image_mean:.3f}, std={image_std:.3f}")
        image = (image - image_mean) / image_std

    # Predict
    logger.info("Running prediction...")
    if verbose:
        print("Running prediction...")
    output = predict_in_parts(image, model, device, z_size)

    # Post-process
    logger.debug("Post-processing output...")
    output = output * 255
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Save
    logger.info(f"Saving output to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_path,
        output,
        compression="zlib",
        compressionargs={"level": 9},
    )
    logger.info("Prediction completed successfully")
    
    if verbose:
        print(f"Saving output to {output_path}...")
        print("Done.")


if __name__ == "__main__":
    # For backward compatibility, allow running predict.py directly
    import sys
    from labelfree.cli import predict_command
    
    # Convert sys.argv to typer format
    if len(sys.argv) > 1:
        # Execute with typer if arguments provided
        from typer.main import run_typer
        run_typer(predict_command)
    else:
        # Show help if no arguments
        print("Use: labelfree predict --help\n")
        from labelfree.cli import app
        app()
