"""Training script for LabelFree model."""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from labelfree.dataset import MicroscopyPatchDataset, find_image_pairs, split_pairs
from labelfree.model import LabelFreeUNet

# Setup logger
logger = logging.getLogger(__name__)


def save_example_predictions(
    model: nn.Module,
    val_dataset: MicroscopyPatchDataset,
    output_dir: Path,
    epoch: int,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    """Save example predictions for visualization."""
    model.eval()
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    indices = random.sample(range(len(val_dataset)), min(num_examples, len(val_dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            tl_patch, fl_patch = val_dataset[idx]
            tl_patch = tl_patch.unsqueeze(0).to(device)

            prediction = model(tl_patch).cpu().numpy()[0, 0]
            prediction = (prediction * 255).clip(0, 255).astype(np.uint8)

            example_dir = examples_dir / f"example_{i + 1}"
            example_dir.mkdir(exist_ok=True)

            if epoch == 0:
                tl_np = tl_patch.cpu().numpy()[0, 0]
                tl_np = (
                    (tl_np - tl_np.min()) / (tl_np.max() - tl_np.min() + 1e-8) * 255
                ).astype(np.uint8)
                tifffile.imwrite(example_dir / "input.tif", tl_np, compression="zlib")

                fl_np = (fl_patch.numpy()[0] * 255).astype(np.uint8)
                tifffile.imwrite(
                    example_dir / "ground_truth.tif", fl_np, compression="zlib"
                )

            tifffile.imwrite(
                example_dir / f"prediction_epoch_{epoch + 1}.tif",
                prediction,
                compression="zlib",
            )

    model.train()


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 4,
    patch_size: tuple[int, int, int] = (16, 256, 256),
    lr: float = 1e-4,
    validation_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    resume: Optional[Path] = None,
    verbose: bool = False,
    mode: Optional[str] = None,
) -> None:
    """Train the LabelFree UNet model.

    Args:
        data_dir: Path to microscopy images folder
        output_dir: Output directory for checkpoints and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        patch_size: Patch size as (Z, Y, X)
        lr: Learning rate
        validation_split: Fraction of data for validation
        seed: Random seed
        num_workers: DataLoader workers
        resume: Path to checkpoint to resume from
        mode: "2d", "3d", or None for custom
        verbose: Enable verbose logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Patch size (Z, Y, X): {patch_size}")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Patch size (Z, Y, X): {patch_size}")

    # Find and split data
    logger.debug("Scanning for image pairs...")
    pairs = find_image_pairs(data_dir)
    logger.info(f"Found {len(pairs)} image pairs")
    
    if verbose:
        print(f"Found {len(pairs)} image pairs")

    if len(pairs) == 0:
        raise ValueError("No image pairs found. Check your data directory structure.")

    logger.debug(f"Splitting data with validation fraction: {validation_split}")
    train_pairs, val_pairs = split_pairs(pairs, validation_split, seed)
    logger.info(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
    
    if verbose:
        print(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")

    # Create datasets
    logger.debug("Creating datasets and dataloaders...")
    train_dataset = MicroscopyPatchDataset(train_pairs, patch_size, augment=True)
    val_dataset = MicroscopyPatchDataset(val_pairs, patch_size, augment=False)
    logger.info(f"Training patches: {len(train_dataset)}, Validation patches: {len(val_dataset)}")
    
    if verbose:
        print(f"Training patches: {len(train_dataset)}, Validation patches: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # Create model
    logger.debug("Creating UNet model...")
    model = LabelFreeUNet(in_channels=1, out_channels=1, z_patch_size=patch_size[0])
    model = model.to(device)

    # Load checkpoint if resuming
    start_epoch = 0
    if resume:
        logger.debug(f"Loading checkpoint from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from checkpoint: {resume}, epoch {start_epoch}")
        if verbose:
            print(f"Resumed from checkpoint: {resume}, epoch {start_epoch}")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logger.debug(f"Optimizer: Adam with lr={lr}")

    # Setup output directory
    logger.debug(f"Setting up output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Setup tensorboard
    logger.debug("Setting up TensorBoard...")
    writer = SummaryWriter(output_dir / "tensorboard")

    # Training loop
    best_val_loss = float("inf")
    logger.info(f"Starting training from epoch {start_epoch + 1} to {epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        logger.debug(f"Starting epoch {epoch + 1}/{epochs}")

        for batch_idx, (tl_batch, fl_batch) in enumerate(train_loader):
            tl_batch = tl_batch.to(device)
            fl_batch = fl_batch.to(device)

            optimizer.zero_grad()
            predictions = model(tl_batch)
            loss = criterion(predictions, fl_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_train_loss = train_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for tl_batch, fl_batch in val_loader:
                tl_batch = tl_batch.to(device)
                fl_batch = fl_batch.to(device)

                predictions = model(tl_batch)
                loss = criterion(predictions, fl_batch)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "z_patch_size": patch_size[0],
        }
        torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoints_dir / "best_model.pt")
            print(f"  New best model saved (val_loss: {avg_val_loss:.6f})")

        # Save example predictions
        save_example_predictions(model, val_dataset, output_dir, epoch, device)

    # Determine model filename based on mode
    if mode == "2d":
        model_filename = "model_2d.pt"
    elif mode == "3d":
        model_filename = "model_3d.pt"
    else:
        model_filename = "model.pt"

    # Save final model with z_patch_size embedded
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "z_patch_size": patch_size[0],
    }
    torch.save(final_checkpoint, output_dir / model_filename)

    writer.close()
    print(f"Training complete. Model saved to {output_dir / model_filename}")


if __name__ == "__main__":
    # For backward compatibility, allow running train.py directly
    import sys
    from labelfree.cli import train_command
    
    # Convert sys.argv to typer format
    if len(sys.argv) > 1:
        # Execute with typer if arguments provided
        from typer.main import run_typer
        run_typer(train_command)
    else:
        # Show help if no arguments
        print("Use: labelfree train --help\n")
        from labelfree.cli import app
        app()
