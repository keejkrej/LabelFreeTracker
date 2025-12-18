"""CLI interface for LabelFree using Typer."""

import logging
from pathlib import Path
from typing import Optional

import torch
import typer

from labelfree.train import train
from labelfree.predict import predict

# Setup logger
logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

def setup_logging(info: bool = False, debug: bool = False) -> None:
    """Configure logging level based on verbosity flags."""
    if debug:
        level = logging.DEBUG
    elif info:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)
    root_logger = logging.getLogger("labelfree")
    root_logger.setLevel(level)

    logger.info(f"Logging level set to {logging.getLevelName(level)}")
    
    if debug:
        logger.debug("Debug logging enabled")


app = typer.Typer(
    name="labelfree",
    help="LabelFree - Predict fluorescence from transmitted light microscopy.",
    no_args_is_help=True,
)


@app.command("train")
def train_command(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Path to microscopy images folder",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("training_output"),
        "--output-dir",
        "-o",
        help="Output directory for checkpoints and logs",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
        min=1,
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        "-b",
        help="Training batch size",
        min=1,
    ),
    patch_size: str = typer.Option(
        "16,256,256",
        "--patch-size",
        "-p",
        help="Patch size as Z,Y,X (ignored if --2d or --3d is set)",
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        "-l",
        help="Learning rate",
        min=1e-6,
        max=1.0,
    ),
    validation_split: float = typer.Option(
        0.2,
        "--validation-split",
        help="Fraction of data for validation",
        min=0.0,
        max=1.0,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        help="DataLoader workers (0 for Windows)",
        min=0,
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to checkpoint to resume from",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    mode_2d: bool = typer.Option(
        False,
        "--2d",
        help="2D mode: auto-set patch size to 1,256,256, output to model_2d.pt",
    ),
    mode_3d: bool = typer.Option(
        False,
        "--3d",
        help="3D mode: auto-set patch size to 16,256,256, output to model_3d.pt",
    ),
    info: bool = typer.Option(
        False,
        "--info",
        help="Enable info-level logging",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug-level logging",
    ),
) -> None:
    """Train a LabelFree UNet model."""
    setup_logging(info=info, debug=debug)
    
    logger.info("Starting LabelFree training")
    logger.debug(f"Parameters: data_dir={data_dir}, output_dir={output_dir}")
    logger.debug(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    try:
        # Handle --2d / --3d mode flags
        if mode_2d and mode_3d:
            raise ValueError("Cannot use both --2d and --3d flags")
        
        if mode_2d:
            patch_size_tuple = (1, 256, 256)
            mode = "2d"
            logger.info("2D mode: using patch size (1, 256, 256)")
        elif mode_3d:
            patch_size_tuple = (16, 256, 256)
            mode = "3d"
            logger.info("3D mode: using patch size (16, 256, 256)")
        else:
            patch_size_tuple = tuple(map(int, patch_size.split(",")))
            if len(patch_size_tuple) != 3:
                raise ValueError("Patch size must have 3 dimensions: Z,Y,X")
            mode = None  # custom mode
        
        logger.debug(f"Patch size: {patch_size_tuple}, mode: {mode}")

        train(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            patch_size=patch_size_tuple,
            lr=lr,
            validation_split=validation_split,
            seed=seed,
            num_workers=num_workers,
            resume=resume,
            verbose=info or debug,
            mode=mode,  # Pass mode for model filename
        )
        
        typer.echo(f"✓ Training complete. Model saved to {output_dir}", color=typer.colors.GREEN)
        logger.info("Training completed successfully")
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        typer.secho(f"Error: {e}", color=typer.colors.RED, bold=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        typer.secho(f"Error during training: {e}", color=typer.colors.RED, bold=True)
        raise typer.Exit(1)


@app.command("predict")
def predict_command(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input TIFF image",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model (model_2d.pt or model_3d.pt)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to save output TIFF image",
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu",
        help="Force CPU inference",
    ),
    mode_2d: bool = typer.Option(
        False,
        "--2d",
        help="Require 2D model (error if model is 3D)",
    ),
    mode_3d: bool = typer.Option(
        False,
        "--3d",
        help="Require 3D model (error if model is 2D)",
    ),
    info: bool = typer.Option(
        False,
        "--info",
        help="Enable info-level logging",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug-level logging",
    ),
) -> None:
    """Run inference on microscopy images."""
    setup_logging(info=info, debug=debug)
    
    logger.info("Starting LabelFree prediction")
    logger.debug(f"Input: {input_path}")
    logger.debug(f"Model: {model_path}")
    logger.debug(f"Output: {output_path}")
    
    try:
        if mode_2d and mode_3d:
            raise ValueError("Cannot use both --2d and --3d flags")
        
        device = torch.device("cpu") if cpu else None
        logger.debug(f"Using device: {device or 'auto-detected'}")
        
        predict(
            input_path=input_path,
            model_path=model_path,
            output_path=output_path,
            device=device,
            verbose=info or debug,
            mode_2d=mode_2d,
            mode_3d=mode_3d,
        )
        
        typer.echo(f"✓ Prediction complete. Output saved to {output_path}", color=typer.colors.GREEN)
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        typer.secho(f"Error during prediction: {e}", color=typer.colors.RED, bold=True)
        raise typer.Exit(1)


@app.command("version")
def version() -> None:
    """Show the version."""
    from labelfree import __version__
    typer.echo(f"labelfree v{__version__}")


def main() -> None:
    """Entry point for CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.secho("\nOperation cancelled by user", color=typer.colors.RED)
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
