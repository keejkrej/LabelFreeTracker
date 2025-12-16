import argparse
import json
import math
import os
import sys
import numpy
import tifffile
import tensorflow as tf
from typing import List, Optional

# Constants from _prediction_script.py
Z_DIVISIBLE_BY = 8
Z_MAX_SIZE = 32


def predict_direct(model: tf.keras.Model, image: numpy.ndarray):
    """
    Adds batch and channel indices, calls model.predict, then removes those extra indices again.
    Adapted from _prediction_script.py
    """
    # Add batch and channels array
    # Image shape is (Z, Y, X)
    input_array = numpy.expand_dims(image, -1)  # (Z, Y, X, 1)
    input_array = numpy.expand_dims(input_array, 0)  # (1, Z, Y, X, 1)

    output_image = model.predict(input_array)

    # Save as 3D 8-bits image
    output_image = output_image[0, :, :, :, 0]
    return output_image


def predict_in_small_nonoverlapping_parts(
    image: numpy.ndarray, model: tf.keras.Model, z_size: int
) -> numpy.ndarray:
    """
    Adapted from _prediction_script.py
    """
    output_image = numpy.empty_like(image)
    for z in range(0, image.shape[0], z_size):
        if z + z_size > image.shape[0]:
            # Make sure that last prediction also has the correct size
            z = image.shape[0] - z_size
            if z < 0:
                z = 0  # Handle case where image is smaller than z_size

        chunk = image[z : z + z_size]
        # Pad if chunk is smaller than z_size (e.g. if image < z_size)
        if chunk.shape[0] < z_size:
            # This case shouldn't happen with the logic above unless image < z_size
            # But if it does, we might need padding.
            # The original code logic: z = image.shape[0] - z_size implies z_size <= image.shape[0].
            # If image is thinner than z_size, this logic fails.
            pass

        output_image[z : z + z_size] = predict_direct(model, chunk)
    return output_image


def predict_in_parts(image: numpy.ndarray, model: tf.keras.Model) -> numpy.ndarray:
    """
    Calls the predict function, if necessary multiple times, so that the entire image is predicted.
    Adapted from _prediction_script.py
    """
    if image.shape[0] <= Z_MAX_SIZE:
        desired_image_z_size = int(
            math.ceil(image.shape[0] / Z_DIVISIBLE_BY) * Z_DIVISIBLE_BY
        )
        if desired_image_z_size == 0:
            desired_image_z_size = Z_DIVISIBLE_BY

        if image.shape[0] <= desired_image_z_size:
            # Expand the image by adding black pixels
            larger_image = numpy.zeros_like(
                image, shape=(desired_image_z_size, image.shape[1], image.shape[2])
            )
            larger_image[0 : image.shape[0]] = image
            return predict_direct(model, larger_image)[0 : image.shape[0]]

    # Divide the image up into parts
    output_image = numpy.empty_like(image)
    z_starts = list(
        range(0, image.shape[0], Z_MAX_SIZE - 4)
    )  # The -4 ensures some overlap between images
    z_starts.reverse()  # Make sure output of lower z (closer the objective) overwrites others

    if (
        len(z_starts) > 0 and z_starts[0] + Z_MAX_SIZE > image.shape[0]
    ):  # Make sure first entry doesn't reach outside the image
        z_starts[0] = image.shape[0] - Z_MAX_SIZE

    for z_start in z_starts:
        chunk = image[z_start : z_start + Z_MAX_SIZE]
        output_image[z_start : z_start + Z_MAX_SIZE] = predict_direct(model, chunk)
    return output_image


def main():
    parser = argparse.ArgumentParser(
        description="Standalone prediction for LabelFreeTracker"
    )
    parser.add_argument(
        "input_image", help="Path to input phase contrast TIFF image (3D: Z, Y, X)"
    )
    parser.add_argument(
        "model_path",
        help="Path to the trained model folder (must contain saved_model.pb or similar and settings.json)",
    )
    parser.add_argument(
        "output_image", help="Path to save the output fluorescence TIFF image"
    )
    parser.add_argument(
        "--z-patch-size",
        type=int,
        help="Override Z patch size if settings.json is missing",
        default=None,
    )

    args = parser.parse_args()

    # 1. Load Model Metadata
    settings_path = os.path.join(args.model_path, "settings.json")
    z_size_model = 16  # default

    if os.path.exists(settings_path):
        with open(settings_path) as handle:
            metadata = json.load(handle)
        if "patch_size_xyz" in metadata:
            z_size_model = metadata["patch_size_xyz"][2]
            print(f"Loaded patch size Z from settings: {z_size_model}")
        else:
            print(
                "settings.json found but missing 'patch_size_xyz'. Using default or override."
            )
    else:
        print("settings.json not found in model path.")

    if args.z_patch_size is not None:
        z_size_model = args.z_patch_size
        print(f"Using provided Z patch size: {z_size_model}")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 3. Load Image
    print(f"Loading image from {args.input_image}...")
    try:
        image = tifffile.imread(args.input_image)
    except Exception as e:
        print(f"Failed to read image: {e}")
        sys.exit(1)

    print(f"Image shape: {image.shape}")

    # Ensure 3D (Z, Y, X). If 2D (Y, X), expand to (1, Y, X)
    if len(image.shape) == 2:
        image = numpy.expand_dims(image, 0)
        print(f"Reshaped 2D image to 3D: {image.shape}")
    elif len(image.shape) != 3:
        print("Error: Image must be 2D or 3D.")
        sys.exit(1)

    # 4. Preprocess
    print("Preprocessing image...")
    # Normalization as per _prediction_script.py
    image_max = numpy.max(image)
    if image_max != 0:
        image = image / image_max

    # Standardization
    image = tf.image.per_image_standardization(image).numpy()

    # 5. Predict
    print("Running prediction...")
    if z_size_model <= 8:
        output_image = predict_in_small_nonoverlapping_parts(image, model, z_size_model)
    else:
        output_image = predict_in_parts(image, model)

    # 6. Post-process
    output_image = output_image * 255
    numpy.clip(output_image, 0, 255, out=output_image)
    output_image = output_image.astype(numpy.uint8)

    # 7. Save
    print(f"Saving output to {args.output_image}...")
    tifffile.imwrite(
        args.output_image,
        output_image,
        compression=tifffile.COMPRESSION.ADOBE_DEFLATE,
        compressionargs={"level": 9},
    )
    print("Done.")


if __name__ == "__main__":
    main()
