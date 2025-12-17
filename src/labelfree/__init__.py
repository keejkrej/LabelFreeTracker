"""
LabelFree - Predict fluorescence from transmitted light microscopy.

This package provides tools for training and using deep learning models
to predict fluorescence images from transmitted light microscopy images.
"""

__version__ = "0.1.0"

from labelfree.model import LabelFreeUNet

__all__ = ["LabelFreeUNet", "__version__"]
