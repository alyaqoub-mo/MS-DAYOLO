"""
MS-DAYOLO: Multiscale Domain Adaptive YOLO for Cross-Domain Object Detection

PyTorch implementation with YOLOv8 base architecture.

Original paper:
    Hnewa, M., & Radha, H. (2021). Multiscale Domain Adaptive YOLO For
    Cross-Domain Object Detection. 2021 IEEE International Conference on
    Image Processing (ICIP), 3323-3327.
"""

__version__ = "1.0.0"
__author__ = "Mazin Hnewa, PyTorch Migration Team"
__license__ = "MIT"

from ms_dayolo import models, data, losses, training, utils, engine

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "data",
    "losses",
    "training",
    "utils",
    "engine",
]
