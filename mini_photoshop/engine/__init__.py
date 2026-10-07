"""
Mini Photoshop Engine Package
"""
from .core import ImageMatrix, ImageMetadata, DocumentState
from .spatial_ops import (
    convolve2d_manual,
    apply_kernel_to_image,
    create_mean_kernel,
    apply_mean_filter,
    create_gaussian_kernel,
    apply_gaussian_filter,
    apply_sharpen_filter,
    apply_edge_roberts,
    apply_edge_sobel,
    apply_median_filter,
    apply_max_filter,
    apply_min_filter,
)
from .noise_ops import (
    add_salt_and_pepper_noise,
    add_gaussian_noise,
    add_speckle_noise,
    evaluate_restoration,
)

__all__ = [
    "ImageMatrix",
    "ImageMetadata",
    "DocumentState",
    "convolve2d_manual",
    "apply_kernel_to_image",
    "create_mean_kernel",
    "apply_mean_filter",
    "create_gaussian_kernel",
    "apply_gaussian_filter",
    "apply_sharpen_filter",
    "apply_edge_roberts",
    "apply_edge_sobel",
    "apply_median_filter",
    "apply_max_filter",
    "apply_min_filter",
    "add_salt_and_pepper_noise",
    "add_gaussian_noise",
    "add_speckle_noise",
    "evaluate_restoration",
]
