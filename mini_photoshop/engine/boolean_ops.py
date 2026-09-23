"""
Boolean / Logic Operations for Mini Photoshop.
Includes Bitwise and Binary logic operations: AND, OR, NOT, XOR, and Masking as taught in P6.
"""

from typing import Tuple
import numpy as np
from .core import ImageMatrix
from .arithmetic_ops import match_dimensions


def bitwise_not(img: ImageMatrix) -> ImageMatrix:
    """
    Computes bitwise NOT on the image:
    f'(x,y) = ~f(x,y) (ekivalen 255 - f(x,y))
    """
    res = np.bitwise_not(img.array)
    return ImageMatrix(res, color_mode=img.color_mode)


def bitwise_and(img1: ImageMatrix, img2: ImageMatrix) -> ImageMatrix:
    """
    Computes bitwise AND between two images:
    C(x,y) = A(x,y) & B(x,y)
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    res = np.bitwise_and(arr1, arr2)
    return ImageMatrix(res, color_mode=color_mode)


def bitwise_or(img1: ImageMatrix, img2: ImageMatrix) -> ImageMatrix:
    """
    Computes bitwise OR between two images:
    C(x,y) = A(x,y) | B(x,y)
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    res = np.bitwise_or(arr1, arr2)
    return ImageMatrix(res, color_mode=color_mode)


def bitwise_xor(img1: ImageMatrix, img2: ImageMatrix) -> ImageMatrix:
    """
    Computes bitwise XOR between two images:
    C(x,y) = A(x,y) ^ B(x,y)
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    res = np.bitwise_xor(arr1, arr2)
    return ImageMatrix(res, color_mode=color_mode)


def mask_image(img: ImageMatrix, mask: ImageMatrix) -> ImageMatrix:
    """
    Applies binary/grayscale mask to an image.
    """
    arr1, arr2, color_mode = match_dimensions(img, mask)
    if arr2.ndim == 3 and arr2.shape[2] == 3:
        arr2 = (0.299 * arr2[:, :, 0] + 0.587 * arr2[:, :, 1] + 0.114 * arr2[:, :, 2]).astype(np.uint8)
        
    mask_normalized = (arr2.astype(np.float32) / 255.0)
    if arr1.ndim == 3:
        mask_normalized = mask_normalized[:, :, np.newaxis]
        
    res = np.clip(arr1.astype(np.float32) * mask_normalized, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)
