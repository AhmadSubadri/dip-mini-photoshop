"""
Arithmetic Operations on Images for Mini Photoshop.
Includes addition, subtraction, multiplication, division, alpha blending,
and scalar arithmetic as taught in P6.
"""

from typing import Tuple
import numpy as np
from PIL import Image
from .core import ImageMatrix


def match_dimensions(img1: ImageMatrix, img2: ImageMatrix) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Matches the spatial dimensions and channel counts of two images.
    Returns (arr1, arr2, resulting_color_mode).
    """
    # 1. Match Color Channels
    if img1.channels == img2.channels:
        arr1 = img1.array.copy()
        arr2 = img2.array.copy()
        mode = img1.color_mode
    elif img1.channels == 3 or img2.channels == 3:
        arr1 = img1.to_rgb()
        arr2 = img2.to_rgb()
        mode = "RGB"
    else:
        arr1 = img1.to_grayscale_array()
        arr2 = img2.to_grayscale_array()
        mode = "GRAYSCALE"

    # 2. Match Dimensions (Resize img2 to match img1 if shapes differ)
    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]

    if (h1, w1) != (h2, w2):
        # Resize arr2 to match arr1 using PIL
        pil2 = Image.fromarray(arr2)
        pil2_resized = pil2.resize((w1, h1), Image.Resampling.BILINEAR)
        arr2 = np.array(pil2_resized)

    return arr1, arr2, mode


def add_images(img1: ImageMatrix, img2: ImageMatrix, mode: str = "clip") -> ImageMatrix:
    """
    Addition of two images:
    - 'clip': C(x,y) = min(255, A(x,y) + B(x,y))
    - 'average': C(x,y) = (A(x,y) + B(x,y)) / 2
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    summed = arr1.astype(np.float32) + arr2.astype(np.float32)
    
    if mode == "average":
        res = summed / 2.0
    else:
        res = summed
        
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)


def subtract_images(img1: ImageMatrix, img2: ImageMatrix, absolute: bool = True) -> ImageMatrix:
    """
    Subtraction of two images:
    - absolute=True: C(x,y) = |A(x,y) - B(x,y)| (Image difference)
    - absolute=False: C(x,y) = max(0, A(x,y) - B(x,y)) (With clipping)
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    diff = arr1.astype(np.float32) - arr2.astype(np.float32)
    
    if absolute:
        res = np.abs(diff)
    else:
        res = diff
        
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)


def multiply_images(img1: ImageMatrix, img2: ImageMatrix) -> ImageMatrix:
    """
    Multiplication of two images / Masking:
    C(x,y) = (A(x,y) * B(x,y)) / 255.0
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    res = (arr1.astype(np.float32) * arr2.astype(np.float32)) / 255.0
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)


def divide_images(img1: ImageMatrix, img2: ImageMatrix) -> ImageMatrix:
    """
    Division of two images:
    C(x,y) = (A(x,y) / (B(x,y) + 1e-5)) * 255.0
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    # Avoid zero division
    res = (arr1.astype(np.float32) / (arr2.astype(np.float32) + 1e-5)) * 255.0
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)


def alpha_blend(img1: ImageMatrix, img2: ImageMatrix, alpha: float = 0.5) -> ImageMatrix:
    """
    Alpha blending of two images:
    C(x,y) = alpha * A(x,y) + (1 - alpha) * B(x,y)
    """
    arr1, arr2, color_mode = match_dimensions(img1, img2)
    alpha = max(0.0, min(1.0, float(alpha)))
    res = alpha * arr1.astype(np.float32) + (1.0 - alpha) * arr2.astype(np.float32)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=color_mode)


def scalar_operation(img: ImageMatrix, op: str, scalar: float) -> ImageMatrix:
    """
    Applies scalar arithmetic: +, -, *, /
    """
    arr = img.array.astype(np.float32)
    if op == "+":
        res = arr + scalar
    elif op == "-":
        res = arr - scalar
    elif op == "*":
        res = arr * scalar
    elif op == "/":
        res = arr / (scalar if scalar != 0 else 1e-5)
    else:
        raise ValueError(f"Unknown scalar operator: {op}")
        
    res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=img.color_mode)
