"""
Geometric Operations for Mini Photoshop.
Includes Translation, Rotation (90, 180, 270, arbitrary angle),
Flipping (horizontal, vertical), and Scaling/Zooming (Nearest Neighbor & Bilinear) as taught in P6.
"""

import math
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from .core import ImageMatrix


def translate(img: ImageMatrix, dx: int, dy: int, fill_value: int = 0) -> ImageMatrix:
    """
    Translates image coordinates:
    x' = x + dx
    y' = y + dy
    """
    arr = img.array
    h, w = arr.shape[:2]
    res = np.full_like(arr, fill_value)
    
    # Calculate source and destination bounding slices
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)

    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)

    if (src_x_end > src_x_start) and (src_y_end > src_y_start):
        res[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[src_y_start:src_y_end, src_x_start:src_x_end]

    return ImageMatrix(res, color_mode=img.color_mode)


def flip_horizontal(img: ImageMatrix) -> ImageMatrix:
    """
    Pencerminan horizontal: x' = width - 1 - x
    """
    arr = np.fliplr(img.array)
    return ImageMatrix(arr, color_mode=img.color_mode)


def flip_vertical(img: ImageMatrix) -> ImageMatrix:
    """
    Pencerminan vertikal: y' = height - 1 - y
    """
    arr = np.flipud(img.array)
    return ImageMatrix(arr, color_mode=img.color_mode)


def rotate_orthogonal(img: ImageMatrix, angle: int) -> ImageMatrix:
    """
    Fast orthogonal rotations: 90, 180, 270 degrees clockwise.
    """
    angle = angle % 360
    arr = img.array
    if angle == 90:
        arr = np.rot90(arr, k=3)  # Clockwise 90
    elif angle == 180:
        arr = np.rot90(arr, k=2)  # 180
    elif angle == 270:
        arr = np.rot90(arr, k=1)  # Clockwise 270
    return ImageMatrix(arr, color_mode=img.color_mode)


def rotate_arbitrary(
    img: ImageMatrix,
    angle_deg: float,
    interpolation: str = "bilinear",
    auto_expand: bool = True,
    fill_value: int = 0
) -> ImageMatrix:
    """
    Rotates image by arbitrary angle (in degrees clockwise) around the center.
    Supports 'nearest' and 'bilinear' interpolation.
    """
    pil_mode = 'L' if img.is_grayscale else ('1' if img.is_binary else 'RGB')
    if img.is_grayscale:
        pil_img = Image.fromarray(img.to_grayscale_array(), mode='L')
    elif img.is_binary:
        pil_img = Image.fromarray(img.to_grayscale_array(), mode='L')
    else:
        pil_img = Image.fromarray(img.to_rgb(), mode='RGB')

    resample = Image.Resampling.NEAREST if interpolation == "nearest" else Image.Resampling.BILINEAR
    # PIL rotate is counter-clockwise, so pass -angle_deg for clockwise
    rotated_pil = pil_img.rotate(-angle_deg, resample=resample, expand=auto_expand, fillcolor=fill_value)
    arr = np.array(rotated_pil)
    return ImageMatrix(arr, color_mode=img.color_mode)


def zoom_scale(
    img: ImageMatrix,
    factor_x: float,
    factor_y: Optional[float] = None,
    interpolation: str = "bilinear"
) -> ImageMatrix:
    """
    Scales / Zooms image by factor_x and factor_y (default factor_y = factor_x).
    Supports Nearest Neighbor and Bilinear interpolation.
    """
    if factor_y is None:
        factor_y = factor_x

    h, w = img.array.shape[:2]
    new_w = max(1, int(round(w * factor_x)))
    new_h = max(1, int(round(h * factor_y)))

    return resize_exact(img, new_w, new_h, interpolation=interpolation)


def resize_exact(
    img: ImageMatrix,
    new_width: int,
    new_height: int,
    interpolation: str = "bilinear"
) -> ImageMatrix:
    """
    Resizes image to exact (new_width, new_height).
    """
    if img.is_grayscale:
        pil_img = Image.fromarray(img.to_grayscale_array(), mode='L')
    elif img.is_binary:
        pil_img = Image.fromarray(img.to_grayscale_array(), mode='L')
    else:
        pil_img = Image.fromarray(img.to_rgb(), mode='RGB')

    resample = Image.Resampling.NEAREST if interpolation == "nearest" else Image.Resampling.BILINEAR
    resized_pil = pil_img.resize((new_width, new_height), resample=resample)
    arr = np.array(resized_pil)
    return ImageMatrix(arr, color_mode=img.color_mode)


def crop(img: ImageMatrix, x: int, y: int, width: int, height: int) -> ImageMatrix:
    """
    Crops rectangular subregion of the image.
    """
    h, w = img.array.shape[:2]
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x + width))
    y2 = max(0, min(h, y + height))

    cropped_arr = img.array[y1:y2, x1:x2]
    return ImageMatrix(cropped_arr, color_mode=img.color_mode)
