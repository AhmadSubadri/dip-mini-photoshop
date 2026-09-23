"""
Point Operations (Aras Titik) for Mini Photoshop.
Covers all point-level transformations taught in P6:
1. Invert / Negative
2. Grayscale Conversion (Average & Luminance/NTSC)
3. Brightness Adjustment with Clipping
4. Contrast Scaling & Linear Contrast Stretching
5. Thresholding / Binarization (Manual & Otsu)
6. Gamma Correction (Power Law)
7. Bit-Depth Quantization / Posterize
8. Solarization
"""

import numpy as np
from .core import ImageMatrix


def invert(img: ImageMatrix) -> ImageMatrix:
    """
    Computes image negative: f'(x,y) = 255 - f(x,y)
    For 1-bit binary image: 0 -> 255, 255 -> 0
    """
    res = 255 - img.array
    return ImageMatrix(res, color_mode=img.color_mode)


def to_grayscale_average(img: ImageMatrix) -> ImageMatrix:
    """
    Converts RGB image to Grayscale using the Average method:
    Y = (R + G + B) / 3
    """
    if img.is_grayscale:
        return img.copy()
    
    rgb = img.to_rgb().astype(np.float32)
    gray = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
    arr = np.clip(gray, 0, 255).astype(np.uint8)
    return ImageMatrix(arr, color_mode="GRAYSCALE")


def to_grayscale_luminance(img: ImageMatrix) -> ImageMatrix:
    """
    Converts RGB image to Grayscale using the NTSC/ITU Luminance method:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    """
    if img.is_grayscale:
        return img.copy()
    
    rgb = img.to_rgb().astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    arr = np.clip(gray, 0, 255).astype(np.uint8)
    return ImageMatrix(arr, color_mode="GRAYSCALE")


def adjust_brightness(img: ImageMatrix, b: int) -> ImageMatrix:
    """
    Adjusts image brightness with clipping to [0, 255]:
    f'(x,y) = clip(f(x,y) + b, 0, 255)
    """
    arr_float = img.array.astype(np.int32) + b
    arr_res = np.clip(arr_float, 0, 255).astype(np.uint8)
    return ImageMatrix(arr_res, color_mode=img.color_mode)


def adjust_contrast(img: ImageMatrix, factor: float) -> ImageMatrix:
    """
    Adjusts contrast relative to mid-gray (128):
    f'(x,y) = clip(factor * (f(x,y) - 128) + 128, 0, 255)
    """
    arr_float = img.array.astype(np.float32)
    arr_res = (factor * (arr_float - 128.0)) + 128.0
    arr_res = np.clip(arr_res, 0, 255).astype(np.uint8)
    return ImageMatrix(arr_res, color_mode=img.color_mode)


def contrast_stretching(img: ImageMatrix, r_min: float = None, r_max: float = None) -> ImageMatrix:
    """
    Performs linear contrast stretching to full dynamic range [0, 255]:
    s = ((r - r_min) / (r_max - r_min)) * 255
    """
    arr = img.array.astype(np.float32)
    
    if img.channels == 1:
        c_min = arr.min() if r_min is None else r_min
        c_max = arr.max() if r_max is None else r_max
        if c_max > c_min:
            stretched = ((arr - c_min) / (c_max - c_min)) * 255.0
        else:
            stretched = arr
    else:
        stretched = np.zeros_like(arr)
        for c in range(img.channels):
            c_min = arr[:, :, c].min() if r_min is None else r_min
            c_max = arr[:, :, c].max() if r_max is None else r_max
            if c_max > c_min:
                stretched[:, :, c] = ((arr[:, :, c] - c_min) / (c_max - c_min)) * 255.0
            else:
                stretched[:, :, c] = arr[:, :, c]
                
    arr_res = np.clip(stretched, 0, 255).astype(np.uint8)
    return ImageMatrix(arr_res, color_mode=img.color_mode)


def threshold_manual(img: ImageMatrix, threshold_val: int) -> ImageMatrix:
    """
    Performs manual thresholding / binarization:
    g(x,y) = 255 if f(x,y) >= T else 0
    """
    gray = img.to_grayscale_array()
    binary = np.where(gray >= threshold_val, 255, 0).astype(np.uint8)
    return ImageMatrix(binary, color_mode="BINARY")


def compute_otsu_threshold(gray_arr: np.ndarray) -> int:
    """
    Computes optimal global threshold using Otsu's maximum variance between classes method.
    """
    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    total_pixels = gray_arr.size
    current_max, threshold = 0.0, 128
    sum_total = np.dot(np.arange(256), hist)
    sum_background, weight_background = 0.0, 0
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Between class variance
        var_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = t
            
    return int(threshold)


def threshold_otsu(img: ImageMatrix) -> ImageMatrix:
    """
    Performs automatic Otsu binarization.
    """
    gray = img.to_grayscale_array()
    t_opt = compute_otsu_threshold(gray)
    return threshold_manual(img, t_opt)


def gamma_correction(img: ImageMatrix, gamma: float, c: float = 1.0) -> ImageMatrix:
    """
    Power law / Gamma correction:
    s = c * (r / 255.0) ^ gamma * 255.0
    """
    arr = img.array.astype(np.float32) / 255.0
    res = c * np.power(arr, gamma) * 255.0
    arr_res = np.clip(res, 0, 255).astype(np.uint8)
    return ImageMatrix(arr_res, color_mode=img.color_mode)


def posterize(img: ImageMatrix, bits: int) -> ImageMatrix:
    """
    Bit-depth quantization (Posterization):
    Reduces number of intensity levels to 2^bits per channel (bits in 1..8).
    """
    bits = max(1, min(8, bits))
    shift = 8 - bits
    mask = (0xFF << shift) & 0xFF
    res = (img.array & mask)
    # Add mid-level quantization offset for natural visual scaling
    if bits < 8:
        offset = (1 << (shift - 1))
        res = np.clip(res.astype(np.int32) + offset, 0, 255).astype(np.uint8)
    return ImageMatrix(res, color_mode=img.color_mode)


def solarize(img: ImageMatrix, threshold: int = 128) -> ImageMatrix:
    """
    Solarization effect: inverts pixel values that are above a given threshold.
    """
    arr = img.array.copy()
    mask = arr >= threshold
    arr[mask] = 255 - arr[mask]
    return ImageMatrix(arr, color_mode=img.color_mode)
