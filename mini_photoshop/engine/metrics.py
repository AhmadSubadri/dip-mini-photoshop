"""
Image Metrics, Histograms, and Statistical Analysis for Mini Photoshop.
Computes intensity distributions and quality metrics (Sharpness, Brightness Mean, Noise Estimation from P2).
"""

from typing import Dict, Any, Tuple
import numpy as np
from .core import ImageMatrix


def compute_histograms(img: ImageMatrix) -> Dict[str, np.ndarray]:
    """
    Computes 256-bin histograms for Grayscale or R, G, B channels.
    Returns raw integer pixel-count arrays of length 256.
    """
    hists = {}
    if img.is_grayscale:
        gray = img.to_grayscale_array()
        h, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hists['Gray'] = h
    else:
        rgb = img.to_rgb()
        hists['R'], _ = np.histogram(rgb[:, :, 0].ravel(), bins=256, range=(0, 256))
        hists['G'], _ = np.histogram(rgb[:, :, 1].ravel(), bins=256, range=(0, 256))
        hists['B'], _ = np.histogram(rgb[:, :, 2].ravel(), bins=256, range=(0, 256))

        # Also compute luminance histogram
        gray = img.to_grayscale_array()
        hists['Luminance'], _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))

    return hists


def compute_normalized_histograms(img: ImageMatrix) -> Dict[str, np.ndarray]:
    """
    Computes normalized 256-bin histograms: h(i) = n(i) / N

    where:
      n(i) = number of pixels with intensity i
      N    = total number of pixels in the image (width * height)

    Each value h(i) represents the probability of a pixel having intensity i.
    All values are in the range [0.0, 1.0] and the sum across all bins
    equals 1.0 for each channel.

    Supports binary, grayscale, and RGB images.
    Returns the same channel keys as compute_histograms().
    """
    raw = compute_histograms(img)
    n = float(img.width * img.height)
    return {key: arr.astype(np.float64) / n for key, arr in raw.items()}


def compute_cumulative_histograms(img: ImageMatrix) -> Dict[str, np.ndarray]:
    """
    Computes cumulative histograms (CDF): P(i <= j) = sum of h(i) for i = 0..j

    where h(i) is the normalized histogram (probability of intensity i).

    Each bin j contains the probability that a pixel has intensity <= j.
    Properties:
      - Same 256-bin domain and channel keys as compute_normalized_histograms()
      - Values are in [0.0, 1.0]
      - Monotonically non-decreasing
      - Final bin (j=255) is approximately 1.0 for any valid image

    Supports binary, grayscale, and RGB images.
    """
    normalized = compute_normalized_histograms(img)
    return {key: np.cumsum(arr) for key, arr in normalized.items()}


def compute_statistics(img: ImageMatrix) -> Dict[str, Any]:
    """
    Computes basic statistical properties of the image matrix.
    """
    gray = img.to_grayscale_array().astype(np.float64)
    
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    var_val = float(np.var(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    median_val = float(np.median(gray))
    
    # Sharpness via Laplacian variance
    # Discrete Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    h, w = gray.shape
    if h > 2 and w > 2:
        lap = (
            gray[0:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, 0:-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))
    else:
        lap_var = 0.0

    # Fast Noise Estimation (Immerkaer's method)
    # Mask N = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    if h > 2 and w > 2:
        mask_conv = (
            gray[0:-2, 0:-2] - 2 * gray[0:-2, 1:-1] + gray[0:-2, 2:] -
            2 * gray[1:-1, 0:-2] + 4 * gray[1:-1, 1:-1] - 2 * gray[1:-1, 2:] +
            gray[2:, 0:-2] - 2 * gray[2:, 1:-1] + gray[2:, 2:]
        )
        sigma = np.sum(np.abs(mask_conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
        noise_est = float(sigma)
    else:
        noise_est = 0.0

    return {
        "mean_intensity": mean_val,
        "std_dev": std_val,
        "variance": var_val,
        "min_intensity": min_val,
        "max_intensity": max_val,
        "median_intensity": median_val,
        "dynamic_range": max_val - min_val,
        "sharpness_laplacian": lap_var,
        "noise_estimate": noise_est,
        "total_pixels": int(img.width * img.height),
        "dimensions": f"{img.width} x {img.height}",
        "channels": img.channels,
        "color_mode": img.color_mode,
        "bit_depth": img.metadata.bit_depth,
        "memory_size_kb": f"{img.array.nbytes / 1024:.2f} KB"
    }
