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


# =============================================================================
# Histogram Equalization & Specification (P10)
# =============================================================================

def _grayscale_cdf(img: ImageMatrix) -> np.ndarray:
    """
    Returns the grayscale CDF for *img* as a float64 array of shape (256,).

    For any color mode (RGB, RGBA, GRAYSCALE, BINARY) the image is first
    reduced to a single luminance channel via ImageMatrix.to_grayscale_array(),
    so the CDF always represents a single-channel grayscale distribution.

    Reuses compute_normalized_histograms() and compute_cumulative_histograms()
    from the existing infrastructure, selecting the correct channel key:
      - GRAYSCALE / BINARY  →  key "Gray"
      - RGB / RGBA          →  key "Luminance"  (grayscale equivalent already
                                computed inside compute_histograms)
    """
    if img.width * img.height == 0:
        raise ValueError("Image has zero pixels.")
    cdfs = compute_cumulative_histograms(img)
    # "Gray" is present for grayscale/binary; "Luminance" for RGB/RGBA.
    gray_cdf = cdfs.get("Gray")
    return gray_cdf if gray_cdf is not None else cdfs["Luminance"]


def histogram_equalization_lut(img: ImageMatrix) -> np.ndarray:
    """
    Computes the 256-entry histogram equalization LUT.

    Implements the lecture (slide P.32) C pseudocode exactly:
        Hist[i]   = normalized histogram (probability) of intensity i
        CDF[i]    = sum(Hist[0..i])
        HistEq[i] = floor(255 * CDF[i])

    The LUT maps every source intensity i (0-255) to a new intensity.

    Input:  ImageMatrix — any color mode; converted to grayscale internally.
    Output: np.ndarray shape (256,) dtype uint8, monotonically non-decreasing,
            values in [0, 255].
    """
    cdf = _grayscale_cdf(img)                              # reuse existing infrastructure
    lut = np.floor(255.0 * cdf).astype(np.uint8)          # floor(255 * CDF) as per lecture
    return lut


def histogram_equalization(img: ImageMatrix) -> ImageMatrix:
    """
    Applies histogram equalization to the image.

    Converts any input to grayscale, applies the equalization LUT
    (see histogram_equalization_lut), and returns a new GRAYSCALE ImageMatrix.

    Output color_mode is always "GRAYSCALE".
    """
    lut = histogram_equalization_lut(img)
    gray = img.to_grayscale_array()
    result = lut[gray]                                     # vectorized LUT lookup
    return ImageMatrix(result, color_mode="GRAYSCALE")


def histogram_specification_lut(img: ImageMatrix, target_prob: np.ndarray) -> np.ndarray:
    """
    Computes the 256-entry histogram specification (matching) LUT.

    Implements the lecture (slides P.38-39, P.48-49) three-step algorithm:

    Step 1 — Equalize source:
        source_CDF[i] = cumsum(source_normalized_histogram)[i]
        HistEq[i]     = floor(255 * source_CDF[i])

    Step 2 — Equalize target:
        target_CDF[j] = cumsum(target_prob)[j]
        SpecEq[j]     = floor(255 * target_CDF[j])

    Step 3 — Inverse nearest-CDF mapping (z = G^-1[T(r)]):
        InvHist[i] = argmin_j |HistEq[i] - SpecEq[j]|
        Ties broken by smallest j (matching lecture's strict '<' comparison).

    Input:
        img         : ImageMatrix — any color mode; converted to grayscale.
        target_prob : np.ndarray shape (256,) — Pz(z), the normalized target
                      histogram. Must satisfy:
                        - shape == (256,)
                        - all values finite
                        - all values >= 0
                        - sum approximately 1.0 (tolerance 1e-6)
                      Passing a non-normalized distribution raises ValueError.
                      The engine does NOT silently normalize.

    Output: np.ndarray shape (256,) dtype uint8
            LUT[i] = output intensity for source intensity i.

    Raises ValueError for invalid target_prob.
    """
    # --- Validate target_prob ---
    target_prob = np.asarray(target_prob, dtype=np.float64)
    if target_prob.shape != (256,):
        raise ValueError(
            f"target_prob must have shape (256,), got {target_prob.shape}"
        )
    if not np.all(np.isfinite(target_prob)):
        raise ValueError("target_prob contains NaN or Inf values.")
    if np.any(target_prob < 0.0):
        raise ValueError("target_prob contains negative values.")
    prob_sum = float(target_prob.sum())
    if prob_sum == 0.0:
        raise ValueError("target_prob is all-zero (no valid distribution).")
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"target_prob must be normalized (sum ≈ 1.0), got sum={prob_sum:.8f}. "
            "Normalize before calling this function."
        )

    # --- Step 1: equalize source (reuse existing CDF infrastructure) ---
    hist_eq = np.floor(255.0 * _grayscale_cdf(img)).astype(np.int32)

    # --- Step 2: equalize target ---
    cdf_tgt = np.cumsum(target_prob)                       # float64 (256,)
    spec_eq = np.floor(255.0 * cdf_tgt).astype(np.int32)  # int32

    # --- Step 3: inverse nearest-CDF mapping ---
    # diff[i, j] = |HistEq[i] - SpecEq[j]|  shape (256, 256) int32
    diff = np.abs(hist_eq[:, np.newaxis] - spec_eq[np.newaxis, :])
    # argmin along axis=1 returns smallest j on tie (np.argmin is left-biased)
    inv_hist = np.argmin(diff, axis=1).astype(np.uint8)    # shape (256,) uint8

    return inv_hist


def histogram_specification(img: ImageMatrix, target_prob: np.ndarray) -> ImageMatrix:
    """
    Applies histogram specification (matching) to the image.

    Converts any input to grayscale, applies the specification LUT
    (see histogram_specification_lut), and returns a new GRAYSCALE ImageMatrix.

    Output color_mode is always "GRAYSCALE".
    """
    lut = histogram_specification_lut(img, target_prob)
    gray = img.to_grayscale_array()
    result = lut[gray]
    return ImageMatrix(result, color_mode="GRAYSCALE")


# ==============================================================================
# Image Quality Assessment Metrics (MSE & PSNR Manual)
# ==============================================================================

def compute_mse(img1: ImageMatrix, img2: ImageMatrix) -> float:
    """
    Menghitung Mean Squared Error (MSE) antara dua citra secara MANUAL.
    
    Rumus Matematis:
        MSE = (1 / N) * Sum ( (I1(x, y) - I2(x, y))^2 )
    
    Nilai MSE = 0 menunjukkan kedua citra identik sempurna.
    Semakin kecil nilai MSE, semakin mirip kedua citra.
    """
    arr1 = img1.array.astype(np.float64)
    arr2 = img2.array.astype(np.float64)

    # Sesuaikan dimensi jika berbeda bentuk
    if arr1.shape != arr2.shape:
        # Jika satu grayscale dan satu RGB, ubah keduanya ke grayscale
        if arr1.ndim == 3 and arr2.ndim == 2:
            arr1 = img1.to_grayscale_array().astype(np.float64)
        elif arr1.ndim == 2 and arr2.ndim == 3:
            arr2 = img2.to_grayscale_array().astype(np.float64)

    # Hitung selisih kuadrat piksel per piksel
    diff_sq = (arr1 - arr2) ** 2
    # Rata-rata dari seluruh elemen (mean)
    mse = float(np.mean(diff_sq))
    return mse


def compute_psnr(img1: ImageMatrix, img2: ImageMatrix, max_pixel_val: float = 255.0) -> float:
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dalam satuan desibel (dB) secara MANUAL.
    
    Rumus Matematis:
        PSNR = 10 * log10( (MAX_I^2) / MSE ) = 20 * log10( MAX_I / sqrt(MSE) )
        
    Di mana MAX_I untuk citra 8-bit adalah 255.
    Jika MSE = 0 (citra identik), PSNR bernilai tak hingga (infinity / float('inf')).
    """
    mse = compute_mse(img1, img2)
    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * np.log10((max_pixel_val ** 2) / mse)
    return float(psnr)

