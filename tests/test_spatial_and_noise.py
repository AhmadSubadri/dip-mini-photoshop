"""
Unit Tests for Manual Spatial Operations & Noise Operations in Mini Photoshop.
Memastikan semua operasi perhitungan manual bekerja 100% akurat tanpa library eksternal.
"""

import numpy as np
import pytest
from mini_photoshop.engine.core import ImageMatrix
from mini_photoshop.engine.spatial_ops import (
    manual_pad2d,
    convolve2d_manual,
    apply_mean_filter,
    apply_gaussian_filter,
    apply_sharpen_filter,
    apply_edge_roberts,
    apply_edge_sobel,
    apply_median_filter,
    apply_max_filter,
    apply_min_filter,
    create_mean_kernel,
    create_gaussian_kernel,
)
from mini_photoshop.engine.noise_ops import (
    add_salt_and_pepper_noise,
    add_gaussian_noise,
    add_speckle_noise,
    evaluate_restoration,
)


def test_manual_pad2d():
    """Menguji penambahan bingkai batas (padding) secara manual."""
    arr = np.array([
        [10, 20],
        [30, 40]
    ], dtype=np.uint8)

    # Replicate padding: 1 piksel di tiap sisi
    padded_rep = manual_pad2d(arr, 1, 1, 1, 1, mode="replicate")
    assert padded_rep.shape == (4, 4)
    # Piksel pojok kiri atas harus mereplikasi arr[0,0] = 10
    assert padded_rep[0, 0] == 10
    # Piksel tengah harus tetap sama
    assert padded_rep[1, 1] == 10
    assert padded_rep[2, 2] == 40

    # Zero padding
    padded_zero = manual_pad2d(arr, 1, 1, 1, 1, mode="zero")
    assert padded_zero[0, 0] == 0
    assert padded_zero[1, 1] == 10


def test_convolve2d_manual():
    """Menguji konvolusi 2D manual dengan kernel identitas dan kernel perataan."""
    channel = np.array([
        [10, 10, 10],
        [10, 50, 10],
        [10, 10, 10]
    ], dtype=np.float32)

    # 1. Identity kernel
    ident = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    out_ident = convolve2d_manual(channel, ident)
    np.testing.assert_allclose(out_ident, channel)

    # 2. Mean kernel 3x3: titik tengah bernilai (10*8 + 50) / 9 = 130 / 9 = 14.444
    mean_k = np.ones((3, 3), dtype=np.float32) / 9.0
    out_mean = convolve2d_manual(channel, mean_k)
    assert abs(out_mean[1, 1] - (130.0 / 9.0)) < 1e-4


def test_linear_filters_grayscale_and_rgb():
    """Menguji filter linier (Mean, Gaussian, Sharpening, Roberts 2x2, Sobel 3x3)."""
    # Citra Grayscale buatan
    gray_arr = np.full((20, 20), 100, dtype=np.uint8)
    gray_img = ImageMatrix(gray_arr, color_mode="GRAYSCALE")

    # Mean 2x2 dan 3x3
    res_m2 = apply_mean_filter(gray_img, kernel_size=2)
    res_m3 = apply_mean_filter(gray_img, kernel_size=3)
    assert res_m2.array.shape == (20, 20)
    assert res_m3.array.shape == (20, 20)
    assert res_m3.array[10, 10] == 100

    # Gaussian 3x3
    res_g = apply_gaussian_filter(gray_img, kernel_size=3, sigma=1.0)
    assert res_g.array[10, 10] == 100

    # Sharpening
    res_s = apply_sharpen_filter(gray_img, mode="standard")
    assert res_s.array.shape == (20, 20)

    # Edge Roberts (2x2) dan Sobel (3x3)
    # Pada citra datar bernilai seragam 100, gradien harus 0
    res_rob = apply_edge_roberts(gray_img)
    res_sob = apply_edge_sobel(gray_img)
    assert res_rob.array[10, 10] == 0
    assert res_sob.array[10, 10] == 0

    # Uji pada citra RGB
    rgb_arr = np.zeros((15, 15, 3), dtype=np.uint8)
    rgb_arr[:, :, 0] = 50
    rgb_arr[:, :, 1] = 100
    rgb_arr[:, :, 2] = 150
    rgb_img = ImageMatrix(rgb_arr, color_mode="RGB")
    res_rgb_mean = apply_mean_filter(rgb_img, kernel_size=3)
    assert res_rgb_mean.array.shape == (15, 15, 3)
    assert res_rgb_mean.array[5, 5, 1] == 100


def test_nonlinear_rank_order_filters():
    """Menguji Median, Max, dan Min Filter manual."""
    # Matriks dengan impuls salt & pepper di tengah
    test_arr = np.full((7, 7), 50, dtype=np.uint8)
    test_arr[3, 3] = 255  # Salt noise di tengah
    test_arr[2, 2] = 0    # Pepper noise

    img = ImageMatrix(test_arr, color_mode="GRAYSCALE")

    # 1. Median filter harus menghilangkan nilai 255 dan 0, mengembalikan ke 50
    med = apply_median_filter(img, kernel_size=3)
    assert med.array[3, 3] == 50
    assert med.array[2, 2] == 50

    # 2. Max filter di sekitar koordinat (3,3) harus menghasilkan 255
    mx = apply_max_filter(img, kernel_size=3)
    assert mx.array[3, 3] == 255
    assert mx.array[3, 4] == 255

    # 3. Min filter di sekitar koordinat (2,2) harus menghasilkan 0
    mn = apply_min_filter(img, kernel_size=3)
    assert mn.array[2, 2] == 0
    assert mn.array[2, 3] == 0


def test_noise_generation_and_restoration():
    """
    Menguji alur penambahan derau dan reduksinya, serta evaluasi restorasi ilmiah
    menggunakan MSE dan PSNR.
    """
    # Buat citra asli
    np.random.seed(42)
    orig_arr = np.random.randint(60, 200, size=(40, 40), dtype=np.uint8)
    orig_img = ImageMatrix(orig_arr, color_mode="GRAYSCALE")

    # 1. Tambahkan Salt and Pepper Noise
    noisy_sp = add_salt_and_pepper_noise(orig_img, amount=0.10, salt_ratio=0.5, seed=42)
    # Pastikan terdapat piksel 0 dan 255
    assert np.any(noisy_sp.array == 0)
    assert np.any(noisy_sp.array == 255)

    # 2. Reduksi derau menggunakan Median Filter
    restored_sp = apply_median_filter(noisy_sp, kernel_size=3)

    # 3. Evaluasi metrik restorasi
    stats = evaluate_restoration(orig_img, noisy_sp, restored_sp)
    # PSNR citra hasil reduksi harus LEBIH TINGGI daripada citra berderau (kualitas meningkat)
    assert stats["psnr_restored"] > stats["psnr_noisy"]
    # MSE citra hasil reduksi harus LEBIH RENDAH daripada citra berderau (error berkurang)
    assert stats["mse_restored"] < stats["mse_noisy"]
    assert stats["delta_psnr"] > 0
    assert stats["is_improved"] is True

    # 4. Tambahkan Gaussian Noise & Speckle Noise
    noisy_gauss = add_gaussian_noise(orig_img, mean=0.0, sigma=20.0, seed=42)
    assert noisy_gauss.array.shape == orig_arr.shape

    noisy_speckle = add_speckle_noise(orig_img, variance=0.05, seed=42)
    assert noisy_speckle.array.shape == orig_arr.shape
