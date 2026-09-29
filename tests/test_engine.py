"""
Comprehensive Unit & Integration Test Suite for Mini Photoshop Engine.
Tests all P1-P6 algorithms, format parsers, and state management.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np

from mini_photoshop.engine.core import ImageMatrix, DocumentState
from mini_photoshop.engine.io_custom import (
    read_pbm, write_pbm, read_pgm, write_pgm,
    read_ppm, write_ppm, read_bmp, write_bmp,
    read_raw, write_raw, load_image_file, save_image_file
)
from mini_photoshop.engine.point_ops import (
    invert, to_grayscale_average, to_grayscale_luminance,
    adjust_brightness, adjust_contrast, contrast_stretching,
    threshold_manual, threshold_otsu, compute_otsu_threshold,
    gamma_correction, posterize, solarize,
    log_transform, inverse_log_transform, gray_level_slicing, bit_plane_slice,
)
from mini_photoshop.engine.arithmetic_ops import (
    add_images, subtract_images, multiply_images, divide_images,
    alpha_blend, scalar_operation
)
from mini_photoshop.engine.boolean_ops import (
    bitwise_not, bitwise_and, bitwise_or, bitwise_xor, mask_image
)
from mini_photoshop.engine.geometry_ops import (
    translate, flip_horizontal, flip_vertical, rotate_orthogonal,
    rotate_arbitrary, zoom_scale, resize_exact, crop
)
from mini_photoshop.engine.metrics import (
    compute_histograms, compute_normalized_histograms,
    compute_cumulative_histograms, compute_statistics
)


class TestMiniPhotoshopEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests_scratch"
        os.makedirs(cls.test_dir, exist_ok=True)
        # Create a standard 64x64 RGB test image
        cls.rgb_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        cls.rgb_arr[:, :32, 0] = 200 # Red half
        cls.rgb_arr[:32, :, 1] = 150 # Green top
        cls.rgb_arr[:, :, 2] = 80    # Blue base
        cls.img_rgb = ImageMatrix(cls.rgb_arr, color_mode="RGB")

        # Create a 64x64 grayscale gradient
        cls.gray_arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        cls.img_gray = ImageMatrix(cls.gray_arr, color_mode="GRAYSCALE")

    def test_01_core_and_undo_redo(self):
        doc = DocumentState(self.img_rgb)
        self.assertEqual(doc.current.width, 64)
        self.assertEqual(doc.current.height, 64)
        self.assertFalse(doc.can_undo())

        # Modify
        modified = invert(doc.current)
        doc.push_state("Invert", modified)
        self.assertTrue(doc.can_undo())
        self.assertEqual(doc.current.array[0, 0, 0], 255 - self.rgb_arr[0, 0, 0])

        # Undo
        doc.undo()
        self.assertEqual(doc.current.array[0, 0, 0], self.rgb_arr[0, 0, 0])
        self.assertTrue(doc.can_redo())

        # Redo
        doc.redo()
        self.assertEqual(doc.current.array[0, 0, 0], 255 - self.rgb_arr[0, 0, 0])

    def test_02_point_operations(self):
        # 1. Invert
        inv = invert(self.img_gray)
        np.testing.assert_array_equal(inv.array, 255 - self.gray_arr)

        # 2. Grayscale conversions
        gray_avg = to_grayscale_average(self.img_rgb)
        self.assertEqual(gray_avg.color_mode, "GRAYSCALE")
        self.assertEqual(gray_avg.channels, 1)

        gray_lum = to_grayscale_luminance(self.img_rgb)
        self.assertEqual(gray_lum.color_mode, "GRAYSCALE")

        # 3. Brightness
        bright = adjust_brightness(self.img_gray, 50)
        self.assertEqual(bright.array[0, 0], 50)
        self.assertEqual(bright.array[0, -1], 255) # clipped

        # 4. Contrast
        cont = adjust_contrast(self.img_gray, 1.5)
        self.assertEqual(cont.shape, self.gray_arr.shape)

        # 5. Contrast stretching
        stretched = contrast_stretching(self.img_gray)
        self.assertEqual(int(stretched.array.min()), 0)
        self.assertEqual(int(stretched.array.max()), 255)

        # 6. Thresholding
        thresh = threshold_manual(self.img_gray, 128)
        self.assertEqual(thresh.color_mode, "BINARY")
        self.assertTrue(set(np.unique(thresh.array)).issubset({0, 255}))

        otsu = threshold_otsu(self.img_gray)
        self.assertEqual(otsu.color_mode, "BINARY")

        # 7. Gamma
        gam = gamma_correction(self.img_gray, 0.5)
        self.assertEqual(gam.shape, self.gray_arr.shape)

        # 8. Posterize
        post = posterize(self.img_gray, 2)
        self.assertLessEqual(len(np.unique(post.array)), 4)

    def test_03_arithmetic_operations(self):
        img_a = ImageMatrix(np.full((32, 32), 100, dtype=np.uint8), color_mode="GRAYSCALE")
        img_b = ImageMatrix(np.full((32, 32), 50, dtype=np.uint8), color_mode="GRAYSCALE")

        # Add
        added = add_images(img_a, img_b, mode="clip")
        self.assertEqual(added.array[0, 0], 150)

        # Subtract
        sub = subtract_images(img_a, img_b)
        self.assertEqual(sub.array[0, 0], 50)

        # Multiply
        mul = multiply_images(img_a, img_b)
        expected_mul = int((100 * 50) / 255.0)
        self.assertEqual(mul.array[0, 0], expected_mul)

        # Blend
        blend = alpha_blend(img_a, img_b, alpha=0.5)
        self.assertEqual(blend.array[0, 0], 75)

    def test_04_boolean_operations(self):
        img_a = ImageMatrix(np.full((16, 16), 0b11001100, dtype=np.uint8))
        img_b = ImageMatrix(np.full((16, 16), 0b10101010, dtype=np.uint8))

        res_and = bitwise_and(img_a, img_b)
        self.assertEqual(res_and.array[0, 0], 0b10001000)

        res_or = bitwise_or(img_a, img_b)
        self.assertEqual(res_or.array[0, 0], 0b11101110)

        res_xor = bitwise_xor(img_a, img_b)
        self.assertEqual(res_xor.array[0, 0], 0b01100110)

        res_not = bitwise_not(img_a)
        self.assertEqual(res_not.array[0, 0], 255 - 0b11001100)

    def test_05_geometry_operations(self):
        # 1. Flip
        fh = flip_horizontal(self.img_gray)
        self.assertEqual(fh.array[0, 0], self.gray_arr[0, -1])

        fv = flip_vertical(self.img_gray)
        self.assertEqual(fv.array[0, 0], self.gray_arr[-1, 0])

        # 2. Orthogonal rotation
        r90 = rotate_orthogonal(self.img_gray, 90)
        self.assertEqual(r90.shape, self.gray_arr.shape)

        # 3. Arbitrary rotation
        r45 = rotate_arbitrary(self.img_gray, 45.0, auto_expand=True)
        self.assertGreater(r45.width, self.img_gray.width)

        # 4. Translation
        tr = translate(self.img_gray, 10, 5)
        self.assertEqual(tr.shape, self.gray_arr.shape)

        # 5. Zoom / Resize
        sc = resize_exact(self.img_gray, 128, 128)
        self.assertEqual(sc.width, 128)
        self.assertEqual(sc.height, 128)

    def test_06_custom_io_formats(self):
        # 1. PBM
        pbm_bin = os.path.join(self.test_dir, "test.pbm")
        write_pbm(self.img_gray, pbm_bin, binary=True)
        pbm_loaded = read_pbm(pbm_bin)
        self.assertEqual(pbm_loaded.color_mode, "BINARY")

        # 2. PGM
        pgm_bin = os.path.join(self.test_dir, "test.pgm")
        write_pgm(self.img_gray, pgm_bin, binary=True)
        pgm_loaded = read_pgm(pgm_bin)
        self.assertEqual(pgm_loaded.shape, self.gray_arr.shape)

        # 3. PPM
        ppm_bin = os.path.join(self.test_dir, "test.ppm")
        write_ppm(self.img_rgb, ppm_bin, binary=True)
        ppm_loaded = read_ppm(ppm_bin)
        self.assertEqual(ppm_loaded.shape, self.rgb_arr.shape)

        # 4. BMP (24-bit RGB)
        bmp_path = os.path.join(self.test_dir, "test_24.bmp")
        write_bmp(self.img_rgb, bmp_path)
        bmp_loaded = read_bmp(bmp_path)
        self.assertEqual(bmp_loaded.shape, self.rgb_arr.shape)
        np.testing.assert_array_equal(bmp_loaded.array, self.rgb_arr)

        # 5. BMP (8-bit Gray)
        bmp_gray_path = os.path.join(self.test_dir, "test_8.bmp")
        write_bmp(self.img_gray, bmp_gray_path)
        bmp_gray_loaded = read_bmp(bmp_gray_path)
        self.assertEqual(bmp_gray_loaded.shape, self.gray_arr.shape)
        np.testing.assert_array_equal(bmp_gray_loaded.array, self.gray_arr)

        # 6. RAW
        raw_path = os.path.join(self.test_dir, "test.raw")
        write_raw(self.img_gray, raw_path)
        raw_loaded = read_raw(raw_path, 64, 64, channels=1)
        self.assertEqual(raw_loaded.shape, (64, 64))

    def test_07_metrics_and_histograms(self):
        hists = compute_histograms(self.img_rgb)
        self.assertIn("R", hists)
        self.assertIn("G", hists)
        self.assertIn("B", hists)
        self.assertEqual(len(hists["R"]), 256)

        stats = compute_statistics(self.img_gray)
        self.assertIn("mean_intensity", stats)
        self.assertIn("sharpness_laplacian", stats)
        self.assertIn("noise_estimate", stats)

    def test_08_normalized_histograms_rgb(self):
        """Normalized histogram for an RGB image: h(i) = n(i)/N per channel."""
        norm = compute_normalized_histograms(self.img_rgb)

        # Expected keys: same as raw histogram
        self.assertIn("R", norm)
        self.assertIn("G", norm)
        self.assertIn("B", norm)
        self.assertIn("Luminance", norm)

        # Each channel histogram has 256 bins
        self.assertEqual(len(norm["R"]), 256)
        self.assertEqual(len(norm["G"]), 256)
        self.assertEqual(len(norm["B"]), 256)

        # Values must be floating-point
        self.assertEqual(norm["R"].dtype, np.float64)
        self.assertEqual(norm["G"].dtype, np.float64)
        self.assertEqual(norm["B"].dtype, np.float64)

        # All values must be in [0, 1]
        for ch in ("R", "G", "B", "Luminance"):
            self.assertGreaterEqual(float(norm[ch].min()), 0.0,
                                    f"Channel {ch}: min value < 0")
            self.assertLessEqual(float(norm[ch].max()), 1.0,
                                 f"Channel {ch}: max value > 1")

        # Sum of each channel must equal 1.0 (within floating-point tolerance)
        for ch in ("R", "G", "B"):
            self.assertAlmostEqual(float(norm[ch].sum()), 1.0, places=9,
                                   msg=f"Channel {ch}: sum != 1.0")

    def test_09_normalized_histograms_grayscale_and_binary(self):
        """Normalized histogram for grayscale and binary images."""
        # ── Grayscale ────────────────────────────────────────────────────────
        norm_gray = compute_normalized_histograms(self.img_gray)

        self.assertIn("Gray", norm_gray)
        self.assertEqual(len(norm_gray["Gray"]), 256)
        self.assertEqual(norm_gray["Gray"].dtype, np.float64)
        self.assertGreaterEqual(float(norm_gray["Gray"].min()), 0.0)
        self.assertLessEqual(float(norm_gray["Gray"].max()), 1.0)
        self.assertAlmostEqual(float(norm_gray["Gray"].sum()), 1.0, places=9)

        # ── Binary ───────────────────────────────────────────────────────────
        binary_arr = np.zeros((32, 32), dtype=np.uint8)
        binary_arr[16:, :] = 255   # bottom half white
        img_bin = ImageMatrix(binary_arr, color_mode="BINARY")

        norm_bin = compute_normalized_histograms(img_bin)

        self.assertIn("Gray", norm_bin)
        self.assertEqual(len(norm_bin["Gray"]), 256)
        self.assertEqual(norm_bin["Gray"].dtype, np.float64)
        self.assertGreaterEqual(float(norm_bin["Gray"].min()), 0.0)
        self.assertLessEqual(float(norm_bin["Gray"].max()), 1.0)
        self.assertAlmostEqual(float(norm_bin["Gray"].sum()), 1.0, places=9)

        # Exactly two non-zero bins: intensity 0 and intensity 255
        nonzero_bins = np.nonzero(norm_bin["Gray"])[0]
        self.assertEqual(len(nonzero_bins), 2)
        self.assertIn(0, nonzero_bins)
        self.assertIn(255, nonzero_bins)

        # Each half contributes 0.5 of total pixels
        self.assertAlmostEqual(float(norm_bin["Gray"][0]),   0.5, places=9)
        self.assertAlmostEqual(float(norm_bin["Gray"][255]), 0.5, places=9)

    def test_10_cumulative_histograms_rgb(self):
        """CDF for an RGB image: P(i<=j) = cumsum of h(i) per channel."""
        cdf = compute_cumulative_histograms(self.img_rgb)

        # Same keys as normalized histogram
        self.assertIn("R", cdf)
        self.assertIn("G", cdf)
        self.assertIn("B", cdf)
        self.assertIn("Luminance", cdf)

        # 256 bins per channel
        for ch in ("R", "G", "B", "Luminance"):
            self.assertEqual(len(cdf[ch]), 256)

        # Values must be float64
        for ch in ("R", "G", "B"):
            self.assertEqual(cdf[ch].dtype, np.float64)

        # All values must be in [0.0, 1.0]
        for ch in ("R", "G", "B", "Luminance"):
            self.assertGreaterEqual(float(cdf[ch].min()), 0.0,
                                    f"Channel {ch}: CDF min < 0")
            self.assertLessEqual(float(cdf[ch].max()), 1.0 + 1e-9,
                                 f"Channel {ch}: CDF max > 1")

        # Monotonically non-decreasing
        for ch in ("R", "G", "B"):
            diffs = np.diff(cdf[ch])
            self.assertTrue(np.all(diffs >= -1e-12),
                            f"Channel {ch}: CDF is not monotonically non-decreasing")

        # Final bin ≈ 1.0
        for ch in ("R", "G", "B"):
            self.assertAlmostEqual(float(cdf[ch][255]), 1.0, places=9,
                                   msg=f"Channel {ch}: CDF[255] != 1.0")

    def test_11_cumulative_histograms_grayscale_and_binary(self):
        """CDF for grayscale and binary images."""
        # ── Grayscale ────────────────────────────────────────────────────────
        cdf_gray = compute_cumulative_histograms(self.img_gray)

        self.assertIn("Gray", cdf_gray)
        self.assertEqual(len(cdf_gray["Gray"]), 256)
        self.assertEqual(cdf_gray["Gray"].dtype, np.float64)
        self.assertGreaterEqual(float(cdf_gray["Gray"].min()), 0.0)
        self.assertLessEqual(float(cdf_gray["Gray"].max()), 1.0 + 1e-9)
        # Monotonically non-decreasing
        self.assertTrue(np.all(np.diff(cdf_gray["Gray"]) >= -1e-12))
        # Final bin ≈ 1.0
        self.assertAlmostEqual(float(cdf_gray["Gray"][255]), 1.0, places=9)

        # ── Binary (50/50 split: top half=0, bottom half=255) ────────────────
        binary_arr = np.zeros((32, 32), dtype=np.uint8)
        binary_arr[16:, :] = 255   # bottom half white
        img_bin = ImageMatrix(binary_arr, color_mode="BINARY")

        cdf_bin = compute_cumulative_histograms(img_bin)

        self.assertIn("Gray", cdf_bin)
        self.assertEqual(len(cdf_bin["Gray"]), 256)
        self.assertEqual(cdf_bin["Gray"].dtype, np.float64)
        self.assertGreaterEqual(float(cdf_bin["Gray"].min()), 0.0)
        self.assertLessEqual(float(cdf_bin["Gray"].max()), 1.0 + 1e-9)
        # Monotonically non-decreasing
        self.assertTrue(np.all(np.diff(cdf_bin["Gray"]) >= -1e-12))
        # Final bin ≈ 1.0
        self.assertAlmostEqual(float(cdf_bin["Gray"][255]), 1.0, places=9)

        # CDF[0] = P(intensity <= 0) = 0.5  (half pixels are 0)
        self.assertAlmostEqual(float(cdf_bin["Gray"][0]), 0.5, places=9)

        # CDF[1..254] = 0.5 (flat plateau — no pixels in range 1..254)
        plateau = cdf_bin["Gray"][1:255]
        for j, val in enumerate(plateau, start=1):
            self.assertAlmostEqual(float(val), 0.5, places=9,
                                   msg=f"CDF[{j}] should be 0.5 (plateau), got {val}")

    def test_12_image_enhancement_operations(self):
        """Validates log transform, inverse log, gray-level slicing, and bit-plane slicing."""

        # ── Helper images ────────────────────────────────────────────────────
        # Grayscale ramp: pixel at column x has intensity x (0..255 across 256 cols)
        ramp = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
        img_ramp = ImageMatrix(ramp, color_mode="GRAYSCALE")

        # Single known pixel image
        def px(v):
            arr = np.full((4, 4), v, dtype=np.uint8)
            return ImageMatrix(arr, color_mode="GRAYSCALE")

        # ── Log Transformation ───────────────────────────────────────────────
        res_log = log_transform(img_ramp, c=1.0)

        # 0 maps to 0
        self.assertEqual(int(res_log.array[0, 0]), 0)

        # output within [0, 255]
        self.assertGreaterEqual(int(res_log.array.min()), 0)
        self.assertLessEqual(int(res_log.array.max()), 255)

        # monotonicity: result must be non-decreasing along the ramp
        row = res_log.array[0].astype(np.int32)
        diffs = np.diff(row)
        self.assertTrue(np.all(diffs >= 0), "log_transform output is not monotonically non-decreasing")

        # c <= 0 must be rejected
        with self.assertRaises(ValueError):
            log_transform(img_ramp, c=0.0)
        with self.assertRaises(ValueError):
            log_transform(img_ramp, c=-1.0)

        # ── Inverse Log Transformation ───────────────────────────────────────
        res_invlog = inverse_log_transform(img_ramp)

        # 0 maps to 0
        self.assertEqual(int(res_invlog.array[0, 0]), 0)

        # 255 maps to 255
        self.assertEqual(int(res_invlog.array[0, 255]), 255)

        # output within [0, 255]
        self.assertGreaterEqual(int(res_invlog.array.min()), 0)
        self.assertLessEqual(int(res_invlog.array.max()), 255)

        # monotonicity: non-decreasing along the ramp
        row_inv = res_invlog.array[0].astype(np.int32)
        diffs_inv = np.diff(row_inv)
        self.assertTrue(np.all(diffs_inv >= 0),
                        "inverse_log_transform output is not monotonically non-decreasing")

        # ── Gray-Level Slicing ───────────────────────────────────────────────
        img_flat = ImageMatrix(
            np.array([[50, 100, 150, 200]], dtype=np.uint8), color_mode="GRAYSCALE"
        )

        # Suppress background (preserve_background=False):
        # lower=99, upper=201 → pixels 100, 150, and 200 are inside (>99 and <201)
        s = gray_level_slicing(img_flat, lower=99, upper=201, preserve_background=False)
        self.assertEqual(int(s.array[0, 0]), 0)    # 50  → outside (50 < 99)  → 0
        self.assertEqual(int(s.array[0, 1]), 255)  # 100 → inside  (99 < 100 < 201) → 255
        self.assertEqual(int(s.array[0, 2]), 255)  # 150 → inside  (99 < 150 < 201) → 255
        self.assertEqual(int(s.array[0, 3]), 255)  # 200 → inside  (99 < 200 < 201) → 255

        # Preserve background (preserve_background=True):
        # lower=99, upper=151 → only pixel 100 and 150 are inside strictly
        p = gray_level_slicing(img_flat, lower=99, upper=151, preserve_background=True)
        self.assertEqual(int(p.array[0, 0]), 50)   # 50  → outside → original
        self.assertEqual(int(p.array[0, 1]), 255)  # 100 → inside  → 255
        self.assertEqual(int(p.array[0, 2]), 255)  # 150 → inside  → 255
        self.assertEqual(int(p.array[0, 3]), 200)  # 200 → outside → original

        # Strict bounds: values exactly equal to bounds are NOT highlighted
        edge = gray_level_slicing(img_flat, lower=100, upper=200, preserve_background=False)
        self.assertEqual(int(edge.array[0, 1]), 0)   # 100 == lower → NOT inside (strict >)
        self.assertEqual(int(edge.array[0, 3]), 0)   # 200 == upper → NOT inside (strict <)
        self.assertEqual(int(edge.array[0, 2]), 255) # 150 → inside → 255

        # Invalid bounds rejected
        with self.assertRaises(ValueError):
            gray_level_slicing(img_flat, lower=200, upper=100)  # lower >= upper
        with self.assertRaises(ValueError):
            gray_level_slicing(img_flat, lower=0, upper=0)      # lower == upper

        # ── Bit-Plane Slicing ────────────────────────────────────────────────
        img_128 = px(128)   # 128 = 0b10000000

        # Bit 7 (MSB) of 128: bit is set → 255
        b7 = bit_plane_slice(img_128, bit=7)
        self.assertEqual(int(b7.array[0, 0]), 255)

        # Bit 0 (LSB) of 128: bit is not set → 0
        b0 = bit_plane_slice(img_128, bit=0)
        self.assertEqual(int(b0.array[0, 0]), 0)

        # Output contains only 0 and 255
        vals = set(np.unique(b7.array))
        self.assertTrue(vals.issubset({0, 255}), f"bit_plane_slice returned unexpected values: {vals}")

        # Full ramp: output for any valid bit must contain only 0 and 255
        for b in range(8):
            plane = bit_plane_slice(img_ramp, bit=b)
            unique = set(np.unique(plane.array))
            self.assertTrue(unique.issubset({0, 255}),
                            f"bit_plane_slice(bit={b}) produced non-binary values: {unique}")

        # Invalid bit rejected
        with self.assertRaises(ValueError):
            bit_plane_slice(img_ramp, bit=8)
        with self.assertRaises(ValueError):
            bit_plane_slice(img_ramp, bit=-1)


if __name__ == "__main__":
    unittest.main()
