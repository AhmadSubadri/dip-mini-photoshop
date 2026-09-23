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
    gamma_correction, posterize, solarize
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
from mini_photoshop.engine.metrics import compute_histograms, compute_statistics


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


if __name__ == "__main__":
    unittest.main()
