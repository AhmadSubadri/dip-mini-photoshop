"""
Adjustment Dialogs with Real-Time Live Preview for Mini Photoshop.
Includes Brightness & Contrast, Thresholding (Manual & Otsu), Gamma, Posterization,
and Image Enhancement operations: Log Transform, Inverse Log Transform,
Gray-Level Slicing, and Bit-Plane Slicing.
"""

from typing import Callable, Optional
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QComboBox,
    QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from ...engine.core import ImageMatrix
from ...engine.point_ops import (
    adjust_brightness, adjust_contrast, threshold_manual,
    threshold_otsu, compute_otsu_threshold, gamma_correction, posterize,
    log_transform, inverse_log_transform, gray_level_slicing, bit_plane_slice,
)
from ...engine.metrics import histogram_specification
from ...engine.io_custom import load_image_file


class BaseLivePreviewDialog(QDialog):
    """Base class for live adjustment modals with preview signal."""
    previewUpdated = pyqtSignal(object)  # Emits ImageMatrix

    def __init__(self, original_img: ImageMatrix, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(360)
        self.original_img = original_img
        self.result_img = original_img.copy()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(12)

        # Content container
        self.content_group = QGroupBox("Parameters")
        self.content_layout = QVBoxLayout(self.content_group)
        self.content_layout.setSpacing(10)
        self.main_layout.addWidget(self.content_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("Apply")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        self.main_layout.addLayout(btn_layout)

    def reject(self):
        # Revert preview
        self.previewUpdated.emit(self.original_img)
        super().reject()


class BrightnessContrastDialog(BaseLivePreviewDialog):
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Brightness & Contrast", parent)

        # 1. Brightness
        b_box = QVBoxLayout()
        b_header = QHBoxLayout()
        b_header.addWidget(QLabel("Brightness:"))
        self.b_spin = QSpinBox()
        self.b_spin.setRange(-255, 255)
        self.b_spin.setValue(0)
        b_header.addWidget(self.b_spin)
        b_box.addLayout(b_header)

        self.b_slider = QSlider(Qt.Orientation.Horizontal)
        self.b_slider.setRange(-255, 255)
        self.b_slider.setValue(0)
        b_box.addWidget(self.b_slider)
        self.content_layout.addLayout(b_box)

        # 2. Contrast
        c_box = QVBoxLayout()
        c_header = QHBoxLayout()
        c_header.addWidget(QLabel("Contrast:"))
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.0, 5.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.setValue(1.0)
        c_header.addWidget(self.c_spin)
        c_box.addLayout(c_header)

        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(0, 500)  # 0.0 to 5.0
        self.c_slider.setValue(100)
        c_box.addWidget(self.c_slider)
        self.content_layout.addLayout(c_box)

        # Connect signals
        self.b_slider.valueChanged.connect(self.b_spin.setValue)
        self.b_spin.valueChanged.connect(self.b_slider.setValue)
        self.b_slider.valueChanged.connect(self._recalculate)

        self.c_slider.valueChanged.connect(lambda v: self.c_spin.setValue(v / 100.0))
        self.c_spin.valueChanged.connect(lambda v: self.c_slider.setValue(int(v * 100)))
        self.c_slider.valueChanged.connect(self._recalculate)

    def _recalculate(self):
        b = self.b_slider.value()
        c = self.c_slider.value() / 100.0

        res = self.original_img
        if b != 0:
            res = adjust_brightness(res, b)
        if c != 1.0:
            res = adjust_contrast(res, c)

        self.result_img = res
        self.previewUpdated.emit(res)


class ThresholdDialog(BaseLivePreviewDialog):
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Threshold / Binarization", parent)

        # Calculate initial Otsu threshold
        gray_arr = original_img.to_grayscale_array()
        self.otsu_t = compute_otsu_threshold(gray_arr)

        t_box = QVBoxLayout()
        t_header = QHBoxLayout()
        t_header.addWidget(QLabel("Threshold Level:"))
        self.t_spin = QSpinBox()
        self.t_spin.setRange(0, 255)
        self.t_spin.setValue(self.otsu_t)
        t_header.addWidget(self.t_spin)
        t_box.addLayout(t_header)

        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(0, 255)
        self.t_slider.setValue(self.otsu_t)
        t_box.addWidget(self.t_slider)
        self.content_layout.addLayout(t_box)

        # Otsu Quick Button
        btn_otsu = QPushButton(f"Auto Otsu ({self.otsu_t})")
        btn_otsu.clicked.connect(lambda: self.t_slider.setValue(self.otsu_t))
        self.content_layout.addWidget(btn_otsu)

        # Connect
        self.t_slider.valueChanged.connect(self.t_spin.setValue)
        self.t_spin.valueChanged.connect(self.t_slider.setValue)
        self.t_slider.valueChanged.connect(self._recalculate)

        # Trigger initial calculation
        self._recalculate()

    def _recalculate(self):
        t = self.t_slider.value()
        res = threshold_manual(self.original_img, t)
        self.result_img = res
        self.previewUpdated.emit(res)


class GammaDialog(BaseLivePreviewDialog):
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Gamma Correction (Power Law)", parent)

        g_box = QVBoxLayout()
        g_header = QHBoxLayout()
        g_header.addWidget(QLabel("Gamma:"))
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(0.05, 6.0)
        self.g_spin.setSingleStep(0.05)
        self.g_spin.setValue(1.0)
        g_header.addWidget(self.g_spin)
        g_box.addLayout(g_header)

        self.g_slider = QSlider(Qt.Orientation.Horizontal)
        self.g_slider.setRange(5, 600)  # 0.05 to 6.0
        self.g_slider.setValue(100)
        g_box.addWidget(self.g_slider)
        self.content_layout.addLayout(g_box)

        self.g_slider.valueChanged.connect(lambda v: self.g_spin.setValue(v / 100.0))
        self.g_spin.valueChanged.connect(lambda v: self.g_slider.setValue(int(v * 100)))
        self.g_slider.valueChanged.connect(self._recalculate)

    def _recalculate(self):
        gamma = self.g_slider.value() / 100.0
        res = gamma_correction(self.original_img, gamma)
        self.result_img = res
        self.previewUpdated.emit(res)


class PosterizeDialog(BaseLivePreviewDialog):
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Bit-Depth Posterization", parent)

        p_box = QVBoxLayout()
        p_header = QHBoxLayout()
        p_header.addWidget(QLabel("Bits per Channel (1 - 8):"))
        self.p_spin = QSpinBox()
        self.p_spin.setRange(1, 8)
        self.p_spin.setValue(4)
        p_header.addWidget(self.p_spin)
        p_box.addLayout(p_header)

        self.p_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_slider.setRange(1, 8)
        self.p_slider.setValue(4)
        p_box.addWidget(self.p_slider)
        self.content_layout.addLayout(p_box)

        self.p_slider.valueChanged.connect(self.p_spin.setValue)
        self.p_spin.valueChanged.connect(self.p_slider.setValue)
        self.p_slider.valueChanged.connect(self._recalculate)

        self._recalculate()

    def _recalculate(self):
        bits = self.p_slider.value()
        res = posterize(self.original_img, bits)
        self.result_img = res
        self.previewUpdated.emit(res)


# =============================================================================
# Image Enhancement Dialogs (P9)
# =============================================================================

class LogTransformDialog(BaseLivePreviewDialog):
    """
    Log Transformation: s = c * log(1 + r)
    Default c ≈ 46 so the full input range maps close to [0, 255].
    """
    # c_auto = 255 / ln(256) ≈ 45.99
    _C_AUTO = 255.0 / float(__import__("math").log(256))

    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Log Transformation  [s = c · ln(1 + r)]", parent)

        row = QHBoxLayout()
        row.addWidget(QLabel("c (scale):"))
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.1, 255.0)
        self.c_spin.setSingleStep(0.5)
        self.c_spin.setDecimals(2)
        self.c_spin.setValue(round(self._C_AUTO, 2))
        row.addWidget(self.c_spin)
        self.content_layout.addLayout(row)

        # Slider maps 0.1–255.0 in steps of 0.1  →  int range 1..2550
        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(1, 2550)
        self.c_slider.setValue(int(round(self._C_AUTO * 10)))
        self.content_layout.addWidget(self.c_slider)

        note = QLabel("c ≈ 46 maps full input [0–255] to output [0–255]")
        note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.content_layout.addWidget(note)

        self.c_slider.valueChanged.connect(lambda v: self.c_spin.setValue(v / 10.0))
        self.c_spin.valueChanged.connect(lambda v: self.c_slider.setValue(int(v * 10)))
        self.c_slider.valueChanged.connect(self._recalculate)
        self._recalculate()

    def _recalculate(self):
        c = self.c_slider.value() / 10.0
        try:
            res = log_transform(self.original_img, c)
        except ValueError:
            return
        self.result_img = res
        self.previewUpdated.emit(res)


class InverseLogTransformDialog(BaseLivePreviewDialog):
    """
    Inverse log transformation using the normalized exponential mapping:
        s = 256^(r / 255) - 1
    Maps r=0 → s=0, r=255 → s=255. No adjustable parameter.
    """
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Inverse Log Transformation  [s = 256^(r/255) − 1]", parent)

        info = QLabel(
            "Formula: s = 256^(r/255) − 1\n"
            "Maps 0 → 0, 255 → 255.\n"
            "Expands bright values; compresses dark values\n"
            "(opposite of log transformation)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.content_layout.addWidget(info)

        res = inverse_log_transform(self.original_img)
        self.result_img = res
        self.previewUpdated.emit(res)


class GrayLevelSlicingDialog(BaseLivePreviewDialog):
    """
    Gray-level slicing (lecture slide p.49-54).
    Highlights pixels strictly between lower and upper bounds.
    """
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Gray-Level Slicing", parent)

        # Lower bound
        lo_row = QHBoxLayout()
        lo_row.addWidget(QLabel("Lower bound:"))
        self.lo_spin = QSpinBox()
        self.lo_spin.setRange(0, 254)
        self.lo_spin.setValue(100)
        lo_row.addWidget(self.lo_spin)
        self.content_layout.addLayout(lo_row)

        self.lo_slider = QSlider(Qt.Orientation.Horizontal)
        self.lo_slider.setRange(0, 254)
        self.lo_slider.setValue(100)
        self.content_layout.addWidget(self.lo_slider)

        # Upper bound
        hi_row = QHBoxLayout()
        hi_row.addWidget(QLabel("Upper bound:"))
        self.hi_spin = QSpinBox()
        self.hi_spin.setRange(1, 255)
        self.hi_spin.setValue(200)
        hi_row.addWidget(self.hi_spin)
        self.content_layout.addLayout(hi_row)

        self.hi_slider = QSlider(Qt.Orientation.Horizontal)
        self.hi_slider.setRange(1, 255)
        self.hi_slider.setValue(200)
        self.content_layout.addWidget(self.hi_slider)

        note = QLabel("Pixels with value > lower AND < upper → 255  (strict bounds, matching lecture)")
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.content_layout.addWidget(note)

        # Mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Preserve Background")
        self.mode_combo.addItem("Suppress Background")
        mode_row.addWidget(self.mode_combo)
        self.content_layout.addLayout(mode_row)

        # Wire sliders ↔ spinboxes
        self.lo_slider.valueChanged.connect(self.lo_spin.setValue)
        self.lo_spin.valueChanged.connect(self.lo_slider.setValue)
        self.hi_slider.valueChanged.connect(self.hi_spin.setValue)
        self.hi_spin.valueChanged.connect(self.hi_slider.setValue)

        self.lo_slider.valueChanged.connect(self._recalculate)
        self.hi_slider.valueChanged.connect(self._recalculate)
        self.mode_combo.currentIndexChanged.connect(self._recalculate)

        self._recalculate()

    def _recalculate(self):
        lower = self.lo_slider.value()
        upper = self.hi_slider.value()
        if lower >= upper:
            return
        preserve = self.mode_combo.currentIndex() == 0
        try:
            res = gray_level_slicing(self.original_img, lower, upper, preserve)
        except ValueError:
            return
        self.result_img = res
        self.previewUpdated.emit(res)


class BitPlaneSlicingDialog(BaseLivePreviewDialog):
    """
    Bit-plane slicing (lecture slide p.56-60).
    Extracts one bit-plane as a binary 0/255 grayscale image.
    Bit 0 = LSB, Bit 7 = MSB.
    """
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Bit-Plane Slicing", parent)

        row = QHBoxLayout()
        row.addWidget(QLabel("Bit plane (0–7):"))
        self.bit_spin = QSpinBox()
        self.bit_spin.setRange(0, 7)
        self.bit_spin.setValue(7)
        row.addWidget(self.bit_spin)
        self.content_layout.addLayout(row)

        self.bit_slider = QSlider(Qt.Orientation.Horizontal)
        self.bit_slider.setRange(0, 7)
        self.bit_slider.setValue(7)
        self.content_layout.addWidget(self.bit_slider)

        note = QLabel("Bit 0 = LSB  |  Bit 7 = MSB\nOutput: binary image (0 or 255 per pixel)")
        note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.content_layout.addWidget(note)

        self.bit_slider.valueChanged.connect(self.bit_spin.setValue)
        self.bit_spin.valueChanged.connect(self.bit_slider.setValue)
        self.bit_slider.valueChanged.connect(self._recalculate)

        self._recalculate()

    def _recalculate(self):
        bit = self.bit_slider.value()
        try:
            res = bit_plane_slice(self.original_img, bit)
        except ValueError:
            return
        self.result_img = res
        self.previewUpdated.emit(res)


# =============================================================================
# Histogram Specification Dialog (P10)
# =============================================================================

class HistogramSpecificationDialog(BaseLivePreviewDialog):
    """
    Histogram Specification (histogram matching) dialog.

    Allows the user to choose a target histogram shape and apply it to the
    source image via histogram_specification().

    Target histogram options:
      - Uniform: flat distribution, 1/256 per bin (equivalent to equalization)
      - From Image: derive target from a reference image loaded from disk

    The source image remains the current document. The reference image is
    only used to compute Spec[256]; it never becomes the active document.
    """

    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Histogram Specification", parent)

        # ── Target selection row ─────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Target Histogram:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Uniform", "From Image..."])
        mode_row.addWidget(self.mode_combo)
        self.content_layout.addLayout(mode_row)

        # ── "Choose Image" button (shown only for From Image mode) ───────────
        self.btn_choose = QPushButton("Choose Reference Image...")
        self.btn_choose.setEnabled(False)
        self.content_layout.addWidget(self.btn_choose)

        # ── Status label ─────────────────────────────────────────────────────
        self.lbl_status = QLabel("Target: Uniform distribution (1/256 per bin)")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.content_layout.addWidget(self.lbl_status)

        # Internal state
        self._target_prob: np.ndarray = np.full(256, 1.0 / 256.0)

        # ── Wire signals ─────────────────────────────────────────────────────
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_choose.clicked.connect(self._on_choose_image)

        # Trigger initial preview
        self._recalculate()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, index: int):
        is_from_image = (index == 1)
        self.btn_choose.setEnabled(is_from_image)
        if index == 0:
            self._target_prob = np.full(256, 1.0 / 256.0)
            self.lbl_status.setText("Target: Uniform distribution (1/256 per bin)")
            self._recalculate()
        # For From Image mode: wait for user to choose a file.

    def _on_choose_image(self):
        filter_str = (
            "All Supported Images (*.png *.jpg *.jpeg *.bmp *.pbm *.pgm *.ppm *.raw *.tif *.tiff);;"
            "All Files (*.*)"
        )
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Choose Reference Image for Target Histogram", "", filter_str
        )
        if not filepath:
            return  # user cancelled — keep existing target

        try:
            ref_img = load_image_file(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Cannot load reference image:\n{exc}")
            return

        # Convert reference to grayscale, build normalized 256-bin histogram
        gray = ref_img.to_grayscale_array()
        raw, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        n = float(gray.size)
        target_prob = raw.astype(np.float64) / n  # normalized, sum=1.0

        import os
        basename = os.path.basename(filepath)
        self.lbl_status.setText(
            f"Target: {basename}  ({ref_img.width}×{ref_img.height} px, "
            f"converted to grayscale)"
        )
        self._target_prob = target_prob
        self._recalculate()

    def _recalculate(self):
        try:
            res = histogram_specification(self.original_img, self._target_prob)
        except ValueError:
            return
        self.result_img = res
        self.previewUpdated.emit(res)
