"""
Adjustment Dialogs with Real-Time Live Preview for Mini Photoshop.
Includes Brightness & Contrast, Thresholding (Manual & Otsu), Gamma, and Posterization.
"""

from typing import Callable, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from ...engine.core import ImageMatrix
from ...engine.point_ops import (
    adjust_brightness, adjust_contrast, threshold_manual,
    threshold_otsu, compute_otsu_threshold, gamma_correction, posterize
)


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
