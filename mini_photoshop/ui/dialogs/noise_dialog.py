"""
Dialog Simulasi Noise & Restorasi Citra (Noise & Restoration Dialogs)
untuk Mini Photoshop.

Menyediakan:
1. AddNoiseDialog: Pembangkitan derau manual (Salt & Pepper, Gaussian, Speckle) dengan Live Preview.
2. NoiseReductionDialog: Reduksi derau menggunakan operasi lokal terarah serta evaluasi kuantitatif
   skor PSNR (Peak Signal-to-Noise Ratio) dan MSE (Mean Squared Error) secara langsung.
"""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTextEdit, QWidget,
    QDialog, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from ...engine.core import ImageMatrix
from ...engine.noise_ops import (
    add_salt_and_pepper_noise,
    add_gaussian_noise,
    add_speckle_noise,
    evaluate_restoration,
)
from ...engine.spatial_ops import (
    apply_median_filter,
    apply_mean_filter,
    apply_gaussian_filter,
)
from .adjust_dialog import BaseLivePreviewDialog


class AddNoiseDialog(BaseLivePreviewDialog):
    """
    Dialog untuk menambahkan derau (noise) buatan secara interaktif dengan Live Preview.
    """
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Simulasi Tambah Derau (Add Noise)", parent)
        self.setMinimumWidth(400)

        # 1. Pilihan Jenis Derau
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Jenis Derau (Noise Type):"))
        self.combo_noise_type = QComboBox()
        self.combo_noise_type.addItems([
            "Salt & Pepper Noise (Impulsive)",
            "Gaussian Noise (Normal Distribution)",
            "Speckle Noise (Multiplicative)",
        ])
        type_row.addWidget(self.combo_noise_type, 1)
        self.content_layout.addLayout(type_row)

        # 2. Kontrol Salt & Pepper
        self.group_sp = QGroupBox("Parameter Salt & Pepper")
        sp_layout = QVBoxLayout(self.group_sp)

        row_sp_amt = QHBoxLayout()
        row_sp_amt.addWidget(QLabel("Kerapatan Derau (%):"))
        self.spin_sp_amount = QSpinBox()
        self.spin_sp_amount.setRange(1, 80)
        self.spin_sp_amount.setValue(5)  # Default 5%
        row_sp_amt.addWidget(self.spin_sp_amount)
        sp_layout.addLayout(row_sp_amt)

        self.slider_sp_amount = QSlider(Qt.Orientation.Horizontal)
        self.slider_sp_amount.setRange(1, 80)
        self.slider_sp_amount.setValue(5)
        sp_layout.addWidget(self.slider_sp_amount)

        self.content_layout.addWidget(self.group_sp)

        # 3. Kontrol Gaussian Noise
        self.group_gauss = QGroupBox("Parameter Gaussian Noise")
        gauss_layout = QVBoxLayout(self.group_gauss)

        row_g_sig = QHBoxLayout()
        row_g_sig.addWidget(QLabel("Standar Deviasi (\u03c3):"))
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(1.0, 100.0)
        self.spin_gauss_sigma.setValue(25.0)
        row_g_sig.addWidget(self.spin_gauss_sigma)
        gauss_layout.addLayout(row_g_sig)

        self.slider_gauss_sigma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gauss_sigma.setRange(1, 100)
        self.slider_gauss_sigma.setValue(25)
        gauss_layout.addWidget(self.slider_gauss_sigma)

        self.content_layout.addWidget(self.group_gauss)

        # 4. Kontrol Speckle Noise
        self.group_speckle = QGroupBox("Parameter Speckle Noise")
        speckle_layout = QVBoxLayout(self.group_speckle)

        row_spec_var = QHBoxLayout()
        row_spec_var.addWidget(QLabel("Varians Derau:"))
        self.spin_speckle_var = QDoubleSpinBox()
        self.spin_speckle_var.setRange(0.01, 1.0)
        self.spin_speckle_var.setSingleStep(0.02)
        self.spin_speckle_var.setValue(0.05)
        row_spec_var.addWidget(self.spin_speckle_var)
        speckle_layout.addLayout(row_spec_var)

        self.content_layout.addWidget(self.group_speckle)

        # 5. Kotak Penjelasan Teori Matematis
        info_group = QGroupBox("Penjelasan Teori")
        info_layout = QVBoxLayout(info_group)
        self.lbl_info = QTextEdit()
        self.lbl_info.setReadOnly(True)
        self.lbl_info.setMaximumHeight(85)
        info_layout.addWidget(self.lbl_info)
        self.content_layout.addWidget(info_group)

        # Sinkronisasi Slider & Spinbox
        self.spin_sp_amount.valueChanged.connect(self.slider_sp_amount.setValue)
        self.slider_sp_amount.valueChanged.connect(self.spin_sp_amount.setValue)
        self.slider_sp_amount.valueChanged.connect(self._on_params_changed)

        self.spin_gauss_sigma.valueChanged.connect(lambda v: self.slider_gauss_sigma.setValue(int(v)))
        self.slider_gauss_sigma.valueChanged.connect(lambda v: self.spin_gauss_sigma.setValue(float(v)))
        self.slider_gauss_sigma.valueChanged.connect(self._on_params_changed)

        self.spin_speckle_var.valueChanged.connect(self._on_params_changed)
        self.combo_noise_type.currentIndexChanged.connect(self._update_visibility)
        self.combo_noise_type.currentIndexChanged.connect(self._on_params_changed)

        self._update_visibility()
        self._on_params_changed()

    def _update_visibility(self):
        idx = self.combo_noise_type.currentIndex()
        self.group_sp.setVisible(idx == 0)
        self.group_gauss.setVisible(idx == 1)
        self.group_speckle.setVisible(idx == 2)

    def _on_params_changed(self):
        idx = self.combo_noise_type.currentIndex()
        seed = 42  # Seed konstan agar live preview slider mulus tanpa flicker drastis

        if idx == 0:
            # Salt & Pepper
            amt = self.spin_sp_amount.value() / 100.0
            self.result_img = add_salt_and_pepper_noise(self.original_img, amount=amt, seed=seed)
            self.lbl_info.setText(
                f"Salt & Pepper Noise ({self.spin_sp_amount.value()}%):\n"
                f"Mengacak {self.spin_sp_amount.value()}% piksel citra menjadi ekstrem:\n"
                f"Putih = 255 (Salt) dan Hitam = 0 (Pepper).\n"
                f"Metode restorasi terbaik: Non-Linear Median Filter."
            )
        elif idx == 1:
            # Gaussian
            sigma = self.spin_gauss_sigma.value()
            self.result_img = add_gaussian_noise(self.original_img, mean=0.0, sigma=sigma, seed=seed)
            self.lbl_info.setText(
                f"Gaussian Noise (\u03c3 = {sigma:.1f}):\n"
                f"f_noisy(x,y) = clip( f(x,y) + N(0, \u03c3\u00b2), 0, 255 )\n"
                f"Menambahkan gangguan distribusi normal ke seluruh piksel.\n"
                f"Metode restorasi terbaik: Linear Mean atau Gaussian Filter."
            )
        else:
            # Speckle
            var = self.spin_speckle_var.value()
            self.result_img = add_speckle_noise(self.original_img, variance=var, seed=seed)
            self.lbl_info.setText(
                f"Speckle Noise (Varians = {var:.2f}):\n"
                f"f_noisy(x,y) = clip( f(x,y) + f(x,y) * N(0, Var), 0, 255 )\n"
                f"Gangguan perkalian (multiplikatif) proporsional terhadap intensitas piksel."
            )

        self.previewUpdated.emit(self.result_img)


class NoiseReductionDialog(BaseLivePreviewDialog):
    """
    Dialog untuk mereduksi derau menggunakan operasi lokal, dilengkapi dengan
    evaluasi kuantitatif ilmiah (PSNR dan MSE) terhadap citra pembanding.
    """
    def __init__(self, current_img: ImageMatrix, reference_img: Optional[ImageMatrix] = None, parent=None):
        super().__init__(current_img, "Reduksi Derau & Evaluasi Restorasi", parent)
        self.setMinimumWidth(440)
        self.reference_img = reference_img if reference_img is not None else current_img

        # 1. Pilihan Metode Reduksi
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metode Reduksi:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "Median Filter (Rekomendasi untuk Salt & Pepper)",
            "Mean / Averaging Filter (Rekomendasi untuk Gaussian)",
            "Gaussian Filter (Penghalusan Berbobot Normal)",
        ])
        method_row.addWidget(self.combo_method, 1)
        self.content_layout.addLayout(method_row)

        # 2. Pilihan Ukuran Kernel
        ksize_row = QHBoxLayout()
        ksize_row.addWidget(QLabel("Ukuran Jendela Lokal:"))
        self.combo_ksize = QComboBox()
        self.combo_ksize.addItems(["3x3", "5x5", "7x7"])
        self.combo_ksize.setCurrentText("3x3")
        ksize_row.addWidget(self.combo_ksize, 1)
        self.content_layout.addLayout(ksize_row)

        # 3. Panel Evaluasi Metrik Restorasi (PSNR & MSE)
        eval_group = QGroupBox("Evaluasi Kualitas Restorasi (PSNR & MSE)")
        eval_layout = QVBoxLayout(eval_group)
        self.lbl_metrics_info = QTextEdit()
        self.lbl_metrics_info.setReadOnly(True)
        self.lbl_metrics_info.setMaximumHeight(110)
        eval_layout.addWidget(self.lbl_metrics_info)
        self.content_layout.addWidget(eval_group)

        # Event
        self.combo_method.currentIndexChanged.connect(self._on_params_changed)
        self.combo_ksize.currentIndexChanged.connect(self._on_params_changed)

        self._on_params_changed()

    def _on_params_changed(self):
        method_idx = self.combo_method.currentIndex()
        ksize = int(self.combo_ksize.currentText().split("x")[0])

        # 1. Terapkan filter reduksi
        if method_idx == 0:
            self.result_img = apply_median_filter(self.original_img, kernel_size=ksize)
        elif method_idx == 1:
            self.result_img = apply_mean_filter(self.original_img, kernel_size=ksize)
        else:
            self.result_img = apply_gaussian_filter(self.original_img, kernel_size=ksize, sigma=1.0)

        # 2. Hitung evaluasi terhadap citra referensi
        stats = evaluate_restoration(
            original=self.reference_img,
            noisy=self.original_img,
            restored=self.result_img
        )

        # 3. Tampilkan hasil kuantitatif
        status_txt = "\u2705 Kualitas Meningkat!" if stats["delta_psnr"] > 0 else "\u2139\ufe0f Filter Menghaluskan Citra"
        self.lbl_metrics_info.setText(
            f"Kondisi Sebelum Reduksi (Noisy):\n"
            f"  - MSE: {stats['mse_noisy']:.2f}  |  PSNR: {stats['psnr_noisy']:.2f} dB\n\n"
            f"Kondisi Setelah Reduksi (Restored):\n"
            f"  - MSE: {stats['mse_restored']:.2f}  |  PSNR: {stats['psnr_restored']:.2f} dB\n"
            f"  - Perubahan Skor: \u0394PSNR = {stats['delta_psnr']:+.2f} dB ({status_txt})"
        )

        self.previewUpdated.emit(self.result_img)
