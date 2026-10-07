"""
Dialog Operasi Spasial Lokal (Spatial / Neighborhood Operations Dialog)
untuk Mini Photoshop.

Menyediakan antarmuka interaktif dengan Live Preview untuk:
1. Operasi Linier: Mean (2x2, 3x3, 5x5), Gaussian, Sharpening, Roberts (2x2), Sobel (3x3), Custom Kernel.
2. Operasi Non-Linier: Median Filter, Max Filter, Min Filter.
3. Box Penjelasan Rumus Matematis Edukatif untuk kemudahan presentasi tugas.
"""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QGridLayout, QLineEdit, QTextEdit,
    QRadioButton, QButtonGroup, QStackedWidget, QWidget,
)
from PyQt6.QtCore import Qt
from ...engine.core import ImageMatrix
from ...engine.spatial_ops import (
    apply_mean_filter,
    apply_gaussian_filter,
    apply_sharpen_filter,
    apply_edge_roberts,
    apply_edge_sobel,
    apply_kernel_to_image,
    apply_median_filter,
    apply_max_filter,
    apply_min_filter,
)
from .adjust_dialog import BaseLivePreviewDialog


class SpatialFilterDialog(BaseLivePreviewDialog):
    """
    Dialog konfigurasi Operasi Spasial Lokal dengan Live Preview real-time.
    """
    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, "Operasi Spasial Lokal (Neighborhood Operations)", parent)
        self.setMinimumWidth(440)

        # ----------------------------------------------------------------------
        # 1. Pilihan Kategori: Linier vs Non-Linier
        # ----------------------------------------------------------------------
        cat_group = QGroupBox("Kategori Operasi")
        cat_layout = QHBoxLayout(cat_group)
        
        self.rb_linear = QRadioButton("Operasi Linier (Konvolusi Kernel)")
        self.rb_nonlinear = QRadioButton("Operasi Non-Linier (Rank-Order Filter)")
        self.rb_linear.setChecked(True)
        
        self.btn_group_cat = QButtonGroup(self)
        self.btn_group_cat.addButton(self.rb_linear, 0)
        self.btn_group_cat.addButton(self.rb_nonlinear, 1)
        
        cat_layout.addWidget(self.rb_linear)
        cat_layout.addWidget(self.rb_nonlinear)
        self.content_layout.addWidget(cat_group)

        # ----------------------------------------------------------------------
        # 2. Stacked Kontrol Parameter
        # ----------------------------------------------------------------------
        self.stack = QStackedWidget()

        # --- A. Halaman Kontrol Linier ---
        self.page_linear = QWidget()
        linear_layout = QVBoxLayout(self.page_linear)
        linear_layout.setContentsMargins(0, 0, 0, 0)

        # Dropdown Jenis Filter Linier
        lin_type_row = QHBoxLayout()
        lin_type_row.addWidget(QLabel("Jenis Filter:"))
        self.combo_linear_type = QComboBox()
        self.combo_linear_type.addItems([
            "Mean / Averaging Blur",
            "Gaussian Blur",
            "Sharpening (Laplacian)",
            "Deteksi Tepi Roberts (Kernel 2x2)",
            "Deteksi Tepi Sobel (Kernel 3x3)",
            "Custom Kernel (3x3 Matrix)",
        ])
        lin_type_row.addWidget(self.combo_linear_type, 1)
        linear_layout.addLayout(lin_type_row)

        # Ukuran Kernel Linier (2x2, 3x3, 5x5, 7x7)
        self.row_lin_ksize = QHBoxLayout()
        self.row_lin_ksize.addWidget(QLabel("Ukuran Kernel:"))
        self.combo_lin_ksize = QComboBox()
        self.combo_lin_ksize.addItems(["2x2", "3x3", "5x5", "7x7"])
        self.combo_lin_ksize.setCurrentText("3x3")
        self.row_lin_ksize.addWidget(self.combo_lin_ksize, 1)
        linear_layout.addLayout(self.row_lin_ksize)

        # Kontrol Sigma untuk Gaussian Blur
        self.row_sigma = QHBoxLayout()
        self.row_sigma.addWidget(QLabel("Gaussian Sigma (\u03c3):"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.2)
        self.spin_sigma.setValue(1.0)
        self.row_sigma.addWidget(self.spin_sigma, 1)
        linear_layout.addLayout(self.row_sigma)

        # Kontrol Mode Sharpening
        self.row_sharp = QHBoxLayout()
        self.row_sharp.addWidget(QLabel("Kekuatan Sharpening:"))
        self.combo_sharp_mode = QComboBox()
        self.combo_sharp_mode.addItems(["Standard (4-Tetangga)", "Strong (8-Tetangga)"])
        self.row_sharp.addWidget(self.combo_sharp_mode, 1)
        linear_layout.addLayout(self.row_sharp)

        # Grid Custom Kernel 3x3
        self.group_custom = QGroupBox("Matriks Kernel 3x3 Custom")
        self.grid_custom = QGridLayout(self.group_custom)
        self.custom_inputs = []
        # Inisialisasi default dengan Identity kernel [[0,0,0],[0,1,0],[0,0,0]]
        default_matrix = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
        for r in range(3):
            row_inputs = []
            for c in range(3):
                le = QLineEdit(str(default_matrix[r][c]))
                le.setAlignment(Qt.AlignmentFlag.AlignCenter)
                le.textChanged.connect(self._on_params_changed)
                self.grid_custom.addWidget(le, r, c)
                row_inputs.append(le)
            self.custom_inputs.append(row_inputs)
        linear_layout.addWidget(self.group_custom)
        self.stack.addWidget(self.page_linear)

        # --- B. Halaman Kontrol Non-Linier ---
        self.page_nonlinear = QWidget()
        nonlin_layout = QVBoxLayout(self.page_nonlinear)
        nonlin_layout.setContentsMargins(0, 0, 0, 0)

        # Dropdown Jenis Filter Non-Linier
        nonlin_type_row = QHBoxLayout()
        nonlin_type_row.addWidget(QLabel("Jenis Filter Non-Linier:"))
        self.combo_nonlin_type = QComboBox()
        self.combo_nonlin_type.addItems([
            "Median Filter (Reduksi Salt & Pepper)",
            "Max Filter (Nilai Maksimum Tetangga)",
            "Min Filter (Nilai Minimum Tetangga)",
        ])
        nonlin_type_row.addWidget(self.combo_nonlin_type, 1)
        nonlin_layout.addLayout(nonlin_type_row)

        # Ukuran Jendela Non-Linier (3x3, 5x5, 7x7)
        nonlin_ksize_row = QHBoxLayout()
        nonlin_ksize_row.addWidget(QLabel("Ukuran Jendela Lokal:"))
        self.combo_nonlin_ksize = QComboBox()
        self.combo_nonlin_ksize.addItems(["3x3", "5x5", "7x7"])
        self.combo_nonlin_ksize.setCurrentText("3x3")
        nonlin_ksize_row.addWidget(self.combo_nonlin_ksize, 1)
        nonlin_layout.addLayout(nonlin_ksize_row)

        nonlin_layout.addStretch()
        self.stack.addWidget(self.page_nonlinear)

        self.content_layout.addWidget(self.stack)

        # ----------------------------------------------------------------------
        # 3. Kotak Penjelasan Teori & Rumus Matematis
        # ----------------------------------------------------------------------
        info_group = QGroupBox("Rumus & Teori Matematis")
        info_layout = QVBoxLayout(info_group)
        self.lbl_formula_desc = QTextEdit()
        self.lbl_formula_desc.setReadOnly(True)
        self.lbl_formula_desc.setMaximumHeight(85)
        info_layout.addWidget(self.lbl_formula_desc)
        self.content_layout.addWidget(info_group)

        # ----------------------------------------------------------------------
        # Hubungkan Sinyal Event
        # ----------------------------------------------------------------------
        self.rb_linear.toggled.connect(self._on_category_changed)
        self.combo_linear_type.currentIndexChanged.connect(self._update_linear_ui_visibility)
        self.combo_linear_type.currentIndexChanged.connect(self._on_params_changed)
        self.combo_lin_ksize.currentIndexChanged.connect(self._on_params_changed)
        self.spin_sigma.valueChanged.connect(self._on_params_changed)
        self.combo_sharp_mode.currentIndexChanged.connect(self._on_params_changed)

        self.combo_nonlin_type.currentIndexChanged.connect(self._on_params_changed)
        self.combo_nonlin_ksize.currentIndexChanged.connect(self._on_params_changed)

        # Update tampilan awal
        self._update_linear_ui_visibility()
        self._on_params_changed()

    def _on_category_changed(self):
        """Berpindah antara halaman kontrol linier dan non-linier."""
        if self.rb_linear.isChecked():
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)
        self._on_params_changed()

    def _update_linear_ui_visibility(self):
        """Menampilkan atau menyembunyikan input sesuai jenis filter linier yang dipilih."""
        lin_type = self.combo_linear_type.currentText()
        is_mean = "Mean" in lin_type
        is_gauss = "Gaussian" in lin_type
        is_sharp = "Sharpening" in lin_type
        is_custom = "Custom" in lin_type

        # Atur visibility kontrol
        self.combo_lin_ksize.setEnabled(is_mean or is_gauss)
        self.spin_sigma.setEnabled(is_gauss)
        self.combo_sharp_mode.setEnabled(is_sharp)
        self.group_custom.setVisible(is_custom)

    def _get_selected_ksize(self, combo_text: str) -> int:
        """Mengonversi teks '3x3' menjadi integer 3."""
        return int(combo_text.split("x")[0])

    def _on_params_changed(self):
        """Menghitung operasi spasial manual dan mengirim sinyal preview real-time."""
        try:
            if self.rb_linear.isChecked():
                # ----------------- OPERASI LINIER -----------------
                lin_choice = self.combo_linear_type.currentText()
                ksize = self._get_selected_ksize(self.combo_lin_ksize.currentText())

                if "Mean" in lin_choice:
                    self.result_img = apply_mean_filter(self.original_img, kernel_size=ksize)
                    self.lbl_formula_desc.setText(
                        f"Mean Filter ({ksize}x{ksize}):\n"
                        f"g(y,x) = (1/{ksize*ksize}) * \u2211 f(y+i, x+j)\n"
                        f"Menghaluskan citra dengan rata-rata piksel tetangga."
                    )

                elif "Gaussian" in lin_choice:
                    sigma = self.spin_sigma.value()
                    self.result_img = apply_gaussian_filter(self.original_img, kernel_size=ksize, sigma=sigma)
                    self.lbl_formula_desc.setText(
                        f"Gaussian Blur ({ksize}x{ksize}, \u03c3={sigma:.1f}):\n"
                        f"G(y,x) = (1 / 2\u03c0\u03c3\u00b2) * exp(-(x\u00b2 + y\u00b2)/(2\u03c3\u00b2))\n"
                        f"Penghalusan berbobot normal, tepi lebih terjaga dibanding mean."
                    )

                elif "Sharpening" in lin_choice:
                    mode = "strong" if "Strong" in self.combo_sharp_mode.currentText() else "standard"
                    self.result_img = apply_sharpen_filter(self.original_img, mode=mode)
                    self.lbl_formula_desc.setText(
                        f"Sharpening Filter (High-Pass Laplacian):\n"
                        f"Kernel tengah bernilai positif tinggi (+5 / +9), tetangga -1.\n"
                        f"Mempertegas transisi kontras pada tepi objek."
                    )

                elif "Roberts" in lin_choice:
                    self.result_img = apply_edge_roberts(self.original_img)
                    self.lbl_formula_desc.setText(
                        f"Deteksi Tepi Roberts Cross (Kernel 2x2):\n"
                        f"Gx = [[1, 0], [0, -1]],  Gy = [[0, 1], [-1, 0]]\n"
                        f"Magnitudo Gradien: G = \u221a(Gx\u00b2 + Gy\u00b2)"
                    )

                elif "Sobel" in lin_choice:
                    self.result_img = apply_edge_sobel(self.original_img)
                    self.lbl_formula_desc.setText(
                        f"Deteksi Tepi Sobel (Kernel 3x3):\n"
                        f"Gx, Gy menggunakan pembobotan diferensial spasial.\n"
                        f"Magnitudo Gradien: G = \u221a(Gx\u00b2 + Gy\u00b2)"
                    )

                elif "Custom" in lin_choice:
                    # Ambil nilai matriks dari tabel grid
                    mat = np.zeros((3, 3), dtype=np.float32)
                    for r in range(3):
                        for c in range(3):
                            val_str = self.custom_inputs[r][c].text().strip()
                            mat[r, c] = float(val_str) if val_str else 0.0
                    self.result_img = apply_kernel_to_image(self.original_img, mat, clip_output=True)
                    self.lbl_formula_desc.setText(
                        f"Custom Kernel Konvolusi 3x3:\n"
                        f"g(y,x) = \u2211\u2211 f(y+i, x+j) * K(i,j)\n"
                        f"Konvolusi spasial manual sesuai bobot matriks yang dimasukkan."
                    )

            else:
                # --------------- OPERASI NON-LINIER ---------------
                nonlin_choice = self.combo_nonlin_type.currentText()
                ksize = self._get_selected_ksize(self.combo_nonlin_ksize.currentText())

                if "Median" in nonlin_choice:
                    self.result_img = apply_median_filter(self.original_img, kernel_size=ksize)
                    self.lbl_formula_desc.setText(
                        f"Median Filter ({ksize}x{ksize}):\n"
                        f"g(y,x) = median( {ksize*ksize} piksel tetangga terurut )\n"
                        f"Sangat efektif mereduksi Salt & Pepper noise tanpa mengaburkan tepi."
                    )

                elif "Max" in nonlin_choice:
                    self.result_img = apply_max_filter(self.original_img, kernel_size=ksize)
                    self.lbl_formula_desc.setText(
                        f"Max Filter ({ksize}x{ksize}):\n"
                        f"g(y,x) = max( piksel tetangga )\n"
                        f"Mempertegas area terang dan menghilangkan bintik hitam (pepper)."
                    )

                elif "Min" in nonlin_choice:
                    self.result_img = apply_min_filter(self.original_img, kernel_size=ksize)
                    self.lbl_formula_desc.setText(
                        f"Min Filter ({ksize}x{ksize}):\n"
                        f"g(y,x) = min( piksel tetangga )\n"
                        f"Mempertegas area gelap dan menghilangkan bintik putih (salt)."
                    )

            # Kirim sinyal update ke canvas untuk Live Preview
            self.previewUpdated.emit(self.result_img)

        except Exception as e:
            # Cegah error saat pengguna sedang mengetik desimal di text input
            pass
