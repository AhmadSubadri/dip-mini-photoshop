"""
Arithmetic and Boolean Operations Dialog for Mini Photoshop.
Allows selecting a second image (from open tabs or file system),
choosing operation mode (Add, Subtract, Multiply, Divide, Blend, AND, OR, XOR),
and live previewing results.
"""

from typing import Optional, List, Dict
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSlider, QFileDialog, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal
from ...engine.core import ImageMatrix
from ...engine.io_custom import load_image_file
from ...engine.arithmetic_ops import (
    add_images, subtract_images, multiply_images, divide_images, alpha_blend
)
from ...engine.boolean_ops import (
    bitwise_and, bitwise_or, bitwise_xor, mask_image
)


class ArithmeticOperationDialog(QDialog):
    previewUpdated = pyqtSignal(object)

    def __init__(
        self,
        img1: ImageMatrix,
        available_documents: Dict[str, ImageMatrix],
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Math & Logic Operations")
        self.setMinimumWidth(420)
        self.img1 = img1
        self.img2: Optional[ImageMatrix] = None
        self.result_img: Optional[ImageMatrix] = None
        self.available_docs = available_documents

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # 1. Image 2 Selector Group
        img2_group = QGroupBox("Select Second Image (Operand B)")
        img2_layout = QVBoxLayout(img2_group)

        self.doc_combo = QComboBox()
        self.doc_combo.addItem("-- Choose from open images --", None)
        for name, doc_img in self.available_docs.items():
            self.doc_combo.addItem(name, doc_img)
        self.doc_combo.currentIndexChanged.connect(self._on_doc_selected)
        img2_layout.addWidget(self.doc_combo)

        browse_layout = QHBoxLayout()
        self.lbl_file = QLabel("Or load image from disk...")
        self.lbl_file.setStyleSheet("color: #888888; font-style: italic;")
        btn_browse = QPushButton("Browse File...")
        btn_browse.clicked.connect(self._browse_file)
        browse_layout.addWidget(self.lbl_file)
        browse_layout.addWidget(btn_browse)
        img2_layout.addLayout(browse_layout)

        main_layout.addWidget(img2_group)

        # 2. Operation Type Group
        op_group = QGroupBox("Operation")
        op_layout = QVBoxLayout(op_group)

        self.op_combo = QComboBox()
        self.op_combo.addItems([
            "Addition (A + B) [Clipped]",
            "Addition (A + B) / 2 [Average]",
            "Subtraction |A - B| [Absolute Difference]",
            "Subtraction (A - B) [Clipped]",
            "Multiplication (A * B / 255)",
            "Division (A / B * 255)",
            "Alpha Blending: α*A + (1-α)*B",
            "Bitwise AND (A & B)",
            "Bitwise OR (A | B)",
            "Bitwise XOR (A ^ B)",
            "Masking (Apply B as mask to A)"
        ])
        self.op_combo.currentIndexChanged.connect(self._recalculate)
        op_layout.addWidget(self.op_combo)

        # Alpha Slider (Visible when Alpha Blending is selected)
        self.alpha_container = QVBoxLayout()
        self.alpha_header = QHBoxLayout()
        self.alpha_header.addWidget(QLabel("Alpha Weight (α):"))
        self.alpha_val_lbl = QLabel("0.50")
        self.alpha_header.addWidget(self.alpha_val_lbl)
        self.alpha_container.addLayout(self.alpha_header)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.alpha_container.addWidget(self.alpha_slider)
        op_layout.addLayout(self.alpha_container)

        main_layout.addWidget(op_group)

        # 3. Dialog Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("Apply Operation")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        main_layout.addLayout(btn_layout)

        # Initial selection if available
        if len(self.available_docs) > 0:
            self.doc_combo.setCurrentIndex(1)

    def _browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Second Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.pbm *.pgm *.ppm *.raw *.tif *.tiff);;All Files (*.*)"
        )
        if filepath:
            try:
                loaded = load_image_file(filepath)
                self.img2 = loaded
                self.lbl_file.setText(os.path.basename(filepath))
                self.lbl_file.setStyleSheet("color: #4ec9b0;")
                self._recalculate()
            except Exception as e:
                self.lbl_file.setText(f"Error: {e}")
                self.lbl_file.setStyleSheet("color: #f48771;")

    def _on_doc_selected(self, index: int):
        data = self.doc_combo.currentData()
        if data is not None:
            self.img2 = data
            self.lbl_file.setText(f"Using open tab: {self.doc_combo.currentText()}")
            self.lbl_file.setStyleSheet("color: #4ec9b0;")
            self._recalculate()

    def _on_alpha_changed(self, val: int):
        alpha = val / 100.0
        self.alpha_val_lbl.setText(f"{alpha:.2f}")
        self._recalculate()

    def _recalculate(self):
        if self.img2 is None:
            return

        op_text = self.op_combo.currentText()
        
        try:
            if "Addition (A + B) [Clipped]" in op_text:
                res = add_images(self.img1, self.img2, mode="clip")
            elif "Addition (A + B) / 2" in op_text:
                res = add_images(self.img1, self.img2, mode="average")
            elif "Subtraction |A - B|" in op_text:
                res = subtract_images(self.img1, self.img2, absolute=True)
            elif "Subtraction (A - B)" in op_text:
                res = subtract_images(self.img1, self.img2, absolute=False)
            elif "Multiplication" in op_text:
                res = multiply_images(self.img1, self.img2)
            elif "Division" in op_text:
                res = divide_images(self.img1, self.img2)
            elif "Alpha Blending" in op_text:
                alpha = self.alpha_slider.value() / 100.0
                res = alpha_blend(self.img1, self.img2, alpha=alpha)
            elif "Bitwise AND" in op_text:
                res = bitwise_and(self.img1, self.img2)
            elif "Bitwise OR" in op_text:
                res = bitwise_or(self.img1, self.img2)
            elif "Bitwise XOR" in op_text:
                res = bitwise_xor(self.img1, self.img2)
            elif "Masking" in op_text:
                res = mask_image(self.img1, self.img2)
            else:
                res = self.img1

            self.result_img = res
            self.previewUpdated.emit(res)
        except Exception as e:
            print(f"Operation error: {e}")

    def reject(self):
        self.previewUpdated.emit(self.img1)
        super().reject()
