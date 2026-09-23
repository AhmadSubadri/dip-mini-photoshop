"""
Geometric Transformation Dialogs for Mini Photoshop.
Includes Rotation (arbitrary angle with interpolation), Translation, and Scaling/Resizing.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from ...engine.core import ImageMatrix
from ...engine.geometry_ops import rotate_arbitrary, translate, zoom_scale, resize_exact


class RotateDialog(QDialog):
    previewUpdated = pyqtSignal(object)

    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rotate Image")
        self.setMinimumWidth(380)
        self.original_img = original_img
        self.result_img = original_img.copy()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        grp = QGroupBox("Rotation Parameters")
        grp_layout = QVBoxLayout(grp)

        # Angle
        angle_header = QHBoxLayout()
        angle_header.addWidget(QLabel("Angle (Degrees Clockwise):"))
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-180.0, 180.0)
        self.spin_angle.setSingleStep(1.0)
        self.spin_angle.setValue(0.0)
        angle_header.addWidget(self.spin_angle)
        grp_layout.addLayout(angle_header)

        self.slider_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_angle.setRange(-180, 180)
        self.slider_angle.setValue(0)
        grp_layout.addWidget(self.slider_angle)

        # Interpolation
        interp_header = QHBoxLayout()
        interp_header.addWidget(QLabel("Interpolation:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["Bilinear (Smooth)", "Nearest Neighbor (Pixelated)"])
        interp_header.addWidget(self.combo_interp)
        grp_layout.addLayout(interp_header)

        # Auto expand canvas
        self.chk_expand = QCheckBox("Auto-expand Canvas to Fit Rotated Image")
        self.chk_expand.setChecked(True)
        grp_layout.addWidget(self.chk_expand)

        layout.addWidget(grp)

        # Quick preset buttons (90, 180, 270)
        preset_layout = QHBoxLayout()
        for deg in [45, 90, 180, -90, -45]:
            btn = QPushButton(f"{deg}°")
            btn.clicked.connect(lambda _, d=deg: self.slider_angle.setValue(d))
            preset_layout.addWidget(btn)
        layout.addLayout(preset_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        # Signals
        self.slider_angle.valueChanged.connect(self.spin_angle.setValue)
        self.spin_angle.valueChanged.connect(lambda v: self.slider_angle.setValue(int(v)))
        self.slider_angle.valueChanged.connect(self._recalculate)
        self.combo_interp.currentIndexChanged.connect(self._recalculate)
        self.chk_expand.toggled.connect(self._recalculate)

    def _recalculate(self):
        angle = self.spin_angle.value()
        interp = "nearest" if "Nearest" in self.combo_interp.currentText() else "bilinear"
        expand = self.chk_expand.isChecked()

        if angle == 0:
            self.result_img = self.original_img.copy()
        else:
            self.result_img = rotate_arbitrary(self.original_img, angle, interpolation=interp, auto_expand=expand)

        self.previewUpdated.emit(self.result_img)

    def reject(self):
        self.previewUpdated.emit(self.original_img)
        super().reject()


class TranslateDialog(QDialog):
    previewUpdated = pyqtSignal(object)

    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Translate Image (Shift)")
        self.setMinimumWidth(380)
        self.original_img = original_img
        self.result_img = original_img.copy()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        grp = QGroupBox("Shift Offsets (Pixels)")
        grp_layout = QVBoxLayout(grp)

        max_dx = original_img.width
        max_dy = original_img.height

        # DX
        dx_hdr = QHBoxLayout()
        dx_hdr.addWidget(QLabel("Horizontal Shift (dx):"))
        self.spin_dx = QSpinBox()
        self.spin_dx.setRange(-max_dx, max_dx)
        self.spin_dx.setValue(0)
        dx_hdr.addWidget(self.spin_dx)
        grp_layout.addLayout(dx_hdr)

        self.slider_dx = QSlider(Qt.Orientation.Horizontal)
        self.slider_dx.setRange(-max_dx, max_dx)
        self.slider_dx.setValue(0)
        grp_layout.addWidget(self.slider_dx)

        # DY
        dy_hdr = QHBoxLayout()
        dy_hdr.addWidget(QLabel("Vertical Shift (dy):"))
        self.spin_dy = QSpinBox()
        self.spin_dy.setRange(-max_dy, max_dy)
        self.spin_dy.setValue(0)
        dy_hdr.addWidget(self.spin_dy)
        grp_layout.addLayout(dy_hdr)

        self.slider_dy = QSlider(Qt.Orientation.Horizontal)
        self.slider_dy.setRange(-max_dy, max_dy)
        self.slider_dy.setValue(0)
        grp_layout.addWidget(self.slider_dy)

        layout.addWidget(grp)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        # Connect
        self.slider_dx.valueChanged.connect(self.spin_dx.setValue)
        self.spin_dx.valueChanged.connect(self.slider_dx.setValue)
        self.slider_dx.valueChanged.connect(self._recalculate)

        self.slider_dy.valueChanged.connect(self.spin_dy.setValue)
        self.spin_dy.valueChanged.connect(self.slider_dy.setValue)
        self.slider_dy.valueChanged.connect(self._recalculate)

    def _recalculate(self):
        dx = self.slider_dx.value()
        dy = self.slider_dy.value()
        self.result_img = translate(self.original_img, dx, dy)
        self.previewUpdated.emit(self.result_img)

    def reject(self):
        self.previewUpdated.emit(self.original_img)
        super().reject()


class ScaleDialog(QDialog):
    previewUpdated = pyqtSignal(object)

    def __init__(self, original_img: ImageMatrix, parent=None):
        super().__init__(original_img, parent)
        self.setWindowTitle("Scale / Resize Image")
        self.setMinimumWidth(380)
        self.original_img = original_img
        self.result_img = original_img.copy()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        grp = QGroupBox("Dimensions")
        grp_layout = QVBoxLayout(grp)

        # Width
        w_hdr = QHBoxLayout()
        w_hdr.addWidget(QLabel("Width (px):"))
        self.spin_w = QSpinBox()
        self.spin_w.setRange(1, 10000)
        self.spin_w.setValue(original_img.width)
        w_hdr.addWidget(self.spin_w)
        grp_layout.addLayout(w_hdr)

        # Height
        h_hdr = QHBoxLayout()
        h_hdr.addWidget(QLabel("Height (px):"))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(1, 10000)
        self.spin_h.setValue(original_img.height)
        h_hdr.addWidget(self.spin_h)
        grp_layout.addLayout(h_hdr)

        # Maintain Aspect Ratio
        self.chk_ratio = QCheckBox("Maintain Aspect Ratio")
        self.chk_ratio.setChecked(True)
        grp_layout.addWidget(self.chk_ratio)

        # Interpolation
        interp_header = QHBoxLayout()
        interp_header.addWidget(QLabel("Interpolation:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["Bilinear (Smooth)", "Nearest Neighbor (Fast / Pixelated)"])
        interp_header.addWidget(self.combo_interp)
        grp_layout.addLayout(interp_header)

        layout.addWidget(grp)

        # Percentage Scale Presets (50%, 75%, 150%, 200%)
        preset_layout = QHBoxLayout()
        for pct in [50, 75, 125, 150, 200]:
            btn = QPushButton(f"{pct}%")
            btn.clicked.connect(lambda _, p=pct: self._apply_percentage(p))
            preset_layout.addWidget(btn)
        layout.addLayout(preset_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        # Connect
        self.spin_w.valueChanged.connect(self._on_width_changed)
        self.spin_h.valueChanged.connect(self._on_height_changed)
        self.combo_interp.currentIndexChanged.connect(self._recalculate)

    def _apply_percentage(self, pct: int):
        new_w = int(round(self.original_img.width * (pct / 100.0)))
        new_h = int(round(self.original_img.height * (pct / 100.0)))
        self.spin_w.blockSignals(True)
        self.spin_h.blockSignals(True)
        self.spin_w.setValue(new_w)
        self.spin_h.setValue(new_h)
        self.spin_w.blockSignals(False)
        self.spin_h.blockSignals(False)
        self._recalculate()

    def _on_width_changed(self, new_w: int):
        if self.chk_ratio.isChecked():
            ratio = self.original_img.height / self.original_img.width
            new_h = int(round(new_w * ratio))
            self.spin_h.blockSignals(True)
            self.spin_h.setValue(new_h)
            self.spin_h.blockSignals(False)
        self._recalculate()

    def _on_height_changed(self, new_h: int):
        if self.chk_ratio.isChecked():
            ratio = self.original_img.width / self.original_img.height
            new_w = int(round(new_h * ratio))
            self.spin_w.blockSignals(True)
            self.spin_w.setValue(new_w)
            self.spin_w.blockSignals(False)
        self._recalculate()

    def _recalculate(self):
        w = self.spin_w.value()
        h = self.spin_h.value()
        interp = "nearest" if "Nearest" in self.combo_interp.currentText() else "bilinear"
        self.result_img = resize_exact(self.original_img, w, h, interpolation=interp)
        self.previewUpdated.emit(self.result_img)

    def reject(self):
        self.previewUpdated.emit(self.original_img)
        super().reject()
