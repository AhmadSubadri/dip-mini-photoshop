"""
RAW Image Import Modal Dialog for Mini Photoshop.
Allows specifying dimensions, channel count, header offset, and data types for raw binary image files.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt


class RawImportDialog(QDialog):
    def __init__(self, file_size_bytes: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import RAW Image")
        self.setMinimumWidth(360)
        self.file_size = file_size_bytes

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_lbl = QLabel(f"Raw File Size: {file_size_bytes:,} bytes")
        info_lbl.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        layout.addWidget(info_lbl)

        grp = QGroupBox("RAW Geometry Specification")
        grp_layout = QVBoxLayout(grp)

        # Width
        w_hdr = QHBoxLayout()
        w_hdr.addWidget(QLabel("Width (px):"))
        self.spin_w = QSpinBox()
        self.spin_w.setRange(1, 10000)
        self.spin_w.setValue(512)
        w_hdr.addWidget(self.spin_w)
        grp_layout.addLayout(w_hdr)

        # Height
        h_hdr = QHBoxLayout()
        h_hdr.addWidget(QLabel("Height (px):"))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(1, 10000)
        self.spin_h.setValue(512)
        h_hdr.addWidget(self.spin_h)
        grp_layout.addLayout(h_hdr)

        # Channels
        ch_hdr = QHBoxLayout()
        ch_hdr.addWidget(QLabel("Channels / Color Mode:"))
        self.combo_ch = QComboBox()
        self.combo_ch.addItems(["1 Channel (8-bit Grayscale)", "3 Channels (24-bit RGB planar/interleaved)"])
        ch_hdr.addWidget(self.combo_ch)
        grp_layout.addLayout(ch_hdr)

        # Header Offset
        off_hdr = QHBoxLayout()
        off_hdr.addWidget(QLabel("Header Offset (Bytes to skip):"))
        self.spin_off = QSpinBox()
        self.spin_off.setRange(0, 1000000)
        self.spin_off.setValue(0)
        off_hdr.addWidget(self.spin_off)
        grp_layout.addLayout(off_hdr)

        layout.addWidget(grp)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("Import RAW")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def get_parameters(self):
        ch = 1 if self.combo_ch.currentIndex() == 0 else 3
        return {
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
            "channels": ch,
            "bit_depth": 8,
            "header_offset": self.spin_off.value()
        }
