"""
Image Information and Metadata Dialog for Mini Photoshop.
Displays resolution, color model, bit depth, file size, min/max intensity,
mean brightness, sharpness, noise estimation, and header info (as required by P5).
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from ...engine.core import ImageMatrix
from ...engine.metrics import compute_statistics


class ImageInfoDialog(QDialog):
    def __init__(self, img: ImageMatrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Properties & Metadata")
        self.setMinimumWidth(480)
        self.setMinimumHeight(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Tabs for Overview vs Raw Header Info
        tabs = QTabWidget()

        # 1. Overview Tab
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)

        stats = compute_statistics(img)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Property", "Value"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        rows = [
            ("File Name", img.metadata.filename),
            ("File Path", img.metadata.filepath or "Memory Canvas (Unsaved)"),
            ("Format", img.metadata.format_type),
            ("Dimensions (W x H)", f"{img.width} x {img.height} pixels"),
            ("Total Pixels", f"{img.width * img.height:,} px"),
            ("Channels", str(img.channels)),
            ("Color Space", img.color_mode),
            ("Bit Depth", f"{img.metadata.bit_depth}-bit"),
            ("Memory Size", stats["memory_size_kb"]),
            ("Min Pixel Value", str(stats["min_intensity"])),
            ("Max Pixel Value", str(stats["max_intensity"])),
            ("Mean Brightness", f"{stats['mean_intensity']:.2f}"),
            ("Standard Deviation", f"{stats['std_dev']:.2f}"),
            ("Sharpness (Laplacian Var)", f"{stats['sharpness_laplacian']:.2f}"),
            ("Estimated Noise (σ)", f"{stats['noise_estimate']:.2f}"),
        ]

        table.setRowCount(len(rows))
        for r_idx, (prop, val) in enumerate(rows):
            item_prop = QTableWidgetItem(prop)
            item_prop.setFlags(item_prop.flags() ^ Qt.ItemFlag.ItemIsEditable)
            item_val = QTableWidgetItem(val)
            item_val.setFlags(item_val.flags() ^ Qt.ItemFlag.ItemIsEditable)
            table.setItem(r_idx, 0, item_prop)
            table.setItem(r_idx, 1, item_val)

        overview_layout.addWidget(table)
        tabs.addTab(overview_widget, "General Properties")

        # 2. Raw Header Tab
        if img.metadata.raw_header_info:
            header_widget = QWidget()
            header_layout = QVBoxLayout(header_widget)
            header_table = QTableWidget()
            header_table.setColumnCount(2)
            header_table.setHorizontalHeaderLabels(["Header Key", "Value"])
            header_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header_table.verticalHeader().setVisible(False)

            header_rows = list(img.metadata.raw_header_info.items())
            header_table.setRowCount(len(header_rows))
            for r_idx, (k, v) in enumerate(header_rows):
                header_table.setItem(r_idx, 0, QTableWidgetItem(str(k)))
                header_table.setItem(r_idx, 1, QTableWidgetItem(str(v)))

            header_layout.addWidget(header_table)
            tabs.addTab(header_widget, "File Header Dump")

        layout.addWidget(tabs)

        # Close button
        btn_layout = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.setObjectName("primaryButton")
        btn_close.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
