"""
High-performance Interactive Histogram Widget for Mini Photoshop.
Renders RGB and Grayscale distributions with antialiased curves, tooltips, and channel filtering.
"""

from typing import Optional, Dict
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QLinearGradient, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF
from ..engine.core import ImageMatrix
from ..engine.metrics import compute_histograms


class HistogramCanvas(QWidget):
    """
    QPainter-based canvas rendering smooth histogram curves.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.histograms: Dict[str, np.ndarray] = {}
        self.selected_channel: str = "RGB"  # "RGB", "Red", "Green", "Blue", "Gray"
        self.hover_bin: Optional[int] = None
        self.hover_count: int = 0

    def set_data(self, img: Optional[ImageMatrix]):
        if img is None:
            self.histograms = {}
        else:
            self.histograms = compute_histograms(img)
        self.update()

    def set_channel(self, channel: str):
        self.selected_channel = channel
        self.update()

    def mouseMoveEvent(self, event):
        margin_l = 30
        margin_r = 10
        w = self.width() - margin_l - margin_r
        x = event.position().x() - margin_l
        if 0 <= x <= w and w > 0:
            bin_idx = int(round((x / w) * 255.0))
            self.hover_bin = max(0, min(255, bin_idx))
            self.hover_count = 0
            # Get max count in that bin
            for ch, h in self.histograms.items():
                if self.hover_bin < len(h):
                    self.hover_count = max(self.hover_count, int(h[self.hover_bin]))
        else:
            self.hover_bin = None
        self.update()

    def leaveEvent(self, event):
        self.hover_bin = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#141414"))

        if not self.histograms:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        margin_l = 32
        margin_r = 12
        margin_t = 12
        margin_b = 24

        plot_w = max(1, self.width() - margin_l - margin_r)
        plot_h = max(1, self.height() - margin_t - margin_b)

        # Draw Gridlines & Axes
        painter.setPen(QPen(QColor("#2c2c2c"), 1, Qt.PenStyle.DashLine))
        for i in range(1, 4):
            y_pos = margin_t + int(plot_h * (i / 4.0))
            painter.drawLine(margin_l, y_pos, margin_l + plot_w, y_pos)
            
        for i in range(1, 4):
            x_pos = margin_l + int(plot_w * (i / 4.0))
            painter.drawLine(x_pos, margin_t, x_pos, margin_t + plot_h)

        # Draw Border
        painter.setPen(QPen(QColor("#3c3c3c"), 1))
        painter.drawRect(margin_l, margin_t, plot_w, plot_h)

        # Determine Maximum bin count for scaling
        max_val = 1
        for name, h in self.histograms.items():
            if name != 'Luminance' or 'Gray' in self.histograms:
                max_val = max(max_val, np.max(h))

        # Channels to draw
        channels_to_draw = []
        if self.selected_channel == "RGB":
            if "R" in self.histograms:
                channels_to_draw = [("B", QColor(40, 120, 255, 140), QColor(40, 120, 255)),
                                    ("G", QColor(40, 220, 80, 140), QColor(40, 220, 80)),
                                    ("R", QColor(255, 50, 50, 140), QColor(255, 50, 50))]
            elif "Gray" in self.histograms:
                channels_to_draw = [("Gray", QColor(200, 200, 200, 140), QColor(220, 220, 220))]
        elif self.selected_channel == "Red" and "R" in self.histograms:
            channels_to_draw = [("R", QColor(255, 50, 50, 160), QColor(255, 70, 70))]
        elif self.selected_channel == "Green" and "G" in self.histograms:
            channels_to_draw = [("G", QColor(40, 220, 80, 160), QColor(60, 240, 100))]
        elif self.selected_channel == "Blue" and "B" in self.histograms:
            channels_to_draw = [("B", QColor(40, 120, 255, 160), QColor(60, 150, 255))]
        elif "Gray" in self.histograms:
            channels_to_draw = [("Gray", QColor(180, 180, 180, 160), QColor(220, 220, 220))]
        elif "Luminance" in self.histograms:
            channels_to_draw = [("Luminance", QColor(180, 180, 180, 160), QColor(220, 220, 220))]

        # Render Curves
        for ch_key, fill_color, stroke_color in channels_to_draw:
            if ch_key not in self.histograms:
                continue
            hist = self.histograms[ch_key]
            
            path = QPainterPath()
            path.moveTo(margin_l, margin_t + plot_h)
            
            for b in range(256):
                x = margin_l + (b / 255.0) * plot_w
                val = hist[b] if b < len(hist) else 0
                y = margin_t + plot_h - (val / max_val) * plot_h
                path.lineTo(x, y)
                
            path.lineTo(margin_l + plot_w, margin_t + plot_h)
            path.closeSubpath()

            # Fill
            painter.fillPath(path, QBrush(fill_color))
            # Outline
            painter.setPen(QPen(stroke_color, 1.5))
            # Draw line only (without bottom closing)
            line_path = QPainterPath()
            for b in range(256):
                x = margin_l + (b / 255.0) * plot_w
                val = hist[b] if b < len(hist) else 0
                y = margin_t + plot_h - (val / max_val) * plot_h
                if b == 0:
                    line_path.moveTo(x, y)
                else:
                    line_path.lineTo(x, y)
            painter.drawPath(line_path)

        # Labels (0, 128, 255)
        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(margin_l - 2, margin_t + plot_h + 14, "0")
        painter.drawText(margin_l + plot_w // 2 - 8, margin_t + plot_h + 14, "128")
        painter.drawText(margin_l + plot_w - 18, margin_t + plot_h + 14, "255")

        # Hover Marker
        if self.hover_bin is not None:
            x_hover = margin_l + (self.hover_bin / 255.0) * plot_w
            painter.setPen(QPen(QColor("#ffffff"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(x_hover), margin_t, int(x_hover), margin_t + plot_h)
            
            # Info box
            tip_text = f"Int: {self.hover_bin} | Count: {self.hover_count:,}"
            painter.setPen(QColor("#00bcd4"))
            painter.drawText(margin_l + 4, margin_t + 14, tip_text)


class HistogramWidget(QWidget):
    """
    Full Histogram Panel with channel combo selector and live canvas.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header controls
        top_bar = QHBoxLayout()
        lbl = QLabel("Histogram")
        lbl.setStyleSheet("font-weight: bold; color: #ffffff;")
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["RGB", "Red", "Green", "Blue", "Luminance / Gray"])
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        
        top_bar.addWidget(lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.channel_combo)
        layout.addLayout(top_bar)

        # Canvas
        self.canvas = HistogramCanvas(self)
        layout.addWidget(self.canvas)

    def update_image(self, img: Optional[ImageMatrix]):
        self.canvas.set_data(img)

    def _on_channel_changed(self, text: str):
        if "Red" in text:
            self.canvas.set_channel("Red")
        elif "Green" in text:
            self.canvas.set_channel("Green")
        elif "Blue" in text:
            self.canvas.set_channel("Blue")
        elif "Luminance" in text or "Gray" in text:
            self.canvas.set_channel("Gray")
        else:
            self.canvas.set_channel("RGB")
