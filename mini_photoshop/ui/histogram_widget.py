"""
High-performance Interactive Histogram Widget for Mini Photoshop.
Renders RGB and Grayscale distributions with antialiased curves, tooltips, and channel filtering.
Supports Normal (raw counts), Normalized (probability h(i) = n(i)/N), and
Cumulative (CDF: P(i<=j) = Σ h(i)) display modes.
"""

from typing import Optional, Dict
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QFrame
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QFont
from PyQt6.QtCore import Qt
from ..engine.core import ImageMatrix
from ..engine.metrics import (
    compute_histograms, compute_normalized_histograms,
    compute_cumulative_histograms, compute_statistics
)


class HistogramCanvas(QWidget):
    """
    QPainter-based canvas rendering smooth histogram curves.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.histograms: Dict[str, np.ndarray] = {}                # raw counts (int)
        self.histograms_normalized: Dict[str, np.ndarray] = {}     # h(i) = n(i)/N (float)
        self.histograms_cumulative: Dict[str, np.ndarray] = {}     # CDF: P(i<=j) (float)
        self.display_mode: str = "normal"  # "normal" | "normalized" | "cumulative"
        self.selected_channel: str = "RGB"  # "RGB", "Red", "Green", "Blue", "Gray"
        self.hover_bin: Optional[int] = None
        self.hover_count: float = 0.0

    def set_data(self, img: Optional[ImageMatrix]):
        if img is None:
            self.histograms = {}
            self.histograms_normalized = {}
            self.histograms_cumulative = {}
        else:
            self.histograms = compute_histograms(img)
            self.histograms_normalized = compute_normalized_histograms(img)
            self.histograms_cumulative = compute_cumulative_histograms(img)
        self.update()

    def set_channel(self, channel: str):
        self.selected_channel = channel
        self.update()

    def set_display_mode(self, mode: str):
        """Set display mode: 'normal' | 'normalized' | 'cumulative'."""
        self.display_mode = mode
        self.update()

    def mouseMoveEvent(self, event):
        margin_l = 30
        margin_r = 10
        w = self.width() - margin_l - margin_r
        x = event.position().x() - margin_l
        if 0 <= x <= w and w > 0:
            bin_idx = int(round((x / w) * 255.0))
            self.hover_bin = max(0, min(255, bin_idx))
            self.hover_count = 0.0
            # Read from whichever histogram dict is currently active
            if self.display_mode == "cumulative":
                active_hists = self.histograms_cumulative
            elif self.display_mode == "normalized":
                active_hists = self.histograms_normalized
            else:
                active_hists = self.histograms
            for ch, h in active_hists.items():
                if self.hover_bin < len(h):
                    self.hover_count = max(self.hover_count, float(h[self.hover_bin]))
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

        # Select the active histogram dict based on the current display mode
        if self.display_mode == "cumulative":
            active_hists = self.histograms_cumulative
        elif self.display_mode == "normalized":
            active_hists = self.histograms_normalized
        else:
            active_hists = self.histograms

        # Determine maximum bin value for Y-axis scaling.
        # Cumulative and normalized modes are bounded by 1.0; raw mode scales to its peak.
        if self.display_mode in ("normalized", "cumulative"):
            max_val = 1.0
        else:
            max_val = 1
            for name, h in active_hists.items():
                if name != 'Luminance' or 'Gray' in active_hists:
                    max_val = max(max_val, float(np.max(h)))

        # Channels to draw (channel selection logic always uses raw histogram keys)
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

        # Render Curves from active histogram dict
        for ch_key, fill_color, stroke_color in channels_to_draw:
            if ch_key not in active_hists:
                continue
            hist = active_hists[ch_key]

            path = QPainterPath()
            path.moveTo(margin_l, margin_t + plot_h)

            for b in range(256):
                x = margin_l + (b / 255.0) * plot_w
                val = float(hist[b]) if b < len(hist) else 0.0
                y = margin_t + plot_h - (val / max_val) * plot_h
                path.lineTo(x, y)

            path.lineTo(margin_l + plot_w, margin_t + plot_h)
            path.closeSubpath()

            # Fill
            painter.fillPath(path, QBrush(fill_color))
            # Outline
            painter.setPen(QPen(stroke_color, 1.5))
            line_path = QPainterPath()
            for b in range(256):
                x = margin_l + (b / 255.0) * plot_w
                val = float(hist[b]) if b < len(hist) else 0.0
                y = margin_t + plot_h - (val / max_val) * plot_h
                if b == 0:
                    line_path.moveTo(x, y)
                else:
                    line_path.lineTo(x, y)
            painter.drawPath(line_path)

        # X-axis labels (0, 128, 255)
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

            # Tooltip: format depends on active mode
            if self.display_mode == "cumulative":
                tip_text = f"Int: {self.hover_bin} | P(i\u2264j): {self.hover_count:.4f}"
            elif self.display_mode == "normalized":
                tip_text = f"Int: {self.hover_bin} | h(i): {self.hover_count:.4f}"
            else:
                tip_text = f"Int: {self.hover_bin} | Count: {int(self.hover_count):,}"
            painter.setPen(QColor("#00bcd4"))
            painter.drawText(margin_l + 4, margin_t + 14, tip_text)


class HistogramWidget(QWidget):
    """
    Full Histogram Panel with channel combo selector, Normal/Normalized/Cumulative mode toggle,
    live histogram canvas, and statistical summary (Mean, Variance, Std Dev).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Row 1: Title + Mode toggle + Channel selector ──────────────────
        top_bar = QHBoxLayout()
        lbl = QLabel("Histogram")
        lbl.setStyleSheet("font-weight: bold; color: #ffffff;")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normal", "Normalized", "Cumulative"])
        self.mode_combo.setToolTip(
            "Normal: raw pixel counts\n"
            "Normalized: h(i) = n(i) / N  (probability, range [0, 1])\n"
            "Cumulative: P(i\u2264j) = \u03a3 h(i)  (CDF, range [0, 1])"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["RGB", "Red", "Green", "Blue", "Luminance / Gray"])
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)

        top_bar.addWidget(lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.mode_combo)
        top_bar.addWidget(self.channel_combo)
        layout.addLayout(top_bar)

        # ── Canvas ─────────────────────────────────────────────────────────
        self.canvas = HistogramCanvas(self)
        layout.addWidget(self.canvas)

        # ── Separator ──────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3c3c3c;")
        layout.addWidget(sep)

        # ── Row 2: Statistics labels (Mean, Variance, Std Dev) ─────────────
        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(8)

        self.lbl_mean = QLabel("Mean (μ): —")
        self.lbl_var  = QLabel("Variance (σ²): —")
        self.lbl_std  = QLabel("Std Dev (σ): —")

        stat_style = "color: #cccccc; font-size: 10px;"
        for lbl_s in (self.lbl_mean, self.lbl_var, self.lbl_std):
            lbl_s.setStyleSheet(stat_style)
            stats_bar.addWidget(lbl_s)

        stats_bar.addStretch()
        layout.addLayout(stats_bar)

    # ── Public API ─────────────────────────────────────────────────────────

    def update_image(self, img: Optional[ImageMatrix]):
        """Refresh the histogram canvas and statistics labels for the given image."""
        self.canvas.set_data(img)
        if img is None:
            self.lbl_mean.setText("Mean (μ): —")
            self.lbl_var.setText("Variance (σ²): —")
            self.lbl_std.setText("Std Dev (σ): —")
        else:
            stats = compute_statistics(img)
            self.lbl_mean.setText(f"Mean (μ): {stats['mean_intensity']:.2f}")
            self.lbl_var.setText(f"Variance (σ²): {stats['variance']:.2f}")
            self.lbl_std.setText(f"Std Dev (σ): {stats['std_dev']:.2f}")

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_mode_changed(self, text: str):
        if text == "Cumulative":
            self.canvas.set_display_mode("cumulative")
        elif text == "Normalized":
            self.canvas.set_display_mode("normalized")
        else:
            self.canvas.set_display_mode("normal")

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
