"""
Interactive Image Canvas for Mini Photoshop.
Provides smooth zooming, panning, pixel inspection, split-screen (Before/After) comparison,
and transparent checkerboard backdrop.
"""

from typing import Optional, Tuple
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap, QPen, QBrush, QWheelEvent, QMouseEvent, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from ..engine.core import ImageMatrix


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Converts a NumPy uint8 array (H, W), (H, W, 3), or (H, W, 4) to QImage."""
    if arr.ndim == 2:
        # Grayscale
        h, w = arr.shape
        # Format_Grayscale8 is available in Qt6
        # Make a contiguous copy
        arr_contig = np.ascontiguousarray(arr)
        qimg = QImage(arr_contig.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c == 3:
            arr_contig = np.ascontiguousarray(arr)
            qimg = QImage(arr_contig.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return qimg.copy()
        elif c == 4:
            arr_contig = np.ascontiguousarray(arr)
            qimg = QImage(arr_contig.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            return qimg.copy()
    raise ValueError(f"Cannot convert array with shape {arr.shape} to QImage")


class ImageCanvas(QWidget):
    """
    High-performance Photoshop-style interactive canvas.
    """
    pixelHovered = pyqtSignal(int, int, int, int, int)  # x, y, r, g, b
    zoomChanged = pyqtSignal(float)                     # zoom_factor (e.g. 1.0 = 100%)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Images
        self.image_matrix: Optional[ImageMatrix] = None
        self.original_matrix: Optional[ImageMatrix] = None
        self.qimage: Optional[QImage] = None
        self.original_qimage: Optional[QImage] = None

        # Viewport Transformations
        self.zoom_factor: float = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning: bool = False
        self.last_mouse_pos = QPointF(0, 0)

        # Split Screen Comparison Mode
        self.split_view_enabled: bool = False
        self.split_ratio: float = 0.5  # 0.0 to 1.0
        self.is_dragging_split: bool = False

        # Checkerboard background tile
        self._create_checker_pattern()

    def _create_checker_pattern(self):
        size = 16
        tile = QPixmap(size, size)
        painter = QPainter(tile)
        c1 = QColor("#1f1f1f")
        c2 = QColor("#282828")
        painter.fillRect(0, 0, size // 2, size // 2, c1)
        painter.fillRect(size // 2, 0, size // 2, size // 2, c2)
        painter.fillRect(0, size // 2, size // 2, size // 2, c2)
        painter.fillRect(size // 2, size // 2, size // 2, size // 2, c1)
        painter.end()
        self.checker_brush = QBrush(tile)

    def set_image(self, current_img: Optional[ImageMatrix], original_img: Optional[ImageMatrix] = None):
        """Updates the active image and optional original reference."""
        self.image_matrix = current_img
        if current_img is not None:
            self.qimage = numpy_to_qimage(current_img.array)
        else:
            self.qimage = None

        self.original_matrix = original_img
        if original_img is not None:
            self.original_qimage = numpy_to_qimage(original_img.array)
        else:
            self.original_qimage = None

        self.update()

    def fit_to_window(self):
        """Fits image to canvas area."""
        if self.image_matrix is None:
            return
        w_img, h_img = self.image_matrix.width, self.image_matrix.height
        w_view, h_view = self.width() - 40, self.height() - 40
        if w_img > 0 and h_img > 0 and w_view > 0 and h_view > 0:
            scale_x = w_view / w_img
            scale_y = h_view / h_img
            self.zoom_factor = min(scale_x, scale_y, 4.0)
            self.center_image()

    def set_actual_size(self):
        """Sets zoom to 100% (1:1)."""
        self.zoom_factor = 1.0
        self.center_image()

    def center_image(self):
        """Centers image in the canvas view."""
        if self.image_matrix is None:
            return
        disp_w = self.image_matrix.width * self.zoom_factor
        disp_h = self.image_matrix.height * self.zoom_factor
        self.pan_offset = QPointF(
            (self.width() - disp_w) / 2.0,
            (self.height() - disp_h) / 2.0
        )
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def set_split_view(self, enabled: bool):
        """Enables or disables Before/After split screen comparison."""
        self.split_view_enabled = enabled
        self.update()

    def _viewport_to_image_coords(self, pos: QPointF) -> Tuple[int, int]:
        """Converts canvas widget coordinates to pixel indices."""
        if self.image_matrix is None:
            return -1, -1
        img_x = int((pos.x() - self.pan_offset.x()) / self.zoom_factor)
        img_y = int((pos.y() - self.pan_offset.y()) / self.zoom_factor)
        return img_x, img_y

    def wheelEvent(self, event: QWheelEvent):
        """Smooth zooming around mouse cursor."""
        if self.image_matrix is None:
            return

        cursor_pos = event.position()
        old_zoom = self.zoom_factor
        delta = event.angleDelta().y()

        if delta > 0:
            new_zoom = min(32.0, self.zoom_factor * 1.15)
        else:
            new_zoom = max(0.05, self.zoom_factor / 1.15)

        if new_zoom != old_zoom:
            # Adjust pan offset so that the pixel under cursor stays at cursor
            scale_ratio = new_zoom / old_zoom
            self.pan_offset = cursor_pos - (cursor_pos - self.pan_offset) * scale_ratio
            self.zoom_factor = new_zoom
            self.zoomChanged.emit(self.zoom_factor)
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            # Check if clicking on split curtain handle
            if self.split_view_enabled and self.image_matrix is not None:
                disp_w = self.image_matrix.width * self.zoom_factor
                split_x = self.pan_offset.x() + disp_w * self.split_ratio
                if abs(event.position().x() - split_x) < 8:
                    self.is_dragging_split = True
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                    return

            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()

        # Handle split dragging
        if self.is_dragging_split and self.image_matrix is not None:
            disp_w = self.image_matrix.width * self.zoom_factor
            if disp_w > 0:
                rel_x = pos.x() - self.pan_offset.x()
                self.split_ratio = max(0.0, min(1.0, rel_x / disp_w))
                self.update()
            return

        # Handle panning
        if self.is_panning:
            delta = pos - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = pos
            self.update()
            return

        # Check split hover cursor
        if self.split_view_enabled and self.image_matrix is not None:
            disp_w = self.image_matrix.width * self.zoom_factor
            split_x = self.pan_offset.x() + disp_w * self.split_ratio
            if abs(pos.x() - split_x) < 8:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

        # Pixel inspection
        img_x, img_y = self._viewport_to_image_coords(pos)
        if self.image_matrix is not None and 0 <= img_x < self.image_matrix.width and 0 <= img_y < self.image_matrix.height:
            arr = self.image_matrix.array
            if arr.ndim == 2:
                v = int(arr[img_y, img_x])
                r, g, b = v, v, v
            else:
                r = int(arr[img_y, img_x, 0])
                g = int(arr[img_y, img_x, 1])
                b = int(arr[img_y, img_x, 2])
            self.pixelHovered.emit(img_x, img_y, r, g, b)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.is_panning = False
        self.is_dragging_split = False
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_factor < 2.0)

        # 1. Background Fill
        painter.fillRect(self.rect(), QColor("#121212"))

        if self.qimage is None or self.image_matrix is None:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image to start editing")
            return

        disp_w = self.image_matrix.width * self.zoom_factor
        disp_h = self.image_matrix.height * self.zoom_factor
        target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), disp_w, disp_h)

        # 2. Checkerboard Backdrop
        painter.fillRect(target_rect, self.checker_brush)

        # 3. Draw Image
        if self.split_view_enabled and self.original_qimage is not None:
            # Draw Original on Left, Current on Right
            split_x = self.pan_offset.x() + disp_w * self.split_ratio

            # Clip Left for Original
            painter.save()
            painter.setClipRect(QRectF(self.pan_offset.x(), self.pan_offset.y(), disp_w * self.split_ratio, disp_h))
            painter.drawImage(target_rect, self.original_qimage)
            painter.restore()

            # Clip Right for Current
            painter.save()
            painter.setClipRect(QRectF(split_x, self.pan_offset.y(), disp_w * (1.0 - self.split_ratio), disp_h))
            painter.drawImage(target_rect, self.qimage)
            painter.restore()

            # Draw Split Divider Line
            painter.setPen(QPen(QColor("#00bcd4"), 2))
            painter.drawLine(int(split_x), int(self.pan_offset.y()), int(split_x), int(self.pan_offset.y() + disp_h))

            # Labels (Original vs Processed)
            painter.setPen(QColor("#ffffff"))
            painter.fillRect(int(self.pan_offset.x() + 8), int(self.pan_offset.y() + 8), 60, 20, QColor(0, 0, 0, 160))
            painter.drawText(int(self.pan_offset.x() + 14), int(self.pan_offset.y() + 22), "Original")

            painter.fillRect(int(split_x + 8), int(self.pan_offset.y() + 8), 70, 20, QColor(0, 0, 0, 160))
            painter.drawText(int(split_x + 14), int(self.pan_offset.y() + 22), "Processed")

        else:
            painter.drawImage(target_rect, self.qimage)

        # 4. Canvas Border
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawRect(target_rect)
