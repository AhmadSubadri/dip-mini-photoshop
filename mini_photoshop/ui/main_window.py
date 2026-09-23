"""
Main Window for Mini Photoshop.
Provides full Photoshop-like Dark Interface, multi-tab canvas, dockable real-time histogram,
properties inspector, status bar with pixel coordinate/RGB inspector, and comprehensive P1-P6 image processing menus.
"""

import os
from typing import Dict, Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QMenu, QToolBar, QStatusBar, QLabel, QFileDialog,
    QMessageBox, QDockWidget, QScrollArea, QDialog, QInputDialog,
    QPushButton, QFrame
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QFont
from PyQt6.QtCore import Qt, QSize

from ..engine.core import ImageMatrix, DocumentState, ImageMetadata
from ..engine.io_custom import load_image_file, save_image_file, read_raw
from ..engine.point_ops import (
    invert, to_grayscale_average, to_grayscale_luminance,
    adjust_brightness, contrast_stretching, threshold_otsu, solarize
)
from ..engine.boolean_ops import bitwise_not
from ..engine.geometry_ops import (
    rotate_orthogonal, flip_horizontal, flip_vertical
)
from ..engine.metrics import compute_statistics
from .canvas import ImageCanvas
from .histogram_widget import HistogramWidget
from .dialogs import (
    BrightnessContrastDialog, ThresholdDialog, GammaDialog,
    PosterizeDialog, ArithmeticOperationDialog, RotateDialog,
    TranslateDialog, ScaleDialog, ImageInfoDialog, RawImportDialog
)
from .styles import PHOTOSHOP_DARK_THEME


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Photoshop - Pengolahan & Analisis Citra Digital")
        self.resize(1280, 800)
        self.setStyleSheet(PHOTOSHOP_DARK_THEME)

        # Multi-document state storage: tab_index -> DocumentState
        self.documents: Dict[int, DocumentState] = {}

        # Setup Components
        self._init_ui()
        self._init_docks()
        self._init_menus()
        self._init_toolbar()
        self._init_statusbar()

        # Load an initial sample if available
        self._load_default_startup_sample()

    # ==========================================================================
    # UI Setup
    # ==========================================================================

    def _init_ui(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tab_widget)

    def _init_docks(self):
        # 1. Histogram Dock
        self.hist_dock = QDockWidget("Histogram", self)
        self.hist_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.hist_widget = HistogramWidget(self)
        self.hist_dock.setWidget(self.hist_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.hist_dock)

        # 2. Quick Properties & Inspector Dock
        self.props_dock = QDockWidget("Image Inspector", self)
        self.props_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        self.props_container = QWidget()
        self.props_layout = QVBoxLayout(self.props_container)
        self.props_layout.setContentsMargins(10, 10, 10, 10)
        self.props_layout.setSpacing(6)

        self.lbl_prop_dim = QLabel("Dimensions: -")
        self.lbl_prop_mode = QLabel("Color Mode: -")
        self.lbl_prop_depth = QLabel("Bit Depth: -")
        self.lbl_prop_mean = QLabel("Mean Intensity: -")
        self.lbl_prop_minmax = QLabel("Min / Max: -")
        self.lbl_prop_sharp = QLabel("Sharpness: -")

        for lbl in [
            self.lbl_prop_dim, self.lbl_prop_mode, self.lbl_prop_depth,
            self.lbl_prop_mean, self.lbl_prop_minmax, self.lbl_prop_sharp
        ]:
            lbl.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            self.props_layout.addWidget(lbl)

        self.props_layout.addStretch()
        self.props_dock.setWidget(self.props_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

    def _init_statusbar(self):
        self.status_bar = self.statusBar()
        
        self.lbl_status_msg = QLabel("Ready")
        self.lbl_status_pos = QLabel("X: -  Y: -")
        self.lbl_status_rgb = QLabel("RGB: -")
        self.lbl_status_zoom = QLabel("Zoom: 100%")
        self.lbl_status_dim = QLabel("Size: -")

        self.status_bar.addWidget(self.lbl_status_msg, 1)
        self.status_bar.addPermanentWidget(self.lbl_status_pos)
        self.status_bar.addPermanentWidget(self.lbl_status_rgb)
        self.status_bar.addPermanentWidget(self.lbl_status_zoom)
        self.status_bar.addPermanentWidget(self.lbl_status_dim)

    # ==========================================================================
    # Menu Bar Setup
    # ==========================================================================

    def _init_menus(self):
        mb = self.menuBar()

        # 1. FILE MENU
        file_menu = mb.addMenu("&File")
        
        act_open = QAction("&Open Image...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.action_open_file)
        file_menu.addAction(act_open)

        act_raw = QAction("Import &RAW Image...", self)
        act_raw.triggered.connect(self.action_import_raw)
        file_menu.addAction(act_raw)

        # Samples Submenu
        samples_menu = file_menu.addMenu("Open &Sample Image")
        self._populate_samples_menu(samples_menu)

        file_menu.addSeparator()

        act_save = QAction("&Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.action_save_file)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save &As... (Export PBM/PGM/PPM/BMP/PNG/JPG)", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self.action_save_as_file)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_info = QAction("Image &Properties / Metadata...", self)
        act_info.setShortcut("Ctrl+I")
        act_info.triggered.connect(self.action_image_info)
        file_menu.addAction(act_info)

        act_close = QAction("&Close Image", self)
        act_close.setShortcut(QKeySequence.StandardKey.Close)
        act_close.triggered.connect(lambda: self._close_tab(self.tab_widget.currentIndex()))
        file_menu.addAction(act_close)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # 2. EDIT MENU
        edit_menu = mb.addMenu("&Edit")

        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.action_undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("&Redo", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.action_redo)
        edit_menu.addAction(self.act_redo)

        edit_menu.addSeparator()

        act_reset = QAction("Revert to &Original", self)
        act_reset.triggered.connect(self.action_revert_original)
        edit_menu.addAction(act_reset)

        # 3. VIEW MENU
        view_menu = mb.addMenu("&View")

        act_zoom_in = QAction("Zoom &In", self)
        act_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        act_zoom_in.triggered.connect(self.action_zoom_in)
        view_menu.addAction(act_zoom_in)

        act_zoom_out = QAction("Zoom &Out", self)
        act_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        act_zoom_out.triggered.connect(self.action_zoom_out)
        view_menu.addAction(act_zoom_out)

        act_fit = QAction("&Fit on Screen", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self.action_fit_screen)
        view_menu.addAction(act_fit)

        act_100 = QAction("&Actual Size (100%)", self)
        act_100.setShortcut("Ctrl+1")
        act_100.triggered.connect(self.action_actual_size)
        view_menu.addAction(act_100)

        view_menu.addSeparator()

        self.act_split = QAction("&Split View (Before / After Comparison)", self)
        self.act_split.setCheckable(True)
        self.act_split.setShortcut("Ctrl+T")
        self.act_split.toggled.connect(self.action_toggle_split_view)
        view_menu.addAction(self.act_split)

        view_menu.addSeparator()
        view_menu.addAction(self.hist_dock.toggleViewAction())
        view_menu.addAction(self.props_dock.toggleViewAction())

        # 4. IMAGE MENU (Aras Titik / Point Operations)
        image_menu = mb.addMenu("&Image")

        act_neg = QAction("Citra &Negatif / Invert", self)
        act_neg.triggered.connect(self.action_invert)
        image_menu.addAction(act_neg)

        gray_menu = image_menu.addMenu("Konversi ke &Grayscale")
        act_gray_avg = QAction("Metode Rata-rata: (R+G+B)/3", self)
        act_gray_avg.triggered.connect(self.action_grayscale_average)
        gray_menu.addAction(act_gray_avg)

        act_gray_lum = QAction("Metode Luminansi (NTSC): 0.299R + 0.587G + 0.114B", self)
        act_gray_lum.triggered.connect(self.action_grayscale_luminance)
        gray_menu.addAction(act_gray_lum)

        image_menu.addSeparator()

        act_thresh = QAction("&Thresholding / Binarization (Manual & Otsu)...", self)
        act_thresh.triggered.connect(self.action_threshold_dialog)
        image_menu.addAction(act_thresh)

        act_auto_otsu = QAction("Auto &Otsu Thresholding", self)
        act_auto_otsu.triggered.connect(self.action_otsu_direct)
        image_menu.addAction(act_auto_otsu)

        image_menu.addSeparator()

        act_gamma = QAction("&Gamma Correction (Power Law)...", self)
        act_gamma.triggered.connect(self.action_gamma_dialog)
        image_menu.addAction(act_gamma)

        act_poster = QAction("Bit-Depth &Posterization (Quantization)...", self)
        act_poster.triggered.connect(self.action_posterize_dialog)
        image_menu.addAction(act_poster)

        act_solar = QAction("&Solarize Effect...", self)
        act_solar.triggered.connect(self.action_solarize)
        image_menu.addAction(act_solar)

        # 5. ADJUSTMENTS MENU
        adj_menu = mb.addMenu("&Adjustments")

        act_bc = QAction("&Brightness & Contrast...", self)
        act_bc.triggered.connect(self.action_brightness_contrast)
        adj_menu.addAction(act_bc)

        act_cstretch = QAction("Linear &Contrast Stretching (Full Dynamic Range)", self)
        act_cstretch.triggered.connect(self.action_contrast_stretch)
        adj_menu.addAction(act_cstretch)

        adj_menu.addSeparator()

        act_bright_p20 = QAction("Quick Brighten (+20)", self)
        act_bright_p20.triggered.connect(lambda: self._apply_quick_point_op("Quick Brighten +20", lambda img: adjust_brightness(img, 20)))
        adj_menu.addAction(act_bright_p20)

        act_bright_m20 = QAction("Quick Darken (-20)", self)
        act_bright_m20.triggered.connect(lambda: self._apply_quick_point_op("Quick Darken -20", lambda img: adjust_brightness(img, -20)))
        adj_menu.addAction(act_bright_m20)

        # 6. OPERATIONS (Arithmetic & Boolean)
        op_menu = mb.addMenu("&Operations")

        act_arith_diag = QAction("&2-Image Arithmetic & Logic Dialog...", self)
        act_arith_diag.triggered.connect(self.action_arithmetic_dialog)
        op_menu.addAction(act_arith_diag)

        op_menu.addSeparator()

        act_not = QAction("Bitwise &NOT (~A)", self)
        act_not.triggered.connect(lambda: self._apply_quick_point_op("Bitwise NOT", bitwise_not))
        op_menu.addAction(act_not)

        # 7. TRANSFORM / GEOMETRY
        geom_menu = mb.addMenu("&Transform")

        act_rot_cw = QAction("Rotate 90° &Clockwise", self)
        act_rot_cw.triggered.connect(lambda: self._apply_quick_point_op("Rotate 90° CW", lambda img: rotate_orthogonal(img, 90)))
        geom_menu.addAction(act_rot_cw)

        act_rot_ccw = QAction("Rotate 90° &Counter-Clockwise", self)
        act_rot_ccw.triggered.connect(lambda: self._apply_quick_point_op("Rotate 90° CCW", lambda img: rotate_orthogonal(img, 270)))
        geom_menu.addAction(act_rot_ccw)

        act_rot_180 = QAction("Rotate &180°", self)
        act_rot_180.triggered.connect(lambda: self._apply_quick_point_op("Rotate 180°", lambda img: rotate_orthogonal(img, 180)))
        geom_menu.addAction(act_rot_180)

        act_rot_arb = QAction("&Arbitrary Angle Rotation...", self)
        act_rot_arb.triggered.connect(self.action_rotate_dialog)
        geom_menu.addAction(act_rot_arb)

        geom_menu.addSeparator()

        act_fliph = QAction("Flip &Horizontal (Pencerminan)", self)
        act_fliph.triggered.connect(lambda: self._apply_quick_point_op("Flip Horizontal", flip_horizontal))
        geom_menu.addAction(act_fliph)

        act_flipv = QAction("Flip &Vertical (Pencerminan)", self)
        act_flipv.triggered.connect(lambda: self._apply_quick_point_op("Flip Vertical", flip_vertical))
        geom_menu.addAction(act_flipv)

        geom_menu.addSeparator()

        act_trans = QAction("&Translate / Shift Coordinates...", self)
        act_trans.triggered.connect(self.action_translate_dialog)
        geom_menu.addAction(act_trans)

        act_scale = QAction("&Scale / Resize Image...", self)
        act_scale.triggered.connect(self.action_scale_dialog)
        geom_menu.addAction(act_scale)

        # 8. HELP MENU
        help_menu = mb.addMenu("&Help")

        act_guide = QAction("Materi Acuan P1 - P6", self)
        act_guide.triggered.connect(self.action_show_guide)
        help_menu.addAction(act_guide)

        act_about = QAction("About &Mini Photoshop", self)
        act_about.triggered.connect(self.action_about)
        help_menu.addAction(act_about)

    def _populate_samples_menu(self, menu: QMenu):
        samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
        if not os.path.exists(samples_dir):
            return

        for fname in sorted(os.listdir(samples_dir)):
            full_path = os.path.join(samples_dir, fname)
            if os.path.isfile(full_path):
                act = QAction(fname, self)
                act.triggered.connect(lambda _, p=full_path: self.open_image_path(p))
                menu.addAction(act)

    # ==========================================================================
    # Tool Bar Setup
    # ==========================================================================

    def _init_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        tb.addAction("Open", self.action_open_file)
        tb.addAction("Save", self.action_save_file)
        tb.addSeparator()

        tb.addAction("Undo", self.action_undo)
        tb.addAction("Redo", self.action_redo)
        tb.addSeparator()

        tb.addAction("Zoom In (+)", self.action_zoom_in)
        tb.addAction("Zoom Out (-)", self.action_zoom_out)
        tb.addAction("Fit", self.action_fit_screen)
        tb.addAction("1:1", self.action_actual_size)
        tb.addSeparator()

        tb.addAction(self.act_split)
        tb.addSeparator()

        tb.addAction("Brightness/Contrast", self.action_brightness_contrast)
        tb.addAction("Grayscale", self.action_grayscale_luminance)
        tb.addAction("Invert", self.action_invert)
        tb.addAction("Info", self.action_image_info)

    # ==========================================================================
    # Tab & Document Management
    # ==========================================================================

    def _get_active_doc(self) -> Optional[DocumentState]:
        idx = self.tab_widget.currentIndex()
        return self.documents.get(idx, None)

    def _get_active_canvas(self) -> Optional[ImageCanvas]:
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, ImageCanvas):
            return widget
        return None

    def add_image_tab(self, img: ImageMatrix):
        doc_state = DocumentState(img)
        canvas = ImageCanvas()
        
        # Connect canvas signals
        canvas.pixelHovered.connect(self._on_pixel_hovered)
        canvas.zoomChanged.connect(self._on_zoom_changed)

        tab_idx = self.tab_widget.addTab(canvas, img.metadata.filename)
        self.documents[tab_idx] = doc_state

        self.tab_widget.setCurrentIndex(tab_idx)
        canvas.set_image(doc_state.current, doc_state.original)
        canvas.fit_to_window()
        self._update_panels_for_active_doc()

    def _close_tab(self, index: int):
        if index < 0 or index >= self.tab_widget.count():
            return
        self.tab_widget.removeTab(index)
        
        # Re-index document dictionary
        new_docs = {}
        curr_i = 0
        for i, doc in sorted(self.documents.items()):
            if i != index:
                new_docs[curr_i] = doc
                curr_i += 1
        self.documents = new_docs
        self._update_panels_for_active_doc()

    def _on_tab_changed(self, index: int):
        self._update_panels_for_active_doc()

    def _update_panels_for_active_doc(self):
        doc = self._get_active_doc()
        canvas = self._get_active_canvas()

        if doc is None or canvas is None:
            self.hist_widget.update_image(None)
            self.lbl_prop_dim.setText("Dimensions: -")
            self.lbl_prop_mode.setText("Color Mode: -")
            self.lbl_prop_depth.setText("Bit Depth: -")
            self.lbl_prop_mean.setText("Mean Intensity: -")
            self.lbl_prop_minmax.setText("Min / Max: -")
            self.lbl_prop_sharp.setText("Sharpness: -")
            self.lbl_status_dim.setText("Size: -")
            self.lbl_status_zoom.setText("Zoom: -")
            self.lbl_status_msg.setText("No open image")
            return

        # Update Canvas
        canvas.set_image(doc.current, doc.original)
        
        # Update Histogram
        self.hist_widget.update_image(doc.current)

        # Update Inspector Stats
        stats = compute_statistics(doc.current)
        self.lbl_prop_dim.setText(f"Dimensions: {doc.current.width} x {doc.current.height}")
        self.lbl_prop_mode.setText(f"Color Space: {doc.current.color_mode}")
        self.lbl_prop_depth.setText(f"Bit Depth: {doc.current.metadata.bit_depth}-bit")
        self.lbl_prop_mean.setText(f"Mean Intensity: {stats['mean_intensity']:.2f}")
        self.lbl_prop_minmax.setText(f"Min: {stats['min_intensity']} | Max: {stats['max_intensity']}")
        self.lbl_prop_sharp.setText(f"Sharpness: {stats['sharpness_laplacian']:.2f}")

        # Update Status Bar
        self.lbl_status_dim.setText(f"{doc.current.width}x{doc.current.height} ({doc.current.color_mode})")
        self.lbl_status_zoom.setText(f"Zoom: {int(canvas.zoom_factor * 100)}%")
        self.lbl_status_msg.setText(f"Active: {doc.current.metadata.filename}")

    def _on_pixel_hovered(self, x: int, y: int, r: int, g: int, b: int):
        self.lbl_status_pos.setText(f"X: {x}  Y: {y}")
        self.lbl_status_rgb.setText(f"R: {r} G: {g} B: {b} (Int: {int((r+g+b)/3)})")

    def _on_zoom_changed(self, zoom_factor: float):
        self.lbl_status_zoom.setText(f"Zoom: {int(zoom_factor * 100)}%")

    # ==========================================================================
    # Action Handlers: File Operations
    # ==========================================================================

    def open_image_path(self, filepath: str):
        try:
            img = load_image_file(filepath)
            self.add_image_tab(img)
            self.lbl_status_msg.setText(f"Opened: {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.critical(self, "Error Opening Image", f"Failed to load image:\n{e}")

    def action_open_file(self):
        filter_str = (
            "All Supported Images (*.png *.jpg *.jpeg *.bmp *.pbm *.pgm *.ppm *.raw *.tif *.tiff *.gif *.webp);;"
            "Netpbm Images (*.pbm *.pgm *.ppm);;"
            "Bitmap Files (*.bmp);;"
            "RAW Binary Files (*.raw);;"
            "PNG / JPEG Images (*.png *.jpg *.jpeg);;"
            "All Files (*.*)"
        )
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", filter_str)
        if filepath:
            self.open_image_path(filepath)

    def action_import_raw(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select RAW File", "", "RAW Files (*.raw);;All Files (*.*)")
        if not filepath:
            return
        
        file_size = os.path.getsize(filepath)
        dlg = RawImportDialog(file_size, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = dlg.get_parameters()
            try:
                img = read_raw(
                    filepath,
                    width=params["width"],
                    height=params["height"],
                    channels=params["channels"],
                    bit_depth=params["bit_depth"],
                    header_offset=params["header_offset"]
                )
                self.add_image_tab(img)
            except Exception as e:
                QMessageBox.critical(self, "RAW Import Failed", str(e))

    def action_save_file(self):
        doc = self._get_active_doc()
        if doc is None:
            return
        if doc.current.metadata.filepath:
            try:
                save_image_file(doc.current, doc.current.metadata.filepath)
                self.lbl_status_msg.setText(f"Saved: {doc.current.metadata.filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))
        else:
            self.action_save_as_file()

    def action_save_as_file(self):
        doc = self._get_active_doc()
        if doc is None:
            return

        filter_str = (
            "PNG Image (*.png);;"
            "JPEG Image (*.jpg);;"
            "Bitmap (*.bmp);;"
            "Portable Graymap (*.pgm);;"
            "Portable Pixmap (*.ppm);;"
            "Portable Bitmap (*.pbm);;"
            "RAW Binary (*.raw);;"
            "All Files (*.*)"
        )
        default_name = doc.current.metadata.filename
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Image As", default_name, filter_str)
        if filepath:
            try:
                save_image_file(doc.current, filepath)
                doc.current.metadata.filepath = filepath
                doc.current.metadata.filename = os.path.basename(filepath)
                self.tab_widget.setTabText(self.tab_widget.currentIndex(), doc.current.metadata.filename)
                self.lbl_status_msg.setText(f"Saved As: {doc.current.metadata.filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))

    def action_image_info(self):
        doc = self._get_active_doc()
        if doc is None:
            return
        dlg = ImageInfoDialog(doc.current, self)
        dlg.exec()

    # ==========================================================================
    # Action Handlers: Undo / Redo / Revert
    # ==========================================================================

    def action_undo(self):
        doc = self._get_active_doc()
        if doc and doc.can_undo():
            act_name = doc.undo()
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText(f"Undo: {act_name}")

    def action_redo(self):
        doc = self._get_active_doc()
        if doc and doc.can_redo():
            act_name = doc.redo()
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText(f"Redo: {act_name}")

    def action_revert_original(self):
        doc = self._get_active_doc()
        if doc:
            doc.reset_to_original()
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText("Reverted to original image")

    # ==========================================================================
    # Action Handlers: View & Zoom
    # ==========================================================================

    def action_zoom_in(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.zoom_factor = min(32.0, canvas.zoom_factor * 1.25)
            canvas.zoomChanged.emit(canvas.zoom_factor)
            canvas.center_image()

    def action_zoom_out(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.zoom_factor = max(0.05, canvas.zoom_factor / 1.25)
            canvas.zoomChanged.emit(canvas.zoom_factor)
            canvas.center_image()

    def action_fit_screen(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.fit_to_window()

    def action_actual_size(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.set_actual_size()

    def action_toggle_split_view(self, checked: bool):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.set_split_view(checked)

    # ==========================================================================
    # Action Handlers: Point Operations (P6)
    # ==========================================================================

    def _apply_quick_point_op(self, action_name: str, func):
        doc = self._get_active_doc()
        if doc is None:
            return
        try:
            res = func(doc.current)
            doc.push_state(action_name, res)
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText(f"Applied: {action_name}")
        except Exception as e:
            QMessageBox.critical(self, "Operation Error", str(e))

    def action_invert(self):
        self._apply_quick_point_op("Citra Negatif (Invert)", invert)

    def action_grayscale_average(self):
        self._apply_quick_point_op("Grayscale (Average)", to_grayscale_average)

    def action_grayscale_luminance(self):
        self._apply_quick_point_op("Grayscale (Luminance NTSC)", to_grayscale_luminance)

    def action_contrast_stretch(self):
        self._apply_quick_point_op("Linear Contrast Stretching", contrast_stretching)

    def action_otsu_direct(self):
        self._apply_quick_point_op("Auto Otsu Threshold", threshold_otsu)

    def action_solarize(self):
        doc = self._get_active_doc()
        if doc is None:
            return
        t, ok = QInputDialog.getInt(self, "Solarize", "Threshold (0 - 255):", 128, 0, 255)
        if ok:
            self._apply_quick_point_op("Solarize", lambda img: solarize(img, t))

    # ==========================================================================
    # Interactive Modals with Live Preview
    # ==========================================================================

    def _run_preview_dialog(self, dialog_class, action_name: str):
        doc = self._get_active_doc()
        canvas = self._get_active_canvas()
        if doc is None or canvas is None:
            return

        dlg = dialog_class(doc.current, self)
        dlg.previewUpdated.connect(lambda preview_img: canvas.set_image(preview_img, doc.original))

        if dlg.exec() == QDialog.DialogCode.Accepted:
            doc.push_state(action_name, dlg.result_img)
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText(f"Applied: {action_name}")
        else:
            # Revert canvas
            canvas.set_image(doc.current, doc.original)

    def action_brightness_contrast(self):
        self._run_preview_dialog(BrightnessContrastDialog, "Brightness & Contrast")

    def action_threshold_dialog(self):
        self._run_preview_dialog(ThresholdDialog, "Thresholding")

    def action_gamma_dialog(self):
        self._run_preview_dialog(GammaDialog, "Gamma Correction")

    def action_posterize_dialog(self):
        self._run_preview_dialog(PosterizeDialog, "Bit-Depth Posterization")

    def action_rotate_dialog(self):
        self._run_preview_dialog(RotateDialog, "Arbitrary Rotation")

    def action_translate_dialog(self):
        self._run_preview_dialog(TranslateDialog, "Translate / Shift")

    def action_scale_dialog(self):
        self._run_preview_dialog(ScaleDialog, "Scale / Resize")

    def action_arithmetic_dialog(self):
        doc = self._get_active_doc()
        canvas = self._get_active_canvas()
        if doc is None or canvas is None:
            return

        # Prepare dictionary of available open documents
        available = {}
        for idx, d in self.documents.items():
            if idx != self.tab_widget.currentIndex():
                available[d.current.metadata.filename] = d.current

        dlg = ArithmeticOperationDialog(doc.current, available, self)
        dlg.previewUpdated.connect(lambda preview_img: canvas.set_image(preview_img, doc.original))

        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_img is not None:
            doc.push_state(f"Arithmetic: {dlg.op_combo.currentText()}", dlg.result_img)
            self._update_panels_for_active_doc()
            self.lbl_status_msg.setText("Applied Arithmetic/Boolean Operation")
        else:
            canvas.set_image(doc.current, doc.original)

    # ==========================================================================
    # Help & About
    # ==========================================================================

    def action_show_guide(self):
        guide_text = (
            "<h3>Mini Photoshop - Acuan Materi P1 s/d P6</h3>"
            "<ul>"
            "<li><b>P1 & P2:</b> Dasar Citra Digital, Representasi Matriks N x M, Kedalaman Bit, Histogram & Noise.</li>"
            "<li><b>P3:</b> Hierarki Image Processing vs Computer Vision & Citra Standar Uji.</li>"
            "<li><b>P5 (Tugas Format Citra):</b> Dukungan format PBM (1-bit), PGM (8-bit gray), PPM (24-bit RGB), BMP (DIB header + Palet), RAW Image, dan Metadata Viewer.</li>"
            "<li><b>P6 (Tugas Operasi Dasar):</b> Operasi Aras Titik (Negatif, Grayscale Rata-rata & Luminansi NTSC, Brightness, Contrast Stretching, Thresholding Manual & Otsu, Gamma, Posterisasi), Operasi Aritmetika (Add, Sub, Mul, Div, Alpha Blending), Operasi Logika (AND, OR, NOT, XOR), dan Operasi Geometri (Translasi, Rotasi, Flipping, Zooming).</li>"
            "</ul>"
        )
        QMessageBox.information(self, "Panduan Materi Perkuliahan", guide_text)

    def action_about(self):
        about_text = (
            "<h2>Mini Photoshop v1.0</h2>"
            "<p>Aplikasi Pemrosesan dan Analisis Citra Digital untuk Tugas Kuliah Semester 3.</p>"
            "<p><b>Fitur:</b></p>"
            "<ul>"
            "<li>Native Netpbm & Windows BMP Parser (PBM, PGM, PPM, BMP, RAW)</li>"
            "<li>Point, Arithmetic, Logic, & Geometry Engines</li>"
            "<li>Real-time Histogram & Pixel Inspector</li>"
            "<li>Split-Screen Before/After Comparison</li>"
            "<li>Multi-tab Document Management & Undo/Redo</li>"
            "</ul>"
        )
        QMessageBox.about(self, "About Mini Photoshop", about_text)

    def _load_default_startup_sample(self):
        # Automatically load lena or baboon or palette test sample on launch
        samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
        for preferred in ["lena.png", "palette_test_24bit.bmp", "gradient_test.pgm"]:
            p = os.path.join(samples_dir, preferred)
            if os.path.exists(p):
                self.open_image_path(p)
                break
