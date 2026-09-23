"""
Dark Theme Stylesheet and Visual Assets for Mini Photoshop.
Provides a modern, professional dark interface inspired by Adobe Photoshop.
"""

PHOTOSHOP_DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
    font-size: 12px;
}

/* Menu Bar */
QMenuBar {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border-bottom: 1px solid #3c3c3c;
    padding: 2px 4px;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
    border-radius: 3px;
}

QMenuBar::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QMenu {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #454545;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px 6px 20px;
    border-radius: 2px;
}

QMenu::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #3c3c3c;
    margin: 4px 6px;
}

/* Tool Bar */
QToolBar {
    background-color: #252526;
    border-bottom: 1px solid #3c3c3c;
    border-top: none;
    spacing: 4px;
    padding: 3px 6px;
}

QToolButton {
    background-color: transparent;
    color: #cccccc;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #3e3e42;
    border: 1px solid #555555;
    color: #ffffff;
}

QToolButton:pressed, QToolButton:checked {
    background-color: #094771;
    border: 1px solid #007acc;
    color: #ffffff;
}

/* Dock Widget & Panels */
QDockWidget {
    color: #ffffff;
    font-weight: bold;
    titlebar-close-icon: url(close.png);
}

QDockWidget::title {
    background-color: #2d2d2d;
    padding: 6px 8px;
    border-bottom: 1px solid #3c3c3c;
}

/* Tab Widget & Tab Bar */
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background-color: #181818;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #999999;
    padding: 6px 16px;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-top: 2px solid #007acc;
}

QTabBar::tab:hover:!selected {
    background-color: #383838;
    color: #e0e0e0;
}

/* Status Bar */
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-weight: 500;
    padding: 2px 8px;
}

QStatusBar QLabel {
    background: transparent;
    color: #ffffff;
    padding: 0 6px;
}

/* Push Buttons */
QPushButton {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #777777;
}

QPushButton:pressed {
    background-color: #0e639c;
    border-color: #007acc;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
    border-color: #333333;
}

QPushButton#primaryButton {
    background-color: #0e639c;
    border-color: #1177bb;
}

QPushButton#primaryButton:hover {
    background-color: #1177bb;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 4px;
    background: #2a2a2a;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #007acc;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #e0e0e0;
    border: 1px solid #555555;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #007acc;
}

/* SpinBox & ComboBox */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #454545;
    border-radius: 3px;
    padding: 4px 6px;
}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid #007acc;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

/* Group Box */
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #e0e0e0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    background-color: #1e1e1e;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background-color: #121212;
}

/* Tooltips */
QToolTip {
    background-color: #252526;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 4px;
    border-radius: 3px;
}
"""
