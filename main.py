"""
Mini Photoshop - Digital Image Processing Application
Course: Pengolahan dan Analisis Citra Digital (P1 - P6)
Author: Ahmad Subadri
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from mini_photoshop.ui.main_window import MainWindow


def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Mini Photoshop")
    app.setOrganizationName("UIN Sunan Kalijaga - Magister Informatika")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
