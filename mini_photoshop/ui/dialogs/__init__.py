"""
Dialogs package for Mini Photoshop
"""
from .adjust_dialog import (
    BrightnessContrastDialog, ThresholdDialog, GammaDialog, PosterizeDialog,
    LogTransformDialog, InverseLogTransformDialog,
    GrayLevelSlicingDialog, BitPlaneSlicingDialog,
    HistogramSpecificationDialog,
)
from .arithmetic_dialog import ArithmeticOperationDialog
from .geometry_dialog import RotateDialog, TranslateDialog, ScaleDialog
from .info_dialog import ImageInfoDialog
from .raw_dialog import RawImportDialog
from .spatial_dialog import SpatialFilterDialog
from .noise_dialog import AddNoiseDialog, NoiseReductionDialog

__all__ = [
    "BrightnessContrastDialog",
    "ThresholdDialog",
    "GammaDialog",
    "PosterizeDialog",
    "LogTransformDialog",
    "InverseLogTransformDialog",
    "GrayLevelSlicingDialog",
    "BitPlaneSlicingDialog",
    "HistogramSpecificationDialog",
    "ArithmeticOperationDialog",
    "RotateDialog",
    "TranslateDialog",
    "ScaleDialog",
    "ImageInfoDialog",
    "RawImportDialog",
    "SpatialFilterDialog",
    "AddNoiseDialog",
    "NoiseReductionDialog",
]
