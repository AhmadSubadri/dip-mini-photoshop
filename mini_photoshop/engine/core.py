"""
Core Image Matrix and State Management for Mini Photoshop.
Handles the dynamic array representation of 1-bit, 8-bit, and 24-bit images,
conversion utilities, and undo/redo history.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import os


@dataclass
class ImageMetadata:
    filename: str = "Untitled"
    filepath: Optional[str] = None
    format_type: str = "RGB"  # 'PBM', 'PGM', 'PPM', 'BMP', 'RAW', 'PNG', 'JPEG', etc.
    bit_depth: int = 24       # 1, 8, 24, 32
    color_mode: str = "RGB"   # 'BINARY', 'GRAYSCALE', 'RGB', 'RGBA'
    file_size_bytes: int = 0
    raw_header_info: Dict[str, Any] = field(default_factory=dict)


class ImageMatrix:
    """
    Represents an image as a multidimensional NumPy matrix (uint8)
    compatible with the data structures taught in P5 (f[y][x] or f[y][x][c]).
    """
    def __init__(
        self,
        array: np.ndarray,
        color_mode: Optional[str] = None,
        metadata: Optional[ImageMetadata] = None
    ):
        # Validate and normalize array
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            # Grayscale or binary (H, W)
            self.array = array
            unique_vals = np.unique(array)
            if color_mode is None:
                if len(unique_vals) <= 2 and (set(unique_vals).issubset({0, 1, 255})):
                    self.color_mode = "BINARY"
                else:
                    self.color_mode = "GRAYSCALE"
            else:
                self.color_mode = color_mode
        elif array.ndim == 3:
            if array.shape[2] == 1:
                self.array = array[:, :, 0]
                self.color_mode = "GRAYSCALE" if color_mode is None else color_mode
            elif array.shape[2] == 3:
                self.array = array
                self.color_mode = "RGB" if color_mode is None else color_mode
            elif array.shape[2] == 4:
                self.array = array
                self.color_mode = "RGBA" if color_mode is None else color_mode
            else:
                raise ValueError(f"Unsupported channel count: {array.shape[2]}")
        else:
            raise ValueError(f"Unsupported array dimension: {array.ndim}")

        # Metadata
        if metadata is None:
            bit_depth = 1 if self.color_mode == "BINARY" else (8 if self.color_mode == "GRAYSCALE" else 24)
            self.metadata = ImageMetadata(
                filename="Untitled",
                format_type="RAW",
                bit_depth=bit_depth,
                color_mode=self.color_mode,
                file_size_bytes=self.array.nbytes
            )
        else:
            self.metadata = metadata
            self.metadata.color_mode = self.color_mode

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def channels(self) -> int:
        return 1 if self.array.ndim == 2 else self.array.shape[2]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def is_grayscale(self) -> bool:
        return self.color_mode in ("GRAYSCALE", "BINARY") or self.channels == 1

    @property
    def is_binary(self) -> bool:
        return self.color_mode == "BINARY"

    def copy(self) -> "ImageMatrix":
        """Creates a deep copy of the image and its metadata."""
        copied_meta = ImageMetadata(
            filename=self.metadata.filename,
            filepath=self.metadata.filepath,
            format_type=self.metadata.format_type,
            bit_depth=self.metadata.bit_depth,
            color_mode=self.metadata.color_mode,
            file_size_bytes=self.metadata.file_size_bytes,
            raw_header_info=dict(self.metadata.raw_header_info)
        )
        return ImageMatrix(self.array.copy(), self.color_mode, copied_meta)

    def to_rgb(self) -> np.ndarray:
        """Returns the image as an RGB uint8 array (H, W, 3)."""
        if self.array.ndim == 2:
            return np.stack([self.array, self.array, self.array], axis=-1)
        elif self.array.shape[2] == 3:
            return self.array.copy()
        elif self.array.shape[2] == 4:
            return self.array[:, :, :3].copy()
        return self.array

    def to_grayscale_array(self) -> np.ndarray:
        """Returns single channel grayscale array (H, W)."""
        if self.array.ndim == 2:
            return self.array.copy()
        # Default luminance weighting
        r = self.array[:, :, 0].astype(np.float32)
        g = self.array[:, :, 1].astype(np.float32)
        b = self.array[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)


class DocumentState:
    """
    Manages the active image, original reference, and Undo/Redo stack.
    """
    def __init__(self, initial_image: ImageMatrix, max_history: int = 30):
        self.original: ImageMatrix = initial_image.copy()
        self.current: ImageMatrix = initial_image.copy()
        self.undo_stack: List[Tuple[str, ImageMatrix]] = []
        self.redo_stack: List[Tuple[str, ImageMatrix]] = []
        self.max_history = max_history

    def push_state(self, action_name: str, new_image: ImageMatrix):
        """Pushes a new state to the undo stack."""
        self.undo_stack.append((action_name, self.current.copy()))
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.current = new_image.copy()

    def undo(self) -> Optional[str]:
        """Reverts to the previous image state."""
        if not self.undo_stack:
            return None
        action_name, previous_image = self.undo_stack.pop()
        self.redo_stack.append((action_name, self.current.copy()))
        self.current = previous_image
        return action_name

    def redo(self) -> Optional[str]:
        """Redoes the reverted image state."""
        if not self.redo_stack:
            return None
        action_name, next_image = self.redo_stack.pop()
        self.undo_stack.append((action_name, self.current.copy()))
        self.current = next_image
        return action_name

    def reset_to_original(self):
        """Resets the current image back to the original loaded state."""
        self.push_state("Reset to Original", self.original.copy())

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0
