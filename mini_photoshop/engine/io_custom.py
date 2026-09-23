"""
Custom and Universal Image I/O Handlers for Mini Photoshop.
Includes native byte-level and ASCII parsers for PBM, PGM, PPM, BMP, and RAW formats (as taught in P5),
with full fallback for modern formats (JPEG, PNG, GIF, TIFF, WEBP).
"""

import os
import struct
import io
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional
from .core import ImageMatrix, ImageMetadata


# ==============================================================================
# Helper Functions for Netpbm ASCII Parsing
# ==============================================================================

def _read_netpbm_tokens(f) -> list:
    """Read tokens from Netpbm stream, skipping comments starting with '#'."""
    tokens = []
    for line in f:
        # Handle bytes or string
        if isinstance(line, bytes):
            line = line.decode('latin1', errors='ignore')
        # Strip comments
        line = line.split('#')[0].strip()
        if line:
            tokens.extend(line.split())
    return tokens


def _parse_netpbm_header(stream) -> Tuple[str, int, int, int, int]:
    """
    Parses Netpbm header (magic, width, height, max_val, header_offset).
    Returns (magic, width, height, max_val, header_end_offset_in_bytes)
    """
    header_bytes = bytearray()
    tokens = []
    magic = None
    width = None
    height = None
    max_val = 255
    
    # We read byte by byte or line by line
    while True:
        line = stream.readline()
        if not line:
            break
        header_bytes.extend(line)
        line_str = line.decode('latin1', errors='ignore')
        line_clean = line_str.split('#')[0].strip()
        if not line_clean:
            continue
        parts = line_clean.split()
        for p in parts:
            if magic is None:
                magic = p
            elif width is None:
                width = int(p)
            elif height is None:
                height = int(p)
            elif magic in ('P2', 'P3', 'P5', 'P6') and max_val is None or (magic in ('P2', 'P3', 'P5', 'P6') and len(tokens) == 3):
                max_val = int(p)
                # Header parsing complete
                return magic, width, height, max_val, len(header_bytes)
            tokens.append(p)
        if magic in ('P1', 'P4') and width is not None and height is not None:
            return magic, width, height, 1, len(header_bytes)
            
    return magic, width, height, max_val, len(header_bytes)


# ==============================================================================
# PBM Parser (P1: ASCII, P4: Binary)
# ==============================================================================

def read_pbm(filepath: str) -> ImageMatrix:
    """Reads PBM (1-bit monochrome) image."""
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        magic_line = f.readline().strip().decode('latin1')
        f.seek(0)
        
        if magic_line.startswith('P1'):
            # ASCII PBM
            f_text = open(filepath, 'r', encoding='latin1')
            tokens = _read_netpbm_tokens(f_text)
            f_text.close()
            
            magic = tokens[0]
            width = int(tokens[1])
            height = int(tokens[2])
            pixel_data = [int(t) for t in tokens[3:3 + width * height]]
            # In PBM: 1 is black (0 intensity), 0 is white (255 intensity)
            arr = np.array(pixel_data, dtype=np.uint8).reshape((height, width))
            # Convert to displayable 8-bit: 0 -> 255 (white), 1 -> 0 (black)
            arr_display = np.where(arr == 1, 0, 255).astype(np.uint8)
            
        elif magic_line.startswith('P4'):
            # Binary PBM
            magic, width, height, _, offset = _parse_netpbm_header(f)
            f.seek(offset)
            row_bytes = (width + 7) // 8
            raw_data = f.read(row_bytes * height)
            
            # Unpack bits
            arr_display = np.zeros((height, width), dtype=np.uint8)
            idx = 0
            for y in range(height):
                row_raw = raw_data[idx:idx + row_bytes]
                idx += row_bytes
                bit_idx = 0
                for byte_val in row_raw:
                    for b in range(7, -1, -1):
                        if bit_idx < width:
                            bit = (byte_val >> b) & 1
                            # 1 is black (0), 0 is white (255)
                            arr_display[y, bit_idx] = 0 if bit == 1 else 255
                            bit_idx += 1
        else:
            raise ValueError(f"Invalid PBM format: {magic_line}")

    meta = ImageMetadata(
        filename=os.path.basename(filepath),
        filepath=filepath,
        format_type="PBM",
        bit_depth=1,
        color_mode="BINARY",
        file_size_bytes=file_size,
        raw_header_info={"magic": magic_line[:2], "width": width, "height": height}
    )
    return ImageMatrix(arr_display, color_mode="BINARY", metadata=meta)


def write_pbm(img: ImageMatrix, filepath: str, binary: bool = True):
    """Writes ImageMatrix as PBM (P1 or P4)."""
    # Threshold if not binary
    gray = img.to_grayscale_array()
    # 0 in PBM is white (>127), 1 is black (<=127)
    pbm_bits = np.where(gray <= 127, 1, 0).astype(np.uint8)
    h, w = pbm_bits.shape

    if binary:
        # P4 Binary
        header = f"P4\n# Created by Mini Photoshop\n{w} {h}\n".encode('latin1')
        row_bytes = (w + 7) // 8
        packed_data = bytearray()
        for y in range(h):
            row = pbm_bits[y]
            for byte_col in range(0, w, 8):
                byte_val = 0
                for b in range(8):
                    if byte_col + b < w:
                        byte_val |= (row[byte_col + b] << (7 - b))
                packed_data.append(byte_val)
        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(packed_data)
    else:
        # P1 ASCII
        with open(filepath, 'w', encoding='latin1') as f:
            f.write(f"P1\n# Created by Mini Photoshop\n{w} {h}\n")
            for y in range(h):
                row_str = " ".join(str(pbm_bits[y, x]) for x in range(w))
                f.write(row_str + "\n")


# ==============================================================================
# PGM Parser (P2: ASCII, P5: Binary)
# ==============================================================================

def read_pgm(filepath: str) -> ImageMatrix:
    """Reads PGM (8-bit grayscale) image."""
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        magic_line = f.readline().strip().decode('latin1')
        f.seek(0)
        
        if magic_line.startswith('P2'):
            # ASCII PGM
            f_text = open(filepath, 'r', encoding='latin1')
            tokens = _read_netpbm_tokens(f_text)
            f_text.close()
            
            magic = tokens[0]
            width = int(tokens[1])
            height = int(tokens[2])
            max_val = int(tokens[3])
            pixel_data = [int(t) for t in tokens[4:4 + width * height]]
            arr = np.array(pixel_data, dtype=np.float32).reshape((height, width))
            if max_val != 255 and max_val > 0:
                arr = (arr / max_val) * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            
        elif magic_line.startswith('P5'):
            # Binary PGM
            magic, width, height, max_val, offset = _parse_netpbm_header(f)
            f.seek(offset)
            raw_data = f.read(width * height)
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
            if max_val != 255 and max_val > 0:
                arr = ((arr.astype(np.float32) / max_val) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Invalid PGM format: {magic_line}")

    meta = ImageMetadata(
        filename=os.path.basename(filepath),
        filepath=filepath,
        format_type="PGM",
        bit_depth=8,
        color_mode="GRAYSCALE",
        file_size_bytes=file_size,
        raw_header_info={"magic": magic_line[:2], "width": width, "height": height, "max_val": max_val}
    )
    return ImageMatrix(arr, color_mode="GRAYSCALE", metadata=meta)


def write_pgm(img: ImageMatrix, filepath: str, binary: bool = True):
    """Writes ImageMatrix as PGM (P2 or P5)."""
    gray = img.to_grayscale_array()
    h, w = gray.shape
    if binary:
        header = f"P5\n# Created by Mini Photoshop\n{w} {h}\n255\n".encode('latin1')
        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(gray.tobytes())
    else:
        with open(filepath, 'w', encoding='latin1') as f:
            f.write(f"P2\n# Created by Mini Photoshop\n{w} {h}\n255\n")
            for y in range(h):
                row_str = " ".join(str(int(gray[y, x])) for x in range(w))
                f.write(row_str + "\n")


# ==============================================================================
# PPM Parser (P3: ASCII, P6: Binary)
# ==============================================================================

def read_ppm(filepath: str) -> ImageMatrix:
    """Reads PPM (24-bit RGB) image."""
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        magic_line = f.readline().strip().decode('latin1')
        f.seek(0)
        
        if magic_line.startswith('P3'):
            # ASCII PPM
            f_text = open(filepath, 'r', encoding='latin1')
            tokens = _read_netpbm_tokens(f_text)
            f_text.close()
            
            magic = tokens[0]
            width = int(tokens[1])
            height = int(tokens[2])
            max_val = int(tokens[3])
            pixel_data = [int(t) for t in tokens[4:4 + width * height * 3]]
            arr = np.array(pixel_data, dtype=np.float32).reshape((height, width, 3))
            if max_val != 255 and max_val > 0:
                arr = (arr / max_val) * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            
        elif magic_line.startswith('P6'):
            # Binary PPM
            magic, width, height, max_val, offset = _parse_netpbm_header(f)
            f.seek(offset)
            raw_data = f.read(width * height * 3)
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            if max_val != 255 and max_val > 0:
                arr = ((arr.astype(np.float32) / max_val) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Invalid PPM format: {magic_line}")

    meta = ImageMetadata(
        filename=os.path.basename(filepath),
        filepath=filepath,
        format_type="PPM",
        bit_depth=24,
        color_mode="RGB",
        file_size_bytes=file_size,
        raw_header_info={"magic": magic_line[:2], "width": width, "height": height, "max_val": max_val}
    )
    return ImageMatrix(arr, color_mode="RGB", metadata=meta)


def write_ppm(img: ImageMatrix, filepath: str, binary: bool = True):
    """Writes ImageMatrix as PPM (P3 or P6)."""
    rgb = img.to_rgb()
    h, w, _ = rgb.shape
    if binary:
        header = f"P6\n# Created by Mini Photoshop\n{w} {h}\n255\n".encode('latin1')
        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(rgb.tobytes())
    else:
        with open(filepath, 'w', encoding='latin1') as f:
            f.write(f"P3\n# Created by Mini Photoshop\n{w} {h}\n255\n")
            for y in range(h):
                tokens = []
                for x in range(w):
                    tokens.extend([str(int(rgb[y, x, 0])), str(int(rgb[y, x, 1])), str(int(rgb[y, x, 2]))])
                f.write(" ".join(tokens) + "\n")


# ==============================================================================
# Native BMP Parser & Writer (14-byte Header, 40-byte DIB Header, Palette)
# ==============================================================================

def read_bmp(filepath: str) -> ImageMatrix:
    """
    Native BMP parser parsing Bitmap File Header (14 bytes) and DIB Header (40 bytes),
    supporting 1-bit, 8-bit indexed, 24-bit RGB, and 32-bit RGBA bitmaps.
    """
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        # 1. Bitmap File Header (14 bytes)
        bfType = f.read(2)
        if bfType != b'BM':
            raise ValueError(f"Not a valid Windows BMP file (Header: {bfType})")
        
        bfSize, bfReserved1, bfReserved2, bfOffBits = struct.unpack('<IHHI', f.read(12))
        
        # 2. DIB Header (BITMAPINFOHEADER 40 bytes)
        biSize = struct.unpack('<I', f.read(4))[0]
        if biSize < 40:
            # Fallback to Pillow for older OS/2 BMP variants
            f.seek(0)
            pil_img = Image.open(filepath)
            arr = np.array(pil_img)
            return ImageMatrix(arr, metadata=ImageMetadata(
                filename=os.path.basename(filepath),
                filepath=filepath,
                format_type="BMP",
                bit_depth=24,
                color_mode="RGB",
                file_size_bytes=file_size
            ))

        (biWidth, biHeight, biPlanes, biBitCount, biCompression,
         biSizeImage, biXPelsPerMeter, biYPelsPerMeter,
         biClrUsed, biClrImportant) = struct.unpack('<iiHHIIiiII', f.read(36))

        # Check orientation: positive height is bottom-up, negative is top-down
        is_bottom_up = biHeight > 0
        height = abs(biHeight)
        width = biWidth

        # Read Palette if 1-bit or 8-bit
        palette = []
        if biBitCount <= 8:
            num_colors = biClrUsed if biClrUsed > 0 else (1 << biBitCount)
            palette_bytes = f.read(num_colors * 4)
            for i in range(num_colors):
                b, g, r, _ = struct.unpack('<BBBB', palette_bytes[i * 4:(i + 1) * 4])
                palette.append((r, g, b))

        # Seek to pixel data offset
        f.seek(bfOffBits)

        # Parse pixel data based on bit count
        if biBitCount == 24:
            # 24-bit BGR with 4-byte row padding
            row_stride = ((width * 3 + 3) // 4) * 4
            raw_pixels = f.read(row_stride * height)
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                row_idx = y if not is_bottom_up else (height - 1 - y)
                offset = y * row_stride
                row_bgr = np.frombuffer(raw_pixels[offset:offset + width * 3], dtype=np.uint8).reshape((width, 3))
                # Convert BGR to RGB
                arr[row_idx, :, 0] = row_bgr[:, 2] # R
                arr[row_idx, :, 1] = row_bgr[:, 1] # G
                arr[row_idx, :, 2] = row_bgr[:, 0] # B
            color_mode = "RGB"

        elif biBitCount == 8:
            # 8-bit Grayscale or Indexed with 4-byte padding
            row_stride = ((width + 3) // 4) * 4
            raw_pixels = f.read(row_stride * height)
            
            # Check if palette is grayscale
            is_gray_palette = all(r == g == b for (r, g, b) in palette) if palette else True
            
            if is_gray_palette:
                arr = np.zeros((height, width), dtype=np.uint8)
                for y in range(height):
                    row_idx = y if not is_bottom_up else (height - 1 - y)
                    offset = y * row_stride
                    arr[row_idx, :] = np.frombuffer(raw_pixels[offset:offset + width], dtype=np.uint8)
                color_mode = "GRAYSCALE"
            else:
                arr = np.zeros((height, width, 3), dtype=np.uint8)
                pal_np = np.array(palette, dtype=np.uint8)
                for y in range(height):
                    row_idx = y if not is_bottom_up else (height - 1 - y)
                    offset = y * row_stride
                    indices = np.frombuffer(raw_pixels[offset:offset + width], dtype=np.uint8)
                    arr[row_idx, :] = pal_np[indices]
                color_mode = "RGB"

        elif biBitCount == 1:
            # 1-bit Monochrome with 4-byte padding
            row_stride = ((width + 31) // 32) * 4
            raw_pixels = f.read(row_stride * height)
            arr = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                row_idx = y if not is_bottom_up else (height - 1 - y)
                offset = y * row_stride
                row_raw = raw_pixels[offset:offset + row_stride]
                bit_idx = 0
                for byte_val in row_raw:
                    for b in range(7, -1, -1):
                        if bit_idx < width:
                            bit = (byte_val >> b) & 1
                            if palette and len(palette) >= 2:
                                arr[row_idx, bit_idx] = palette[bit][0]
                            else:
                                arr[row_idx, bit_idx] = 255 if bit == 1 else 0
                            bit_idx += 1
            color_mode = "BINARY"
        else:
            # Fallback for 32-bit or compressed BMP
            f.seek(0)
            pil_img = Image.open(filepath)
            arr = np.array(pil_img)
            color_mode = "RGB"

    header_info = {
        "bfType": "BM",
        "bfSize": bfSize,
        "bfOffBits": bfOffBits,
        "biWidth": width,
        "biHeight": height,
        "biBitCount": biBitCount,
        "biCompression": biCompression,
        "biClrUsed": biClrUsed
    }
    meta = ImageMetadata(
        filename=os.path.basename(filepath),
        filepath=filepath,
        format_type="BMP",
        bit_depth=biBitCount,
        color_mode=color_mode,
        file_size_bytes=file_size,
        raw_header_info=header_info
    )
    return ImageMatrix(arr, color_mode=color_mode, metadata=meta)


def write_bmp(img: ImageMatrix, filepath: str):
    """
    Native BMP writer creating standard Windows 24-bit RGB or 8-bit Grayscale BMP.
    """
    if img.is_grayscale:
        gray = img.to_grayscale_array()
        h, w = gray.shape
        row_stride = ((w + 3) // 4) * 4
        pixel_array_size = row_stride * h
        palette_size = 256 * 4  # 256 RGBQUAD entries
        offset_bits = 14 + 40 + palette_size
        file_size = offset_bits + pixel_array_size

        # File header (14 bytes)
        file_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, offset_bits)
        # DIB header (40 bytes, 11 fields: biSize, biWidth, biHeight, biPlanes, biBitCount, biCompression, biSizeImage, biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant)
        dib_header = struct.pack('<IiiHHIIiiII', 40, w, h, 1, 8, 0, pixel_array_size, 2835, 2835, 256, 0)
        # 256 Grayscale Palette
        palette = bytearray()
        for i in range(256):
            palette.extend([i, i, i, 0])  # B, G, R, Reserved

        # Pixel data (bottom-up, 4-byte aligned)
        padded_pixels = bytearray()
        pad_bytes = bytes(row_stride - w)
        for y in range(h - 1, -1, -1):
            padded_pixels.extend(gray[y].tobytes())
            if pad_bytes:
                padded_pixels.extend(pad_bytes)

        with open(filepath, 'wb') as f:
            f.write(file_header)
            f.write(dib_header)
            f.write(palette)
            f.write(padded_pixels)
    else:
        # 24-bit RGB
        rgb = img.to_rgb()
        h, w, _ = rgb.shape
        row_stride = ((w * 3 + 3) // 4) * 4
        pixel_array_size = row_stride * h
        offset_bits = 14 + 40
        file_size = offset_bits + pixel_array_size

        file_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, offset_bits)
        dib_header = struct.pack('<IiiHHIIiiII', 40, w, h, 1, 24, 0, pixel_array_size, 2835, 2835, 0, 0)

        # Pixel data: bottom-up BGR
        padded_pixels = bytearray()
        pad_bytes = bytes(row_stride - (w * 3))
        for y in range(h - 1, -1, -1):
            row_rgb = rgb[y]
            # Convert RGB to BGR bytes
            row_bgr = np.empty((w, 3), dtype=np.uint8)
            row_bgr[:, 0] = row_rgb[:, 2] # B
            row_bgr[:, 1] = row_rgb[:, 1] # G
            row_bgr[:, 2] = row_rgb[:, 0] # R
            padded_pixels.extend(row_bgr.tobytes())
            if pad_bytes:
                padded_pixels.extend(pad_bytes)

        with open(filepath, 'wb') as f:
            f.write(file_header)
            f.write(dib_header)
            f.write(padded_pixels)


# ==============================================================================
# RAW Image Parser & Writer
# ==============================================================================

def read_raw(
    filepath: str,
    width: int,
    height: int,
    channels: int = 1,
    bit_depth: int = 8,
    header_offset: int = 0
) -> ImageMatrix:
    """Reads raw uncompressed binary image stream."""
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        f.seek(header_offset)
        total_bytes = width * height * channels * (bit_depth // 8)
        raw_bytes = f.read(total_bytes)
        
        dtype = np.uint8 if bit_depth == 8 else np.uint16
        arr = np.frombuffer(raw_bytes, dtype=dtype)
        if channels == 1:
            arr = arr.reshape((height, width))
            color_mode = "GRAYSCALE"
        else:
            arr = arr.reshape((height, width, channels))
            color_mode = "RGB" if channels == 3 else "RGBA"

    meta = ImageMetadata(
        filename=os.path.basename(filepath),
        filepath=filepath,
        format_type="RAW",
        bit_depth=bit_depth * channels,
        color_mode=color_mode,
        file_size_bytes=file_size,
        raw_header_info={"width": width, "height": height, "channels": channels, "offset": header_offset}
    )
    return ImageMatrix(arr, color_mode=color_mode, metadata=meta)


def write_raw(img: ImageMatrix, filepath: str):
    """Exports raw binary array bytes."""
    with open(filepath, 'wb') as f:
        f.write(img.array.tobytes())


# ==============================================================================
# Universal Load & Save Dispatcher
# ==============================================================================

def load_image_file(filepath: str) -> ImageMatrix:
    """
    Universal image loader dispatcher. Automatically detects format
    and selects the appropriate native or universal decoder.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pbm':
        return read_pbm(filepath)
    elif ext == '.pgm':
        return read_pgm(filepath)
    elif ext == '.ppm':
        return read_ppm(filepath)
    elif ext == '.bmp':
        return read_bmp(filepath)
    elif ext == '.raw':
        # Default assumption 512x512 grayscale if opened directly,
        # otherwise user uses RAW Import Dialog
        file_size = os.path.getsize(filepath)
        side = int(np.sqrt(file_size))
        if side * side == file_size:
            return read_raw(filepath, width=side, height=side, channels=1)
        elif side * side * 3 == file_size:
            return read_raw(filepath, width=side, height=side, channels=3)
        return read_raw(filepath, width=512, height=512, channels=1)
    else:
        # Universal Pillow loader for PNG, JPG, JPEG, GIF, TIFF, WEBP
        pil_img = Image.open(filepath)
        file_size = os.path.getsize(filepath)
        mode = pil_img.mode
        arr = np.array(pil_img)
        
        if mode in ('1', 'L'):
            color_mode = "BINARY" if mode == '1' else "GRAYSCALE"
            bit_depth = 1 if mode == '1' else 8
        elif mode == 'RGBA':
            color_mode = "RGBA"
            bit_depth = 32
        else:
            pil_img = pil_img.convert('RGB')
            arr = np.array(pil_img)
            color_mode = "RGB"
            bit_depth = 24

        meta = ImageMetadata(
            filename=os.path.basename(filepath),
            filepath=filepath,
            format_type=ext.replace('.', '').upper(),
            bit_depth=bit_depth,
            color_mode=color_mode,
            file_size_bytes=file_size,
            raw_header_info=dict(pil_img.info)
        )
        return ImageMatrix(arr, color_mode=color_mode, metadata=meta)


def save_image_file(img: ImageMatrix, filepath: str):
    """
    Universal image saver dispatcher based on file extension.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pbm':
        write_pbm(img, filepath, binary=True)
    elif ext == '.pgm':
        write_pgm(img, filepath, binary=True)
    elif ext == '.ppm':
        write_ppm(img, filepath, binary=True)
    elif ext == '.bmp':
        write_bmp(img, filepath)
    elif ext == '.raw':
        write_raw(img, filepath)
    else:
        # Pillow for PNG, JPEG, TIFF, etc.
        if img.color_mode == "GRAYSCALE":
            pil_img = Image.fromarray(img.to_grayscale_array(), mode='L')
        elif img.color_mode == "BINARY":
            pil_img = Image.fromarray(img.to_grayscale_array(), mode='L').convert('1')
        elif img.color_mode == "RGBA":
            pil_img = Image.fromarray(img.array, mode='RGBA')
        else:
            pil_img = Image.fromarray(img.to_rgb(), mode='RGB')
        
        # Quality parameters
        if ext in ('.jpg', '.jpeg'):
            pil_img.save(filepath, quality=95)
        else:
            pil_img.save(filepath)
