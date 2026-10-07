"""
Spatial / Neighborhood Operations (Operasi Ketetanggaan Piksel / Operasi Spasial Lokal)
untuk Mini Photoshop.

Seluruh perhitungan matematika dalam modul ini dibuat MANDIRI (FROM SCRATCH)
hanya menggunakan manipulasi matriks numerik NumPy murni TANPA menggunakan library eksternal
seperti OpenCV (cv2), SciPy, ataupun PIL ImageFilter.

Daftar Operasi yang Diimplementasikan:
1. Padding Manual (Zero Padding, Replicate/Border Padding, Reflect Padding)
2. Konvolusi Spasial 2D Manual (Sliding Window Convolution)
3. Operasi Linier:
   - Mean / Averaging Filter (Mendukung Kernel 2x2, 3x3, 5x5, dst.)
   - Gaussian Blur Filter (Pembobotan Distribusi Normal Gauss Matematis)
   - Sharpening Filter (Penajaman Tepi Berbasis High-Pass Laplacian)
   - Deteksi Tepi Roberts (Kernel 2x2)
   - Deteksi Tepi Sobel & Prewitt (Kernel 3x3)
   - Custom Kernel Convolution (Pengguna Memasukkan Matriks Kernel Bebas)
4. Operasi Non-Linier:
   - Median Filter (Pengurutan / Ranking Nilai Tetangga untuk Reduksi Noise)
   - Max Filter (Mencari Nilai Maksimum Lokal / Mirip Dilatasi Morfologi)
   - Min Filter (Mencari Nilai Minimum Lokal / Mirip Erosi Morfologi)
"""

from typing import Tuple, Union, Optional
import numpy as np
from .core import ImageMatrix


# ==============================================================================
# 1. PADDING MANUAL CITRA (BOUNDARY HANDLING)
# ==============================================================================

def manual_pad2d(
    channel: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    mode: str = "replicate"
) -> np.ndarray:
    """
    Melakukan padding (penambahan bingkai batas) pada matriks 2D secara manual.
    Tujuan: Menjaga ukuran citra keluaran agar sama persis dengan citra masukan
            setelah dilakukan operasi konvolusi atau filtering jendela lokal.

    Parameter:
    - channel: Matriks 2D (Tinggi x Lebar) bertipe float atau uint8.
    - pad_top, pad_bottom, pad_left, pad_right: Ketebalan bingkai di tiap sisi.
    - mode:
      * 'replicate' : Menduplikasi piksel batas terluar citra (menghindari artefak gelap).
      * 'zero'      : Mengisi bingkai dengan nilai 0 (hitam).
      * 'reflect'   : Mencerminkan piksel di sekitar batas.

    Kembalian:
    - Matriks 2D baru dengan dimensi: (H + pad_top + pad_bottom, W + pad_left + pad_right).
    """
    # 1. Ambil tinggi (H) dan lebar (W) citra asli
    height, width = channel.shape[:2]

    # 2. Hitung dimensi matriks setelah ditambahkan padding
    new_h = height + pad_top + pad_bottom
    new_w = width + pad_left + pad_right

    # 3. Alokasikan matriks baru berisi nol dengan tipe data yang sama
    padded = np.zeros((new_h, new_w), dtype=channel.dtype)

    # 4. Salin citra asli ke bagian tengah matriks padding
    padded[pad_top : pad_top + height, pad_left : pad_left + width] = channel

    # 5. Jika mode adalah 'zero', cukup biarkan bingkai bernilai 0
    if mode == "zero":
        return padded

    # 6. Jika mode adalah 'replicate' (duplikasi nilai tepi):
    elif mode == "replicate":
        # a. Duplikasi batas atas jika ada padding atas
        if pad_top > 0:
            padded[:pad_top, pad_left : pad_left + width] = channel[0:1, :]
        
        # b. Duplikasi batas bawah jika ada padding bawah
        if pad_bottom > 0:
            padded[pad_top + height :, pad_left : pad_left + width] = channel[-1:, :]
        
        # c. Duplikasi batas kiri untuk seluruh baris (termasuk sudut atas & bawah)
        if pad_left > 0:
            padded[:, :pad_left] = padded[:, pad_left : pad_left + 1]
        
        # d. Duplikasi batas kanan untuk seluruh baris
        if pad_right > 0:
            padded[:, pad_left + width :] = padded[:, pad_left + width - 1 : pad_left + width]

        return padded

    # 7. Jika mode adalah 'reflect' (pencerminan piksel di tepi)
    elif mode == "reflect":
        # Cermin atas
        if pad_top > 0:
            padded[:pad_top, pad_left : pad_left + width] = np.flipud(channel[:pad_top, :])
        # Cermin bawah
        if pad_bottom > 0:
            padded[pad_top + height :, pad_left : pad_left + width] = np.flipud(channel[-pad_bottom:, :])
        # Cermin kiri
        if pad_left > 0:
            padded[:, :pad_left] = np.fliplr(padded[:, pad_left : pad_left + pad_left])
        # Cermin kanan
        if pad_right > 0:
            padded[:, pad_left + width :] = np.fliplr(padded[:, pad_left + width - pad_right : pad_left + width])
        return padded

    else:
        # Default fallback jika mode tidak dikenali
        return padded


def _compute_padding_offsets(kernel_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Menghitung ukuran padding (top, bottom, left, right) berdasarkan ukuran kernel.
    Mendukung kernel ganjil (seperti 3x3, 5x5) maupun kernel genap (seperti 2x2).
    
    Untuk kernel ganjil (K=3): pad = K // 2 = 1 (simetris: 1 atas, 1 bawah, 1 kiri, 1 kanan).
    Untuk kernel genap (K=2): pad_top=0, pad_bottom=1, pad_left=0, pad_right=1.
    """
    kh, kw = kernel_shape
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


# ==============================================================================
# 2. OPERASI LINIER: KONVOLUSI SPASIAL 2D MANUAL
# ==============================================================================

def convolve2d_manual(
    channel: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "replicate"
) -> np.ndarray:
    """
    Menghitung Konvolusi Spasial 2D / Cross-Correlation secara MANUAL.
    
    Rumus Matematis Konvolusi:
        g(y, x) = Sum_{i=0}^{kh-1} Sum_{j=0}^{kw-1} [ f(y + i, x + j) * kernel(i, j) ]
    
    Alur Komputasi:
    1. Siapkan padding pada citra agar ukuran hasil keluaran tidak berubah.
    2. Geser jendela (sliding window) berukuran (kh x kw) di setiap koordinat (y, x).
    3. Kalikan sub-matriks tetangga dengan matriks kernel secara elemen demi elemen (Hadamard product).
    4. Jumlahkan seluruh hasil perkalian bobot tersebut untuk memperoleh nilai piksel baru g(y, x).
    
    Parameter:
    - channel: Matriks 2D citra kanal tunggal (float32 atau uint8).
    - kernel: Matriks bobot 2D (misal ukuran 2x2, 3x3, 5x5).
    - pad_mode: Mode penanganan tepi ('replicate', 'zero', 'reflect').
    
    Kembalian:
    - Matriks 2D hasil konvolusi bertipe float32.
    """
    # 1. Pastikan input channel berupa float32 untuk mencegah overflow/underflow
    src = channel.astype(np.float32)

    # 2. Ambil ukuran citra (H, W) dan ukuran kernel (kh, kw)
    h, w = src.shape[:2]
    kh, kw = kernel.shape[:2]

    # 3. Hitung tebal padding yang dibutuhkan
    pad_t, pad_b, pad_l, pad_r = _compute_padding_offsets((kh, kw))

    # 4. Tambahkan padding pada citra menggunakan fungsi manual kita
    padded = manual_pad2d(src, pad_t, pad_b, pad_l, pad_r, mode=pad_mode)

    # 5. Ekstraksi jendela lokal (Sliding Window) secara vektorisasi NumPy:
    #    np.lib.stride_tricks.sliding_window_view menghasilkan view memory matriks (H, W, kh, kw)
    #    tanpa menyalin data berulang-ulang, merepresentasikan pergeseran jendela tetangga f(y+i, x+j).
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
        # windows berukuran: (H, W, kh, kw)
        # Kalikan elemen jendela dengan bobot kernel: windows * kernel
        # Jumlahkan pada 2 dimensi kernel terakhir (axis -1 dan -2)
        output = np.sum(windows * kernel, axis=(-2, -1), dtype=np.float32)
    except Exception:
        # Fallback: Loop iterasi baris dan kolom manual jika sliding_window_view tidak tersedia
        output = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # Ambil sub-matriks tetangga sebesar ukuran kernel
                sub_region = padded[y : y + kh, x : x + kw]
                # Hitung perkalian bobot dan jumlahkan
                output[y, x] = np.sum(sub_region * kernel)

    return output


def apply_kernel_to_image(
    img: ImageMatrix,
    kernel: np.ndarray,
    clip_output: bool = True
) -> ImageMatrix:
    """
    Menerapkan matriks kernel konvolusi ke seluruh kanal citra (Grayscale atau RGB).
    
    Langkah:
    - Jika citra Grayscale (2D), konvolusi langsung dilakukan pada 1 kanal.
    - Jika citra RGB/RGBA (3D), konvolusi dilakukan secara independen pada setiap kanal warna.
    - Jika clip_output=True, nilai dipotong ke rentang valid piksel [0, 255] dan dibulatkan ke uint8.
    """
    arr = img.array
    kernel = np.asarray(kernel, dtype=np.float32)

    if img.is_grayscale or arr.ndim == 2:
        # 1. Konvolusi untuk citra 1 kanal (Grayscale/Monokrom)
        conv = convolve2d_manual(arr, kernel)
        if clip_output:
            out_arr = np.clip(conv, 0, 255).astype(np.uint8)
        else:
            out_arr = conv
        return ImageMatrix(out_arr, color_mode="GRAYSCALE")

    else:
        # 2. Konvolusi untuk citra multi-kanal (RGB atau RGBA)
        channels = []
        num_channels = arr.shape[2]
        
        # Proses setiap kanal (R, G, B) secara terpisah
        for c in range(min(num_channels, 3)):
            ch_conv = convolve2d_manual(arr[:, :, c], kernel)
            channels.append(ch_conv)

        # Gabungkan kembali kanal-kanal warna
        out_channels = np.stack(channels, axis=2)

        # Jika citra memiliki kanal Alpha (transparansi), pertahankan nilai alpha aslinya
        if num_channels == 4:
            alpha = arr[:, :, 3:4].astype(np.float32)
            out_channels = np.concatenate([out_channels, alpha], axis=2)

        if clip_output:
            out_arr = np.clip(out_channels, 0, 255).astype(np.uint8)
        else:
            out_arr = out_channels

        return ImageMatrix(out_arr, color_mode=img.color_mode)


# ==============================================================================
# 3. IMPLEMENTASI FILTER LINIER KHUSUS
# ==============================================================================

def create_mean_kernel(size: int = 3) -> np.ndarray:
    """
    Membuat Matriks Kernel Perataan (Mean / Averaging Filter).
    
    Rumus:
        K = (1 / (size * size)) * Matriks_Satu(size, size)
    
    Contoh:
    - Ukuran 2x2:
        K = 1/4 * [[1, 1],
                   [1, 1]]
    - Ukuran 3x3:
        K = 1/9 * [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]
    """
    total_elements = size * size
    # Setiap elemen kernel bernilai sama rata yaitu 1 / N
    kernel = np.ones((size, size), dtype=np.float32) / float(total_elements)
    return kernel


def apply_mean_filter(img: ImageMatrix, kernel_size: int = 3) -> ImageMatrix:
    """
    Menerapkan Linear Mean Filter (Averaging / Blur).
    Menghaluskan citra dengan merata-ratakan intensitas piksel tetangga.
    Dapat digunakan untuk mereduksi derau Gaussian.
    """
    kernel = create_mean_kernel(kernel_size)
    return apply_kernel_to_image(img, kernel, clip_output=True)


def create_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Membuat Matriks Kernel Gaussian 2D Berdasarkan Rumus Distribusi Normal Matematis:
    
        G(y, x) = (1 / (2 * pi * sigma^2)) * exp( - ( (x - cx)^2 + (y - cy)^2 ) / (2 * sigma^2) )
    
    Lalu seluruh elemen dinormalisasi sehingga: Sum(Kernel) = 1.0.
    """
    # 1. Tentukan koordinat titik pusat kernel
    center_y = size // 2
    center_x = size // 2

    kernel = np.zeros((size, size), dtype=np.float32)

    # 2. Hitung bobot Gauss untuk setiap sel (y, x)
    two_sigma_sq = 2.0 * (sigma ** 2)
    coeff = 1.0 / (np.pi * two_sigma_sq)

    for y in range(size):
        for x in range(size):
            # Jarak kuadrat dari titik pusat (r^2 = dx^2 + dy^2)
            dx = x - center_x
            dy = y - center_y
            dist_sq = (dx ** 2) + (dy ** 2)
            # Evaluasi eksponensial Gauss
            kernel[y, x] = coeff * np.exp(-dist_sq / two_sigma_sq)

    # 3. Normalisasi agar penjumlahan seluruh elemen kernel bernilai tepat 1
    sum_kernel = np.sum(kernel)
    if sum_kernel > 0:
        kernel /= sum_kernel

    return kernel


def apply_gaussian_filter(
    img: ImageMatrix,
    kernel_size: int = 3,
    sigma: float = 1.0
) -> ImageMatrix:
    """
    Menerapkan Linear Gaussian Filter.
    Memberikan efek penghalusan (blur) yang lebih natural dibanding mean filter
    karena piksel yang lebih dekat ke pusat diberi bobot lebih besar.
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return apply_kernel_to_image(img, kernel, clip_output=True)


def apply_sharpen_filter(img: ImageMatrix, mode: str = "standard") -> ImageMatrix:
    """
    Menerapkan Sharpening Filter (Penajaman Citra) Berbasis Kernel High-Pass Laplacian.
    
    Kernel Standard (4-tetangga):
        K = [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]]
             
    Kernel Strong (8-tetangga):
        K = [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]]
    """
    if mode == "strong":
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
    else:
        # Standard laplacian sharpening
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)

    return apply_kernel_to_image(img, kernel, clip_output=True)


def apply_edge_roberts(img: ImageMatrix) -> ImageMatrix:
    """
    Deteksi Tepi Operator Roberts Cross (Kernel Ukuran 2x2).
    
    Kernel Roberts:
        Gx = [[ 1,  0],
              [ 0, -1]]
              
        Gy = [[ 0,  1],
              [-1,  0]]
              
    Magnitudo Gradien:
        G = sqrt(Gx^2 + Gy^2)  atau pendekatan  G = |Gx| + |Gy|
    """
    # 1. Definisikan matriks kernel 2x2 Roberts
    gx_kernel = np.array([[ 1.0,  0.0],
                          [ 0.0, -1.0]], dtype=np.float32)
                          
    gy_kernel = np.array([[ 0.0,  1.0],
                          [-1.0,  0.0]], dtype=np.float32)

    # 2. Proses pada citra grayscale atau per kanal
    arr = img.array
    if img.is_grayscale or arr.ndim == 2:
        gx = convolve2d_manual(arr, gx_kernel)
        gy = convolve2d_manual(arr, gy_kernel)
        # Hitung magnitudo gradien: sqrt(Gx^2 + Gy^2)
        magnitude = np.sqrt(gx**2 + gy**2)
        out_arr = np.clip(magnitude, 0, 255).astype(np.uint8)
        return ImageMatrix(out_arr, color_mode="GRAYSCALE")
    else:
        # Untuk RGB, hitung gradien pada setiap kanal warna
        channels = []
        for c in range(3):
            ch = arr[:, :, c]
            gx = convolve2d_manual(ch, gx_kernel)
            gy = convolve2d_manual(ch, gy_kernel)
            mag = np.sqrt(gx**2 + gy**2)
            channels.append(mag)
        out_rgb = np.stack(channels, axis=2)
        out_arr = np.clip(out_rgb, 0, 255).astype(np.uint8)
        return ImageMatrix(out_arr, color_mode="RGB")


def apply_edge_sobel(img: ImageMatrix) -> ImageMatrix:
    """
    Deteksi Tepi Operator Sobel (Kernel Ukuran 3x3).
    
    Kernel Sobel:
        Gx = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]
              
        Gy = [[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]
              
    Magnitudo Gradien:
        G = sqrt(Gx^2 + Gy^2)
    """
    # 1. Definisikan matriks kernel 3x3 Sobel
    gx_kernel = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32)

    gy_kernel = np.array([
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ], dtype=np.float32)

    arr = img.array
    if img.is_grayscale or arr.ndim == 2:
        gx = convolve2d_manual(arr, gx_kernel)
        gy = convolve2d_manual(arr, gy_kernel)
        magnitude = np.sqrt(gx**2 + gy**2)
        out_arr = np.clip(magnitude, 0, 255).astype(np.uint8)
        return ImageMatrix(out_arr, color_mode="GRAYSCALE")
    else:
        channels = []
        for c in range(3):
            ch = arr[:, :, c]
            gx = convolve2d_manual(ch, gx_kernel)
            gy = convolve2d_manual(ch, gy_kernel)
            mag = np.sqrt(gx**2 + gy**2)
            channels.append(mag)
        out_rgb = np.stack(channels, axis=2)
        out_arr = np.clip(out_rgb, 0, 255).astype(np.uint8)
        return ImageMatrix(out_arr, color_mode="RGB")


# ==============================================================================
# 4. OPERASI NON-LINIER: RANK-ORDER / ORDER-STATISTIC FILTERS MANUAL
# ==============================================================================

def rank_order_filter2d_manual(
    channel: np.ndarray,
    kernel_size: int,
    operation: str = "median",
    pad_mode: str = "replicate"
) -> np.ndarray:
    """
    Menghitung Filter Statistik Peringkat / Non-Linier secara MANUAL.
    Operasi ini TIDAK menggunakan kombinasi perkalian linier, melainkan
    mengambil sampel jendela lokal tetangga (kh x kw), lalu menerapkan
    fungsi pemeringkatan (ranking/sorting):
    
    1. 'median' : Mengurutkan seluruh elemen di jendela, lalu mengambil nilai tengah.
                  g(y, x) = median( { f(y+i, x+j) } )
                  -> Sangat efektif menghilangkan Salt & Pepper noise.
                  
    2. 'max'    : Mengambil nilai intensitas paling tinggi di dalam jendela.
                  g(y, x) = max( { f(y+i, x+j) } )
                  -> Memperluas area terang / menghilangkan pepper (bintik hitam).
                  
    3. 'min'    : Mengambil nilai intensitas paling rendah di dalam jendela.
                  g(y, x) = min( { f(y+i, x+j) } )
                  -> Memperluas area gelap / menghilangkan salt (bintik putih).
    """
    src = channel.astype(np.float32)
    h, w = src.shape[:2]
    kh = kw = kernel_size

    # 1. Hitung padding agar output berdimensi sama dengan input
    pad_t, pad_b, pad_l, pad_r = _compute_padding_offsets((kh, kw))
    padded = manual_pad2d(src, pad_t, pad_b, pad_l, pad_r, mode=pad_mode)

    # 2. Ekstraksi jendela geser (sliding window view)
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
        # windows shape: (H, W, kh, kw)

        if operation == "median":
            # Perhitungan median manual pada axis jendela lokal (-2, -1)
            output = np.median(windows, axis=(-2, -1))
        elif operation == "max":
            output = np.max(windows, axis=(-2, -1))
        elif operation == "min":
            output = np.min(windows, axis=(-2, -1))
        else:
            raise ValueError(f"Operasi tidak dikenal: {operation}")

    except Exception:
        # Fallback: Loop iterasi 2D baris dan kolom manual jika sliding_window_view gagal
        output = np.zeros((h, w), dtype=np.float32)
        mid_idx = (kh * kw) // 2

        for y in range(h):
            for x in range(w):
                # Ambil seluruh tetangga pada jendela lokal
                sub = padded[y : y + kh, x : x + kw]
                # Ratakan (flatten) menjadi daftar 1 dimensi
                flat = sub.flatten()

                if operation == "median":
                    # Urutkan nilai piksel secara menaik (sorting)
                    sorted_vals = np.sort(flat)
                    # Ambil elemen di posisi tengah
                    output[y, x] = sorted_vals[mid_idx]
                elif operation == "max":
                    output[y, x] = np.max(flat)
                elif operation == "min":
                    output[y, x] = np.min(flat)

    return output


def apply_median_filter(img: ImageMatrix, kernel_size: int = 3) -> ImageMatrix:
    """
    Menerapkan Non-Linear Median Filter.
    Metode standar terbaik untuk mereduksi derau impulsif (Salt & Pepper Noise)
    karena tidak mengaburkan tepi setajam filter linier (edge-preserving).
    """
    arr = img.array
    if img.is_grayscale or arr.ndim == 2:
        out = rank_order_filter2d_manual(arr, kernel_size, operation="median")
        return ImageMatrix(np.clip(out, 0, 255).astype(np.uint8), color_mode="GRAYSCALE")
    else:
        channels = [
            rank_order_filter2d_manual(arr[:, :, c], kernel_size, operation="median")
            for c in range(3)
        ]
        out_rgb = np.stack(channels, axis=2)
        if arr.shape[2] == 4:
            out_rgb = np.concatenate([out_rgb, arr[:, :, 3:4]], axis=2)
        return ImageMatrix(np.clip(out_rgb, 0, 255).astype(np.uint8), color_mode=img.color_mode)


def apply_max_filter(img: ImageMatrix, kernel_size: int = 3) -> ImageMatrix:
    """
    Menerapkan Non-Linear Max Filter (Order-Statistic: Maximum).
    Menggantikan nilai piksel tengah dengan nilai terbesar di lingkungannya.
    Efek: Mencerahkan citra, memperbesar objek terang, dan menghilangkan bintik hitam (pepper noise).
    """
    arr = img.array
    if img.is_grayscale or arr.ndim == 2:
        out = rank_order_filter2d_manual(arr, kernel_size, operation="max")
        return ImageMatrix(np.clip(out, 0, 255).astype(np.uint8), color_mode="GRAYSCALE")
    else:
        channels = [
            rank_order_filter2d_manual(arr[:, :, c], kernel_size, operation="max")
            for c in range(3)
        ]
        out_rgb = np.stack(channels, axis=2)
        if arr.shape[2] == 4:
            out_rgb = np.concatenate([out_rgb, arr[:, :, 3:4]], axis=2)
        return ImageMatrix(np.clip(out_rgb, 0, 255).astype(np.uint8), color_mode=img.color_mode)


def apply_min_filter(img: ImageMatrix, kernel_size: int = 3) -> ImageMatrix:
    """
    Menerapkan Non-Linear Min Filter (Order-Statistic: Minimum).
    Menggantikan nilai piksel tengah dengan nilai terkecil di lingkungannya.
    Efek: Menggelapkan citra, memperkecil objek terang, dan menghilangkan bintik putih (salt noise).
    """
    arr = img.array
    if img.is_grayscale or arr.ndim == 2:
        out = rank_order_filter2d_manual(arr, kernel_size, operation="min")
        return ImageMatrix(np.clip(out, 0, 255).astype(np.uint8), color_mode="GRAYSCALE")
    else:
        channels = [
            rank_order_filter2d_manual(arr[:, :, c], kernel_size, operation="min")
            for c in range(3)
        ]
        out_rgb = np.stack(channels, axis=2)
        if arr.shape[2] == 4:
            out_rgb = np.concatenate([out_rgb, arr[:, :, 3:4]], axis=2)
        return ImageMatrix(np.clip(out_rgb, 0, 255).astype(np.uint8), color_mode=img.color_mode)
