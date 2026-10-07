"""
Noise Generation & Image Restoration Operations (Penambahan & Reduksi Derau)
untuk Mini Photoshop.

Seluruh formula pembangkitan derau dalam modul ini ditulis secara MANDIRI (FROM SCRATCH)
sesuai kaidah teori Pengolahan Citra Digital, TANPA menggunakan library eksternal
seperti OpenCV atau scikit-image.

Fitur Modul:
1. Pembangkitan Derau Impulsif (Salt and Pepper Noise)
2. Pembangkitan Derau Normal Gaussian (Additive Gaussian Noise)
3. Pembangkitan Derau Multiplikatif (Speckle Noise)
4. Evaluasi Kualitas Restorasi (Perhitungan Perbaikan Skor PSNR & MSE)
"""

from typing import Optional, Dict, Any
import numpy as np
from .core import ImageMatrix
from .metrics import compute_mse, compute_psnr


def add_salt_and_pepper_noise(
    img: ImageMatrix,
    amount: float = 0.05,
    salt_ratio: float = 0.5,
    seed: Optional[int] = None
) -> ImageMatrix:
    """
    Menambahkan Derau Impulsif (Salt and Pepper Noise) secara MANUAL.
    
    Konsep Teori:
    - Derau Salt & Pepper memodelkan gangguan transmisi data atau sensor di mana
      sejumlah piksel secara acak rusak menjadi nilai ekstrem:
      * 'Salt'   : Nilai intensitas maksimum (255 / putih terang).
      * 'Pepper' : Nilai intensitas minimum (0 / hitam pekat).
      
    Parameter:
    - img        : Objek ImageMatrix citra asli.
    - amount     : Proporsi total piksel yang terkontaminasi derau (0.0 s.d. 1.0, misal 0.05 = 5%).
    - salt_ratio : Perbandingan antara bintik garam (putih) terhadap merica (hitam) (default 0.5 = 50:50).
    - seed       : Seed bilangan acak untuk reprodusibilitas (opsional).
    """
    # 1. Pasang seed generator jika ditentukan untuk hasil yang konsisten
    rng = np.random.default_rng(seed)

    # 2. Salin array citra agar data citra asli tidak termodifikasi
    arr = img.array.copy()
    h, w = arr.shape[:2]
    total_pixels = h * w

    # 3. Hitung jumlah total piksel yang akan diubah menjadi derau
    num_noise_pixels = int(total_pixels * amount)

    # 4. Hitung proporsi masing-masing piksel Salt (putih) dan Pepper (hitam)
    num_salt = int(num_noise_pixels * salt_ratio)
    num_pepper = num_noise_pixels - num_salt

    # 5. Acak indeks koordinat 1D untuk posisi piksel yang terkena derau
    #    replace=False memastikan satu posisi piksel tidak terpilih dua kali
    chosen_indices = rng.choice(total_pixels, size=num_noise_pixels, replace=False)

    # 6. Bagi indeks menjadi kelompok Salt dan kelompok Pepper
    salt_indices = chosen_indices[:num_salt]
    pepper_indices = chosen_indices[num_salt:]

    # 7. Konversi indeks 1D menjadi koordinat 2D (baris y, kolom x)
    salt_y, salt_x = np.unravel_index(salt_indices, (h, w))
    pepper_y, pepper_x = np.unravel_index(pepper_indices, (h, w))

    # 8. Terapkan nilai ekstrem pada koordinat tersebut:
    #    Jika citra Grayscale (2D array):
    if arr.ndim == 2:
        arr[salt_y, salt_x] = 255      # Titik Salt (Putih)
        arr[pepper_y, pepper_x] = 0     # Titik Pepper (Hitam)
    #    Jika citra RGB / Multi-kanal (3D array):
    else:
        # Ubah seluruh kanal RGB pada titik tersebut
        arr[salt_y, salt_x, :3] = 255
        arr[pepper_y, pepper_x, :3] = 0

    return ImageMatrix(arr, color_mode=img.color_mode)


def add_gaussian_noise(
    img: ImageMatrix,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = None
) -> ImageMatrix:
    """
    Menambahkan Derau Normal Gauss Aditif (Additive Gaussian Noise) secara MANUAL.
    
    Rumus Matematis:
        f_noisy(y, x) = clip( f(y, x) + n(y, x), 0, 255 )
        di mana n(y, x) ~ N(mu, sigma^2)
        
    Konsep Teori:
    - Memodelkan derau sensor elektronik yang disebabkan oleh fluktuasi termal.
    - Setiap piksel ditambahkan nilai gangguan acak yang terdistribusi normal (lonceng Gauss).
    """
    # 1. Inisialisasi generator acak
    rng = np.random.default_rng(seed)

    # 2. Konversi citra asli ke tipe float32 agar proses penjumlahan tidak overflow
    src_float = img.array.astype(np.float32)

    # 3. Bangkitkan matriks noise acak dengan ukuran yang sama persis seperti citra
    #    Distribusi normal: rata-rata = mean, deviasi standar = sigma
    noise = rng.normal(loc=mean, scale=sigma, size=src_float.shape)

    # 4. Tambahkan noise aditif ke citra asli
    noisy_float = src_float + noise

    # 5. Potong nilai (clipping) ke rentang intensitas valid [0, 255]
    #    dan kembalikan tipe data ke uint8 (8-bit unsigned integer)
    noisy_arr = np.clip(noisy_float, 0, 255).astype(np.uint8)

    return ImageMatrix(noisy_arr, color_mode=img.color_mode)


def add_speckle_noise(
    img: ImageMatrix,
    variance: float = 0.04,
    seed: Optional[int] = None
) -> ImageMatrix:
    """
    Menambahkan Derau Bintik Multiplikatif (Multiplicative Speckle Noise) secara MANUAL.
    
    Rumus Matematis:
        f_noisy(y, x) = clip( f(y, x) + f(y, x) * n(y, x), 0, 255 )
        di mana n ~ N(0, variance)
        
    Konsep Teori:
    - Umum terjadi pada sistem pencitraan radar (SAR) atau citra ultrasonografi medis (USG).
    - Besarnya gangguan sebanding dengan tingkat kecerahan piksel asli.
    """
    rng = np.random.default_rng(seed)
    src_float = img.array.astype(np.float32)

    # Standar deviasi = akar kuadrat dari varians
    sigma = np.sqrt(variance)

    # Bangkitkan derau berdistribusi normal dengan rata-rata 0
    noise = rng.normal(loc=0.0, scale=sigma, size=src_float.shape)

    # Derau dikalikan dengan nilai piksel citra (multiplikatif) lalu ditambahkan
    noisy_float = src_float + (src_float * noise)

    noisy_arr = np.clip(noisy_float, 0, 255).astype(np.uint8)
    return ImageMatrix(noisy_arr, color_mode=img.color_mode)


def evaluate_restoration(
    original: ImageMatrix,
    noisy: ImageMatrix,
    restored: ImageMatrix
) -> Dict[str, Any]:
    """
    Mengevaluasi Kualitas Reduksi Derau / Restorasi Citra secara Ilmiah Kuantitatif.
    
    Menghitung:
    1. MSE dan PSNR Citra Tercemar Noise terhadap Citra Asli (Kondisi Sebelum Restorasi)
    2. MSE dan PSNR Citra Hasil Restorasi terhadap Citra Asli (Kondisi Setelah Restorasi)
    3. Delta PSNR (+dB): Selisih peningkatan kualitas sinyal terhadap derau.
    """
    # 1. Hitung skor citra noisy vs citra asli
    mse_noisy = compute_mse(original, noisy)
    psnr_noisy = compute_psnr(original, noisy)

    # 2. Hitung skor citra hasil reduksi vs citra asli
    mse_restored = compute_mse(original, restored)
    psnr_restored = compute_psnr(original, restored)

    # 3. Hitung selisih perbaikan (dB)
    delta_psnr = psnr_restored - psnr_noisy
    delta_mse = mse_noisy - mse_restored

    return {
        "mse_noisy": mse_noisy,
        "psnr_noisy": psnr_noisy,
        "mse_restored": mse_restored,
        "psnr_restored": psnr_restored,
        "delta_psnr": delta_psnr,
        "delta_mse": delta_mse,
        "is_improved": delta_psnr > 0,
    }
