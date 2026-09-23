# 📖 Dokumentasi Teknis & Panduan Pengembangan Mini Photoshop

Dokumen ini ditujukan sebagai panduan komprehensif bagi anggota tim pengembang untuk memahami arsitektur, struktur kode, alur data citra (*pipeline*), detail setiap fungsi, serta panduan langkah demi langkah dalam menambahkan fitur baru.

---

## 📑 Daftar Isi
1. [Arsitektur Sistem & Aliran Data](#-arsitektur-sistem--aliran-data)
2. [Dokumentasi Lengkap Berkas & Fungsi](#-dokumentasi-lengkap-berkas--fungsi)
   - [A. Entry Point](#a-entry-point)
   - [B. Engine Layer (Logika Citra Murni)](#b-engine-layer-logika-citra-murni)
   - [C. UI Layer (Antarmuka & Interaksi)](#c-ui-layer-antarmuka--interaksi)
   - [D. Test Suite](#d-test-suite)
3. [Panduan Menambahkan Fitur Baru (How-To Develop)](#-panduan-menambahkan-fitur-baru-how-to-develop)
   - [1. Menambahkan Operasi/Filter Citra Baru ke Engine](#1-menambahkan-operasifilter-citra-baru-ke-engine)
   - [2. Menghubungkan Operasi ke Menu Bar & UI](#2-menghubungkan-operasi-ke-menu-bar--ui)
   - [3. Membuat Dialog Interaktif dengan Live Preview](#3-membuat-dialog-interaktif-dengan-live-preview)
   - [4. Menambahkan Unit Test](#4-menambahkan-unit-test)
4. [Konvensi & Standar Kode Tim](#-konvensi--standar-kode-tim)

---

## 🏛️ Arsitektur Sistem & Aliran Data

Aplikasi ini menggunakan pola **Decoupled Engine-UI Architecture** yang memisahkan logika matematika citra dari tampilan:

```
                  ┌──────────────────────────────────────────────┐
                  │                 USER (GUI)                   │
                  └──────────────────────┬───────────────────────┘
                                         │ Action Trigger
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ UI LAYER (PyQt6)                                                                │
│  - MainWindow       : Mengelola Menu, Toolbar, Shortcut, dan Tab Dokumen        │
│  - CanvasWidget     : Merender QImage, Panning, Zooming, Split View             │
│  - HistogramWidget  : Visualisasi Kurva Frekuensi RGB/Grayscale                 │
│  - Dialogs          : Mengambil input pengguna & mengirim sinyal live preview   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Mengirim / Mengambil ImageMatrix
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ENGINE LAYER (Pure NumPy & Matrix Math)                                         │
│  - core.py          : ImageMatrix (Wrapper np.ndarray uint8) + DocumentState    │
│  - io_custom.py     : Parser & Writer Biner/ASCII (PBM, PGM, PPM, BMP, RAW)     │
│  - point_ops.py     : Invert, Grayscale, Brightness, Contrast, Otsu, Gamma      │
│  - arithmetic_ops.py: Add, Subtract, Multiply, Divide, Alpha Blend              │
│  - boolean_ops.py   : AND, OR, NOT, XOR, Masking                                │
│  - geometry_ops.py  : Translate, Rotate, Flip, Scale, Crop                      │
│  - metrics.py       : Histogram, Mean, Variance, Sharpness, Noise               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Dokumentasi Lengkap Berkas & Fungsi

---

### A. Entry Point

#### 📄 `main.py`
* **Tujuan**: Titik masuk utama eksekusi program.
* **Fungsi**:
  * `main()`:
    * Mengaktifkan *High DPI scaling* (`Qt.HighDpiScaleFactorRoundingPolicy.PassThrough`).
    * Menginisialisasi `QApplication`.
    * Membuka jendela `MainWindow`.
    * Menjalankan siklus `app.exec()`.

---

### B. Engine Layer (Logika Citra Murni)

#### 📄 `mini_photoshop/engine/core.py`
* **Tujuan**: Mendefinisikan struktur data citra dan manajemen riwayat Undo/Redo.
* **Kelas & Fungsi**:
  * **`class ImageMatrix`**:
    * Menyimpan array data matriks NumPy `np.ndarray` dengan tipe data `np.uint8`.
    * Menyimpan metadata: `color_mode` (`"GRAYSCALE"`, `"RGB"`, `"RGBA"`), `width`, `height`, dan `channels`.
    * `copy()`: Membuat salinan mendalam (*deep copy*) dari citra.
    * `to_qimage()`: Mengonversi matriks NumPy ke format `QImage` agar siap digambar oleh canvas PyQt6.
    * `from_qimage(qimg)` *(classmethod)*: Mengonversi `QImage` kembali menjadi objek `ImageMatrix`.
  * **`class DocumentState`**:
    * Mengelola riwayat editan satu tab dokumen.
    * `push_state(img_matrix)`: Menambahkan *state* baru ke tumpukan riwayat (menghapus stack redo jika ada editan baru).
    * `undo()`: Memundurkan citra ke status sebelum editan terakhir.
    * `redo()`: Mengulang kembali editan yang sempat dibatalkan.
    * `can_undo()` / `can_redo()`: Mengetahui ketersediaan aksi Undo/Redo untuk mengaktifkan/menonaktifkan tombol toolbar.

---

#### 📄 `mini_photoshop/engine/io_custom.py`
* **Tujuan**: Parser dan writer mandiri tingkat byte & ASCII tanpa pustaka eksternal pihak ketiga (Materi P5).
* **Fungsi**:
  * `read_pbm(filepath)` / `write_pbm(filepath, img, ascii_mode=False)`:
    * Membaca dan menulis format Netpbm PBM 1-bit (`P1` teks ASCII, `P4` biner bitwise).
  * `read_pgm(filepath)` / `write_pgm(filepath, img, ascii_mode=False)`:
    * Membaca dan menulis format Netpbm PGM 8-bit Grayscale (`P2` teks ASCII, `P5` biner byte).
  * `read_ppm(filepath)` / `write_ppm(filepath, img, ascii_mode=False)`:
    * Membaca dan menulis format Netpbm PPM 24-bit TrueColor RGB (`P3` teks ASCII, `P6` biner byte).
  * `read_bmp(filepath)` / `write_bmp(filepath, img)`:
    * Parser native BMP: membaca 14-byte File Header, 40-byte DIB Info Header, color palette, serta menghitung padding baris 4-byte (`((w * bpp + 31) // 32) * 4`).
  * `read_raw(filepath, w, h, channels, bit_depth, offset)` / `write_raw(filepath, img)`:
    * Parser citra mentah biner tanpa header dengan parameter dimensi kustom.
  * `load_image_file(filepath)`:
    * Router pembaca yang otomatis memilih parser native jika ekstensi berkas berupa `.pbm`, `.pgm`, `.ppm`, `.bmp`, atau menggunakan backend universal (Pillow/Qt) untuk `.png`, `.jpg`, `.tiff`, `.webp`.
  * `save_image_file(filepath, img, **kwargs)`:
    * Router penyimpan berkas citra sesuai ekstensi target.

---

#### 📄 `mini_photoshop/engine/point_ops.py`
* **Tujuan**: Operasi aras titik di mana nilai piksel baru dihitung secara independen dari piksel sekitarnya (Materi P6).
* **Fungsi**:
  * `invert(img)`: Citra negatif, $f'(x, y) = 255 - f(x, y)$.
  * `to_grayscale_average(img)`: Konversi grayscale metode rata-rata: $(R+G+B)/3$.
  * `to_grayscale_luminance(img)`: Konversi grayscale luminansi standar NTSC (bobot mata manusia): $0.299R + 0.587G + 0.114B$.
  * `adjust_brightness(img, value)`: Menambah/mengurangi kecerahan dengan *clipping* rentang $[0, 255]$.
  * `adjust_contrast(img, factor)`: Mengatur kontras citra terhadap nilai tengah 128: $128 + factor \times (f(x,y) - 128)$.
  * `contrast_stretching(img)`: Normalisasi dinamis rentang intensitas minimum-maksimum ke skala penuh $[0, 255]$.
  * `threshold_manual(img, thresh)`: Binerisasi dengan nilai ambang $T$ manual.
  * `compute_otsu_threshold(img)`: Algoritma Otsu untuk mencari ambang $T^*$ optimal yang memaksimalkan varians antar-kelas.
  * `threshold_otsu(img)`: Menerapkan binerisasi otomatis menggunakan ambang dari `compute_otsu_threshold`.
  * `gamma_correction(img, gamma)`: Transformasi daya (*Power-Law*): $s = c \cdot r^\gamma$.
  * `posterize(img, bits)`: Penurunan kedalaman bit warna (kuantisasi tingkat keabuan).
  * `solarize(img, threshold)`: Efek solarisasi (membalikkan hanya piksel yang di atas nilai ambang tertentu).

---

#### 📄 `mini_photoshop/engine/arithmetic_ops.py`
* **Tujuan**: Operasi aljabar dan penggabungan matriks antara dua citra (Materi P6).
* **Fungsi**:
  * `add_images(img1, img2, mode="clipped")`: Penjumlahan citra. Pilihan mode: `"clipped"` ($A+B$) atau `"average"` ($(A+B)/2$).
  * `subtract_images(img1, img2, mode="abs")`: Pengurangan citra. Pilihan mode: `"abs"` ($|A-B|$) atau `"clipped"` ($A-B$).
  * `multiply_images(img1, img2)`: Perkalian citra ternormalisasi: $(A \times B) / 255$.
  * `divide_images(img1, img2)`: Pembagian citra dengan pencegahan pembagian dengan nol.
  * `alpha_blend(img1, img2, alpha)`: Penggabungan citra berbobot transparansi: $\alpha A + (1-\alpha) B$.
  * `scalar_operation(img, scalar, op)`: Operasi aritmetika citra tunggal dengan nilai skalar konstan.

---

#### 📄 `mini_photoshop/engine/boolean_ops.py`
* **Tujuan**: Operasi logika biner bitwise antar citra (Materi P6).
* **Fungsi**:
  * `bitwise_not(img)`: Pembalikan bitwise logika NOT (`~`).
  * `bitwise_and(img1, img2)`: Logika bitwise AND (`&`).
  * `bitwise_or(img1, img2)`: Logika bitwise OR (`|`).
  * `bitwise_xor(img1, img2)`: Logika bitwise XOR (`^`).
  * `mask_image(img, mask_img)`: Menerapkan citra masker biner untuk mengisolasi Region of Interest (ROI).

---

#### 📄 `mini_photoshop/engine/geometry_ops.py`
* **Tujuan**: Transformasi spasial geometrik pada matriks koordinat citra (Materi P6).
* **Fungsi**:
  * `translate(img, dx, dy, fill_value=0)`: Menggeser koordinat citra sejauh $\Delta x$ dan $\Delta y$.
  * `flip_horizontal(img)` / `flip_vertical(img)`: Pencerminan matriks horizontal ($X$-axis) atau vertikal ($Y$-axis).
  * `rotate_orthogonal(img, angle)`: Rotasi cepat sudut siku-siku ($90^\circ, 180^\circ, 270^\circ$).
  * `rotate_arbitrary(img, angle, interpolation="bilinear", expand=True)`: Rotasi sudut sembarang $(\theta)$ menggunakan matriks rotasi dan interpolasi *Bilinear* atau *Nearest Neighbor*.
  * `zoom_scale(img, scale_x, scale_y, interpolation="bilinear")`: Penskalaan citra dengan faktor perkalian skala.
  * `resize_exact(img, new_w, new_h, interpolation="bilinear")`: Mengubah ukuran citra ke dimensi piksel target tertentu.
  * `crop(img, x, y, w, h)`: Memotong wilayah persegi panjang tertentu dari citra.

---

#### 📄 `mini_photoshop/engine/metrics.py`
* **Tujuan**: Kalkulasi analisis data statistik, histogram, dan metrik kualitas citra (Materi P1–P3).
* **Fungsi**:
  * `compute_histograms(img)`: Menghitung array sebaran 256 tingkat intensitas untuk channel R, G, B, dan Grayscale.
  * `compute_statistics(img)`: Menghitung nilai piksel minimum, maksimum, rata-rata (mean), standar deviasi, dan entropi informasi citra.
  * `estimate_sharpness(img)`: Mengestimasi tingkat ketajaman gambar berdasarkan varians operator Laplacian ($\nabla^2$).
  * `estimate_noise(img)`: Mengestimasi tingkat noise citra menggunakan algoritma cepat berbasis Immerkaer.

---

### C. UI Layer (Antarmuka & Interaksi)

#### 📄 `mini_photoshop/ui/main_window.py`
* **Tujuan**: Controller utama GUI aplikasi.
* **Tanggung Jawab**:
  * Membuat susunan Menu Bar (File, Edit, View, Image, Operations, Geometry, Analysis, Help) dan Toolbar.
  * Mengelola *Tab Widget* untuk membuka banyak dokumen sekaligus.
  * Menghubungkan setiap sinyal klik menu ke fungsi engine pemrosesan yang sesuai.
  * Mengelola panel samping *Dock Widget* (Panel Histogram & Panel Info Metadata).
  * Menghubungkan pembaruan kursor dari canvas ke status bar (*Pixel Inspector*).

---

#### 📄 `mini_photoshop/ui/canvas.py`
* **Tujuan**: Widget viewport interaktif untuk merender citra dengan performa tinggi.
* **Fitur Utama**:
  * `wheelEvent()`: Zooming halus berbasis posisi kursor mouse.
  * `mousePressEvent()` / `mouseMoveEvent()`: Panning (menggeser kanvas) saat tombol mouse ditekan.
  * `enable_split_view(original_img)`: Mengaktifkan mode tirai pembanding *Before/After*. Menggambar garis pemisah interaktif yang dapat digeser oleh pengguna.
  * `pixelHovered`: Sinyal PyQt yang dipancarkan saat mouse melintas di atas piksel, membawa data `(x, y, r, g, b)`.

---

#### 📄 `mini_photoshop/ui/histogram_widget.py`
* **Tujuan**: Widget custom pembina kurva grafik histogram.
* **Fitur Utama**:
  * Merender grafik distribusi warna RGB dan Grayscale dengan kurva kurva semi-transparan ber-antialiasing halus.
  * Interaksi hover mouse: menampilkan garis vertikal penunjuk intensitas ($0-255$) beserta jumlah piksel pada posisi tersebut via tooltip.

---

#### 📄 `mini_photoshop/ui/styles.py`
* **Tujuan**: Menyediakan konfigurasi visual, palet warna, dan stylesheet CSS untuk tema gelap (*Photoshop Dark Theme*).

---

#### 📁 `mini_photoshop/ui/dialogs/`
* `adjust_dialog.py`: Dialog slider untuk operasi Brightness, Contrast, Threshold, Gamma, Posterize, dan Solarize lengkap dengan sinyal *live preview*.
* `arithmetic_dialog.py`: Dialog pemilihan citra kedua dari tab aktif atau dari berkas komputer untuk operasi aljabar 2 citra dan operasi logika bitwise.
* `geometry_dialog.py`: Dialog input parameter translasi, rotasi sudut bebas, dan penskalaan dimensi kanvas.
* `info_dialog.py`: Dialog laporan metadata detail berkas, resolusi, color space, dan metrik kualitas citra.
* `raw_dialog.py`: Dialog konfigurasi parameter dimensi, channels, dan header offset saat mengimpor citra biner mentah (*RAW*).

---

### D. Test Suite

#### 📄 `tests/test_engine.py`
* **Tujuan**: Kumpulan pengujian otomatis (*Unit Testing*) berbasis `unittest` / `pytest` untuk memverifikasi keakuratan algoritma engine tanpa perlu membuka GUI.
* **Daftar Tes**:
  * `test_01_core_and_undo_redo`: Validasi objek `ImageMatrix` dan stack Undo/Redo.
  * `test_02_native_io_pbm_pgm_ppm_bmp_raw`: Validasi siklus simpan-dan-baca native parser.
  * `test_03_point_operations`: Validasi matematis rumus invert, grayscale, brightness, contrast, dan Otsu.
  * `test_04_arithmetic_and_blending`: Validasi penjumlahan, pengurangan, dan alpha blending citra.
  * `test_05_boolean_and_masking`: Validasi gerbang logika bitwise dan masking citra.
  * `test_06_geometry_ops`: Validasi translasi, flipping, rotasi, dan penskalaan.
  * `test_07_metrics_and_histogram`: Validasi kalkulasi sebaran histogram dan metrik statistik.

---

## 🛠️ Panduan Menambahkan Fitur Baru (How-To Develop)

Berikut adalah panduan bagi rekan tim yang ingin menambahkan fitur atau algoritma baru ke dalam Mini Photoshop.

### 1. Menambahkan Operasi/Filter Citra Baru ke Engine
Misalnya, kita ingin menambahkan fitur **Filter Konvolusi Spasial** (misal: Gaussian Blur / Sobel Edge).

Buka file yang relevan di `mini_photoshop/engine/` (atau buat file baru, misal `spatial_filters.py`):
```python
# mini_photoshop/engine/spatial_filters.py
import numpy as np
from .core import ImageMatrix

def apply_custom_filter(img: ImageMatrix, kernel: np.ndarray) -> ImageMatrix:
    """
    Menerapkan konvolusi 2D pada citra menggunakan matriks kernel.
    """
    arr = img.array.astype(np.float32)
    # Lakukan komputasi matematika NumPy di sini...
    # Pastikan hasil akhir di-clip [0, 255] dan dikonversi ke np.uint8
    result_arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return ImageMatrix(result_arr, color_mode=img.color_mode)
```

---

### 2. Menghubungkan Operasi ke Menu Bar & UI
Buka `mini_photoshop/ui/main_window.py`:

1. **Tambahkan Aksi ke Menu**:
   ```python
   # Di dalam method setup_menus()
   act_custom = QAction("Custom Filter...", self)
   act_custom.triggered.connect(self.action_custom_filter)
   filter_menu.addAction(act_custom)
   ```

2. **Buat Handler Aksi**:
   ```python
   def action_custom_filter(self):
       doc = self.get_current_document()
       if not doc:
           return
       
       # Panggil fungsi engine
       new_img = apply_custom_filter(doc.current, my_kernel)
       
       # Simpan ke riwayat Undo/Redo & perbarui canvas
       doc.push_state(new_img)
       self.refresh_current_tab()
   ```

---

### 3. Membuat Dialog Interaktif dengan Live Preview
Jika operasi memerlukan parameter slider dari pengguna (misal ukuran kernel atau nilai ambang):

1. Buat class turunan dari `QDialog` di dalam `mini_photoshop/ui/dialogs/`.
2. Pasang sinyal `previewUpdated = pyqtSignal(ImageMatrix)`.
3. Setiap kali slider digeser (`valueChanged`), hitung citra sementara dan pancarkan sinyal:
   ```python
   def on_slider_changed(self, val):
       preview_img = apply_custom_filter(self.original_img, val)
       self.previewUpdated.emit(preview_img)
   ```
4. Di `MainWindow`, hubungkan sinyal `dialog.previewUpdated.connect(canvas.set_preview_image)`.

---

### 4. Menambahkan Unit Test
Buka `tests/test_engine.py` dan tambahkan fungsi uji baru:

```python
def test_custom_filter(self):
    dummy_img = ImageMatrix(np.zeros((32, 32), dtype=np.uint8))
    result = apply_custom_filter(dummy_img, my_kernel)
    self.assertEqual(result.width, 32)
    self.assertEqual(result.height, 32)
```

Jalankan pengujian via terminal:
```bash
pytest
```

---

## 📏 Konvensi & Standar Kode Tim

1. **Tipe Data Matriks**:
   - Selalu pastikan data citra di dalam `ImageMatrix` bertipe `np.uint8` dengan nilai piksel $[0, 255]$.
   - Lakukan operasi kalkulasi perantara dalam `np.float32` atau `np.int32` untuk mencegah *integer overflow*, lalu gunakan `np.clip(result, 0, 255).astype(np.uint8)` sebelum mengembalikan `ImageMatrix`.
2. **Prinsip Immutability (Non-destructive)**:
   - Jangan pernah mengubah `img.array` secara *in-place*. Selalu buat array baru (`arr.copy()` atau buat array hasil komputasi baru).
3. **Pemisahan Engine & UI**:
   - Modul di dalam `mini_photoshop/engine/` **tidak boleh** mengimpor komponen GUI PyQt6 (seperti `QWidget`, `QAction`, dll). Modul engine harus murni berisi komputasi numerik NumPy / Python.
4. **Dokumentasi Docstring**:
   - Sertakan *docstring* singkat di setiap fungsi yang menjelaskan rumus matematika, parameter input, dan tipe kembalian.
