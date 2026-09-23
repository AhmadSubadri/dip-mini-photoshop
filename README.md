<div align="center">

# 🎨 Mini Photoshop
### Desktop Digital Image Processing Application & Framework

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![NumPy](https://img.shields.io/badge/Matrix_Engine-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(7%2F7)-brightgreen?style=for-the-badge&logo=pytest)](tests/)

*Aplikasi desktop pengolahan citra digital modern dengan antarmuka gelap profesional (Adobe Photoshop-style UX), mendukung live preview, multi-tab document, split-screen before/after comparison, histogram real-time, serta parser/writer mandiri untuk format citra standar dan biner mentah (RAW).*

[Fitur Utama](#-fitur-utama) •
[Instalasi](#-instalasi--menjalankan) •
[Matematika & Algoritma](#-pemetaan-algoritma--matematika-citra) •
[Struktur Kode](#-arsitektur--struktur-proyek) •
[Pintasan Keyboard](#-pintasan-keyboard-shortcuts)

</div>

---

## 🌟 Gambaran Umum (Overview)

**Mini Photoshop** dikembangkan sebagai aplikasi komprehensif untuk pengolahan dan analisis citra digital tingkat lanjut. Seluruh operasi piksel, manipulasi matriks numerik, dan parser format citra dibangun secara mandiri menggunakan operasi tensor **NumPy** bertenaga tinggi, dipadukan dengan performa antarmuka grafis **PyQt6**.

---

## ✨ Fitur Utama

### 1. 🖼️ Native Format Parsers & Image I/O
- **Parser & Writer Mandiri (Tanpa Library Eksternal)**:
  - **Netpbm PBM**: Citra monokrom 1-bit (`.pbm` tipe P1 ASCII & P4 Binary).
  - **Netpbm PGM**: Citra grayscale 8-bit (`.pgm` tipe P2 ASCII & P5 Binary).
  - **Netpbm PPM**: Citra RGB 24-bit (`.ppm` tipe P3 ASCII & P6 Binary).
  - **Windows BMP**: Format DIB 24-bit RGB & 8-bit Grayscale lengkap dengan kalkulasi byte row-padding (4-byte boundary).
  - **RAW Image Import/Export**: Konfigurasi fleksibel dimensi ($W \times H$), channels, bit depth, dan header byte offset.
- **Universal Formats**: Integrasi seamless untuk JPEG, PNG, TIFF, GIF, WEBP.
- **Image Metadata Viewer**: Inspeksi resolusi, rasio aspek, kedalaman bit, ukuran memori/file, min/max intensitas piksel, rata-rata kanal, dan dump header.

### 2. ⚡ Operasi Aras Titik (Point Operations)
- **Inversi / Citra Negatif**: Membalikkan seluruh tingkat intensitas warna.
- **Konversi Grayscale Presisi**:
  - *Metode Rata-rata*: $(R + G + B) / 3$
  - *Metode Luminansi Standar NTSC (ITU-R BT.601)*: $0.299R + 0.587G + 0.114B$
- **Kecerahan (Brightness) & Kontras**: Pengaturan interaktif dengan *live preview* dan proteksi *clipping*.
- **Peregangan Kontras (Contrast Stretching)**: Normalisasi dinamis rentang intensitas ke $[0, 255]$.
- **Binerisasi / Thresholding**:
  - *Manual Thresholding*: Slider ambang dinamis $[0, 255]$.
  - *Otsu Thresholding*: Binerisasi otomatis berbasis maksimasi varians antar kelas (*inter-class variance*).
- **Koreksi Gamma (Power-Law Transformation)**: Transformasi kurva pencahayaan non-linear.
- **Posterisasi & Solarize**: Kuantisasi kedalaman bit $(1, 2, 4, 8\text{ bit})$ dan efek solarisasi artistik.

### 3. ➕ Operasi Aritmetika & Logika Citra
- **Aritmetika Dua Citra**:
  - Penjumlahan (*Addition*): Mode *Clipped* ($A+B$) dan *Average Blending* ($(A+B)/2$).
  - Pengurangan (*Subtraction*): Mode *Absolute Difference* ($|A-B|$) dan *Directional Difference*.
  - Perkalian (*Multiplication*) & Pembagian (*Division*): Normalisasi dinamis $255$.
  - **Alpha Blending**: Penggabungan dua citra bertahap dengan bobot $\alpha \in [0.0, 1.0]$.
- **Operasi Logika Bitwise**:
  - Gerbang logika biner `AND`, `OR`, `NOT`, `XOR`.
  - Masking citra spasial dengan citra biner / ROI (*Region of Interest*).

### 4. 📐 Transformasi Geometri (Spatial Transformations)
- **Translasi Spasial**: Pergeseran koordinat horizontal ($\Delta X$) dan vertikal ($\Delta Y$).
- **Rotasi Ortogonal & Bebas**:
  - Rotasi cepat $90^\circ$ Searah Jarum Jam, $90^\circ$ Berlawanan Jarum Jam, $180^\circ$.
  - Rotasi sudut bebas sembarang ($\theta$) dengan interpolasi *Bilinear* / *Nearest Neighbor* dan opsi *Auto-expand Canvas*.
- **Pencerminan (Flip)**: Horizontal Flip ($X$-axis mirror) dan Vertical Flip ($Y$-axis mirror).
- **Zooming & Scaling**: Resizing dimensi bebas ($W \times H$) maupun skala persentase.

### 5. 📊 Analisis Citra & Metrik Kualitas
- **Live RGB & Grayscale Histogram**: Visualisasi sebaran intensitas piksel real-time, grafik kurva antialiased, dan tooltip nilai frekuensi saat hover.
- **Pixel Inspector (Real-time HUD)**: Koordinat kursor $(X, Y)$, nilai intensitas kanal $[R, G, B]$, rasio perbesaran layar, dan dimensi kanvas pada status bar.
- **Kalkulasi Kualitas Citra**:
  - *Sharpness Estimation*: Varians operator Laplacian ($\nabla^2 f$).
  - *Noise Level Estimation*: Algoritma estimasi derau berbasis Immerkaer.

### 6. 🖥️ Photoshop-Grade User Experience (UX)
- **Photoshop Dark UI**: Desain antarmuka gelap modern, elegan, dan ramah untuk penggunaan jangka panjang.
- **Multi-Tab Document Workspace**: Kemampuan membuka dan menyunting banyak citra sekaligus.
- **Interactive Canvas Engine**: Panning halus (klik-seret kursor) dan zooming berbasis titik kursor (*mouse wheel zoom*).
- **Split-Screen Comparison**: Tirai pemisah vertikal *Before / After* interaktif untuk membandingkan citra sebelum dan sesudah diedit.
- **Unlimited Undo / Redo History**: Manajemen riwayat *snapshot* berbasis *Document State Management*.
- **Bundled Benchmark Samples**: Dilengkapi citra uji standar (Lena, Baboon, Fruits, PBM, PGM, PPM, BMP).

---

## 🚀 Instalasi & Menjalankan

### Prasyarat
- **Python 3.10** atau versi lebih baru (Disarankan Python 3.11)
- Pip package manager

### 1. Kloning Repositori
```bash
git clone https://github.com/username/mini-photoshop.git
cd mini-photoshop/mini_photoshop_project
```

### 2. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
python main.py
```

### 4. Menjalankan Unit Tests Otomatis
```bash
pytest
```

---

## 🧮 Pemetaan Algoritma & Matematika Citra

| Fitur / Operasi | Formulasi Matematika |
| :--- | :--- |
| **Citra Negatif** | $f'(x, y) = 255 - f(x, y)$ |
| **Grayscale (Luminansi NTSC)** | $Y(x, y) = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$ |
| **Penyesuaian Kecerahan** | $f'(x, y) = \min(255, \max(0, f(x, y) + b))$ |
| **Penyesuaian Kontras** | $f'(x, y) = \text{clip}\left(128 + c \cdot (f(x, y) - 128), 0, 255\right)$ |
| **Contrast Stretching** | $f'(x, y) = \frac{f(x, y) - f_{\min}}{f_{\max} - f_{\min}} \times 255$ |
| **Koreksi Gamma** | $s = 255 \times \left(\frac{r}{255}\right)^\gamma$ |
| **Ambang Batas Otsu** | $\sigma_B^2(t) = \omega_0(t)\omega_1(t)\left[\mu_0(t) - \mu_1(t)\right]^2 \rightarrow \max$ |
| **Alpha Blending** | $C(x, y) = \alpha \cdot A(x, y) + (1 - \alpha) \cdot B(x, y)$ |
| **Rotasi Matriks** | $\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x - x_c \\ y - y_c \end{bmatrix} + \begin{bmatrix} x_c \\ y_c \end{bmatrix}$ |

---

## 🏗️ Arsitektur & Struktur Proyek

```text
mini_photoshop_project/
│
├── main.py                     # Entry point eksekusi aplikasi
├── requirements.txt            # Daftar pustaka dependensi (PyQt6, NumPy, Pillow, PySide6-addons)
├── README.md                   # Dokumentasi teknis repositori
├── .gitignore                  # Konfigurasi filter Git
│
├── mini_photoshop/             # Paket utama aplikasi
│   ├── engine/                 # Core logic & algoritma pemrosesan citra murni
│   │   ├── core.py             # ImageMatrix & DocumentState (Undo/Redo stack)
│   │   ├── io_custom.py        # Parser & Writer biner/ASCII (PBM, PGM, PPM, BMP, RAW)
│   │   ├── point_ops.py        # Algoritma aras titik (Negatif, Grayscale, Brightness, dll)
│   │   ├── arithmetic_ops.py   # Operasi aritmetika 2 citra & alpha blend
│   │   ├── boolean_ops.py      # Operasi logika bitwise (AND, OR, NOT, XOR, Mask)
│   │   ├── geometry_ops.py     # Transformasi geometri (Translasi, Rotasi, Scaling, Flip)
│   │   └── metrics.py          # Kalkulasi histogram, statistik, & estimasi kualitas
│   │
│   ├── ui/                     # Komponen antarmuka pengguna (PyQt6)
│   │   ├── main_window.py      # Jendela utama, Menu bar, Toolbar, Tab manager
│   │   ├── canvas.py           # Canvas interaktif (Pan, Zoom, Split-screen Before/After)
│   │   ├── histogram_widget.py # Live RGB/Grayscale histogram visualization
│   │   ├── styles.py           # Tema Photoshop Dark QSS
│   │   └── dialogs/            # Dialog parameter interaktif & live preview
│   │       ├── adjust_dialog.py
│   │       ├── arithmetic_dialog.py
│   │       ├── geometry_dialog.py
│   │       ├── info_dialog.py
│   │       └── raw_dialog.py
│   │
│   └── samples/                # Koleksi citra uji benchmark bawaan
│
└── tests/
    └── test_engine.py          # Unit & integration tests otomatis
```

---

## ⌨️ Pintasan Keyboard (Shortcuts)

| Pintasan | Aksi |
| :--- | :--- |
| `Ctrl + N` | Buat Kanvas Baru (New Document) |
| `Ctrl + O` | Buka Berkas Citra (Open Image) |
| `Ctrl + S` | Simpan Citra (Save Image) |
| `Ctrl + Shift + S` | Simpan Sebagai (Save As) |
| `Ctrl + Z` | Batalkan Perubahan (Undo) |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Ulangi Perubahan (Redo) |
| `Ctrl + T` | Beralih Tampilan Tirai Pembanding (*Toggle Split View Before/After*) |
| `Ctrl + +` / `Mouse Wheel Up` | Perbesar Citra (Zoom In) |
| `Ctrl + -` / `Mouse Wheel Down` | Perkecil Citra (Zoom Out) |
| `Ctrl + 0` | Sesuaikan Ukuran Layar (*Fit on Screen*) |
| `Ctrl + 1` | Ukuran Asli 100% (*Actual Size*) |
| `Ctrl + I` | Buka Dialog Informasi & Metadata Citra |
| `Ctrl + W` | Tutup Tab Aktif |
| `Ctrl + Q` | Keluar dari Aplikasi |

---

## 👨‍💻 Penulis & Pengembang

* **Ahmad Subadri**
* **Program Studi**: Magister Informatika / Ilmu Komputer (S2)
* **Mata Kuliah**: Pengolahan dan Analisis Citra Digital

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE) - bebas digunakan, dimodifikasi, dan didistribusikan untuk kepentingan akademik maupun pengembangan perangkat lunak.
