# 🎬 Naskah Presentasi & Panduan Rekaman Video (Screen Record)
## Fitur Baru: Operasi Spasial Lokal (Linier & Non-Linier), Simulasi Derau (Noise), dan Evaluasi Restorasi (PSNR & MSE)

> **Catatan Penggunaan:**
> Dokumen ini dirancang khusus agar Anda dapat membacanya langsung di layar laptop saat melakukan perekaman layar (*screen recording*) pada monitor eksternal.
> 
> Teks terbagi menjadi 2 bagian utama:
> 1. **Bagian I: Naskah Demo Aplikasi (Skenario Aksi Layar + Teks Bacaan Lisan)**
> 2. **Bagian II: Naskah Penjelasan Kode Sumber (Source Code Walkthrough per Fungsi)**

---

# BAGIAN I: NASKAH DEMO APLIKASI (LIVE DEMO)

---

### 📍 Scene 1: Pembukaan & Pengantar
* **[Aksi Layar]:** Tampilkan jendela Mini Photoshop yang sudah terbuka di layar penuh, dengan citra uji standar (misal `lena.png` atau buka lewat menu *File -> Open Sample Image -> lena.png*).
* **[Teks Bacaan / Ucapkan]:**
> "Assalamu’alaikum warahmatullahi wabarakatuh, selamat pagi/siang rekan-rekan dan Bapak Dosen.
> 
> Pada kesempatan kali ini, saya akan mendemonstrasikan penambahan fitur baru pada aplikasi Mini Photoshop, yaitu **Operasi Spasial Lokal (Neighborhood Operations)** yang mencakup operasi linier dan non-linier, serta fitur **Simulasi dan Reduksi Derau (Noise Restoration)**.
> 
> Seluruh fitur ini dibangun secara **mandiri (from scratch)** menggunakan perhitungan matematika dan manipulasi matriks numerik NumPy murni, **tanpa menggunakan library pengolahan citra seperti OpenCV, SciPy, ataupun PIL ImageFilter**.
> 
> Mari kita mulai demonstrasi dari fitur Operasi Lokal terlebih dahulu."

---

### 📍 Scene 2: Demo Operasi Spasial Lokal Linier (Mean, Gaussian, Sharpening, Roberts, Sobel)
* **[Aksi Layar]:** Klik menu **Filter** di menubar atas, lalu pilih **Operasi Spasial Lokal (Dialog Lengkap)...** (atau tunjukkan shortcut `Ctrl + F`). Jendela dialog modal akan muncul di atas canvas.
* **[Teks Bacaan / Ucapkan]:**
> "Pertama, kita buka menu **Filter**, lalu pilih **Operasi Spasial Lokal**.
> 
> Dialog ini dilengkapi fitur **Live Preview real-time**. Artinya, setiap kali kita mengubah parameter, perubahan langsung terlihat di kanvas tanpa harus menutup dialog. Di bagian bawah juga terdapat kotak teori yang menampilkan rumus matematika yang sedang dieksekusi secara transparan.
> 
> Pada kategori **Operasi Linier**, piksel baru dihitung dari konvolusi spasial antara tetangga piksel dengan matriks kernel berbobot.
> 
> 1. **Mean Filter:** Kita bisa memilih ukuran kernel fleksibel, misalnya **2x2** atau **3x3**. Saat saya pilih kernel 3x3, citra menjadi halus secara merata karena setiap piksel dirata-ratakan dengan sembilan tetangganya dengan bobot 1/9.
> 2. **Gaussian Blur:** Sekarang kita coba Gaussian Blur. Berbeda dengan mean, filter ini menggunakan pembobotan kurva distribusi normal Gauss. Kita bisa mengatur nilai **Sigma**. Efek blur yang dihasilkan terlihat jauh lebih natural dan halus.
> 3. **Sharpening:** Selanjutnya, jika kita pilih filter penajaman, kernel High-Pass Laplacian akan mempertegas kontras pada garis tepi objek.
> 4. **Deteksi Tepi Roberts (Kernel 2x2):** Sesuai materi, kita sediakan operator Roberts yang menggunakan kernel 2x2 gradien silang diagonal. Garis tepi terdeteksi tajam dengan komputasi yang sangat cepat.
> 5. **Deteksi Tepi Sobel (Kernel 3x3):** Sedangkan Sobel menggunakan kernel 3x3 diferensial berbobot untuk mendeteksi tepi horizontal dan vertikal.
> 
> Selain itu, pengguna juga bisa memilih **Custom Kernel** untuk mengetikkan sendiri angka-angka matriks konvolusi 3x3 bebas sesuai kebutuhan.
> 
> Sekarang kita batalkan dulu dengan klik **Cancel** untuk mengembalikan citra ke kondisi awal."

* **[Aksi Layar]:** Klik tombol **Cancel** pada dialog (tunjukkan kanvas kembali ke citra asli).

---

### 📍 Scene 3: Demo Operasi Spasial Lokal Non-Linier (Median, Max, Min Filter)
* **[Aksi Layar]:** Buka kembali dialog dengan menekan `Ctrl + F`. Klik radio button **Operasi Non-Linier (Rank-Order Filter)**.
* **[Teks Bacaan / Ucapkan]:**
> "Sekarang kita beralih ke kategori kedua, yaitu **Operasi Non-Linier**.
> 
> Berbeda dengan operasi linier yang mengalikan bobot kernel, operasi non-linier menggunakan teknik pemeringkatan atau pengurutan (*sorting*) nilai intensitas piksel tetangga di jendela lokal.
> 
> 1. **Median Filter:** Mengurutkan piksel tetangga lalu mengambil nilai median (tengah). Ini adalah metode terbaik untuk menghilangkan noise impulsif.
> 2. **Max Filter:** Jika saya pilih Max Filter, citra akan digantikan dengan nilai intensitas tertinggi di lingkungannya. Terlihat area terang membesar dan bintik-bintik gelap hilang.
> 3. **Min Filter:** Sebaliknya, Min Filter mengambil nilai intensitas terendah, sehingga area gelap meluas dan bintik-bintik putih tereduksi.
> 
> Kita klik **Cancel** kembali untuk bersiap masuk ke skenario restorasi derau."

* **[Aksi Layar]:** Klik tombol **Cancel**.

---

### 📍 Scene 4: Demo Simulasi Penambahan Derau (Add Noise)
* **[Aksi Layar]:** Klik menu **Noise & Restorasi** di menubar atas, pilih **Simulasi Tambah Derau (Add Noise)...**.
* **[Teks Bacaan / Ucapkan]:**
> "Fitur selanjutnya adalah simulasi penambahan derau buatan untuk menguji keandalan restorasi citra.
> 
> Kita buka menu **Noise & Restorasi**, lalu pilih **Simulasi Tambah Derau**.
> 
> Di sini tersedia tiga jenis derau:
> 1. **Salt & Pepper Noise:** Memodelkan kerusakan transmisi data di mana piksel secara acak rusak menjadi putih pekat (nilai 255) atau hitam pekat (nilai 0).
> 2. **Gaussian Noise:** Menambahkan gangguan acak berdistribusi normal Gauss ke seluruh piksel.
> 3. **Speckle Noise:** Derau perkalian (*multiplicative*) proporsional kecerahan citra.
> 
> Untuk pengujian tugas ini, kita pilih **Salt & Pepper Noise** dengan kerapatan derau **10%**. Terlihat pada live preview citra dipenuhi bintik-bintik hitam putih. Sekarang kita terapkan dengan menekan tombol **Apply**."

* **[Aksi Layar]:** Atur slider Salt & Pepper ke **10%**, lalu klik **Apply**. Citra di kanvas sekarang tampak berderau bintik-bintik.

---

### 📍 Scene 5: Demo Reduksi Derau & Evaluasi Kualitas Restorasi (PSNR & MSE)
* **[Aksi Layar]:** Buka menu **Noise & Restorasi**, pilih **Reduksi Derau & Evaluasi Restorasi (PSNR/MSE)...**.
* **[Teks Bacaan / Ucapkan]:**
> "Sekarang citra kita sudah terkontaminasi derau Salt & Pepper 10%. Selanjutnya kita akan mereduksinya sekaligus membuktikan hasil perbaikannya secara ilmiah dan kuantitatif.
> 
> Kita buka menu **Reduksi Derau & Evaluasi Restorasi**.
> 
> Di dialog ini, sistem langsung membandingkan citra berderau saat ini dengan citra asli sebelum diberi derau menggunakan rumus **MSE (Mean Squared Error)** dan **PSNR (Peak Signal-to-Noise Ratio)**.
> 
> Perhatikan pada kotak evaluasi:
> - Pada kondisi noisy, nilai MSE sangat tinggi (sekitar ribuan) dan nilai PSNR anjlok ke angka sekitar **15 desibel (dB)**.
> - Begitu kita pilih metode **Median Filter (3x3)**, bintik-bintik derau langsung bersih total, MSE turun drastis, dan skor PSNR melonjak naik menjadi di atas **28 hingga 30 dB**.
> - Di sini tertulis status: **Kualitas Meningkat! dengan delta perbaikan mencapai lebih dari +13 dB**.
> 
> Hal ini membuktikan secara ilmiah bahwa Median Filter adalah solusi paling tepat dan efektif untuk mereduksi derau Salt & Pepper tanpa merusak ketajaman citra.
> 
> Kita klik **Apply** untuk menyimpan hasil restorasi."

* **[Aksi Layar]:** Klik tombol **Apply**. Tunjukkan citra bersih kembali. Lalu tekan `Ctrl + T` (Split View Before/After) dan geser garis pemisah untuk memperlihatkan fitur pembanding.
* **[Teks Bacaan / Ucapkan]:**
> "Kita juga bisa menggunakan fitur Split View dengan menekan tombol Ctrl + T untuk membandingkan secara langsung kondisi sebelum dan sesudah restorasi.
> 
> Demikian demonstrasi langsung pada antarmuka aplikasi. Selanjutnya, saya akan memperlihatkan dan menjelaskan baris kode implementasi per fungsi di kode program."

---

# BAGIAN II: NASKAH PENJELASAN KODE SUMBER (SOURCE CODE)

* **[Aksi Layar]:** Pindahkan tampilan layar monitor ke editor kode (VS Code / IDE) dan buka berkas `mini_photoshop/engine/spatial_ops.py`.

---

### 📄 Berkas 1: `mini_photoshop/engine/spatial_ops.py`

#### 1. Fungsi `manual_pad2d()`
* **[Aksi Layar]:** Scroll ke fungsi `manual_pad2d()` (sekitar baris 30).
* **[Teks Bacaan / Ucapkan]:**
> "Fungsi pertama pada engine adalah `manual_pad2d()`.
> 
> **Kegunaan:** Menambahkan bingkai batas piksel tambahan di sekeliling matriks citra sebelum dilakukan proses konvolusi atau filtering jendela lokal. Tujuannya adalah agar ukuran citra keluaran tetap sama persis dengan citra masukan, dan piksel di tepi tidak terpotong.
> 
> **Penjelasan Baris Kode:**
> - Di baris 53, kita alokasikan matriks baru berisi nol berukuran `(H + padding, W + padding)`.
> - Di baris 56, citra asli disalin tepat di tengah matriks padding tersebut.
> - Di baris 64 sampai 77, pada mode `replicate`, kita duplikasi piksel baris terluar atas, bawah, kiri, dan kanan. Dengan menduplikasi batas tepi, kita menghindari terjadinya artefak bingkai gelap yang biasanya timbul jika menggunakan zero-padding."

#### 2. Fungsi `convolve2d_manual()` & `apply_kernel_to_image()`
* **[Aksi Layar]:** Scroll ke fungsi `convolve2d_manual()` (sekitar baris 118).
* **[Teks Bacaan / Ucapkan]:**
> "Fungsi kedua adalah inti dari operasi linier, yaitu `convolve2d_manual()`.
> 
> **Kegunaan:** Menghitung konvolusi spasial 2D manual berdasarkan rumus:
> `g(y, x) = jumlah perkalian f(y+i, x+j) dengan kernel(i, j)`.
> 
> **Penjelasan Baris Kode:**
> - Pertama, citra dikonversi ke tipe data `float32` agar kalkulasi perkalian tidak mengalami overflow di atas angka 255.
> - Kemudian kita panggil `manual_pad2d()` sesuai ukuran kernel yang digunakan.
> - Untuk efisiensi pergeseran jendela (*sliding window*), kita menggunakan `sliding_window_view` dari NumPy. Ini merepresentasikan pergeseran jendela tetangga f(y+i, x+j) di setiap koordinat piksel tanpa duplikasi memori.
> - Kemudian jendela dikalikan dengan kernel dan dijumlahkan menggunakan `np.sum(windows * kernel, axis=(-2, -1))`.
> - Fungsi pendukung `apply_kernel_to_image()` bertugas menerapkan konvolusi ini ke setiap kanal warna (kanal Grayscale tunggal, maupun kanal R, G, B pada citra berwarna), lalu memotong nilai (*clip*) ke rentang valid [0, 255] bertipe uint8."

#### 3. Fungsi Filter Linier: `apply_mean_filter()`, `apply_gaussian_filter()`, dll.
* **[Aksi Layar]:** Tunjukkan fungsi `create_mean_kernel()` dan `create_gaussian_kernel()`.
* **[Teks Bacaan / Ucapkan]:**
> "Berikutnya adalah pembuatan matriks kernel:
> - Pada `create_mean_kernel(size)`: seluruh elemen matriks berukuran size x size diisi nilai yang sama rata, yaitu `1 / (size * size)`. Untuk ukuran 2x2 nilainya 1/4, dan untuk 3x3 nilainya 1/9.
> - Pada `create_gaussian_kernel(size, sigma)`: kita hitung bobot tiap sel menggunakan rumus fungsi Gauss 2 dimensi: `(1 / 2*pi*sigma^2) * exp(-(dx^2 + dy^2)/(2*sigma^2))`. Di akhir, seluruh elemen dinormalisasi dengan membaginya terhadap jumlah total elemen agar total bobotnya bernilai tepat 1.0.
> - Untuk `apply_edge_roberts()`: kita definisikan kernel gradien 2x2 Gx dan Gy, lalu magnitudo gradien dihitung dengan rumus Pythagoras: `akar dari (Gx kuadrat + Gy kuadrat)`.
> - Begitu juga dengan `apply_edge_sobel()`, kita gunakan matriks diferensial 3x3 Sobel horizontal dan vertikal."

#### 4. Fungsi Filter Non-Linier: `rank_order_filter2d_manual()` & `apply_median_filter()`
* **[Aksi Layar]:** Scroll ke fungsi `rank_order_filter2d_manual()` (sekitar baris 315).
* **[Teks Bacaan / Ucapkan]:**
> "Selanjutnya adalah operasi non-linier pada fungsi `rank_order_filter2d_manual()`.
> 
> **Kegunaan:** Menghitung filter statistik urutan (*order-statistic*). Operasi ini sama sekali tidak menggunakan perkalian bobot kernel.
> 
> **Penjelasan Baris Kode:**
> - Di setiap jendela lokal berukuran k x k (misal 3x3 = 9 piksel), kita mengambil seluruh piksel tetangga.
> - Untuk **Median Filter**: piksel-piksel tetangga diurutkan dari nilai terkecil ke terbesar (*sorting*), kemudian kita mengambil nilai di posisi tengah atau median. Bintik derau 0 (hitam pekat) atau 255 (putih pekat) akan terlempar ke posisi paling ujung dan tidak akan pernah terpilih, sehingga noise hilang secara otomatis tanpa mengaburkan tepi objek.
> - Untuk **Max Filter**: kita mengambil nilai `np.max` di jendela lokal.
> - Untuk **Min Filter**: kita mengambil nilai `np.min` di jendela lokal."

---

### 📄 Berkas 2: `mini_photoshop/engine/noise_ops.py`
* **[Aksi Layar]:** Buka berkas `mini_photoshop/engine/noise_ops.py`.

#### 1. Fungsi `add_salt_and_pepper_noise()`
* **[Aksi Layar]:** Tunjukkan fungsi `add_salt_and_pepper_noise()`.
* **[Teks Bacaan / Ucapkan]:**
> "Pada berkas `noise_ops.py`, kita mengimplementasikan generator derau manual:
> 
> Pada `add_salt_and_pepper_noise()`:
> - Kita hitung total piksel yang rusak berdasarkan persentase `amount`. Misalnya jika amount 10% pada citra 512x512, maka ada 26.214 piksel yang akan dirusak.
> - Kita acak indeks koordinat piksel menggunakan `rng.choice(replace=False)` agar tidak ada koordinat yang dobel.
> - Setengah dari indeks tersebut kita ubah nilainya menjadi **255 (Salt / putih)**, dan setengahnya lagi kita ubah nilainya menjadi **0 (Pepper / hitam)**."

#### 2. Fungsi `add_gaussian_noise()` & `add_speckle_noise()`
* **[Aksi Layar]:** Scroll ke fungsi `add_gaussian_noise()`.
* **[Teks Bacaan / Ucapkan]:**
> "- Pada `add_gaussian_noise()`: kita membangkitkan matriks bilangan acak terdistribusi normal dengan rata-rata 0 dan standar deviasi sigma menggunakan `rng.normal()`. Lalu nilai derau ini ditambahkan ke piksel asli dan dipotong dengan `np.clip(0, 255)`.
> - Pada `add_speckle_noise()`: nilai derau dikalikan dengan intensitas citra asli sebelum ditambahkan, memodelkan derau perkalian seperti pada citra USG atau Radar."

---

### 📄 Berkas 3: `mini_photoshop/engine/metrics.py`
* **[Aksi Layar]:** Buka berkas `mini_photoshop/engine/metrics.py` dan scroll ke bagian paling bawah (baris 275).

#### 1. Fungsi `compute_mse()` dan `compute_psnr()`
* **[Aksi Layar]:** Tunjukkan fungsi `compute_mse()` dan `compute_psnr()`.
* **[Teks Bacaan / Ucapkan]:**
> "Terakhir, pada berkas `metrics.py`, kita menambahkan fungsi evaluasi kuantitatif kualitas citra manual:
> 
> 1. **`compute_mse(img1, img2)`**: Menghitung Mean Squared Error. Baris kodenya menghitung selisih kuadrat antara piksel citra asli dan citra pembanding: `diff_sq = (arr1 - arr2) ** 2`, lalu diambil nilai rata-ratanya menggunakan `np.mean()`.
> 2. **`compute_psnr(img1, img2)`**: Menghitung Peak Signal-to-Noise Ratio dalam satuan desibel (dB) dengan rumus: `10 * log10( (255^2) / MSE )`. Jika MSE bernilai 0 (citra identik sempurna), PSNR bernilai tak hingga (*infinity*). Semakin tinggi angka PSNR, semakin mirip citra hasil restorasi dengan citra aslinya.
> 
> Nilai-nilai inilah yang ditampilkan secara otomatis di dialog reduksi derau tadi untuk membuktikan keberhasilan algoritma secara matematis."

---

### 📍 Scene Penutup
* **[Aksi Layar]:** Kembali ke tampilan jendela utama Mini Photoshop atau buka terminal yang menampilkan hasil `pytest` (19 passed).
* **[Teks Bacaan / Ucapkan]:**
> "Seluruh fungsi yang telah saya jelaskan ini telah diuji dengan unit testing otomatis, dan seluruh sembilan belas pengujian lulus seratus persen.
> 
> Demikian presentasi penambahan fitur Operasi Spasial Lokal, Simulasi Derau, dan Restorasi Citra pada aplikasi Mini Photoshop ini. Terima kasih atas perhatian rekan-rekan dan Bapak Dosen.
> 
> Wassalamu’alaikum warahmatullahi wabarakatuh."

---
*(Dokumen selesai. Anda dapat langsung menjalankan rekaman layar sambil membaca teks ini.)*
