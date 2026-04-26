# 🖼️ Image Upscaler — Pemrograman Berbasis Objek

**UTS Project | Pemrograman Berbasis Objek**  
Program upscaling gambar interaktif menggunakan Python dengan konsep OOP lengkap.

---

## 📌 Deskripsi

Program ini mengimplementasikan berbagai algoritma upscaling gambar menggunakan prinsip **Pemrograman Berbasis Objek (OOP)**:

- **Inheritance** — `NearestNeighborUpscaler`, `BilinearUpscaler`, dll. mewarisi `BaseUpscaler`
- **Polymorphism** — setiap subclass mengimplementasikan metode `upscale()` sendiri
- **Encapsulation** — logika tersimpan rapi dalam class
- **Composition** — `ImageUpscalerManager` mengelola banyak upscaler sekaligus

---

## 🗂️ Struktur Proyek

```
image_upscaler/
│
├── image_upscaler.py     # Program utama
├── requirements.txt      # Dependensi
├── README.md             # Dokumentasi
│
└── output/               # Hasil otomatis dibuat
    ├── demo_input_nearest_2x.png
    ├── demo_input_bilinear_2x.png
    ├── demo_input_bicubic_2x.png
    ├── demo_input_lanczos_2x.png
    ├── dashboard.png          # Dashboard visualisasi
    └── report.json            # Laporan JSON
```

---

## ⚙️ Metode Upscaling

| Metode | Kecepatan | Kualitas | Deskripsi |
|---|---|---|---|
| Nearest Neighbor | ⚡⚡⚡ Tercepat | ⭐ Rendah | Sederhana, cocok untuk pixel art |
| Bilinear | ⚡⚡ Cepat | ⭐⭐ Sedang | Interpolasi linear 2D |
| Bicubic | ⚡ Sedang | ⭐⭐⭐ Baik | Interpolasi cubic, lebih halus |
| Lanczos | ⚡ Sedang | ⭐⭐⭐⭐ Terbaik | Filter sinc, detail paling tajam |

---

## 📊 Visualisasi (Objek Interaktif)

Dashboard otomatis menghasilkan:

1. **Grafik Tren** — Waktu proses setiap run (time series)
2. **Grafik Perbandingan** — Perbandingan kecepatan antar metode
3. **Scatter Plot** — Kualitas vs Kecepatan
4. **KPI Panel** — Key Performance Indicators (PSNR terbaik, tercepat, dll.)

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/[NamaAnda]/image-upscaler.git
cd image-upscaler
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Program
```bash
python image_upscaler.py
```

### 4. Gunakan Gambar Sendiri
Edit bagian `main()` di `image_upscaler.py`:
```python
# Ganti path ini dengan path gambar kamu
results = manager.process_image("foto_kamu.jpg")
```

---

## 📈 Contoh Output

```
=======================================================
   IMAGE UPSCALER - Pemrograman Berbasis Objek
=======================================================

  Memproses: demo_input.png
  Ukuran input: 200x200 px
  Scale factor: 2x

  ✓ Nearest Neighbor    | Waktu: 0.0012s | PSNR: 30.21 dB | Output: 400x400
  ✓ Bilinear            | Waktu: 0.0023s | PSNR: 34.87 dB | Output: 400x400
  ✓ Bicubic             | Waktu: 0.0031s | PSNR: 36.54 dB | Output: 400x400
  ✓ Lanczos             | Waktu: 0.0034s | PSNR: 37.92 dB | Output: 400x400
```

---

## 🧠 Konsep OOP yang Diterapkan

```
BaseUpscaler (Abstract Base Class)
│   - name, scale_factor, metrics
│   - upscale() [abstract]
│   - process_and_record()
│   - calculate_psnr()
│   - get_average_metrics()
│
├── NearestNeighborUpscaler
├── BilinearUpscaler
├── BicubicUpscaler
└── LanczosUpscaler

ImageUpscalerManager (Composition)
│   - Mengelola semua upscaler
│   - process_image()
│   - benchmark()
│   - save_report()

UpscalerDashboard (Visualization)
    - create_dashboard()
```

---

## 📋 Metrik Performa

- **PSNR (Peak Signal-to-Noise Ratio)** — Mengukur kualitas gambar hasil upscaling (semakin tinggi semakin baik, satuan dB)
- **Processing Time** — Waktu yang dibutuhkan untuk upscaling (detik)

---

## 👤 Informasi Mahasiswa

- **Nama**: [Nama Anda]
- **NIM**: [NIM Anda]
- **Mata Kuliah**: Pemrograman Berbasis Objek
- **Dosen**: [Nama Dosen]

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan akademik.
