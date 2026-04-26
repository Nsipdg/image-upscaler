# 🔬 Image Quality Analyzer
**UTS PBO · 100% GRATIS · Tanpa API Key**

Analisis kualitas foto menggunakan matematika & algoritma PIL/NumPy.
Tidak perlu bayar, tidak perlu API key apapun.

---

## ✨ Fitur
- Upload foto JPG/PNG/WEBP/BMP
- Analisis otomatis: ketajaman (Laplacian), noise, kecerahan, kontras, warna dominan
- Skor kualitas 0–100 + Grade A/B/C/D
- Rekomendasi upscale
- 5 KPI cards dengan delta perbandingan
- Grafik tren garis (time series)
- Grafik batang perbandingan antar foto
- Riwayat lengkap dengan thumbnail

---

## 🚀 Deploy Gratis via Streamlit Cloud (dari HP Android)

### 1. Buat repo GitHub
- Buka **github.com** di browser HP
- Buat akun (gratis) → **New repository** → nama: `image-analyzer`
- Upload `app.py` dan `requirements.txt`

### 2. Deploy
- Buka **share.streamlit.io** → login pakai GitHub
- Klik **New app** → pilih repo → pilih `app.py` → **Deploy**
- Tunggu ~2 menit → dapat link publik ✓

### 3. Selesai!
Tidak perlu isi API key apapun. Langsung pakai.

---

## 📱 Jalankan di Android via Termux

```bash
pkg update && pkg install python
pip install streamlit pillow numpy pandas plotly
streamlit run app.py
# Buka browser: http://localhost:8501
```

---

## 🧮 Cara Kerja Analisis (tanpa AI)

| Metrik | Algoritma |
|--------|-----------|
| Ketajaman | Laplacian variance pada grayscale |
| Noise | Selisih gambar asli vs Gaussian blur |
| Kecerahan | Mean piksel grayscale |
| Kontras | Standard deviation piksel |
| Warna dominan | Mean RGB → klasifikasi |
| Skor kualitas | Weighted sum semua metrik |

---

*UTS PBO · 100% Gratis · No API Key*
