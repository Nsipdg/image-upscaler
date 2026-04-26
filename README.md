# 🔍 Image Quality Analyzer
**UTS Pemrograman Berbasis Objek — AI-powered Image Analysis**

---

## ✨ Fitur
- **Upload foto** JPG, PNG, WEBP, BMP
- **Analisis AI** menggunakan Claude (Anthropic) — skor kualitas, ketajaman, noise, warna, dan rekomendasi upscale
- **KPI Dashboard** — 5 kartu metrik utama dengan delta perbandingan
- **Grafik Tren Garis** — time series skor kualitas, ketajaman, dan noise
- **Grafik Batang** — perbandingan antar foto
- **Riwayat** — semua foto yang pernah dianalisis dengan thumbnail

---

## 🚀 Deploy ke Streamlit Cloud (GRATIS, tanpa Windows)

### Langkah dari HP Android:

1. **Buat akun GitHub** di [github.com](https://github.com) (pakai browser HP)
2. **Buat repository baru** → nama: `image-quality-analyzer`
3. **Upload kedua file ini** (`app.py` dan `requirements.txt`) ke repo tersebut
4. **Buka** [share.streamlit.io](https://share.streamlit.io) dan login pakai GitHub
5. Klik **"New app"** → pilih repo → pilih `app.py` → klik **Deploy**
6. Tunggu ~2 menit → aplikasi online dan bisa diakses siapapun!

### Setelah deploy:
- Masukkan **Anthropic API Key** di bagian atas app
- Dapatkan API key gratis di [console.anthropic.com](https://console.anthropic.com)

---

## 💻 Jalankan Lokal di Android (via Termux)

```bash
# Install Termux dari F-Droid
pkg update && pkg install python
pip install streamlit anthropic Pillow pandas plotly
streamlit run app.py
# Buka browser → http://localhost:8501
```

---

## 📁 Struktur File

```
image-quality-analyzer/
├── app.py              ← Aplikasi utama
└── requirements.txt    ← Daftar library
```

---

## 📋 Output untuk UTS
- **File program**: `app.py` + `requirements.txt`
- **Screenshot**: tampilan dashboard setelah analisis foto
- **Tautan publik**: URL dari Streamlit Cloud setelah deploy

---

*Dibuat untuk UTS PBO | Powered by Claude AI (Anthropic)*
