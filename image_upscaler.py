"""
Image Upscaler - Object-Oriented Implementation
Proyek UTS Pemrograman Berbasis Objek
Program untuk upscale gambar dengan berbagai metode interpolasi
"""

import os
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# BASE CLASS (Abstrak)
# ============================================================
class BaseUpscaler:
    """Base class untuk semua metode upscaling"""

    def __init__(self, name: str, scale_factor: int = 2):
        self.name = name
        self.scale_factor = scale_factor
        self.metrics = {
            "processing_time": [],
            "input_sizes": [],
            "output_sizes": [],
            "psnr_scores": []
        }

    def upscale(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("Subclass harus mengimplementasikan metode upscale()")

    def calculate_psnr(self, original: Image.Image, upscaled: Image.Image) -> float:
        """Hitung Peak Signal-to-Noise Ratio (PSNR)"""
        orig_resized = original.resize(upscaled.size, Image.LANCZOS)
        orig_arr = np.array(orig_resized, dtype=np.float64)
        upscaled_arr = np.array(upscaled, dtype=np.float64)
        mse = np.mean((orig_arr - upscaled_arr) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def process_and_record(self, image: Image.Image) -> dict:
        """Proses gambar dan catat metrik"""
        start = time.time()
        result = self.upscale(image)
        elapsed = time.time() - start

        psnr = self.calculate_psnr(image, result)

        self.metrics["processing_time"].append(round(elapsed, 4))
        self.metrics["input_sizes"].append(image.size)
        self.metrics["output_sizes"].append(result.size)
        self.metrics["psnr_scores"].append(round(psnr, 2))

        return {
            "method": self.name,
            "processing_time": elapsed,
            "psnr": psnr,
            "input_size": image.size,
            "output_size": result.size,
            "image": result
        }

    def get_average_metrics(self) -> dict:
        """Rata-rata metrik semua proses"""
        if not self.metrics["processing_time"]:
            return {}
        return {
            "method": self.name,
            "avg_time": round(sum(self.metrics["processing_time"]) / len(self.metrics["processing_time"]), 4),
            "avg_psnr": round(sum(self.metrics["psnr_scores"]) / len(self.metrics["psnr_scores"]), 2),
            "total_processed": len(self.metrics["processing_time"])
        }


# ============================================================
# SUBCLASS - Metode Nearest Neighbor
# ============================================================
class NearestNeighborUpscaler(BaseUpscaler):
    """Upscaling dengan metode Nearest Neighbor - cepat tapi kualitas rendah"""

    def __init__(self, scale_factor: int = 2):
        super().__init__("Nearest Neighbor", scale_factor)

    def upscale(self, image: Image.Image) -> Image.Image:
        new_width = image.width * self.scale_factor
        new_height = image.height * self.scale_factor
        return image.resize((new_width, new_height), Image.NEAREST)


# ============================================================
# SUBCLASS - Metode Bilinear
# ============================================================
class BilinearUpscaler(BaseUpscaler):
    """Upscaling dengan interpolasi Bilinear"""

    def __init__(self, scale_factor: int = 2):
        super().__init__("Bilinear", scale_factor)

    def upscale(self, image: Image.Image) -> Image.Image:
        new_width = image.width * self.scale_factor
        new_height = image.height * self.scale_factor
        return image.resize((new_width, new_height), Image.BILINEAR)


# ============================================================
# SUBCLASS - Metode Bicubic
# ============================================================
class BicubicUpscaler(BaseUpscaler):
    """Upscaling dengan interpolasi Bicubic - kualitas lebih baik"""

    def __init__(self, scale_factor: int = 2):
        super().__init__("Bicubic", scale_factor)

    def upscale(self, image: Image.Image) -> Image.Image:
        new_width = image.width * self.scale_factor
        new_height = image.height * self.scale_factor
        return image.resize((new_width, new_height), Image.BICUBIC)


# ============================================================
# SUBCLASS - Metode Lanczos (Kualitas Terbaik)
# ============================================================
class LanczosUpscaler(BaseUpscaler):
    """Upscaling dengan filter Lanczos - kualitas terbaik"""

    def __init__(self, scale_factor: int = 2):
        super().__init__("Lanczos", scale_factor)

    def upscale(self, image: Image.Image) -> Image.Image:
        new_width = image.width * self.scale_factor
        new_height = image.height * self.scale_factor
        return image.resize((new_width, new_height), Image.LANCZOS)


# ============================================================
# KELAS UTAMA - Image Upscaler Manager
# ============================================================
class ImageUpscalerManager:
    """
    Manajer utama untuk mengelola semua upscaler.
    Menggunakan konsep OOP: komposisi dan polimorfisme.
    """

    def __init__(self, scale_factor: int = 2, output_dir: str = "output"):
        self.scale_factor = scale_factor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Inisialisasi semua upscaler (polimorfisme)
        self.upscalers = {
            "nearest": NearestNeighborUpscaler(scale_factor),
            "bilinear": BilinearUpscaler(scale_factor),
            "bicubic": BicubicUpscaler(scale_factor),
            "lanczos": LanczosUpscaler(scale_factor)
        }

        self.history = []  # Riwayat semua proses

    def process_image(self, image_path: str, methods: list = None) -> dict:
        """
        Proses satu gambar dengan satu atau lebih metode upscaling
        
        Args:
            image_path: Path ke file gambar
            methods: List metode ['nearest', 'bilinear', 'bicubic', 'lanczos']
                     Jika None, gunakan semua metode
        Returns:
            Dict berisi hasil semua metode
        """
        if methods is None:
            methods = list(self.upscalers.keys())

        image = Image.open(image_path).convert("RGB")
        filename = Path(image_path).stem
        results = {}

        print(f"\n{'='*55}")
        print(f"  Memproses: {Path(image_path).name}")
        print(f"  Ukuran input: {image.size[0]}x{image.size[1]} px")
        print(f"  Scale factor: {self.scale_factor}x")
        print(f"{'='*55}")

        for method_key in methods:
            if method_key not in self.upscalers:
                print(f"  [!] Metode '{method_key}' tidak dikenal, dilewati.")
                continue

            upscaler = self.upscalers[method_key]
            result = upscaler.process_and_record(image)

            # Simpan hasil
            out_path = self.output_dir / f"{filename}_{method_key}_{self.scale_factor}x.png"
            result["image"].save(str(out_path))

            results[method_key] = {
                "method": result["method"],
                "processing_time": round(result["processing_time"], 4),
                "psnr": round(result["psnr"], 2),
                "output_size": result["output_size"],
                "output_path": str(out_path)
            }

            print(f"  ✓ {result['method']:18s} | "
                  f"Waktu: {result['processing_time']:.4f}s | "
                  f"PSNR: {result['psnr']:.2f} dB | "
                  f"Output: {result['output_size'][0]}x{result['output_size'][1]}")

        # Simpan ke history
        self.history.append({
            "input": str(image_path),
            "input_size": image.size,
            "results": results
        })

        return results

    def benchmark(self, image_path: str, runs: int = 5) -> dict:
        """
        Benchmark semua metode dengan banyak run untuk mendapatkan data tren
        
        Args:
            image_path: Path gambar untuk benchmark
            runs: Jumlah percobaan per metode
        Returns:
            Dict hasil benchmark
        """
        image = Image.open(image_path).convert("RGB")
        benchmark_results = {k: {"times": [], "psnr": []} for k in self.upscalers}

        print(f"\n{'='*55}")
        print(f"  BENCHMARK - {runs} runs per metode")
        print(f"{'='*55}")

        for run in range(1, runs + 1):
            print(f"  Run {run}/{runs}...")
            for key, upscaler in self.upscalers.items():
                result = upscaler.process_and_record(image)
                benchmark_results[key]["times"].append(round(result["processing_time"], 4))
                benchmark_results[key]["psnr"].append(round(result["psnr"], 2))

        # Hitung rata-rata
        for key in benchmark_results:
            times = benchmark_results[key]["times"]
            psnrs = benchmark_results[key]["psnr"]
            benchmark_results[key]["avg_time"] = round(sum(times) / len(times), 4)
            benchmark_results[key]["avg_psnr"] = round(sum(psnrs) / len(psnrs), 2)
            benchmark_results[key]["min_time"] = min(times)
            benchmark_results[key]["max_time"] = max(times)

        return benchmark_results

    def save_report(self, filename: str = "report.json"):
        """Simpan laporan lengkap dalam format JSON"""
        report = {
            "scale_factor": self.scale_factor,
            "total_images_processed": len(self.history),
            "upscaler_summary": {k: v.get_average_metrics() for k, v in self.upscalers.items()},
            "history": self.history
        }
        report_path = self.output_dir / filename
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Laporan disimpan ke: {report_path}")
        return report


# ============================================================
# VISUALISASI - Dashboard dengan Matplotlib
# ============================================================
class UpscalerDashboard:
    """Kelas untuk membuat visualisasi dashboard performa upscaler"""

    def __init__(self, benchmark_data: dict):
        self.data = benchmark_data
        self.methods = list(benchmark_data.keys())
        self.colors = {
            "nearest": "#FF6B6B",
            "bilinear": "#4ECDC4",
            "bicubic": "#45B7D1",
            "lanczos": "#96CEB4"
        }

    def create_dashboard(self, output_path: str = "output/dashboard.png"):
        """Buat dashboard lengkap dengan semua grafik"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  [!] matplotlib tidak ditemukan. Install: pip install matplotlib")
            return

        fig = plt.figure(figsize=(16, 10), facecolor="#0D1117")
        fig.suptitle("Image Upscaler — Performance Dashboard",
                     fontsize=20, color="white", fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        colors = [self.colors.get(m, "#888") for m in self.methods]
        labels = [self.data[m].get("method", m) for m in self.methods]

        # --- Grafik 1: Tren Waktu Proses (Time Series) ---
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor("#161B22")
        for i, method in enumerate(self.methods):
            times = self.data[method]["times"]
            ax1.plot(range(1, len(times) + 1), times,
                     marker='o', color=colors[i],
                     label=labels[i], linewidth=2, markersize=5)
        ax1.set_title("Tren Waktu Proses per Run", color="white", fontsize=13)
        ax1.set_xlabel("Run ke-", color="#8B949E")
        ax1.set_ylabel("Waktu (detik)", color="#8B949E")
        ax1.tick_params(colors="#8B949E")
        ax1.legend(facecolor="#21262D", labelcolor="white", fontsize=9)
        ax1.grid(alpha=0.2, color="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363D")

        # --- Grafik 2: KPI Card - PSNR ---
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor("#161B22")
        avg_psnrs = [self.data[m]["avg_psnr"] for m in self.methods]
        bars = ax2.barh(labels, avg_psnrs, color=colors, height=0.5)
        for bar, val in zip(bars, avg_psnrs):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f} dB", va='center', color="white", fontsize=9)
        ax2.set_title("KPI: Rata-rata PSNR", color="white", fontsize=13)
        ax2.set_xlabel("PSNR (dB)", color="#8B949E")
        ax2.tick_params(colors="#8B949E")
        ax2.grid(alpha=0.2, color="white", axis='x')
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363D")

        # --- Grafik 3: Perbandingan Waktu (Bar Chart) ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor("#161B22")
        avg_times = [self.data[m]["avg_time"] for m in self.methods]
        bars3 = ax3.bar(labels, avg_times, color=colors, width=0.5)
        for bar, val in zip(bars3, avg_times):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                     f"{val:.4f}s", ha='center', color="white", fontsize=8)
        ax3.set_title("Perbandingan Waktu Rata-rata", color="white", fontsize=12)
        ax3.set_ylabel("Detik", color="#8B949E")
        ax3.tick_params(colors="#8B949E", rotation=15)
        ax3.grid(alpha=0.2, color="white", axis='y')
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363D")

        # --- Grafik 4: Scatter PSNR vs Waktu ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor("#161B22")
        for i, method in enumerate(self.methods):
            ax4.scatter(self.data[method]["avg_time"],
                        self.data[method]["avg_psnr"],
                        color=colors[i], s=120, label=labels[i], zorder=5)
            ax4.annotate(labels[i],
                         (self.data[method]["avg_time"], self.data[method]["avg_psnr"]),
                         textcoords="offset points", xytext=(6, 4),
                         color=colors[i], fontsize=8)
        ax4.set_title("Kualitas vs Kecepatan", color="white", fontsize=12)
        ax4.set_xlabel("Waktu (s)", color="#8B949E")
        ax4.set_ylabel("PSNR (dB)", color="#8B949E")
        ax4.tick_params(colors="#8B949E")
        ax4.grid(alpha=0.2, color="white")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#30363D")

        # --- Grafik 5: KPI Summary ---
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor("#161B22")
        ax5.axis("off")
        kpi_lines = ["⚡ KEY PERFORMANCE INDICATORS\n"]
        best_psnr_method = self.methods[avg_psnrs.index(max(avg_psnrs))]
        best_speed_method = self.methods[avg_times.index(min(avg_times))]
        kpi_lines.append(f"🏆 Kualitas Terbaik:\n   {labels[self.methods.index(best_psnr_method)]}")
        kpi_lines.append(f"   PSNR: {max(avg_psnrs):.2f} dB\n")
        kpi_lines.append(f"⚡ Tercepat:\n   {labels[self.methods.index(best_speed_method)]}")
        kpi_lines.append(f"   Waktu: {min(avg_times):.4f}s\n")
        kpi_lines.append(f"📊 Total Metode Diuji: {len(self.methods)}")
        ax5.text(0.05, 0.95, "\n".join(kpi_lines),
                 transform=ax5.transAxes, color="white", fontsize=10,
                 va='top', linespacing=1.6,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D", edgecolor="#30363D"))
        ax5.set_title("Ringkasan KPI", color="white", fontsize=12)

        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1117", edgecolor="none")
        print(f"\n  Dashboard disimpan ke: {output_path}")
        plt.close()


# ============================================================
# DEMO UTAMA
# ============================================================
def main():
    print("\n" + "="*55)
    print("   IMAGE UPSCALER - Pemrograman Berbasis Objek")
    print("   UTS Project | Python Implementation")
    print("="*55)

    # Buat gambar contoh jika tidak ada input
    demo_path = "demo_input.png"
    if not os.path.exists(demo_path):
        print("\n  Membuat gambar demo...")
        img = Image.new("RGB", (200, 200))
        pixels = img.load()
        for i in range(200):
            for j in range(200):
                pixels[i, j] = (
                    int(255 * i / 200),
                    int(255 * j / 200),
                    int(128 + 127 * ((i + j) % 100) / 100)
                )
        img.save(demo_path)
        print(f"  Gambar demo dibuat: {demo_path}")

    # Inisialisasi manager
    manager = ImageUpscalerManager(scale_factor=2, output_dir="output")

    # Proses gambar dengan semua metode
    results = manager.process_image(demo_path)

    # Benchmark
    benchmark_data = manager.benchmark(demo_path, runs=5)

    # Simpan laporan JSON
    manager.save_report("report.json")

    # Buat dashboard visualisasi
    dashboard = UpscalerDashboard(benchmark_data)
    dashboard.create_dashboard("output/dashboard.png")

    print("\n" + "="*55)
    print("  ✅ Selesai! Cek folder 'output/' untuk hasil.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
