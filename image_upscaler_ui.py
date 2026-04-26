"""
Image Upscaler - Interactive UI Version
Pemrograman Berbasis Objek | UTS Project
Jalankan di Google Colab untuk UI interaktif lengkap
"""

# ============================================================
# INSTALL & IMPORT
# ============================================================
import os, time, io, json, base64
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output


# ============================================================
# BASE CLASS
# ============================================================
class BaseUpscaler:
    def __init__(self, name: str, scale_factor: int = 2):
        self.name = name
        self.scale_factor = scale_factor
        self.metrics = {"processing_time": [], "psnr_scores": []}

    def upscale(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def calculate_psnr(self, original: Image.Image, upscaled: Image.Image) -> float:
        orig_r = original.resize(upscaled.size, Image.LANCZOS)
        o = np.array(orig_r, dtype=np.float64)
        u = np.array(upscaled, dtype=np.float64)
        mse = np.mean((o - u) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def process(self, image: Image.Image) -> dict:
        start = time.time()
        result = self.upscale(image)
        elapsed = time.time() - start
        psnr = self.calculate_psnr(image, result)
        self.metrics["processing_time"].append(round(elapsed, 4))
        self.metrics["psnr_scores"].append(round(psnr, 2))
        return {"method": self.name, "time": elapsed, "psnr": psnr, "image": result}


# ============================================================
# SUBCLASSES
# ============================================================
class NearestNeighborUpscaler(BaseUpscaler):
    def __init__(self, s): super().__init__("Nearest Neighbor", s)
    def upscale(self, img):
        return img.resize((img.width * self.scale_factor, img.height * self.scale_factor), Image.NEAREST)

class BilinearUpscaler(BaseUpscaler):
    def __init__(self, s): super().__init__("Bilinear", s)
    def upscale(self, img):
        return img.resize((img.width * self.scale_factor, img.height * self.scale_factor), Image.BILINEAR)

class BicubicUpscaler(BaseUpscaler):
    def __init__(self, s): super().__init__("Bicubic", s)
    def upscale(self, img):
        return img.resize((img.width * self.scale_factor, img.height * self.scale_factor), Image.BICUBIC)

class LanczosUpscaler(BaseUpscaler):
    def __init__(self, s): super().__init__("Lanczos", s)
    def upscale(self, img):
        return img.resize((img.width * self.scale_factor, img.height * self.scale_factor), Image.LANCZOS)


# ============================================================
# MANAGER
# ============================================================
class ImageUpscalerManager:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.upscalers = {
            "nearest":  NearestNeighborUpscaler(scale_factor),
            "bilinear": BilinearUpscaler(scale_factor),
            "bicubic":  BicubicUpscaler(scale_factor),
            "lanczos":  LanczosUpscaler(scale_factor),
        }

    def set_scale(self, scale_factor):
        self.scale_factor = scale_factor
        for key in self.upscalers:
            self.upscalers[key].scale_factor = scale_factor

    def run(self, image: Image.Image, methods: list) -> dict:
        results = {}
        for key in methods:
            if key in self.upscalers:
                results[key] = self.upscalers[key].process(image)
        return results


# ============================================================
# UI INTERAKTIF
# ============================================================
class UpscalerUI:
    """
    UI interaktif berbasis ipywidgets.
    Mendukung upload gambar, pilih metode, pilih scale factor,
    tampilkan hasil perbandingan, dan dashboard performa.
    """

    COLORS = {
        "nearest":  "#E24B4A",
        "bilinear": "#1D9E75",
        "bicubic":  "#378ADD",
        "lanczos":  "#BA7517",
    }

    def __init__(self):
        self.manager = ImageUpscalerManager(scale_factor=2)
        self.current_image = None
        self.results = {}
        self._build_ui()

    # ─────────────────── BUILD UI ───────────────────
    def _build_ui(self):
        # Header
        header = HTML("""
        <div style="
            background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
            border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
            border: 1px solid #30363D;">
          <h2 style="color:#58A6FF; margin:0; font-family:monospace; font-size:20px;">
            🖼️ Image Upscaler
          </h2>
          <p style="color:#8B949E; margin:6px 0 0; font-size:13px;">
            Upload gambar → pilih metode → lihat hasil & performa
          </p>
        </div>
        """)

        # Upload
        self.upload_btn = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.bmp,.webp',
            multiple=False,
            description='Upload Gambar',
            style={'button_color': '#238636'},
            layout=widgets.Layout(width='200px')
        )
        self.upload_btn.observe(self._on_upload, names='value')

        self.img_info = HTML("<span style='color:#8B949E; font-size:12px;'>Belum ada gambar</span>")

        # Scale factor
        self.scale_slider = widgets.IntSlider(
            value=2, min=2, max=4, step=1,
            description='Scale:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='260px')
        )
        self.scale_label = HTML("<span style='color:#58A6FF; font-weight:bold; font-size:15px;'>2×</span>")
        self.scale_slider.observe(self._on_scale_change, names='value')

        # Method checkboxes
        self.method_checks = {
            "nearest":  widgets.Checkbox(value=True,  description='Nearest Neighbor', style={'description_width': 'initial'}),
            "bilinear": widgets.Checkbox(value=True,  description='Bilinear',         style={'description_width': 'initial'}),
            "bicubic":  widgets.Checkbox(value=True,  description='Bicubic',          style={'description_width': 'initial'}),
            "lanczos":  widgets.Checkbox(value=True,  description='Lanczos (best)',   style={'description_width': 'initial'}),
        }

        # Run button
        self.run_btn = widgets.Button(
            description='▶ Jalankan Upscaling',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.run_btn.on_click(self._on_run)

        # Progress
        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='',
            bar_style='info',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
        self.status_label = HTML("")

        # Output areas
        self.out_preview    = widgets.Output()
        self.out_comparison = widgets.Output()
        self.out_dashboard  = widgets.Output()

        # Tabs untuk hasil
        self.tabs = widgets.Tab(children=[
            self.out_comparison,
            self.out_dashboard,
        ])
        self.tabs.set_title(0, '📸 Perbandingan Hasil')
        self.tabs.set_title(1, '📊 Dashboard Performa')

        # Layout
        row_upload = widgets.HBox([self.upload_btn, self.img_info],
                                  layout=widgets.Layout(align_items='center', gap='12px'))
        row_scale  = widgets.HBox([self.scale_slider, self.scale_label],
                                  layout=widgets.Layout(align_items='center', gap='8px'))
        col_checks = widgets.VBox(list(self.method_checks.values()),
                                  layout=widgets.Layout(gap='2px'))

        panel = widgets.VBox([
            HTML("<b style='color:#C9D1D9'>Pengaturan</b>"),
            HTML("<hr style='border-color:#30363D; margin:4px 0'>"),
            HTML("<span style='color:#8B949E; font-size:12px'>Scale Factor</span>"),
            row_scale,
            HTML("<span style='color:#8B949E; font-size:12px; margin-top:8px'>Metode Upscaling</span>"),
            col_checks,
            HTML("<div style='height:8px'></div>"),
            self.run_btn,
        ], layout=widgets.Layout(
            padding='16px', background='#161B22',
            border='1px solid #30363D', border_radius='8px',
            width='240px', gap='6px'
        ))

        main_area = widgets.VBox([
            row_upload,
            HTML("<div style='height:8px'></div>"),
            self.out_preview,
            HTML("<div style='height:8px'></div>"),
            self.progress,
            self.status_label,
            HTML("<div style='height:8px'></div>"),
            self.tabs,
        ], layout=widgets.Layout(flex='1'))

        root = widgets.VBox([
            header,
            widgets.HBox([panel, main_area],
                         layout=widgets.Layout(gap='16px', align_items='flex-start')),
        ])

        display(root)

    # ─────────────────── EVENTS ───────────────────
    def _on_upload(self, change):
        if not change['new']:
            return
        data = list(change['new'].values())[0]['content']
        self.current_image = Image.open(io.BytesIO(bytes(data))).convert("RGB")
        w, h = self.current_image.size
        self.img_info.value = (
            f"<span style='color:#3FB950; font-size:12px'>✓ {w}×{h} px loaded</span>"
        )
        with self.out_preview:
            clear_output()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(self.current_image)
            ax.set_title("Preview input", fontsize=11, color='#C9D1D9')
            ax.axis('off')
            fig.patch.set_facecolor('#0D1117')
            plt.tight_layout()
            plt.show()

    def _on_scale_change(self, change):
        self.scale_label.value = f"<span style='color:#58A6FF; font-weight:bold; font-size:15px;'>{change['new']}×</span>"

    def _on_run(self, _):
        if self.current_image is None:
            self.status_label.value = "<span style='color:#F85149'>⚠ Upload gambar dulu!</span>"
            return

        selected = [k for k, v in self.method_checks.items() if v.value]
        if not selected:
            self.status_label.value = "<span style='color:#F85149'>⚠ Pilih minimal 1 metode!</span>"
            return

        scale = self.scale_slider.value
        self.manager.set_scale(scale)

        # Progress
        self.progress.layout.visibility = 'visible'
        self.progress.value = 0
        self.status_label.value = "<span style='color:#8B949E'>Memproses...</span>"

        self.results = {}
        step = 100 // len(selected)

        for i, key in enumerate(selected):
            self.status_label.value = f"<span style='color:#58A6FF'>⏳ Menjalankan {self.manager.upscalers[key].name}...</span>"
            self.results[key] = self.manager.upscalers[key].process(self.current_image)
            self.progress.value = (i + 1) * step

        self.progress.value = 100
        self.status_label.value = "<span style='color:#3FB950'>✓ Selesai!</span>"

        self._show_comparison(selected, scale)
        self._show_dashboard(selected)
        self.tabs.selected_index = 0

    # ─────────────────── COMPARISON ───────────────────
    def _show_comparison(self, selected, scale):
        n = len(selected) + 1
        with self.out_comparison:
            clear_output()
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            fig.patch.set_facecolor('#0D1117')

            if n == 2:
                axes = [axes[0], axes[1]] if hasattr(axes, '__iter__') else [axes]

            # Original
            ax0 = axes[0]
            ax0.imshow(self.current_image)
            ax0.set_title(f"Original\n{self.current_image.size[0]}×{self.current_image.size[1]}",
                          color='#C9D1D9', fontsize=10)
            ax0.axis('off')

            for i, key in enumerate(selected):
                r = self.results[key]
                ax = axes[i + 1]
                ax.imshow(r["image"])
                color = self.COLORS.get(key, "#888")
                ax.set_title(
                    f"{r['method']}\n{r['image'].size[0]}×{r['image'].size[1]} | "
                    f"PSNR: {r['psnr']:.1f} dB\n⏱ {r['time']*1000:.1f} ms",
                    color=color, fontsize=9
                )
                ax.axis('off')
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)

            fig.suptitle(f"Hasil Upscaling {scale}×", color='white', fontsize=13, y=1.01)
            plt.tight_layout()
            plt.show()

            # Download buttons (simpan ke file)
            os.makedirs("output", exist_ok=True)
            saved = []
            for key in selected:
                path = f"output/upscaled_{key}_{scale}x.png"
                self.results[key]["image"].save(path)
                saved.append(path)

            print("\n✅ File tersimpan di folder 'output/':")
            for p in saved:
                print(f"   📁 {p}")

    # ─────────────────── DASHBOARD ───────────────────
    def _show_dashboard(self, selected):
        with self.out_dashboard:
            clear_output()
            data = {k: self.results[k] for k in selected}
            labels  = [data[k]["method"] for k in selected]
            times   = [data[k]["time"] * 1000 for k in selected]   # ms
            psnrs   = [data[k]["psnr"] for k in selected]
            colors  = [self.COLORS.get(k, "#888") for k in selected]

            fig = plt.figure(figsize=(12, 8), facecolor='#0D1117')
            fig.suptitle("Performance Dashboard", color='white', fontsize=14, y=0.98)
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

            # ── KPI cards (text panel) ──
            ax_kpi = fig.add_subplot(gs[0, 2])
            ax_kpi.set_facecolor('#161B22')
            ax_kpi.axis('off')
            best_psnr_i = psnrs.index(max(psnrs))
            best_time_i = times.index(min(times))
            kpi_text = (
                f"KEY PERFORMANCE INDICATORS\n\n"
                f"Kualitas Terbaik\n"
                f"  {labels[best_psnr_i]}\n"
                f"  {psnrs[best_psnr_i]:.2f} dB\n\n"
                f"Tercepat\n"
                f"  {labels[best_time_i]}\n"
                f"  {times[best_time_i]:.2f} ms\n\n"
                f"Metode diuji: {len(selected)}\n"
                f"Scale factor: {self.manager.scale_factor}×"
            )
            ax_kpi.text(0.05, 0.95, kpi_text, transform=ax_kpi.transAxes,
                        color='white', fontsize=9, va='top', linespacing=1.7,
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262D',
                                  edgecolor='#30363D'))
            ax_kpi.set_title("KPI", color='white', fontsize=11)

            # ── Bar: waktu ──
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor('#161B22')
            bars = ax1.bar(labels, times, color=colors, width=0.5)
            for bar, val in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.05,
                         f"{val:.1f} ms", ha='center', color='white', fontsize=8)
            ax1.set_title("Waktu Proses", color='white', fontsize=11)
            ax1.set_ylabel("ms", color='#8B949E')
            ax1.tick_params(colors='#8B949E', rotation=15, labelsize=8)
            ax1.grid(alpha=0.15, color='white', axis='y')
            for sp in ax1.spines.values(): sp.set_edgecolor('#30363D')

            # ── Bar: PSNR ──
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor('#161B22')
            bars2 = ax2.bar(labels, psnrs, color=colors, width=0.5)
            for bar, val in zip(bars2, psnrs):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         f"{val:.1f}", ha='center', color='white', fontsize=8)
            ax2.set_title("Kualitas (PSNR)", color='white', fontsize=11)
            ax2.set_ylabel("dB", color='#8B949E')
            ax2.tick_params(colors='#8B949E', rotation=15, labelsize=8)
            ax2.grid(alpha=0.15, color='white', axis='y')
            for sp in ax2.spines.values(): sp.set_edgecolor('#30363D')

            # ── Scatter: PSNR vs Waktu ──
            ax3 = fig.add_subplot(gs[1, :2])
            ax3.set_facecolor('#161B22')
            for i, key in enumerate(selected):
                ax3.scatter(times[i], psnrs[i], color=colors[i], s=120, zorder=5)
                ax3.annotate(labels[i], (times[i], psnrs[i]),
                             textcoords='offset points', xytext=(6, 4),
                             color=colors[i], fontsize=9)
            ax3.set_title("Kualitas vs Kecepatan", color='white', fontsize=11)
            ax3.set_xlabel("Waktu (ms)", color='#8B949E')
            ax3.set_ylabel("PSNR (dB)", color='#8B949E')
            ax3.tick_params(colors='#8B949E')
            ax3.grid(alpha=0.15, color='white')
            ax3.annotate("← Lebih cepat", xy=(0.02, 0.05), xycoords='axes fraction',
                         color='#3FB950', fontsize=8)
            ax3.annotate("Kualitas lebih baik →", xy=(0.6, 0.95), xycoords='axes fraction',
                         color='#58A6FF', fontsize=8)
            for sp in ax3.spines.values(): sp.set_edgecolor('#30363D')

            # ── Pie: proporsi waktu ──
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.set_facecolor('#161B22')
            ax4.pie(times, labels=labels, colors=colors,
                    autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 8},
                    pctdistance=0.75)
            ax4.set_title("Proporsi Waktu", color='white', fontsize=11)

            plt.savefig("output/dashboard.png", dpi=130, bbox_inches='tight',
                        facecolor='#0D1117')
            plt.show()
            print("\n📊 Dashboard disimpan: output/dashboard.png")


# ============================================================
# JALANKAN UI
# ============================================================
Path("output").mkdir(exist_ok=True)
print("🚀 Memuat Image Upscaler UI...")
ui = UpscalerUI()
