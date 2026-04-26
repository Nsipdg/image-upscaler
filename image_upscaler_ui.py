"""
Image Upscaler - Interactive UI Version
Pemrograman Berbasis Objek | UTS Project
Kompatibel dengan ipywidgets 7.x dan 8.x
"""

import os, time, io
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


# ============================================================
# HELPER: baca bytes dari FileUpload (support v7 & v8)
# ============================================================
def read_upload_bytes(upload_widget):
    val = upload_widget.value
    # ipywidgets 8.x: val adalah tuple/list of dict
    if isinstance(val, (list, tuple)) and len(val) > 0:
        item = val[0]
        content = item.get('content', None)
        if content is not None:
            return bytes(content)
    # ipywidgets 7.x: val adalah dict {filename: {content: bytes}}
    if isinstance(val, dict) and len(val) > 0:
        item = list(val.values())[0]
        content = item.get('content', None)
        if content is not None:
            return bytes(content)
    return None


# ============================================================
# UI INTERAKTIF
# ============================================================
class UpscalerUI:
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

    def _build_ui(self):
        header = HTML("""
        <div style="background:#161B22; border-radius:12px; padding:20px 24px;
                    margin-bottom:16px; border:1px solid #30363D;">
          <h2 style="color:#58A6FF; margin:0; font-family:monospace; font-size:20px;">
            Image Upscaler
          </h2>
          <p style="color:#8B949E; margin:6px 0 0; font-size:13px;">
            Upload gambar, pilih metode, lihat hasil dan performa
          </p>
        </div>
        """)

        self.upload_btn = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.bmp,.webp',
            multiple=False,
            layout=widgets.Layout(width='180px')
        )
        self.upload_btn.observe(self._on_upload, names='value')

        self.img_info = HTML(
            "<span style='color:#8B949E; font-size:12px;'>Belum ada gambar</span>"
        )

        self.scale_slider = widgets.IntSlider(
            value=2, min=2, max=4, step=1,
            description='Scale:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='240px')
        )
        self.scale_label = HTML(
            "<span style='color:#58A6FF; font-weight:bold; font-size:15px;'>2x</span>"
        )
        self.scale_slider.observe(self._on_scale_change, names='value')

        self.method_checks = {
            "nearest":  widgets.Checkbox(value=True, description='Nearest Neighbor',
                                         style={'description_width': 'initial'}),
            "bilinear": widgets.Checkbox(value=True, description='Bilinear',
                                         style={'description_width': 'initial'}),
            "bicubic":  widgets.Checkbox(value=True, description='Bicubic',
                                         style={'description_width': 'initial'}),
            "lanczos":  widgets.Checkbox(value=True, description='Lanczos (best)',
                                         style={'description_width': 'initial'}),
        }

        self.run_btn = widgets.Button(
            description='Jalankan Upscaling',
            button_style='success',
            layout=widgets.Layout(width='190px', height='38px')
        )
        self.run_btn.on_click(self._on_run)

        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            bar_style='info',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
        self.status_label = HTML("")

        self.out_preview    = widgets.Output()
        self.out_comparison = widgets.Output()
        self.out_dashboard  = widgets.Output()

        self.tabs = widgets.Tab(children=[self.out_comparison, self.out_dashboard])
        self.tabs.set_title(0, 'Perbandingan Hasil')
        self.tabs.set_title(1, 'Dashboard Performa')

        row_upload = widgets.HBox(
            [self.upload_btn, self.img_info],
            layout=widgets.Layout(align_items='center', gap='12px')
        )
        row_scale = widgets.HBox(
            [self.scale_slider, self.scale_label],
            layout=widgets.Layout(align_items='center', gap='8px')
        )

        panel = widgets.VBox([
            HTML("<b style='color:#C9D1D9'>Pengaturan</b>"),
            HTML("<hr style='border-color:#30363D; margin:4px 0'>"),
            HTML("<span style='color:#8B949E; font-size:12px'>Scale Factor</span>"),
            row_scale,
            HTML("<span style='color:#8B949E; font-size:12px'>Metode</span>"),
            widgets.VBox(list(self.method_checks.values()),
                         layout=widgets.Layout(gap='2px')),
            HTML("<div style='height:6px'></div>"),
            self.run_btn,
        ], layout=widgets.Layout(padding='16px', width='250px', gap='6px',
                                  border='1px solid #30363D'))

        main_area = widgets.VBox([
            row_upload,
            self.out_preview,
            self.progress,
            self.status_label,
            self.tabs,
        ], layout=widgets.Layout(flex='1', gap='8px'))

        root = widgets.VBox([
            header,
            widgets.HBox([panel, main_area],
                         layout=widgets.Layout(gap='16px', align_items='flex-start')),
        ])

        display(root)

    # ─── EVENTS ───
    def _on_upload(self, change):
        raw = read_upload_bytes(self.upload_btn)
        if raw is None:
            return
        self.current_image = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = self.current_image.size
        self.img_info.value = (
            f"<span style='color:#3FB950; font-size:12px'>OK {w}x{h} px</span>"
        )
        with self.out_preview:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.imshow(self.current_image)
            ax.set_title(f"Input: {w}x{h}", fontsize=10)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

    def _on_scale_change(self, change):
        self.scale_label.value = (
            f"<span style='color:#58A6FF; font-weight:bold; font-size:15px;'>"
            f"{change['new']}x</span>"
        )

    def _on_run(self, _):
        if self.current_image is None:
            self.status_label.value = (
                "<span style='color:#F85149'>Upload gambar dulu!</span>"
            )
            return

        selected = [k for k, v in self.method_checks.items() if v.value]
        if not selected:
            self.status_label.value = (
                "<span style='color:#F85149'>Pilih minimal 1 metode!</span>"
            )
            return

        scale = self.scale_slider.value
        self.manager.set_scale(scale)

        self.progress.layout.visibility = 'visible'
        self.progress.value = 0
        self.results = {}
        step = 100 // len(selected)

        for i, key in enumerate(selected):
            name = self.manager.upscalers[key].name
            self.status_label.value = (
                f"<span style='color:#58A6FF'>Memproses {name}...</span>"
            )
            self.results[key] = self.manager.upscalers[key].process(self.current_image)
            self.progress.value = (i + 1) * step

        self.progress.value = 100
        self.status_label.value = "<span style='color:#3FB950'>Selesai!</span>"

        self._show_comparison(selected, scale)
        self._show_dashboard(selected)
        self.tabs.selected_index = 0

    def _show_comparison(self, selected, scale):
        n = len(selected) + 1
        with self.out_comparison:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            if n == 2:
                axes = list(axes)
            elif n == 1:
                axes = [axes]
            fig.patch.set_facecolor('#0D1117')

            axes[0].imshow(self.current_image)
            axes[0].set_title(
                f"Original\n{self.current_image.size[0]}x{self.current_image.size[1]}",
                color='#C9D1D9', fontsize=10)
            axes[0].axis('off')

            for i, key in enumerate(selected):
                r = self.results[key]
                c = self.COLORS.get(key, "#888")
                axes[i + 1].imshow(r["image"])
                axes[i + 1].set_title(
                    f"{r['method']}\n"
                    f"{r['image'].size[0]}x{r['image'].size[1]}\n"
                    f"PSNR: {r['psnr']:.1f} dB | {r['time']*1000:.1f} ms",
                    color=c, fontsize=9)
                axes[i + 1].axis('off')

            fig.suptitle(f"Hasil Upscaling {scale}x", color='white', fontsize=13)
            plt.tight_layout()
            plt.show()

            os.makedirs("output", exist_ok=True)
            print("File tersimpan di folder output/:")
            for key in selected:
                path = f"output/upscaled_{key}_{scale}x.png"
                self.results[key]["image"].save(path)
                print(f"  {path}")

    def _show_dashboard(self, selected):
        with self.out_dashboard:
            clear_output(wait=True)
            data   = {k: self.results[k] for k in selected}
            labels = [data[k]["method"] for k in selected]
            times  = [data[k]["time"] * 1000 for k in selected]
            psnrs  = [data[k]["psnr"] for k in selected]
            colors = [self.COLORS.get(k, "#888") for k in selected]

            fig = plt.figure(figsize=(11, 7), facecolor='#0D1117')
            fig.suptitle("Performance Dashboard", color='white', fontsize=14)
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor('#161B22')
            bars = ax1.bar(labels, times, color=colors, width=0.5)
            for bar, val in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.05,
                         f"{val:.1f}ms", ha='center', color='white', fontsize=8)
            ax1.set_title("Waktu Proses", color='white', fontsize=11)
            ax1.set_ylabel("ms", color='#8B949E')
            ax1.tick_params(colors='#8B949E', rotation=15, labelsize=8)
            ax1.grid(alpha=0.15, color='white', axis='y')
            for sp in ax1.spines.values(): sp.set_edgecolor('#30363D')

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor('#161B22')
            bars2 = ax2.bar(labels, psnrs, color=colors, width=0.5)
            for bar, val in zip(bars2, psnrs):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.1,
                         f"{val:.1f}", ha='center', color='white', fontsize=8)
            ax2.set_title("Kualitas (PSNR)", color='white', fontsize=11)
            ax2.set_ylabel("dB", color='#8B949E')
            ax2.tick_params(colors='#8B949E', rotation=15, labelsize=8)
            ax2.grid(alpha=0.15, color='white', axis='y')
            for sp in ax2.spines.values(): sp.set_edgecolor('#30363D')

            ax3 = fig.add_subplot(gs[0, 2])
            ax3.set_facecolor('#161B22')
            ax3.axis('off')
            bi = psnrs.index(max(psnrs))
            ti = times.index(min(times))
            kpi = (
                f"KPI SUMMARY\n\n"
                f"Kualitas Terbaik\n  {labels[bi]}\n  {psnrs[bi]:.2f} dB\n\n"
                f"Tercepat\n  {labels[ti]}\n  {times[ti]:.2f} ms\n\n"
                f"Metode diuji: {len(selected)}\n"
                f"Scale: {self.manager.scale_factor}x"
            )
            ax3.text(0.05, 0.95, kpi, transform=ax3.transAxes,
                     color='white', fontsize=9, va='top', fontfamily='monospace',
                     linespacing=1.7,
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='#21262D', edgecolor='#30363D'))
            ax3.set_title("KPI", color='white', fontsize=11)

            ax4 = fig.add_subplot(gs[1, :2])
            ax4.set_facecolor('#161B22')
            for i, key in enumerate(selected):
                ax4.scatter(times[i], psnrs[i], color=colors[i], s=120, zorder=5)
                ax4.annotate(labels[i], (times[i], psnrs[i]),
                             textcoords='offset points', xytext=(6, 4),
                             color=colors[i], fontsize=9)
            ax4.set_title("Kualitas vs Kecepatan", color='white', fontsize=11)
            ax4.set_xlabel("Waktu (ms)", color='#8B949E')
            ax4.set_ylabel("PSNR (dB)", color='#8B949E')
            ax4.tick_params(colors='#8B949E')
            ax4.grid(alpha=0.15, color='white')
            for sp in ax4.spines.values(): sp.set_edgecolor('#30363D')

            ax5 = fig.add_subplot(gs[1, 2])
            ax5.set_facecolor('#161B22')
            ax5.pie(times, labels=labels, colors=colors, autopct='%1.1f%%',
                    textprops={'color': 'white', 'fontsize': 8}, pctdistance=0.75)
            ax5.set_title("Proporsi Waktu", color='white', fontsize=11)

            os.makedirs("output", exist_ok=True)
            plt.savefig("output/dashboard.png", dpi=120, bbox_inches='tight',
                        facecolor='#0D1117')
            plt.show()
            print("Dashboard disimpan: output/dashboard.png")


# ============================================================
# JALANKAN
# ============================================================
Path("output").mkdir(exist_ok=True)
print("Memuat Image Upscaler UI...")
ui = UpscalerUI()
