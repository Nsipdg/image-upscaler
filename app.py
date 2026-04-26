import streamlit as st
import io, base64, time
from PIL import Image, ImageFilter, ImageEnhance, ImageStat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Image Upscaler", page_icon="🖼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1000px; }

h1 { font-size: 1.4rem; font-weight: 600; color: #111; margin: 0 0 4px; }
.sub { font-size: 0.78rem; color: #999; margin: 0 0 1.5rem; }

.kpi { background: #f8f8f8; border-radius: 8px; padding: 1rem; text-align: center; }
.kpi-v { font-size: 1.6rem; font-weight: 600; color: #111; margin: 0; }
.kpi-l { font-size: 0.68rem; color: #aaa; margin: 4px 0 0; text-transform: uppercase; letter-spacing: .05em; }

.result-row { display: flex; justify-content: space-between; padding: 6px 0;
  border-bottom: 1px solid #f0f0f0; font-size: 0.82rem; }
.result-row:last-child { border: none; }
.rk { color: #888; }
.rv { font-weight: 500; color: #111; }

.grade { font-size: 2rem; font-weight: 700; width: 52px; height: 52px;
  display: flex; align-items: center; justify-content: center; border-radius: 8px; }
.gA,.gB { background: #e8f5e9; color: #2e7d32; }
.gC { background: #fff8e1; color: #f57f17; }
.gD { background: #ffebee; color: #c62828; }

.note { font-size: 0.78rem; color: #888; background: #fafafa;
  border-left: 3px solid #ddd; padding: 8px 12px; margin-top: 10px; border-radius: 0 6px 6px 0; }
.sec { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .08em;
  color: #bbb; margin: 0 0 10px; }
hr { border: none; border-top: 1px solid #f0f0f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Analysis (no ImageFilter.Kernel) ─────────────────────────────────────────
def analyze(img: Image.Image, name: str, raw_size: int) -> dict:
    rgb  = img.convert("RGB")
    gray = rgb.convert("L")
    arr  = np.array(gray, dtype=np.float32)
    w, h = rgb.size

    # Sharpness — Laplacian via numpy (no PIL Kernel)
    lap = (
        -arr[:-2,:-2] - arr[:-2,1:-1] - arr[:-2,2:]
        - arr[1:-1,:-2] + 8*arr[1:-1,1:-1] - arr[1:-1,2:]
        - arr[2:,:-2]  - arr[2:,1:-1]  - arr[2:,2:]
    )
    sharpness = min(100, int(np.var(lap) / 80))

    # Noise — gaussian blur difference via PIL (safe)
    blurred = np.array(gray.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32)
    noise_level = min(100, int(np.mean(np.abs(arr - blurred)) * 4))

    # Brightness & contrast
    mean_b = float(np.mean(arr))
    std_b  = float(np.std(arr))
    brightness = "gelap" if mean_b < 60 else ("terang" if mean_b > 190 else "normal")
    contrast   = "rendah" if std_b < 35 else ("tinggi" if std_b > 75 else "normal")

    # Dominant color
    rgb_arr = np.array(rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    r, g_c, b = rgb_arr
    if max(r,g_c,b) > 200 and min(r,g_c,b) > 150: dom = "Putih"
    elif max(r,g_c,b) < 60: dom = "Hitam"
    elif r > g_c and r > b: dom = "Merah"
    elif g_c > r and g_c > b: dom = "Hijau"
    elif b > r and b > g_c: dom = "Biru"
    else: dom = "Netral"

    # Resolution
    mp = (w * h) / 1_000_000
    res = "tinggi" if mp >= 4 else ("sedang" if mp >= 1 else "rendah")

    # Quality score (weighted)
    q = int(
        sharpness * 0.40 +
        (100 - noise_level) * 0.25 +
        (80 if brightness == "normal" else 45) * 0.10 +
        (80 if contrast == "normal" else 50) * 0.10 +
        (100 if res == "tinggi" else 70 if res == "sedang" else 40) * 0.15
    )
    q = max(0, min(100, q))
    grade = "A" if q >= 80 else ("B" if q >= 65 else ("C" if q >= 45 else "D"))

    upscale = "tidak perlu" if (q >= 80 and res == "tinggi") else \
              ("sangat disarankan" if (q < 50 or res == "rendah") else "disarankan")

    issues = []
    if sharpness < 40: issues.append("blur")
    if noise_level > 60: issues.append("noise tinggi")
    if brightness != "normal": issues.append(brightness)
    if contrast == "rendah": issues.append("kontras rendah")
    notes = "Kualitas baik." if not issues else "Perhatikan: " + ", ".join(issues) + "."

    return dict(
        name=name, size_kb=round(raw_size/1024,1),
        width=w, height=h,
        datetime=datetime.now().strftime("%d %b %H:%M"),
        quality_score=q, sharpness=sharpness, noise_level=noise_level,
        brightness=brightness, contrast=contrast,
        dominant_color=dom, resolution_estimate=res,
        upscale_recommendation=upscale,
        quality_grade=grade, notes=notes,
    )

# ── Upscale (simple PIL resize) ───────────────────────────────────────────────
def upscale_image(img: Image.Image, scale: int) -> Image.Image:
    nw, nh = img.width * scale, img.height * scale
    up = img.resize((nw, nh), Image.LANCZOS)
    # sharpen after upscale
    up = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    return up

def grade_col(g):
    return {"A":"#2e7d32","B":"#388e3c","C":"#f57f17","D":"#c62828"}.get(g,"#888")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>Image Upscaler & Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload foto · Analisis kualitas · Upscale · Lihat tren</p>", unsafe_allow_html=True)

# ── Upload + Result ────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown("<p class='sec'>Upload Foto</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("foto", type=["jpg","jpeg","png","webp","bmp"],
                                label_visibility="collapsed")
    if uploaded:
        raw = uploaded.read()
        img = Image.open(io.BytesIO(raw))
        st.image(img, use_container_width=True)
        st.caption(f"{img.width}×{img.height} px · {len(raw)/1024:.1f} KB · {uploaded.type.split('/')[1].upper()}")

        scale = st.selectbox("Upscale", ["2× (default)", "3×", "4×"], label_visibility="visible")
        scale_val = int(scale[0])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔬 Analisis", use_container_width=True, type="primary"):
                with st.spinner("Menganalisis..."):
                    result = analyze(img, uploaded.name, len(raw))
                    thumb = img.copy(); thumb.thumbnail((100,100))
                    buf = io.BytesIO(); thumb.save(buf, format="PNG")
                    result["thumb_b64"] = base64.b64encode(buf.getvalue()).decode()
                    st.session_state.history.insert(0, result)
                    st.rerun()

        with col_b:
            if st.button("⬆ Upscale & Unduh", use_container_width=True):
                with st.spinner(f"Upscaling {scale_val}×..."):
                    up_img = upscale_image(img, scale_val)
                    buf2 = io.BytesIO()
                    fmt = "PNG" if uploaded.type == "image/png" else "JPEG"
                    up_img.save(buf2, format=fmt, quality=95)
                    st.download_button(
                        label=f"💾 Download ({up_img.width}×{up_img.height})",
                        data=buf2.getvalue(),
                        file_name=f"upscaled_{scale_val}x_{uploaded.name}",
                        mime=uploaded.type,
                        use_container_width=True
                    )

with col_r:
    if st.session_state.history:
        latest = st.session_state.history[0]
        g = latest["quality_grade"]
        st.markdown("<p class='sec'>Hasil Analisis</p>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"<div class='grade g{g}'>{g}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='padding:4px 0;'>
              <div style='font-size:1.8rem;font-weight:600;color:#111;line-height:1;'>{latest['quality_score']}<span style='font-size:0.9rem;color:#bbb;'>/100</span></div>
              <div style='font-size:0.72rem;color:#aaa;margin-top:2px;'>{latest['name'][:30]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        for lbl, val in [
            ("Ketajaman", f"{latest['sharpness']}/100"),
            ("Noise", f"{latest['noise_level']}/100"),
            ("Kecerahan", latest['brightness']),
            ("Kontras", latest['contrast']),
            ("Warna dominan", latest['dominant_color']),
            ("Resolusi", f"{latest['resolution_estimate']} ({latest['width']}×{latest['height']})"),
            ("Upscale", latest['upscale_recommendation']),
        ]:
            st.markdown(f"<div class='result-row'><span class='rk'>{lbl}</span><span class='rv'>{val}</span></div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='note'>💡 {latest['notes']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;align-items:center;justify-content:center;height:260px;'>
          <p style='color:#ccc;font-size:0.85rem;text-align:center;'>Upload foto dan klik Analisis</p>
        </div>""", unsafe_allow_html=True)

# ── KPI ────────────────────────────────────────────────────────────────────────
hist = st.session_state.history
n = len(hist)

if n > 0:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='sec'>KPI</p>", unsafe_allow_html=True)

    scores = [x["quality_score"] for x in hist]
    sharps = [x["sharpness"] for x in hist]
    avg_q  = round(sum(scores)/n)

    delta = ""
    if n >= 2:
        d = scores[0] - scores[1]
        delta = f"{'▲' if d>0 else '▼'} {abs(d)}"

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, str(n), "Total Foto"),
        (c2, str(avg_q), f"Rata-rata {delta}"),
        (c3, str(max(scores)), "Tertinggi"),
        (c4, str(min(scores)), "Terendah"),
        (c5, str(round(sum(sharps)/n)), "Rata-rata Ketajaman"),
    ]:
        with col:
            st.markdown(f"<div class='kpi'><p class='kpi-v'>{val}</p><p class='kpi-l'>{lbl}</p></div>",
                        unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr>", unsafe_allow_html=True)

    df = pd.DataFrame([{
        "label": (x["name"][:10]+"…" if len(x["name"])>10 else x["name"]),
        "Kualitas": x["quality_score"],
        "Ketajaman": x["sharpness"],
        "Noise": x["noise_level"],
    } for x in reversed(hist)])

    tab1, tab2 = st.tabs(["Tren Garis", "Perbandingan"])

    chart_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#aaa"),
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#f5f5f5", linecolor="#eee", tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f5f5f5", linecolor="#eee", range=[0,108], tickfont=dict(size=10)),
        hovermode="x unified",
    )

    with tab1:
        if n < 2:
            st.info("Upload minimal 2 foto untuk melihat tren.")
        else:
            fig = go.Figure()
            for col_name, color, dash in [("Kualitas","#4a90e2","solid"),
                                           ("Ketajaman","#3fcf8e","dot"),
                                           ("Noise","#e24a4a","dash")]:
                fig.add_trace(go.Scatter(
                    x=df["label"], y=df[col_name], name=col_name,
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(color=color, size=6),
                    fill="tozeroy" if col_name=="Kualitas" else "none",
                    fillcolor="rgba(74,144,226,0.05)" if col_name=="Kualitas" else None,
                ))
            fig.update_layout(**chart_layout)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        fig2 = go.Figure()
        for col_name, color in [("Kualitas","#4a90e2"),("Ketajaman","#3fcf8e"),("Noise","#e24a4a")]:
            fig2.add_trace(go.Bar(name=col_name, x=df["label"], y=df[col_name],
                                  marker_color=color, marker_line_width=0))
        layout2 = dict(**chart_layout)
        layout2.update(barmode="group", bargap=0.2, bargroupgap=0.05)
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

# ── History ───────────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr>", unsafe_allow_html=True)
    hc, bc = st.columns([3,1])
    with hc:
        st.markdown("<p class='sec'>Riwayat</p>", unsafe_allow_html=True)
    with bc:
        if st.button("Hapus semua", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for e in hist:
        c1,c2,c3 = st.columns([1,4,1])
        with c1:
            thumb = Image.open(io.BytesIO(base64.b64decode(e["thumb_b64"])))
            st.image(thumb, use_container_width=True)
        with c2:
            st.markdown(f"""
            <div style='padding:2px 0;'>
              <p style='font-size:0.85rem;font-weight:500;color:#111;margin:0 0 2px;'>{e['name']}</p>
              <p style='font-size:0.72rem;color:#aaa;margin:0;'>{e['datetime']} · {e['size_kb']} KB · {e['width']}×{e['height']}</p>
              <p style='font-size:0.72rem;color:#aaa;margin:2px 0 0;'>{e['notes']}</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            g = e["quality_grade"]
            st.markdown(f"""
            <div style='text-align:right;'>
              <div style='font-size:1.3rem;font-weight:700;color:{grade_col(g)};'>{g}</div>
              <div style='font-size:0.72rem;color:#aaa;'>{e['quality_score']}/100</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
