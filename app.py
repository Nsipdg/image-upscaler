import streamlit as st
import io, math, base64, time
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.block-container { padding: 1.2rem 1.8rem 3rem; max-width: 1080px; }

.hero { padding: 1.2rem 0 1rem; border-bottom: 1px solid #e5e0d8; margin-bottom: 1.4rem; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700;
  color: #1a1a1a; margin: 0; letter-spacing: -0.04em; line-height: 1.1; }
.hero-sub { font-size: 0.75rem; color: #999; margin: 5px 0 0;
  text-transform: uppercase; letter-spacing: 0.1em; }

.kpi-wrap { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.4rem; }
.kpi { flex: 1; min-width: 120px; background: #faf9f7; border: 1px solid #e5e0d8;
  border-radius: 10px; padding: 0.85rem 1rem; }
.kpi-lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: #aaa; margin: 0 0 5px; }
.kpi-val { font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700;
  color: #1a1a1a; margin: 0; line-height: 1; }
.kpi-note { font-size: 0.68rem; color: #aaa; margin: 4px 0 0; }

.grade-box { display:inline-flex; align-items:center; justify-content:center;
  width:52px; height:52px; border-radius:10px; font-family:'Space Mono',monospace;
  font-size:1.6rem; font-weight:700; }
.gA { background:#d4f5e2; color:#1a7a4a; }
.gB { background:#d9f0d4; color:#2d6e22; }
.gC { background:#fff3cd; color:#856404; }
.gD { background:#ffe0e0; color:#a02020; }

.res-table { width:100%; border-collapse:collapse; }
.res-table td { padding:6px 0; border-bottom:1px solid #f0ece6; font-size:0.82rem; }
.res-table td:first-child { color:#888; }
.res-table td:last-child { text-align:right; font-family:'Space Mono',monospace;
  font-size:0.8rem; color:#1a1a1a; }
.res-table tr:last-child td { border:none; }

.bar-wrap { height:10px; background:#f0ece6; border-radius:5px; overflow:hidden; margin-top:3px; }
.bar-inner { height:100%; border-radius:5px; }

.hist-row { display:flex; align-items:center; gap:10px; padding:8px 0;
  border-bottom:1px solid #f0ece6; }
.hist-row:last-child { border:none; }
.hist-name { font-size:0.82rem; font-weight:500; color:#1a1a1a; margin:0 0 2px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:160px; }
.hist-meta { font-size:0.68rem; color:#aaa; margin:0; }
.hist-score { font-family:'Space Mono',monospace; font-size:1rem; font-weight:700; }

.sec-lbl { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
  color:#aaa; margin:0 0 0.7rem; font-family:'Space Mono',monospace; }
.notes { background:#faf9f7; border-left:3px solid #d0c8bc; border-radius:0 8px 8px 0;
  padding:0.6rem 0.9rem; font-size:0.78rem; color:#666; margin-top:0.8rem; }
.divider { border:none; border-top:1px solid #e5e0d8; margin:1.4rem 0; }
.badge { display:inline-block; font-size:0.7rem; padding:2px 9px;
  border-radius:20px; font-family:'Space Mono',monospace; }
.badge-ok { background:#d4f5e2; color:#1a7a4a; }
.badge-warn { background:#fff3cd; color:#856404; }
.badge-bad { background:#ffe0e0; color:#a02020; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Core analysis (100% PIL, no API) ─────────────────────────────────────────
def analyze(img: Image.Image, filename: str, size_bytes: int) -> dict:
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)
    gray = np.array(rgb.convert("L"), dtype=np.float32)
    w, h = rgb.size

    # --- Sharpness via Laplacian variance ---
    lap = np.array(rgb.convert("L").filter(ImageFilter.Kernel(
        size=3, kernel=[-1,-1,-1,-1,8,-1,-1,-1,-1], scale=1, offset=128
    )), dtype=np.float32)
    lap_var = float(np.var(lap))
    sharpness = min(100, int(lap_var / 8))

    # --- Noise via high-freq std ---
    blurred = np.array(rgb.convert("L").filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32)
    noise_map = np.abs(gray - blurred)
    noise_raw = float(np.mean(noise_map))
    noise_level = min(100, int(noise_raw * 4))

    # --- Brightness ---
    brightness_val = float(np.mean(gray))
    if brightness_val < 60:
        brightness = "gelap"
    elif brightness_val > 190:
        brightness = "terang"
    else:
        brightness = "normal"

    # --- Contrast via std ---
    contrast_val = float(np.std(gray))
    if contrast_val < 35:
        contrast = "rendah"
    elif contrast_val > 75:
        contrast = "tinggi"
    else:
        contrast = "normal"

    # --- Dominant color ---
    pixels = arr.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)
    r, g_c, b = mean_rgb
    if r > g_c and r > b:
        dom = "Merah / Hangat"
    elif g_c > r and g_c > b:
        dom = "Hijau"
    elif b > r and b > g_c:
        dom = "Biru / Dingin"
    elif r > 180 and g_c > 180 and b > 180:
        dom = "Putih / Terang"
    elif r < 60 and g_c < 60 and b < 60:
        dom = "Hitam / Gelap"
    else:
        dom = "Netral / Abu-abu"

    # --- Resolution rating ---
    mp = (w * h) / 1_000_000
    if mp >= 4:
        res_est = "tinggi"
    elif mp >= 1:
        res_est = "sedang"
    else:
        res_est = "rendah"

    # --- Quality score (weighted) ---
    sharp_w   = sharpness * 0.40
    noise_w   = (100 - noise_level) * 0.25
    bright_w  = (80 if brightness == "normal" else 50) * 0.10
    contrast_w= (80 if contrast == "normal" else (60 if contrast == "tinggi" else 40)) * 0.10
    res_w     = (100 if res_est == "tinggi" else (70 if res_est == "sedang" else 40)) * 0.15
    quality   = int(sharp_w + noise_w + bright_w + contrast_w + res_w)
    quality   = max(0, min(100, quality))

    # --- Grade ---
    if quality >= 80: grade = "A"
    elif quality >= 65: grade = "B"
    elif quality >= 45: grade = "C"
    else: grade = "D"

    # --- Upscale recommendation ---
    if quality >= 80 and res_est == "tinggi":
        upscale = "tidak perlu"
    elif quality < 50 or res_est == "rendah":
        upscale = "sangat disarankan"
    else:
        upscale = "disarankan"

    # --- Notes ---
    issues = []
    if sharpness < 40: issues.append("gambar blur")
    if noise_level > 60: issues.append("noise tinggi")
    if brightness == "gelap": issues.append("terlalu gelap")
    if brightness == "terang": issues.append("overexposed")
    if contrast == "rendah": issues.append("kontras rendah")
    notes = "Foto berkualitas baik." if not issues else "Perhatikan: " + ", ".join(issues) + "."

    return {
        "name": filename,
        "size_kb": round(size_bytes / 1024, 1),
        "width": w, "height": h,
        "datetime": datetime.now().strftime("%d %b %H:%M"),
        "quality_score": quality,
        "sharpness": sharpness,
        "noise_level": noise_level,
        "brightness_val": round(brightness_val, 1),
        "contrast_val": round(contrast_val, 1),
        "brightness": brightness,
        "contrast": contrast,
        "dominant_color": dom,
        "resolution_estimate": res_est,
        "upscale_recommendation": upscale,
        "quality_grade": grade,
        "notes": notes,
    }

def grade_col(g):
    return {"A":"#1a7a4a","B":"#2d6e22","C":"#856404","D":"#a02020"}.get(g,"#888")

def score_bar(val, color="#4a90e2"):
    return f"""<div class="bar-wrap"><div class="bar-inner" style="width:{val}%;background:{color};"></div></div>"""

def badge_upscale(u):
    if u == "tidak perlu": return '<span class="badge badge-ok">tidak perlu</span>'
    if u == "disarankan":  return '<span class="badge badge-warn">disarankan</span>'
    return '<span class="badge badge-bad">sangat disarankan</span>'

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">Image Quality<br>Analyzer</p>
  <p class="hero-sub">UTS PBO · Gratis · No API Key · Powered by PIL + NumPy</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_up, col_res = st.columns([1, 1.1], gap="large")

with col_up:
    st.markdown("<p class='sec-lbl'>Upload Foto</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih foto", type=["jpg","jpeg","png","webp","bmp"],
                                 label_visibility="collapsed")

    if uploaded:
        raw = uploaded.read()
        img = Image.open(io.BytesIO(raw))
        st.image(img, use_container_width=True)

        w, h = img.size
        kb = len(raw)/1024
        st.markdown(f"""
        <div style='display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 12px;'>
          <span style='font-size:0.7rem;color:#888;background:#f5f3f0;padding:2px 9px;border-radius:20px;border:1px solid #e5e0d8;'>{w}×{h} px</span>
          <span style='font-size:0.7rem;color:#888;background:#f5f3f0;padding:2px 9px;border-radius:20px;border:1px solid #e5e0d8;'>{kb:.1f} KB</span>
          <span style='font-size:0.7rem;color:#888;background:#f5f3f0;padding:2px 9px;border-radius:20px;border:1px solid #e5e0d8;'>{uploaded.type.split("/")[1].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔬 Analisis Kualitas", use_container_width=True, type="primary"):
            with st.spinner("Menganalisis..."):
                result = analyze(img, uploaded.name, len(raw))
                # simpan thumbnail
                thumb = img.copy(); thumb.thumbnail((100,100))
                buf = io.BytesIO(); thumb.save(buf, format="PNG")
                result["thumb_b64"] = base64.b64encode(buf.getvalue()).decode()
                st.session_state.history.insert(0, result)
                st.rerun()

with col_res:
    if st.session_state.history:
        latest = st.session_state.history[0]
        g = latest["quality_grade"]
        qs = latest["quality_score"]
        sh = latest["sharpness"]
        nl = latest["noise_level"]

        st.markdown("<p class='sec-lbl'>Hasil Analisis</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:14px;margin-bottom:1rem;'>
          <div class='grade-box g{g}'>{g}</div>
          <div>
            <div style='font-family:"Space Mono",monospace;font-size:2rem;font-weight:700;color:#1a1a1a;line-height:1;'>{qs}<span style='font-size:0.9rem;color:#aaa;'>/100</span></div>
            <div style='font-size:0.75rem;color:#aaa;margin-top:2px;'>{latest.get("name","")[:28]}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Skor bars
        st.markdown(f"""
        <div style='margin-bottom:1rem;'>
          <div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#888;margin-bottom:2px;'>
            <span>Ketajaman</span><span style='font-family:"Space Mono",monospace;'>{sh}</span>
          </div>
          {score_bar(sh, "#4a90e2")}
          <div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#888;margin:8px 0 2px;'>
            <span>Noise (lebih rendah = lebih baik)</span><span style='font-family:"Space Mono",monospace;'>{nl}</span>
          </div>
          {score_bar(nl, "#e24a4a")}
          <div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#888;margin:8px 0 2px;'>
            <span>Kualitas Keseluruhan</span><span style='font-family:"Space Mono",monospace;'>{qs}</span>
          </div>
          {score_bar(qs, "#3fcf8e")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <table class='res-table'>
          <tr><td>Kecerahan</td><td>{latest["brightness"]} ({latest["brightness_val"]})</td></tr>
          <tr><td>Kontras</td><td>{latest["contrast"]} (σ={latest["contrast_val"]})</td></tr>
          <tr><td>Warna Dominan</td><td>{latest["dominant_color"]}</td></tr>
          <tr><td>Resolusi</td><td>{latest["resolution_estimate"]} ({latest["width"]}×{latest["height"]})</td></tr>
          <tr><td>Upscale</td><td>{badge_upscale(latest["upscale_recommendation"])}</td></tr>
        </table>
        <div class='notes'>💡 {latest["notes"]}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:300px;'>
          <div style='font-size:3rem;opacity:0.15;margin-bottom:12px;'>🔬</div>
          <p style='font-size:0.85rem;color:#bbb;text-align:center;'>Upload foto dan klik Analisis<br>untuk melihat hasil di sini</p>
        </div>
        """, unsafe_allow_html=True)

# ── KPI ────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<p class='sec-lbl'>Key Performance Indicators</p>", unsafe_allow_html=True)

hist = st.session_state.history
n = len(hist)

if n > 0:
    scores  = [x["quality_score"] for x in hist]
    sharps  = [x["sharpness"] for x in hist]
    noises  = [x["noise_level"] for x in hist]
    avg_q   = round(sum(scores)/n)
    avg_s   = round(sum(sharps)/n)
    avg_n   = round(sum(noises)/n)

    delta_str = ""
    delta_col = "#aaa"
    if n >= 2:
        d = scores[0] - scores[1]
        delta_str = f"{'▲' if d>0 else '▼'} {abs(d)} vs sebelumnya"
        delta_col = "#1a7a4a" if d > 0 else "#a02020"

    grade_map = {}
    for x in hist:
        grade_map[x["quality_grade"]] = grade_map.get(x["quality_grade"],0)+1
    best_g = max(grade_map, key=grade_map.get)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, lbl, val, note, col_hex in [
        (c1, "Total Foto",        str(n),    "dianalisis",     "#1a1a1a"),
        (c2, "Rata-rata Kualitas",str(avg_q), delta_str or "—", delta_col if delta_str else "#aaa"),
        (c3, "Skor Tertinggi",    str(max(scores)), "terbaik",  "#1a7a4a"),
        (c4, "Skor Terendah",     str(min(scores)), "perlu upscale","#a02020"),
        (c5, "Rata-rata Ketajaman",str(avg_s),"sharpness",     "#4a90e2"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi'>
              <p class='kpi-lbl'>{lbl}</p>
              <p class='kpi-val' style='color:{col_hex};'>{val}</p>
              <p class='kpi-note'>{note}</p>
            </div>""", unsafe_allow_html=True)
else:
    st.info("Belum ada data. Upload dan analisis foto untuk melihat KPI.", icon="📊")

# ── Charts ─────────────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    df = pd.DataFrame([{
        "label": (x["name"][:10]+"…" if len(x["name"])>10 else x["name"]),
        "datetime": x["datetime"],
        "Kualitas": x["quality_score"],
        "Ketajaman": x["sharpness"],
        "Noise": x["noise_level"],
    } for x in reversed(hist)])

    tab1, tab2 = st.tabs(["📈 Tren Garis", "📊 Perbandingan Batang"])

    with tab1:
        st.markdown("<p class='sec-lbl' style='margin-top:0.5rem;'>Tren skor dari waktu ke waktu</p>", unsafe_allow_html=True)
        if n < 2:
            st.info("Upload minimal 2 foto untuk melihat tren.", icon="ℹ️")
        else:
            fig = go.Figure()
            configs = [
                ("Kualitas",  "#4a90e2", "solid",  dict(size=7)),
                ("Ketajaman", "#3fcf8e", "dot",    dict(size=6, symbol="diamond")),
                ("Noise",     "#e24a4a", "dash",   dict(size=6, symbol="square")),
            ]
            for col_name, color, dash, marker in configs:
                fig.add_trace(go.Scatter(
                    x=df["label"], y=df[col_name], name=col_name,
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(color=color, **marker),
                    fill=("tozeroy" if col_name=="Kualitas" else "none"),
                    fillcolor="rgba(74,144,226,0.05)" if col_name=="Kualitas" else None,
                ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Syne", size=12, color="#888"),
                height=300, margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                xaxis=dict(gridcolor="#f0ece6", linecolor="#e5e0d8", tickangle=-25, tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#f0ece6", linecolor="#e5e0d8", range=[0,105], tickfont=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        st.markdown("<p class='sec-lbl' style='margin-top:0.5rem;'>Perbandingan metrik antar foto</p>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for col_name, color in [("Kualitas","#4a90e2"),("Ketajaman","#3fcf8e"),("Noise","#e24a4a")]:
            fig2.add_trace(go.Bar(
                name=col_name, x=df["label"], y=df[col_name],
                marker_color=color, marker_line_width=0
            ))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Syne", size=12, color="#888"),
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            xaxis=dict(gridcolor="#f0ece6", linecolor="#e5e0d8", tickangle=-20, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#f0ece6", linecolor="#e5e0d8", range=[0,110], tickfont=dict(size=10)),
            bargap=0.18, bargroupgap=0.05,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

# ── History ───────────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    h_col, btn_col = st.columns([3,1])
    with h_col:
        st.markdown("<p class='sec-lbl'>Riwayat Analisis</p>", unsafe_allow_html=True)
    with btn_col:
        if st.button("🗑 Hapus Semua", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for entry in st.session_state.history:
        g = entry["quality_grade"]
        c_thumb, c_info, c_score = st.columns([1, 3.5, 1])
        with c_thumb:
            thumb = Image.open(io.BytesIO(base64.b64decode(entry["thumb_b64"])))
            st.image(thumb, use_container_width=True)
        with c_info:
            st.markdown(f"""
            <p class='hist-name'>{entry["name"]}</p>
            <p class='hist-meta'>{entry["datetime"]} · {entry["size_kb"]} KB · {entry["width"]}×{entry["height"]} · {entry.get("dominant_color","—")}</p>
            <p class='hist-meta' style='margin-top:3px;'>{entry["notes"]}</p>
            """, unsafe_allow_html=True)
        with c_score:
            st.markdown(f"""
            <div style='text-align:right;padding:4px 0;'>
              <div style='font-family:"Space Mono",monospace;font-size:1.5rem;font-weight:700;color:{grade_col(g)};'>{g}</div>
              <div style='font-size:0.72rem;color:#aaa;'>{entry["quality_score"]}/100</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #f0ece6;margin:4px 0;'>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2.5rem;text-align:center;font-size:0.68rem;color:#ccc;border-top:1px solid #f0ece6;padding-top:1rem;'>
  Image Quality Analyzer · UTS PBO · 100% Gratis · No API Key
</div>
""", unsafe_allow_html=True)
