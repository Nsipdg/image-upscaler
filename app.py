import streamlit as st
import anthropic
import base64
import json
import io
import os
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Quality Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #0f0f11; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1100px; }

/* Header */
.hero-title {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #f0ede8;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.85rem;
    color: #6b6b72;
    margin: 0.4rem 0 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-wrap {
    padding: 1.5rem 0 1.2rem;
    border-bottom: 1px solid #1e1e24;
    margin-bottom: 1.5rem;
}

/* KPI Cards */
.kpi-row { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #16161a;
    border: 1px solid #222228;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.kpi-label { font-size: 0.7rem; color: #5a5a64; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 600; color: #f0ede8; margin: 0; font-family: 'DM Mono', monospace; }
.kpi-delta { font-size: 0.72rem; margin: 4px 0 0; }
.kpi-delta.up { color: #3fcf8e; }
.kpi-delta.down { color: #f76e6e; }
.kpi-delta.neu { color: #6b6b72; }

/* Grade Badge */
.grade-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    padding: 2px 14px;
    border-radius: 20px;
}
.grade-A { background: #0d2e1f; color: #3fcf8e; border: 1px solid #1a5c3a; }
.grade-B { background: #1a2710; color: #7ec845; border: 1px solid #2d4a18; }
.grade-C { background: #2b2010; color: #e8a838; border: 1px solid #4a3518; }
.grade-D { background: #2b1010; color: #f76e6e; border: 1px solid #4a2020; }

/* Result table */
.res-table { width: 100%; border-collapse: collapse; }
.res-table td { padding: 7px 0; border-bottom: 1px solid #1e1e24; font-size: 0.85rem; }
.res-table td:first-child { color: #6b6b72; }
.res-table td:last-child { color: #f0ede8; text-align: right; font-family: 'DM Mono', monospace; }
.res-table tr:last-child td { border-bottom: none; }

/* History card */
.hist-card {
    background: #16161a;
    border: 1px solid #222228;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hist-info { flex: 1; min-width: 0; }
.hist-name { font-size: 0.85rem; font-weight: 500; color: #f0ede8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0 0 2px; }
.hist-meta { font-size: 0.72rem; color: #5a5a64; margin: 0; }
.hist-score { font-family: 'DM Mono', monospace; font-size: 1.1rem; font-weight: 500; color: #f0ede8; }

/* Upload zone */
.upload-note { font-size: 0.78rem; color: #5a5a64; margin: 0.3rem 0 0; }

/* Section title */
.sec-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5a5a64; margin: 0 0 0.8rem; font-family: 'DM Mono', monospace; }

/* Divider */
.divider { border: none; border-top: 1px solid #1e1e24; margin: 1.5rem 0; }

/* Recommendation pill */
.rec-pill {
    display: inline-block; font-size: 0.75rem; padding: 3px 10px;
    border-radius: 20px; background: #1e2535; color: #5b9cf6;
    border: 1px solid #2a3a5c; font-family: 'DM Mono', monospace;
}
.rec-pill.rec-yes { background: #2b1a10; color: #f0883e; border-color: #4a2e18; }
.rec-pill.rec-no { background: #0d2e1f; color: #3fcf8e; border-color: #1a5c3a; }

/* Notes box */
.notes-box {
    background: #1a1a20; border-left: 3px solid #3a3a44;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
    font-size: 0.8rem; color: #9a9aaa; font-style: italic;
    margin-top: 0.8rem;
}

/* Stframe fix */
[data-testid="stFileUploader"] label { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ── Helpers ───────────────────────────────────────────────────────────────────
def img_to_b64(img_bytes: bytes) -> str:
    return base64.standard_b64encode(img_bytes).decode()

def grade_color(g: str) -> str:
    return {"A": "#3fcf8e", "B": "#7ec845", "C": "#e8a838", "D": "#f76e6e"}.get(g, "#9a9aaa")

def score_color(s: int) -> str:
    if s >= 75: return "#3fcf8e"
    if s >= 50: return "#e8a838"
    return "#f76e6e"

def analyze_image(api_key: str, img_bytes: bytes, mime: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    b64 = img_to_b64(img_bytes)
    prompt = """Kamu adalah sistem analisis kualitas gambar profesional.
Analisis gambar ini dan berikan output HANYA dalam format JSON valid berikut (tanpa teks lain, tanpa markdown):
{
  "quality_score": <0-100>,
  "sharpness": <0-100>,
  "noise_level": <0-100, 0=sangat bersih>,
  "brightness": <"gelap"|"normal"|"terang">,
  "contrast": <"rendah"|"normal"|"tinggi">,
  "dominant_color": <warna dominan dalam bahasa Indonesia>,
  "subject": <subjek utama foto dalam bahasa Indonesia, maks 6 kata>,
  "resolution_estimate": <"rendah"|"sedang"|"tinggi">,
  "upscale_recommendation": <"tidak perlu"|"disarankan"|"sangat disarankan">,
  "quality_grade": <"A"|"B"|"C"|"D">,
  "notes": <catatan singkat dalam bahasa Indonesia, maks 20 kata>
}"""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def make_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["quality_score"],
        mode="lines+markers", name="Skor Kualitas",
        line=dict(color="#5b9cf6", width=2.5),
        marker=dict(size=7, color="#5b9cf6"),
        fill="tozeroy", fillcolor="rgba(91,156,246,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["sharpness"],
        mode="lines+markers", name="Ketajaman",
        line=dict(color="#f0883e", width=2, dash="dot"),
        marker=dict(size=6, color="#f0883e", symbol="diamond")
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["noise_level"],
        mode="lines+markers", name="Noise",
        line=dict(color="#f76e6e", width=2, dash="dash"),
        marker=dict(size=6, color="#f76e6e", symbol="square")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#9a9aaa", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="#1e1e24", linecolor="#1e1e24",
            tickfont=dict(size=10), tickangle=-30
        ),
        yaxis=dict(
            gridcolor="#1e1e24", linecolor="#1e1e24",
            range=[0, 105], tickfont=dict(size=10)
        ),
        hovermode="x unified",
    )
    return fig

def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Kualitas", x=df["label"], y=df["quality_score"],
        marker_color="#5b9cf6", marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        name="Ketajaman", x=df["label"], y=df["sharpness"],
        marker_color="#3fcf8e", marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        name="Noise", x=df["label"], y=df["noise_level"],
        marker_color="#f76e6e", marker_line_width=0
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#9a9aaa", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(gridcolor="#1e1e24", linecolor="#1e1e24", tickfont=dict(size=10), tickangle=-20),
        yaxis=dict(gridcolor="#1e1e24", linecolor="#1e1e24", range=[0, 110], tickfont=dict(size=10)),
        bargap=0.18, bargroupgap=0.05,
    )
    return fig

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">Image Quality<br>Analyzer</p>
  <p class="hero-sub">UTS Pemrograman Berbasis Objek · AI-powered</p>
</div>
""", unsafe_allow_html=True)

# ── API Key input ──────────────────────────────────────────────────────────────
with st.expander("⚙ Konfigurasi API Key", expanded=(not st.session_state.api_key)):
    st.markdown("""
    <p style='font-size:0.82rem;color:#6b6b72;margin:0 0 8px;'>
    Masukkan Anthropic API Key kamu. Bisa didapat gratis di
    <a href='https://console.anthropic.com' target='_blank' style='color:#5b9cf6;'>console.anthropic.com</a>
    </p>
    """, unsafe_allow_html=True)
    key_input = st.text_input("API Key", type="password", value=st.session_state.api_key,
                               placeholder="sk-ant-...", label_visibility="collapsed")
    if key_input:
        st.session_state.api_key = key_input
        st.success("API Key tersimpan ✓", icon="🔑")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Layout: 2 kolom utama ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.05], gap="large")

with col_left:
    st.markdown("<p class='sec-title'>Upload Foto</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload foto",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_bytes = uploaded.read()
        mime = uploaded.type or "image/jpeg"

        # Buka gambar
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        size_kb = len(img_bytes) / 1024

        st.image(img, use_container_width=True)
        st.markdown(f"""
        <div style='display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;'>
          <span style='font-size:0.72rem;color:#5a5a64;background:#16161a;padding:3px 10px;border-radius:20px;border:1px solid #222228;'>{w}×{h} px</span>
          <span style='font-size:0.72rem;color:#5a5a64;background:#16161a;padding:3px 10px;border-radius:20px;border:1px solid #222228;'>{size_kb:.1f} KB</span>
          <span style='font-size:0.72rem;color:#5a5a64;background:#16161a;padding:3px 10px;border-radius:20px;border:1px solid #222228;'>{mime.split("/")[1].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        btn_analyze = st.button("🔍 Analisis Kualitas", use_container_width=True, type="primary")

        if btn_analyze:
            if not st.session_state.api_key:
                st.error("Masukkan API Key dulu di bagian atas!")
            else:
                with st.spinner("Menganalisis dengan AI..."):
                    try:
                        result = analyze_image(st.session_state.api_key, img_bytes, mime)
                        # Simpan thumbnail kecil
                        thumb = img.copy()
                        thumb.thumbnail((120, 120))
                        buf = io.BytesIO()
                        thumb.save(buf, format="PNG")
                        thumb_b64 = img_to_b64(buf.getvalue())

                        entry = {
                            "id": int(time.time() * 1000),
                            "name": uploaded.name,
                            "time": datetime.now().strftime("%H:%M"),
                            "date": datetime.now().strftime("%d %b"),
                            "datetime": datetime.now().strftime("%d %b %H:%M"),
                            "size_kb": round(size_kb, 1),
                            "width": w,
                            "height": h,
                            "thumb_b64": thumb_b64,
                            **result
                        }
                        st.session_state.history.insert(0, entry)
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("Gagal parse respons AI. Coba lagi.")
                    except Exception as e:
                        st.error(f"Error: {e}")

with col_right:
    if st.session_state.history:
        latest = st.session_state.history[0]

        st.markdown("<p class='sec-title'>Hasil Analisis Terbaru</p>", unsafe_allow_html=True)

        grade = latest.get("quality_grade", "—")
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:14px;margin-bottom:1rem;'>
          <div style='font-family:"DM Mono",monospace;font-size:2.8rem;font-weight:500;color:{grade_color(grade)};line-height:1;'>{grade}</div>
          <div>
            <div style='font-size:1.4rem;font-weight:600;color:#f0ede8;font-family:"DM Mono",monospace;'>{latest.get("quality_score",0)}<span style='font-size:0.85rem;color:#5a5a64;'>/100</span></div>
            <div style='font-size:0.75rem;color:#5a5a64;margin-top:2px;'>{latest.get("subject","—")}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <table class='res-table'>
          <tr><td>Ketajaman</td><td style='color:{score_color(latest.get("sharpness",0))};'>{latest.get("sharpness","—")}/100</td></tr>
          <tr><td>Noise Level</td><td>{latest.get("noise_level","—")}/100</td></tr>
          <tr><td>Kecerahan</td><td>{latest.get("brightness","—")}</td></tr>
          <tr><td>Kontras</td><td>{latest.get("contrast","—")}</td></tr>
          <tr><td>Warna Dominan</td><td>{latest.get("dominant_color","—")}</td></tr>
          <tr><td>Resolusi</td><td>{latest.get("resolution_estimate","—")}</td></tr>
          <tr><td>Upscale</td><td>{latest.get("upscale_recommendation","—")}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        notes = latest.get("notes", "")
        if notes:
            st.markdown(f"<div class='notes-box'>💬 {notes}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem;'>
          <div style='font-size:2.5rem;margin-bottom:0.8rem;opacity:0.3;'>🖼</div>
          <p style='font-size:0.85rem;color:#5a5a64;text-align:center;margin:0;'>Upload dan analisis foto<br>untuk melihat hasil di sini</p>
        </div>
        """, unsafe_allow_html=True)

# ── KPI Section ───────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<p class='sec-title'>Key Performance Indicators</p>", unsafe_allow_html=True)

h = st.session_state.history
n = len(h)

if n > 0:
    scores = [x["quality_score"] for x in h]
    sharp  = [x["sharpness"] for x in h]
    noise  = [x["noise_level"] for x in h]

    avg_q = round(sum(scores) / n)
    max_q = max(scores)
    min_q = min(scores)
    avg_s = round(sum(sharp) / n)

    grade_counts = {}
    for x in h:
        g = x.get("quality_grade", "?")
        grade_counts[g] = grade_counts.get(g, 0) + 1
    top_grade = max(grade_counts, key=grade_counts.get)

    # Delta vs foto sebelumnya
    if n >= 2:
        delta_q = scores[0] - scores[1]
        delta_sym = "▲" if delta_q > 0 else ("▼" if delta_q < 0 else "—")
        delta_cls = "up" if delta_q > 0 else ("down" if delta_q < 0 else "neu")
        delta_str = f"{delta_sym} {abs(delta_q)} vs sebelumnya"
    else:
        delta_str = "foto pertama"
        delta_cls = "neu"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
          <p class='kpi-label'>Total Foto</p>
          <p class='kpi-value'>{n}</p>
          <p class='kpi-delta neu'>dianalisis</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
          <p class='kpi-label'>Rata-rata Kualitas</p>
          <p class='kpi-value'>{avg_q}</p>
          <p class='kpi-delta {delta_cls}'>{delta_str}</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
          <p class='kpi-label'>Skor Tertinggi</p>
          <p class='kpi-value' style='color:#3fcf8e;'>{max_q}</p>
          <p class='kpi-delta neu'>terbaik</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card'>
          <p class='kpi-label'>Skor Terendah</p>
          <p class='kpi-value' style='color:#f76e6e;'>{min_q}</p>
          <p class='kpi-delta neu'>perlu perbaikan</p>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='kpi-card'>
          <p class='kpi-label'>Rata-rata Ketajaman</p>
          <p class='kpi-value'>{avg_s}</p>
          <p class='kpi-delta neu'>sharpness</p>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='kpi-card' style='text-align:center;'>
      <p style='color:#5a5a64;font-size:0.85rem;margin:0;'>Belum ada data. Upload dan analisis foto untuk melihat KPI.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Chart Section ──────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    df_hist = pd.DataFrame([{
        "label": f"{x['name'][:12]}…" if len(x['name']) > 12 else x['name'],
        "datetime": x["datetime"],
        "quality_score": x["quality_score"],
        "sharpness": x["sharpness"],
        "noise_level": x["noise_level"],
        "quality_grade": x.get("quality_grade", "—"),
    } for x in reversed(h)])  # kronologis

    tab_trend, tab_bar = st.tabs(["📈 Tren Garis", "📊 Perbandingan Batang"])

    with tab_trend:
        st.markdown("<p class='sec-title' style='margin-top:0.5rem;'>Tren skor kualitas dari waktu ke waktu</p>", unsafe_allow_html=True)
        if n < 2:
            st.info("Upload minimal 2 foto untuk melihat tren.", icon="ℹ️")
        else:
            fig_trend = make_trend_chart(df_hist)
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    with tab_bar:
        st.markdown("<p class='sec-title' style='margin-top:0.5rem;'>Perbandingan metrik antar foto</p>", unsafe_allow_html=True)
        fig_bar = make_bar_chart(df_hist)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ── History Section ────────────────────────────────────────────────────────────
if n > 0:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='sec-title'>Riwayat Analisis</p>", unsafe_allow_html=True)

    col_hist, col_clear = st.columns([3, 1])
    with col_clear:
        if st.button("🗑 Hapus Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for entry in st.session_state.history:
        grade = entry.get("quality_grade", "—")
        qs = entry.get("quality_score", 0)
        upscale = entry.get("upscale_recommendation", "—")

        c_img, c_info, c_score = st.columns([1, 3, 1])
        with c_img:
            thumb_bytes = base64.b64decode(entry["thumb_b64"])
            thumb_img = Image.open(io.BytesIO(thumb_bytes))
            st.image(thumb_img, use_container_width=True)

        with c_info:
            st.markdown(f"""
            <div style='padding:4px 0;'>
              <p style='font-size:0.88rem;font-weight:500;color:#f0ede8;margin:0 0 2px;'>{entry["name"]}</p>
              <p style='font-size:0.72rem;color:#5a5a64;margin:0 0 4px;'>{entry["datetime"]} · {entry["size_kb"]} KB · {entry["width"]}×{entry["height"]}</p>
              <p style='font-size:0.75rem;color:#9a9aaa;margin:0;'>{entry.get("subject","—")}</p>
            </div>
            """, unsafe_allow_html=True)

        with c_score:
            st.markdown(f"""
            <div style='text-align:right;padding:4px 0;'>
              <div style='font-family:"DM Mono",monospace;font-size:1.4rem;font-weight:500;color:{grade_color(grade)};'>{grade}</div>
              <div style='font-size:0.78rem;color:#5a5a64;'>{qs}/100</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid #1e1e24;margin:4px 0;'>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding-top:1rem;border-top:1px solid #1e1e24;
            text-align:center;font-size:0.72rem;color:#3a3a44;'>
  Image Quality Analyzer · UTS PBO · Powered by Claude AI
</div>
""", unsafe_allow_html=True)
