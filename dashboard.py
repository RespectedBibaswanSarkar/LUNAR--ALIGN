#!/usr/bin/env python3
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.registration.pipeline import LunarAlignPipeline, build_csv_pair_manifest, discover_image_dataset_pairs

st.set_page_config(page_title="LUNAR-ALIGN Dashboard", page_icon="🌙", layout="wide")

ROOT = Path(__file__).resolve().parent
base_dir = ROOT / "outputs"
base_dir.mkdir(parents=True, exist_ok=True)

st.markdown(
    """
    <style>
    :root {
        --bg: #020d1f;
        --bg-2: #061a2d;
        --panel: rgba(10, 23, 38, 0.95);
        --panel-soft: rgba(18, 35, 53, 0.9);
        --panel-border: rgba(98, 164, 255, 0.32);
        --text: #eaf6ff;
        --muted: #a7bfd6;
        --primary: #5ec9ff;
        --primary-2: #2a9af5;
        --mint: #74f0b1;
        --warning: #f8b368;
        --shadow: rgba(1, 8, 20, 0.7);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    div[data-testid="stHeader"] { background: rgba(2,13,31,0.75); }
    .stApp > header { background: rgba(5,19,31,0.9); }
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .panel {
        background: linear-gradient(180deg, rgba(14,28,42,0.96), rgba(7,18,31,0.96));
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        box-shadow: 0 18px 42px var(--shadow);
        padding: 1.25rem 1.5rem;
    }
    .hero-panel {
        position: relative;
        overflow: hidden;
        min-height: 420px;
        background: radial-gradient(circle at 30% 30%, rgba(65,149,224,0.6), rgba(5,19,32,0.95) 40%, rgba(1,10,16,0.98) 72%),
                    linear-gradient(180deg, rgba(5,15,24,0.8), rgba(2,10,18,0.9));
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1.5rem 1.4rem;
    }
    .hero-panel::before {
        content: "";
        position: absolute;
        right: 10%;
        top: 8%;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, rgba(242,245,252,0.78), rgba(128,143,160,0.66) 28%, rgba(32,41,52,0.96) 55%, rgba(10,15,22,0.95) 70%);
        box-shadow: 0 0 80px rgba(111,157,255,0.25);
        opacity: 0.96;
    }
    .hero-panel::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 65% 50%, rgba(255,255,255,0.05), transparent 35%);
    }
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 60%;
        padding-top: 1rem;
    }
    .hero-title {
        font-size: clamp(2.0rem, 3vw, 4rem);
        font-weight: 700;
        line-height: 1.0;
        color: #edf7ff;
        letter-spacing: -0.04em;
        margin: 0;
    }
    .hero-title .accent {
        display: block;
        color: #dff4ff;
    }
    .hero-sub {
        margin-top: 1rem;
        color: var(--muted);
        font-size: 1.1rem;
        max-width: 520px;
    }
    .primary-btn {
        margin-top: 1.3rem;
        background: linear-gradient(90deg, var(--primary-2), var(--primary));
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 8px 22px rgba(40,130,246,0.35);
        cursor: pointer;
    }
    .sensor-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(100px, 1fr));
        gap: 12px;
        margin-top: 1rem;
    }
    .sensor-card {
        background: rgba(17, 35, 54, 0.74);
        border: 1px solid rgba(163, 196, 255, 0.2);
        border-radius: 12px;
        padding: 0.9rem 0.8rem;
        text-align: center;
        color: #dfeeff;
        min-height: 82px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }
    .nav-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(11, 22, 36, 0.9);
        border: 1px solid var(--panel-border);
        border-radius: 18px 18px 0 0;
        padding: 0.8rem 1rem;
        margin-top: 1.2rem;
    }
    .nav-bar {
        background: rgba(8, 20, 31, 0.9);
        border: 1px solid rgba(141, 192, 255, 0.2);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.7rem;
        font-weight: 700;
        color: #edf7ff;
    }
    .nav-brand .orb {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #e9f7ff, #7aa7d7 35%, #26538c 70%, #0e1b2d 100%);
        box-shadow: 0 0 22px rgba(124,178,255,0.8);
    }
    .nav-links {
        display: flex;
        gap: 1.2rem;
        color: var(--muted);
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-badge {
        background: rgba(30,50,68,0.8);
        border: 1px solid rgba(117, 182, 255, 0.3);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        color: #bcedcc;
        font-size: 0.72rem;
    }
    .section-card {
        background: linear-gradient(180deg, rgba(11,24,39,0.98), rgba(7,18,29,0.98));
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin-top: 1.1rem;
        box-shadow: 0 18px 42px rgba(1,9,18,0.7);
    }
    .section-title {
        font-size: 1.7rem; 
        font-weight: 700;
        margin: 0;
        color: var(--text);
    }
    .muted {
        color: var(--muted);
    }
    .upload-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    .upload-card {
        background: rgba(14, 31, 47, 0.88);
        border: 1px solid rgba(148, 194, 255, 0.18);
        border-radius: 13px;
        padding: 1rem 0.9rem;
        min-height: 220px;
    }
    .upload-label {
        display: block;
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 0.8rem;
    }
    .image-drop {
        border: 1px dashed rgba(121, 194, 255, 0.5);
        border-radius: 12px;
        min-height: 160px;
        background: rgba(21,42,60,0.76);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        color: var(--muted);
        text-align: center;
        gap: 0.4rem;
    }
    .tiny {
        font-size: 0.72rem;
        color: var(--muted);
    }
    .option-box {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: rgba(19,31,47,0.7);
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(118,172,255,0.2);
        margin-top: 0.9rem;
        color: var(--text);
    }
    .check {
        width: 14px;
        height: 14px;
        border-radius: 4px;
        background: rgba(90,226,170,0.9);
        box-shadow: 0 0 12px rgba(90,226,170,0.6);
    }
    .register-btn {
        width: 100%;
        margin-top: 0.9rem;
        background: linear-gradient(90deg, #2b83f4, #4ec1ff);
        border: none;
        border-radius: 10px;
        padding: 0.82rem 1rem;
        color: white;
        font-weight: 700;
        font-size: 1rem;
    }
    .progress-panel {
        display: grid;
        grid-template-columns: 1.65fr 0.8fr;
        gap: 1rem;
        margin-top: 1.1rem;
    }
    .progress-box {
        background: rgba(11, 23, 38, 0.9);
        border: 1px solid rgba(116, 164, 255, 0.22);
        border-radius: 18px;
        padding: 1rem 1.2rem;
    }
    .progress-row {
        display: grid;
        grid-template-columns: 1.2fr 2fr 0.7fr;
        align-items: center;
        gap: 0.75rem;
        margin: 0.7rem 0;
        color: var(--text);
    }
    .progress-bar {
        height: 8px;
        border-radius: 100px;
        background: rgba(137, 176, 221, 0.2);
        overflow: hidden;
    }
    .progress-bar > span {
        display: block;
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #6ee7ff, #60d5ff, #5d9aff);
    }
    .donut-wrap {
        position: relative;
        width: 150px;
        height: 150px;
        margin: 0 auto;
        border-radius: 50%;
        background: conic-gradient(#60d0ff 0 72%, rgba(137,176,221,0.2) 72% 100%);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .donut-wrap::before {
        content: "";
        position: absolute;
        inset: 17px;
        border-radius: 50%;
        background: rgba(8, 17, 28, 0.95);
        border: 1px solid rgba(117, 170, 255, 0.25);
    }
    .donut-center {
        position: relative;
        z-index: 1;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .metric-box {
        background: rgba(18, 36, 55, 0.75);
        border: 1px solid rgba(117, 170, 255, 0.2);
        border-radius: 12px;
        padding: 0.9rem 0.8rem;
    }
    .metric-box .name {
        color: var(--muted);
        font-size: 0.7rem;
        display: block;
    }
    .metric-box .val {
        display: block;
        margin-top: 0.3rem;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .match-preview {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    .preview-card {
        background: rgba(11, 23, 38, 0.9);
        border: 1px solid rgba(117, 170, 255, 0.2);
        border-radius: 12px;
        padding: 0.7rem;
    }
    .preview-thumb {
        border-radius: 10px;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(120,160,190,0.5), rgba(16,25,35,0.9));
        min-height: 180px;
    }
    .preview-thumb img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .download-list {
        display: grid;
        gap: 0.55rem;
    }
    .download-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(18,36,55,0.8);
        border: 1px solid rgba(117,170,255,0.2);
        border-radius: 10px;
        padding: 0.75rem 0.8rem;
        color: var(--text);
    }
    .download-item .label { color: #dfeeff; }
    .download-item .tag {
        background: rgba(94,201,255,0.12);
        border: 1px solid rgba(94,201,255,0.35);
        color: #bfeeff;
        padding: 0.2rem 0.45rem;
        border-radius: 6px;
        font-size: 0.7rem;
    }
    @media (max-width: 1100px) {
        .hero-content { max-width: 100%; }
        .progress-panel, .upload-grid, .match-preview { grid-template-columns: 1fr; }
        .sensor-row { grid-template-columns: repeat(2, 1fr); }
        .nav-strip { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

manifest = build_csv_pair_manifest(ROOT / "coordinates_ohrc.csv", ROOT / "coordinates_tmc2.csv", max_pairs=6)
image_pairs = discover_image_dataset_pairs(ROOT, max_pairs=6)
all_pairs = (manifest or []) + [{**pair, "ohrc": Path(pair["reference"]).name, "tmc": Path(pair["target"]).name} for pair in (image_pairs or [])]

if not all_pairs:
    st.error("No valid dataset pairs were discovered. Add reference/target image folders or valid coordinates CSVs.")
    st.stop()

pair_labels = [f"{item.get('ohrc', Path(item.get('reference', '')).name)} ↔ {item.get('tmc', Path(item.get('target', '')).name)}" for item in all_pairs]
selected_label = st.selectbox("Select image pair to correspond", pair_labels, key="pair_select")
selected_idx = pair_labels.index(selected_label)
selected_pair = all_pairs[selected_idx]

reference_path = selected_pair.get("reference") if selected_pair.get("reference") else selected_pair.get("ohrc")
target_path = selected_pair.get("target") if selected_pair.get("target") else selected_pair.get("tmc")

if "result_cache" not in st.session_state:
    st.session_state.result_cache = {}

if selected_label not in st.session_state.result_cache:
    pipeline = LunarAlignPipeline(output_dir=str(base_dir / selected_pair.get("pair_id", "pair_selected")))
    if reference_path and Path(reference_path).exists() and target_path and Path(target_path).exists():
        st.session_state.result_cache[selected_label] = pipeline.run(reference_path, target_path, pair_metadata=selected_pair)
    else:
        st.session_state.result_cache[selected_label] = pipeline.run_csv_pair(selected_pair, metadata_only=False)

result = st.session_state.result_cache[selected_label]

st.markdown("""
<div class='nav-bar'>
  <div style='display:flex; align-items:center; gap:12px;'><span class='orb' style='width:18px;height:18px;border-radius:50%; background: radial-gradient(circle at 35% 35%, #f1f9ff, #87b2e8 30%, #2c5f9e 68%, #091c2a 100%); box-shadow: 0 0 18px rgba(117,176,255,0.7);'></span><span style='font-size:1.6rem; font-weight:700; color:#eff8ff;'>LUNAR-ALIGN</span></div>
  <div style='display:flex; align-items:center; gap:2rem; color:#cfdff4; font-size:0.9rem; font-weight:600;'><span>Home</span><span>Register</span><span>Results</span><span>Documentation</span></div>
  <div class='status-badge'>● System Ready</div>
</div>
""", unsafe_allow_html=True)

hero, register = st.columns([1.25, 1.1])
with hero:
    st.markdown(
        """
        <div class='hero-panel'>
            <div class='hero-content'>
                <div class='hero-title'>Physics-Guided Multimodal <span class='accent'>Lunar Image Registration</span></div>
                <div class='hero-sub'>Register Chandrayaan-2 imagery across sensors, scales, viewpoints and illumination conditions.</div>
                <button class='primary-btn'>Start Registration →</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with register:
    st.markdown(
        """
        <div class='section-card'>
            <div style='display:flex; align-items:center; justify-content:space-between;'><h2 style='margin:0; font-size:1.6rem;'>Register Lunar Images</h2></div>
            <div class='muted' style='margin-top:0.4rem;'>Upload image pairs, select overlap settings and confirm processing options.</div>
            <div class='upload-grid'>
                <div class='upload-card'>
                    <span class='upload-label'>Source Image (Chandrayaan-2)</span>
                    <div class='image-drop'>
                        <div>🌙</div>
                        <div>Drag & drop image here</div>
                        <div class='tiny'>or click to browse</div>
                    </div>
                    <div class='tiny' style='margin-top:0.7rem;'>Source: OHRC (3.2 m)</div>
                </div>
                <div class='upload-card'>
                    <span class='upload-label'>Reference Image</span>
                    <div class='image-drop'>
                        <div>🌙</div>
                        <div>Drag & drop image here</div>
                        <div class='tiny'>or click to browse</div>
                    </div>
                    <div class='tiny' style='margin-top:0.7rem;'>Source: LRO NAC (0.5 m)</div>
                </div>
            </div>
            <div class='option-box'><div class='check'></div>Sensor-aware preprocessing</div>
            <div class='option-box'><div class='check'></div>Multiscale pyramid</div>
            <div class='option-box'><div class='check'></div>Adaptive matching</div>
            <div class='option-box'><div class='check'></div>Geometric verification</div>
            <button class='register-btn'>Register Images →</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

sensor_cols = st.columns(4)
for col, label in zip(sensor_cols, ["OHRC (PAN)", "TMC-2", "IIRS", "LRO NAC"]):
    with col:
        st.markdown(f"<div class='sensor-card'>{label}</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:0.7rem;'><h3 class='section-title'>3. Registration in Progress</h3><div class='status-badge' style='color:#d8ffee;'>● In progress</div></div>", unsafe_allow_html=True)
    prog_left, prog_right = st.columns([1.7, 0.9])
    with prog_left:
        st.markdown("<div class='progress-box'>", unsafe_allow_html=True)
        stages = [
            ("Data preprocessing", 100),
            ("Geometric normalization", 100),
            ("Adaptive matching", 72),
            ("Geometric verification", 58),
            ("Sub-pixel refinement", 20),
            ("Registration", 10),
        ]
        for name, pct in stages:
            st.markdown(
                f"""
                <div class='progress-row'>
                    <div style='color:#dfeeff; font-size:0.8rem;'>{name}</div>
                    <div class='progress-bar'><span style='width:{pct}%;'></span></div>
                    <div style='color:#dfeeff; text-align:right; font-size:0.8rem;'>{pct}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with prog_right:
        st.markdown("<div class='progress-box' style='height:100%; display:flex; align-items:center; justify-content:center;'><div class='donut-wrap'><div class='donut-center'>72%</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; align-items:center; justify-content:space-between;'><h3 class='section-title'>Registration Results</h3><div class='status-badge'>● System Ready</div></div>", unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metrics_map = [
        ("RMSE", f"{result['rmse']:.2f}px"),
        ("Inlier Ratio", f"{result['inlier_ratio']*100:.1f}%"),
        ("Matches", str(int(len(result['matches']['points1']) if result.get('matches') else 0))),
        ("Coverage", f"{result['coverage']*100:.0f}%"),
        ("Runtime", f"{result['runtime']:.2f}s"),
    ]
    for col, (name, value) in zip(metric_cols, metrics_map):
        with col:
            st.markdown(f"<div class='metric-box'><span class='name'>{name}</span><span class='val'>{value}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    match_cols = st.columns([1.5, 1.5, 0.8])
    with match_cols[0]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex; align-items:center; justify-content:space-between;'><div class='muted'>Source (OHRC)</div><div class='muted'>Reference (LRO)</div></div>", unsafe_allow_html=True)
        first, second = st.columns(2)
        with first:
            st.markdown("<div class='preview-card'><div class='preview-thumb'>" + (f"<img src='data:image/png;base64,{Path(base_dir / selected_pair.get('pair_id', 'pair_selected') / 'matches.png').read_bytes()[:0] if False else ''}' />" if False else "") + "</div></div>", unsafe_allow_html=True)
        with second:
            st.markdown("<div class='preview-card'><div class='preview-thumb' style='background:linear-gradient(180deg, rgba(132,152,176,0.35), rgba(10,20,30,0.9));'></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with match_cols[1]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex; align-items:center; justify-content:space-between;'><div class='muted'>Detected matches</div><div class='status-badge'>● Stable</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='preview-thumb' style='margin-top:0.8rem; min-height:220px; background:radial-gradient(circle at center, rgba(157,212,255,0.2), rgba(12,22,34,0.8));'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with match_cols[2]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='muted' style='margin-bottom:0.8rem;'>Download Results</div>", unsafe_allow_html=True)
        st.markdown("<div class='download-list'><div class='download-item'><span class='label'>Registered Image</span><span class='tag'>PNG</span></div><div class='download-item'><span class='label'>Match Points</span><span class='tag'>CSV</span></div><div class='download-item'><span class='label'>Error Map</span><span class='tag'>PNG</span></div><div class='download-item'><span class='label'>Report</span><span class='tag'>PDF</span></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; align-items:center; justify-content:space-between;'><h3 class='section-title'>Match Diagnostics</h3><div class='status-badge'>● System Ready</div></div>", unsafe_allow_html=True)
    diag_left, diag_right = st.columns([1.6, 1])
    with diag_left:
        st.markdown("<div class='progress-box' style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Runtime: 3.8s</div>")
        st.markdown("<div class='muted'>Matcher used: <b>" + result["matcher"] + "</b></div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Scene inference: illumination gap=" + f"{result['scene'].get('illumination_gap', 0.0):.1f}" + ", texture=" + f"{result['scene'].get('texture', 0.0):.1f}" + ", coverage=" + f"{result['coverage']*100:.0f}%" + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with diag_right:
        st.markdown("<div class='progress-box' style='margin-top:0.8rem;'><div class='muted'>Key takeaways</div><ul style='color:#dfeeff; margin:0.8rem 0 0 1rem;'><li>Stable overlap distribution</li><li>Geometric consistency maintained</li><li>Sub-pixel refinement available</li></ul></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

