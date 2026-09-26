"""Hoichoi P4 — Creative Reformatting Engine (Streamlit UI)."""
import streamlit as st
from pathlib import Path
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import ASPECT_RATIOS, OUTPUTS_DIR
from src.validation import validate_image, validate_video

st.set_page_config(page_title="Hoichoi Reformatter", layout="wide")
st.title("🎬 Creative Reformatting Engine")
st.caption("Subject-aware image & video reformatting — Hoichoi Hackathon P4")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])

# ── Image tab ──────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader("Upload a master image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        # Save temp
        tmp_dir = OUTPUTS_DIR / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.read())

        st.image(str(tmp_path), caption="Original", use_container_width=True)

        if st.button("Generate Crops", key="gen_img"):
            from src.image_pipeline import process_image

            with st.spinner("Detecting subjects & cropping…"):
                results = process_image(tmp_path)

            st.success(f"Generated {len(results)} variants")

            for label, path in results.items():
                rw, rh = ASPECT_RATIOS[label]
                val = validate_image(path, expected_ratio=(rw, rh))

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(str(path), caption=f"{label}", use_container_width=True)
                with col2:
                    status = "✅" if val["ok"] else "❌"
                    st.markdown(f"**{label}** {status}")
                    for c in val["checks"]:
                        icon = "✓" if c["passed"] else "✗"
                        st.text(f"  {icon} {c['name']}")

# ── Video tab ──────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader("Upload a master video", type=["mp4", "avi", "mov", "mkv"])
    if vid_file:
        tmp_dir = OUTPUTS_DIR / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_vid = tmp_dir / vid_file.name
        tmp_vid.write_bytes(vid_file.read())

        st.video(str(tmp_vid))

        sample_interval = st.slider("Analysis sample interval (frames)", 5, 30, 10)

        if st.button("Generate 9:16 Reel", key="gen_vid"):
            from src.video_pipeline import analyse_video, render_reel

            cache = OUTPUTS_DIR / f"{tmp_vid.stem}_analysis.json"

            with st.spinner("Analysing video (face detection on proxy frames)…"):
                samples = analyse_video(tmp_vid, sample_interval=sample_interval, cache_path=cache)

            st.info(f"Analysed {len(samples)} sample points")

            # Show detection summary
            faces_detected = sum(1 for s in samples if s["n_faces"] > 0)
            st.metric("Frames with faces", f"{faces_detected}/{len(samples)}")

            with st.spinner("Rendering reel…"):
                reel_path = render_reel(tmp_vid, samples, sample_interval=sample_interval)

            val = validate_video(reel_path)
            status = "✅" if val["ok"] else "❌"
            st.markdown(f"**Reel validation:** {status}")
            for c in val["checks"]:
                icon = "✓" if c["passed"] else "✗"
                st.text(f"  {icon} {c['name']}  {c.get('detail', '')}")

            st.video(str(reel_path))

st.divider()
st.caption("Built for Hoichoi Hackathon 2026 — P4 Creative Reformatting Engine")
