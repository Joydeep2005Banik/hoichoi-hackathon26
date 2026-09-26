"""Hoichoi P4 — Creative Reformatting Engine (Streamlit UI)."""
import streamlit as st
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import ASPECT_RATIOS, OUTPUTS_DIR

st.set_page_config(page_title="Hoichoi Reformatter", layout="wide")
st.title("🎬 Creative Reformatting Engine")
st.caption("Subject-aware image & video reformatting — Hoichoi Hackathon P4")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])

# ── Image tab ──────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader("Upload a master image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        tmp_dir = OUTPUTS_DIR / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())

        st.image(str(tmp_path), caption="Original", use_container_width=True)

        if st.button("Generate Crops", key="gen_img"):
            from src.image_pipeline import process_image

            with st.spinner("Detecting subjects & generating crops…"):
                results = process_image(tmp_path)

            # Count ready assets
            ready_count = sum(1 for r in results.values() if r["ready"])
            st.success(f"Generated {len(results)} variants · {ready_count} ready")

            # Detection summary
            first_meta = next(iter(results.values()))["metadata"]
            subjects = first_meta["detected_subjects"]
            n_faces = sum(1 for s in subjects if s["category"] == "face")
            det_source = subjects[0]["source"] if subjects else "none"
            st.info(
                f"Detected **{len(subjects)}** subjects "
                f"({n_faces} faces) via **{det_source}** · "
                f"focus ({first_meta['focus']['x']:.2f}, {first_meta['focus']['y']:.2f})"
            )

            for label, data in results.items():
                val = data["validation"]
                meta = data["metadata"]
                crop_box = meta["crop_box"]
                ready = data["ready"]

                # Status badge
                if val["status"] == "PASS":
                    status_badge = "✅ PASS"
                    status_color = "green"
                else:
                    status_badge = "❌ FAIL"
                    status_color = "red"

                st.subheader(f"{label} {'✓' if ready else '⚠'}")
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.image(str(data["path"]), use_container_width=True)

                with col2:
                    st.markdown(f"**Validation:** :{status_color}[{status_badge}]")

                    # Show all checks with severity
                    for c in val["checks"]:
                        icon = "✓" if c["passed"] else "✗"
                        severity = c.get("severity", "info")
                        sev_badge = {"critical": "🔴", "error": "🟠", "warning": "🟡"}.get(severity, "")
                        detail = f" — {c['detail']}" if c.get("detail") else ""
                        st.text(f"  {icon} {c['name']} {sev_badge}{detail}")

                    # Warnings
                    if val["warnings"]:
                        st.markdown("**Warnings:**")
                        for w in val["warnings"]:
                            st.warning(f"{w['message']}: {w.get('detail', '')}", icon="⚠️")

                    # Metrics
                    if val["metrics"]:
                        with st.expander("Metrics"):
                            st.json(val["metrics"])

                    # Crop box
                    st.markdown("**Crop box**")
                    st.text(
                        f"  ({crop_box['x1']}, {crop_box['y1']}) → "
                        f"({crop_box['x2']}, {crop_box['y2']})\n"
                        f"  {meta['crop_size']['w']}×{meta['crop_size']['h']}"
                    )

                    # Face visibility
                    face_vis = meta.get("face_visibility", [])
                    if face_vis:
                        vis_count = sum(face_vis)
                        st.markdown(f"**Faces visible:** {vis_count}/{len(face_vis)}")

                    with st.expander("Full metadata JSON"):
                        st.json(meta)

# ── Video tab ──────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader("Upload a master video", type=["mp4", "avi", "mov", "mkv"])
    if vid_file:
        tmp_dir = OUTPUTS_DIR / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_vid = tmp_dir / vid_file.name
        tmp_vid.write_bytes(vid_file.getvalue())

        st.video(str(tmp_vid))

        sample_interval = st.slider("Analysis sample interval (frames)", 5, 30, 10)

        if st.button("Generate 9:16 Reel", key="gen_vid"):
            from src.video_pipeline import analyse_video, render_reel

            cache = OUTPUTS_DIR / f"{tmp_vid.stem}_analysis.json"

            with st.spinner("Analysing video (face detection on proxy frames)…"):
                samples = analyse_video(
                    tmp_vid, sample_interval=sample_interval, cache_path=cache
                )

            st.info(f"Analysed {len(samples)} sample points")
            faces_detected = sum(1 for s in samples if s["n_faces"] > 0)
            st.metric("Frames with faces", f"{faces_detected}/{len(samples)}")

            with st.spinner("Rendering reel…"):
                reel_path = render_reel(tmp_vid, samples, sample_interval=sample_interval)

            from src.validation import validate_video

            val = validate_video(reel_path)
            status = "✅" if val["ok"] else "❌"
            st.markdown(f"**Reel validation:** {status}")
            for c in val["checks"]:
                icon = "✓" if c["passed"] else "✗"
                st.text(f"  {icon} {c['name']}  {c.get('detail', '')}")

            st.video(str(reel_path))

st.divider()
st.caption("Built for Hoichoi Hackathon 2026 — P4 Creative Reformatting Engine")
