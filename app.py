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
    vid_file = st.file_uploader("Upload a master video", type=["mp4", "avi", "mov", "mkv", "webm"])
    if vid_file:
        tmp_dir = OUTPUTS_DIR / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_vid = tmp_dir / vid_file.name
        tmp_vid.write_bytes(vid_file.getvalue())

        st.video(str(tmp_vid))

        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            sample_fps = st.slider("Analysis sample rate (FPS)", 2.0, 10.0, 4.5, 0.5)
        with col2:
            smoothing = st.slider("Crop smoothing window", 3, 15, 5, 2)

        if st.button("Generate 9:16 Reel", key="gen_vid"):
            from src.video_pipeline import analyse_video, render_reel, extract_still

            cache = OUTPUTS_DIR / f"{tmp_vid.stem}_analysis.json"

            # Analysis phase
            progress_placeholder = st.empty()
            with st.spinner("Analyzing video (face detection + tracking)…"):
                progress_placeholder.info("🔍 Detecting faces on sampled frames...")
                analysis = analyse_video(tmp_vid, sample_fps=sample_fps, cache_path=cache)

            # Show analysis summary
            meta = analysis["metadata"]
            st.success(
                f"Analyzed {meta['total_frames']} frames "
                f"({meta['duration_sec']:.1f}s @ {meta['fps']:.1f} FPS) · "
                f"sampled at ~{meta['sample_fps']:.1f} FPS"
            )

            # Detection summary
            det_summary = analysis["detections_summary"]
            total_dets = sum(d["n_detections"] for d in det_summary)
            frames_with_faces = sum(1 for d in det_summary if d["n_detections"] > 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples analyzed", len(det_summary))
            with col2:
                st.metric("Faces detected", total_dets)
            with col3:
                st.metric("Frames with faces", f"{frames_with_faces}/{len(det_summary)}")

            primary_id = analysis.get("primary_track_id")
            if primary_id is not None:
                st.info(f"📍 Primary subject: Track #{primary_id}")
            else:
                st.warning("⚠ No faces tracked, using center crop fallback")

            # Render phase
            with st.spinner("Rendering 9:16 reel at full resolution…"):
                progress_placeholder.info("🎬 Rendering reel...")
                reel_path = render_reel(tmp_vid, analysis)

            # Extract still
            with st.spinner("Extracting representative still frame…"):
                still_path = extract_still(tmp_vid, analysis)

            # Validation
            from src.validation import validate_video
            val = validate_video(reel_path)
            status = "✅" if val["ok"] else "❌"

            st.markdown(f"**Reel validation:** {status}")
            for c in val["checks"]:
                icon = "✓" if c["passed"] else "✗"
                st.text(f"  {icon} {c['name']}  {c.get('detail', '')}")

            # Show outputs
            st.subheader("Generated Reel")
            st.video(str(reel_path))

            st.subheader("Representative Still")
            st.image(str(still_path), use_container_width=True)

            # Show crop path visualization
            with st.expander("Crop path analysis"):
                crop_path = analysis["crop_path"]
                st.write(f"{len(crop_path)} keyframes in crop path")

                # Simple time-series data
                times = [p["time"] for p in crop_path[:50]]  # First 50 for display
                focus_x = [p["focus_x"] for p in crop_path[:50]]
                focus_y = [p["focus_y"] for p in crop_path[:50]]

                st.line_chart({"focus_x": focus_x, "focus_y": focus_y})

                with st.expander("Full analysis JSON"):
                    st.json(analysis)

st.divider()
st.caption("Built for Hoichoi Hackathon 2026 — P4 Creative Reformatting Engine")
