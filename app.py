"""Hoichoi Hackathon 2026 (P4) — Creative Reformatting Engine.

Final Streamlit UI (Phase 5):
- Single-page workflow supporting both master images and videos
- Automatic media type detection & session-safe disk streaming
- Subject-aware 16:9, 1:1, 4:5, 9:16 multi-crop generation with P4 QA validation
- Active-speaker 9:16 vertical reel + representative still generation
- Detailed visual analysis: speaker timeline, dialogue segments, crop path curves, and debug overlay
- Direct one-click download buttons for all media outputs and machine-readable JSON sidecars
"""
from __future__ import annotations
import streamlit as st
from pathlib import Path
import json
import uuid
import sys
import os

# Ensure repo root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import ASPECT_RATIOS, OUTPUTS_DIR

# ── Session & Storage Setup ─────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]

SESSION_ID = st.session_state["session_id"]
SESSION_DIR = OUTPUTS_DIR / f"session_{SESSION_ID}"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Creative Reformatting Engine — Hoichoi P4",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom header banner
st.title("🎬 Creative Reformatting Engine")
st.markdown(
    "**Hoichoi Hackathon 2026 · Problem 4 (P4)** — Subject-aware multi-aspect reformatting "
    "for master images and dynamic active-speaker tracking 9:16 reels for video."
)
st.divider()

# ── File Upload & Auto-Detection ────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

uploaded_file = st.file_uploader(
    "Upload Master Asset (Image or Video)",
    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"],
    help="Upload an image (JPG, PNG, WebP) or video (MP4, MOV, AVI, MKV, WebM up to 400MB).",
)

if not uploaded_file:
    st.info("👆 Upload an image or video above to begin creative reformatting.")

    # Showcase demo cards for judges
    st.markdown("### 🌟 Key Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🖼 Smart Image Reformatting")
        st.caption("Auto-detects faces/subjects via lightweight CPU cascade (YuNet → Haar → Saliency) and computes mathematically exact crops for **16:9**, **1:1**, **4:5**, and **9:16**.")
    with c2:
        st.markdown("#### 🎥 Speaker-Aware 9:16 Reels")
        st.caption("Analyzes dialogue at proxy resolution (~4.5 FPS), tracks faces, estimates active speaker via mouth motion & audio VAD, and applies smooth, causal camera tracking.")
    with c3:
        st.markdown("#### 🛡 Machine QA Validation")
        st.caption("Every generated asset passes automated P4 validation checks: dimension integrity, aspect ratio tolerance, subject visibility, safe margins, and metadata consistency.")

else:
    file_ext = Path(uploaded_file.name).suffix.lower()
    is_image = file_ext in IMAGE_EXTENSIONS
    is_video = file_ext in VIDEO_EXTENSIONS

    # Stream file to disk in chunks to avoid RAM bloat
    safe_filename = Path(uploaded_file.name).name
    temp_input_path = SESSION_DIR / f"input_{safe_filename}"

    if not temp_input_path.exists() or temp_input_path.stat().st_size != uploaded_file.size:
        with open(temp_input_path, "wb") as f:
            uploaded_file.seek(0)
            while chunk := uploaded_file.read(65536):
                f.write(chunk)

    # ═══════════════════════════════════════════════════════════
    # IMAGE WORKFLOW
    # ═══════════════════════════════════════════════════════════
    if is_image:
        st.subheader("📷 Master Image Detected")

        col_preview, col_action = st.columns([2, 1])
        with col_preview:
            st.image(str(temp_input_path), caption=f"Original: {safe_filename}", use_container_width=True)
        with col_action:
            st.markdown("#### Reformatting Target")
            st.markdown("- **16:9** · Landscape / Banner\n- **1:1** · Square / Feed\n- **4:5** · Portrait / Social Post\n- **9:16** · Story / Vertical")
            st.caption("Crops prioritize detected faces & subject saliency, preserving face visibility without center-locking.")
            generate_img_btn = st.button("⚡ Generate Multi-Crop Assets", type="primary", use_container_width=True)

        if generate_img_btn or f"img_results_{safe_filename}" in st.session_state:
            from src.image_pipeline import process_image

            if f"img_results_{safe_filename}" not in st.session_state:
                with st.spinner("Analyzing image, detecting subjects, and computing subject-aware crops…"):
                    try:
                        img_output_dir = SESSION_DIR / f"crops_{Path(safe_filename).stem}"
                        results = process_image(temp_input_path, output_dir=img_output_dir)
                        st.session_state[f"img_results_{safe_filename}"] = {
                            k: {
                                "path": str(v["path"]),
                                "meta_path": str(v["meta_path"]),
                                "crop_box": v["crop_box"],
                                "metadata": v["metadata"],
                                "validation": v["validation"],
                                "ready": v["ready"],
                            }
                            for k, v in results.items()
                        }
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        results = {}
            else:
                results = st.session_state[f"img_results_{safe_filename}"]

            if results:
                ready_count = sum(1 for r in results.values() if r["ready"])
                first_meta = next(iter(results.values()))["metadata"]
                subjects = first_meta["detected_subjects"]
                n_faces = sum(1 for s in subjects if s.get("category") == "face")
                det_source = subjects[0]["source"] if subjects else "none"

                st.success(f"✓ Generated {len(results)} subject-aware variants · {ready_count}/{len(results)} Passed QA Validation")
                st.info(
                    f"**Detection Cascade**: Identified **{len(subjects)}** subject(s) "
                    f"({n_faces} face(s)) via `{det_source}` · Focus point: "
                    f"`({first_meta['focus']['x']:.2f}, {first_meta['focus']['y']:.2f})`"
                )

                st.markdown("### 🖼 Generated Formats")

                # Render 4 aspect ratios in a clean 4-column responsive grid
                cols = st.columns(4)
                ratio_labels = ["16:9", "1:1", "4:5", "9:16"]

                for idx, label in enumerate(ratio_labels):
                    if label not in results:
                        continue
                    data = results[label]
                    val = data["validation"]
                    meta = data["metadata"]
                    crop_box = meta["crop_box"]
                    ready = data["ready"]
                    crop_path = Path(data["path"])

                    with cols[idx]:
                        st.markdown(f"#### **{label}**")
                        if crop_path.exists():
                            st.image(str(crop_path), use_container_width=True)

                            # Validation badge
                            if val["status"] == "PASS":
                                st.success("✅ QA: PASS")
                            else:
                                st.error("❌ QA: FAIL")

                            st.caption(f"Dimensions: **{meta['crop_size']['w']}×{meta['crop_size']['h']}**")

                            # Download Crop Image without loading whole file into memory
                            with open(crop_path, "rb") as img_f:
                                st.download_button(
                                    label=f"⬇ Download {label}",
                                    data=img_f,
                                    file_name=crop_path.name,
                                    mime="image/jpeg",
                                    key=f"dl_img_{label}_{safe_filename}",
                                    use_container_width=True,
                                )

                            # Validation details expander
                            with st.expander("Validation Details"):
                                for c in val["checks"]:
                                    icon = "✓" if c["passed"] else "✗"
                                    detail = f" — {c['detail']}" if c.get("detail") else ""
                                    st.text(f"{icon} {c['name']}{detail}")
                                if val.get("warnings"):
                                    for w in val["warnings"]:
                                        st.warning(f"{w['message']}: {w.get('detail', '')}")

                            # Metadata JSON download
                            meta_path = Path(data["meta_path"])
                            if meta_path.exists():
                                st.download_button(
                                    label="📄 JSON Metadata",
                                    data=meta_path.read_bytes(),
                                    file_name=meta_path.name,
                                    mime="application/json",
                                    key=f"dl_meta_{label}_{safe_filename}",
                                    use_container_width=True,
                                )

    # ═══════════════════════════════════════════════════════════
    # VIDEO WORKFLOW
    # ═══════════════════════════════════════════════════════════
    elif is_video:
        st.subheader("🎥 Master Video Detected")

        cache_key = f"vid_results_{safe_filename}_{False}"  # default check

        col_vid_prev, col_vid_config = st.columns([2, 1])
        with col_vid_prev:
            # Memory safeguard: after reel is generated, collapse input video to avoid dual 1080p video player buffers in browser
            any_results_ready = any(k.startswith(f"vid_results_{safe_filename}") for k in st.session_state)
            if any_results_ready:
                with st.expander("▶ Original Master Video (Click to expand)", expanded=False):
                    st.video(str(temp_input_path))
                st.info(f"Master Video: `{safe_filename}` ({temp_input_path.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                st.video(str(temp_input_path))
        with col_vid_config:
            st.markdown("#### Video Reformatting Settings")
            sample_fps = st.slider(
                "Analysis Sample Rate (FPS)",
                min_value=2.0,
                max_value=10.0,
                value=4.5,
                step=0.5,
                help="Analyzes dialogue at proxy resolution (≤480p) at this sampling rate (default 4.5 FPS).",
            )
            debug_overlay_mode = st.checkbox(
                "🔍 Generate Debug Overlay Reel",
                value=False,
                help="Overlays face bounding boxes, active speaker IDs, confidence scores, and the 9:16 crop window.",
            )
            generate_vid_btn = st.button("⚡ Generate 9:16 Active-Speaker Reel", type="primary", use_container_width=True)

        target_cache_key = f"vid_results_{safe_filename}_{debug_overlay_mode}"

        if generate_vid_btn or target_cache_key in st.session_state:
            from src.video_pipeline import analyse_video, render_reel, extract_still
            from src.validation import validate_video

            if target_cache_key not in st.session_state:
                vid_cache_file = SESSION_DIR / f"{Path(safe_filename).stem}_analysis.json"

                # Multi-step progress narration
                progress_box = st.empty()
                progress_box.info("Step 1/3: Extracting audio VAD and tracking faces on proxy frames…")

                try:
                    with st.spinner("Running lightweight active-speaker detection & tracking…"):
                        analysis = analyse_video(temp_input_path, sample_fps=sample_fps, cache_path=vid_cache_file)

                    progress_box.info("Step 2/3: Rendering 9:16 full-resolution reel with adaptive smoothing…")
                    with st.spinner("Rendering vertical video via FFmpeg…"):
                        out_reel_name = f"{Path(safe_filename).stem}_reel_debug.mp4" if debug_overlay_mode else f"{Path(safe_filename).stem}_reel.mp4"
                        out_reel_path = SESSION_DIR / out_reel_name
                        reel_path = render_reel(temp_input_path, analysis, output_path=out_reel_path, debug_overlay=debug_overlay_mode)

                    progress_box.info("Step 3/3: Extracting representative still and running QA validation…")
                    out_still_path = SESSION_DIR / f"{Path(safe_filename).stem}_still.jpg"
                    still_path = extract_still(temp_input_path, analysis, output_path=out_still_path)

                    val_result = validate_video(reel_path)

                    progress_box.empty()

                    st.session_state[target_cache_key] = {
                        "analysis": analysis,
                        "reel_path": str(reel_path),
                        "still_path": str(still_path),
                        "validation": val_result,
                    }
                except Exception as e:
                    progress_box.empty()
                    st.error(f"Error during video processing: {e}")
                    analysis = None

            if target_cache_key in st.session_state:
                res_data = st.session_state[target_cache_key]
                analysis = res_data["analysis"]
                reel_path = Path(res_data["reel_path"])
                still_path = Path(res_data["still_path"])
                val = res_data["validation"]

                meta = analysis["metadata"]
                audio_status = "🔊 Audio Speech VAD Enabled" if meta.get("audio_detected") else "🔇 Video Motion Analysis"

                st.success(
                    f"✓ Processed **{meta['total_frames']}** frames ({meta['duration_sec']:.1f}s) · "
                    f"Sampled at **~{meta['sample_fps']:.1f} FPS** · {audio_status}"
                )

                # Output presentation: 2-column layout for Reel + Still
                st.markdown("### 🎬 Generated Video Deliverables")
                col_reel, col_still = st.columns([3, 2])

                with col_reel:
                    st.markdown("#### **9:16 Vertical Reel**" + (" *(Debug Overlay)*" if debug_overlay_mode else ""))
                    if reel_path.exists():
                        st.video(str(reel_path))

                        # Validation badge
                        if val.get("ok"):
                            st.success("✅ Video Validation: PASS (1080×1920 stream valid)")
                        else:
                            st.error("❌ Video Validation: FAIL")

                        with open(reel_path, "rb") as rf:
                            st.download_button(
                                label="⬇ Download 9:16 Reel (.mp4)",
                                data=rf,
                                file_name=reel_path.name,
                                mime="video/mp4",
                                key=f"dl_reel_{safe_filename}",
                                use_container_width=True,
                            )

                with col_still:
                    st.markdown("#### **Representative Keyframe Still**")
                    if still_path.exists():
                        st.image(str(still_path), use_container_width=True)
                        st.caption("Extracted from peak active-speaker keyframe.")

                        with open(still_path, "rb") as sf:
                            st.download_button(
                                label="⬇ Download Still (.jpg)",
                                data=sf,
                                file_name=still_path.name,
                                mime="image/jpeg",
                                key=f"dl_still_{safe_filename}",
                                use_container_width=True,
                            )

                # ── "Show Analysis" Section ──────────────────────────────────
                st.divider()
                st.markdown("### 🔍 Video Dialogue & Active-Speaker Analysis")

                tab_timeline, tab_curve, tab_sidecars = st.tabs([
                    "🗣 Speaker Segments & Dialogue Turns",
                    "📈 Crop Path & Tracking Curve",
                    "📦 Machine-Readable Sidecars",
                ])

                with tab_timeline:
                    speaker_segments = analysis.get("speaker_segments", [])
                    primary_speaker = analysis.get("primary_track_id")

                    st.markdown(f"**Primary Detected Subject/Speaker**: `Track #{primary_speaker}`")
                    st.markdown(f"**Total Dialogue Segments Identified**: `{len(speaker_segments)}`")

                    if speaker_segments:
                        # Render structured table of speaker segments
                        seg_table = [
                            {
                                "Segment #": i + 1,
                                "Speaker": f"Track #{s['speaker_track_id']}" if s.get("speaker_track_id") is not None else "Ambiguous / Group",
                                "Start Time (s)": f"{s['start_time']:.1f}s",
                                "End Time (s)": f"{s['end_time']:.1f}s",
                                "Duration (s)": f"{s['end_time'] - s['start_time']:.1f}s",
                                "Confidence": f"{s['avg_confidence'] * 100:.0f}%",
                            }
                            for i, s in enumerate(speaker_segments)
                        ]
                        st.dataframe(seg_table, use_container_width=True)

                with tab_curve:
                    crop_path = analysis.get("crop_path", [])
                    st.caption(f"Smoothed horizontal (focus_x) and vertical (focus_y) camera tracking across {len(crop_path)} keyframes:")

                    if crop_path:
                        chart_data = {
                            "Horizontal Center (focus_x)": [p["focus_x"] for p in crop_path],
                            "Vertical Center (focus_y)": [p["focus_y"] for p in crop_path],
                        }
                        st.line_chart(chart_data)

                with tab_sidecars:
                    st.markdown("#### Generated JSON Sidecars")
                    st.caption("Machine-readable outputs generated alongside media assets:")

                    col_sc1, col_sc2 = st.columns(2)

                    # Speaker Segments Sidecar
                    seg_path_str = analysis.get("speaker_segments_path")
                    if seg_path_str and Path(seg_path_str).exists():
                        with col_sc1:
                            st.download_button(
                                label="⬇ Download speaker_segments.json",
                                data=Path(seg_path_str).read_bytes(),
                                file_name="speaker_segments.json",
                                mime="application/json",
                                key=f"dl_seg_json_{safe_filename}",
                                use_container_width=True,
                            )
                            with st.expander("Preview speaker_segments.json"):
                                st.json(json.loads(Path(seg_path_str).read_text())[:10])

                    # Crop Path Sidecar
                    crop_path_str = analysis.get("crop_path_path")
                    if crop_path_str and Path(crop_path_str).exists():
                        with col_sc2:
                            st.download_button(
                                label="⬇ Download crop_path.json",
                                data=Path(crop_path_str).read_bytes(),
                                file_name="crop_path.json",
                                mime="application/json",
                                key=f"dl_crop_json_{safe_filename}",
                                use_container_width=True,
                            )
                            with st.expander("Preview crop_path.json"):
                                st.json(json.loads(Path(crop_path_str).read_text())[:10])

st.divider()
st.caption("Hoichoi Hackathon 2026 — P4 Creative Reformatting Engine · Built with Python, OpenCV, FFmpeg & Streamlit")
