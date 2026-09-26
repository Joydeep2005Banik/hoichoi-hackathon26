"""Video pipeline — active-speaker estimation, speaker tracking, and reel rendering.

Flow:
  1. Extract audio & compute speech activity (VAD / RMS energy)
  2. Analyze at low-resolution proxy (~4-5 FPS sampling)
  3. Detect faces & track persistent subjects across frames
  4. Track mouth-region motion for each face
  5. Estimate active speaker per sample segment with temporal hysteresis
  6. Compute smoothed crop path targeting the active speaker (or stable group/conversational crop)
  7. Save sidecars: speaker_segments.json and crop_path.json
  8. Render 9:16 reel at full resolution (with optional debug overlay mode)
  9. Extract representative still frame
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import subprocess
import cv2
import numpy as np

from src.detection import detect_faces
from src.tracking import SimpleTracker, SpeakerEstimator, Track
from src.audio import extract_audio, compute_rms_energy, get_speech_score_at_time
from src.utils import PROXY_MAX_DIM, REEL_WIDTH, REEL_HEIGHT, REEL_FPS, OUTPUTS_DIR
from src.utils.geometry import compute_crop_box


def analyse_video(
    video_path: str | Path,
    sample_fps: float = 4.5,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze video with active-speaker estimation & tracking.

    Produces canonical analysis dictionary, speaker segments, and crop path.
    """
    video_path = Path(video_path)
    output_dir = OUTPUTS_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache — require Phase 4 schema (speaker_segments + active_speaker_track_id in crop_path)
    if cache_path and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text())
            cp0 = (cached_data.get("crop_path") or [{}])[0]
            is_p4 = (
                isinstance(cached_data, dict)
                and "metadata" in cached_data
                and "crop_path" in cached_data
                and "speaker_segments" in cached_data
                and "active_speaker_track_id" in cp0
            )
            if is_p4:
                return cached_data
            else:
                print(f"Cache at {cache_path} is pre-Phase-4 schema. Re-analyzing...")
        except Exception as e:
            print(f"Warning: Failed to load cache at {cache_path} ({e}). Re-analyzing...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1. Extract audio & VAD
    wav_path = output_dir / f"{video_path.stem}.wav"
    extracted_wav = extract_audio(video_path, wav_path)
    audio_energy = compute_rms_energy(extracted_wav, window_ms=100) if extracted_wav else []

    # 2. Proxy setup
    scale = min(1.0, PROXY_MAX_DIM / max(src_w, src_h))
    proxy_w = int(src_w * scale)
    proxy_h = int(src_h * scale)
    sample_interval = max(1, int(fps / sample_fps))

    print(f"Analyzing {video_path.name}: {src_w}×{src_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Proxy: {proxy_w}×{proxy_h}, sampling every {sample_interval} frames (~{fps/sample_interval:.1f} FPS)")

    tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
    speaker_estimator = SpeakerEstimator(switch_threshold=0.15, hold_frames=8)

    detections_log: List[Dict] = []
    speaker_timeline: List[Dict] = []
    raw_crop_path: List[Dict] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            t = frame_idx / fps
            proxy = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
            proxy_gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)

            # Detect faces on proxy
            boxes_proxy = detect_faces(proxy)

            # Scale boxes back to source resolution
            boxes_src = [
                (int(b[0] / scale), int(b[1] / scale),
                 int(b[2] / scale), int(b[3] / scale))
                for b in boxes_proxy
            ]

            # Update tracker with mouth extraction on proxy
            # Mouth extraction needs proxy-scale boxes (frame_gray is proxy resolution)
            boxes_proxy_for_mouth = [
                (int(b[0] * scale), int(b[1] * scale),
                 int(b[2] * scale), int(b[3] * scale))
                for b in boxes_src
            ] if boxes_src else []
            active_tracks = tracker.update(boxes_src, frame_idx,
                                           frame_gray=proxy_gray,
                                           mouth_boxes=boxes_proxy_for_mouth)

            # Audio speech score at timestamp t
            speech_score = get_speech_score_at_time(audio_energy, t)

            # Estimate active speaker
            active_track_id, speaker_conf, all_scores = speaker_estimator.step(
                active_tracks=active_tracks,
                audio_speech_score=speech_score,
                frame_idx=frame_idx,
                img_w=src_w,
                img_h=src_h,
            )

            # Determine focus target for this sample
            if active_track_id is not None:
                track = tracker.get_track_by_id(active_track_id)
                if track is not None:
                    target_box = track.current_box
                    target_cx = (target_box[0] + target_box[2]) / 2 / src_w
                    target_cy = (target_box[1] + target_box[3]) / 2 / src_h
                else:
                    target_cx, target_cy = 0.5, 0.5
            elif active_tracks:
                # Ambiguous: use group center of valid active faces
                valid_boxes = [t.current_box for t in active_tracks if t.center[1] <= 0.78 * src_h]
                if valid_boxes:
                    min_x = min(b[0] for b in valid_boxes)
                    max_x = max(b[2] for b in valid_boxes)
                    min_y = min(b[1] for b in valid_boxes)
                    max_y = max(b[3] for b in valid_boxes)
                    target_cx = (min_x + max_x) / 2 / src_w
                    target_cy = (min_y + max_y) / 2 / src_h
                else:
                    target_cx, target_cy = 0.5, 0.5
            else:
                target_cx, target_cy = 0.5, 0.5

            raw_crop_path.append({
                "frame": frame_idx,
                "time": round(t, 3),
                "focus_x": float(target_cx),
                "focus_y": float(target_cy),
                "active_speaker_track_id": active_track_id,
                "speaker_confidence": round(speaker_conf, 2),
                "n_faces": len(boxes_src),
            })

            detections_log.append({
                "frame": frame_idx,
                "time": round(t, 3),
                "n_detections": len(boxes_src),
                "n_tracks": len(active_tracks),
                "active_speaker_id": active_track_id,
                "speaker_confidence": round(speaker_conf, 2),
                "scores": all_scores,
                "speech_score": round(speech_score, 2),
            })

            speaker_timeline.append({
                "frame": frame_idx,
                "time": round(t, 3),
                "active_speaker_track_id": active_track_id,
                "confidence": round(speaker_conf, 2),
                "is_speech": speech_score > 0.4,
            })

        frame_idx += 1

    cap.release()

    # Temporal smoothing on crop path
    smoothed_crop_path = _smooth_crop_path(raw_crop_path, window=5)

    # Convert timeline into consolidated speaker segments
    speaker_segments = _consolidate_speaker_segments(speaker_timeline)

    # Primary track (most frequent active speaker or longest track)
    speaker_counts: Dict[int, int] = {}
    for pt in smoothed_crop_path:
        st_id = pt.get("active_speaker_track_id")
        if st_id is not None:
            speaker_counts[st_id] = speaker_counts.get(st_id, 0) + 1

    primary_speaker_id = max(speaker_counts, key=speaker_counts.get) if speaker_counts else None

    # Sidecars
    segments_path = output_dir / f"{video_path.stem}_speaker_segments.json"
    segments_path.write_text(json.dumps(speaker_segments, indent=2))

    crop_path_file = output_dir / f"{video_path.stem}_crop_path.json"
    crop_path_file.write_text(json.dumps(smoothed_crop_path, indent=2))

    analysis = {
        "source": str(video_path),
        "metadata": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2),
            "resolution": {"w": src_w, "h": src_h},
            "sample_interval": sample_interval,
            "sample_fps": round(fps / sample_interval, 2),
            "audio_detected": bool(audio_energy),
        },
        "detections_summary": detections_log,
        "speaker_segments": speaker_segments,
        "speaker_segments_path": str(segments_path),
        "crop_path_path": str(crop_path_file),
        "primary_track_id": primary_speaker_id,
        "crop_path": smoothed_crop_path,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(analysis, indent=2))
        print(f"Cached analysis to {cache_path.name}")

    return analysis


def _consolidate_speaker_segments(timeline: List[Dict]) -> List[Dict]:
    """Group frame-level speaker samples into contiguous time segments."""
    if not timeline:
        return []

    segments: List[Dict] = []
    cur_speaker = timeline[0]["active_speaker_track_id"]
    cur_start = timeline[0]["time"]
    cur_start_frame = timeline[0]["frame"]
    confidences = [timeline[0]["confidence"]]

    for sample in timeline[1:]:
        spk = sample["active_speaker_track_id"]
        if spk == cur_speaker:
            confidences.append(sample["confidence"])
        else:
            segments.append({
                "speaker_track_id": cur_speaker,
                "start_time": round(cur_start, 2),
                "end_time": round(sample["time"], 2),
                "start_frame": cur_start_frame,
                "end_frame": sample["frame"],
                "avg_confidence": round(sum(confidences) / len(confidences), 2),
            })
            cur_speaker = spk
            cur_start = sample["time"]
            cur_start_frame = sample["frame"]
            confidences = [sample["confidence"]]

    # Final segment
    segments.append({
        "speaker_track_id": cur_speaker,
        "start_time": round(cur_start, 2),
        "end_time": round(timeline[-1]["time"], 2),
        "start_frame": cur_start_frame,
        "end_frame": timeline[-1]["frame"],
        "avg_confidence": round(sum(confidences) / len(confidences), 2),
    })

    return segments


def _smooth_crop_path(path: List[Dict], window: int = 5) -> List[Dict]:
    """Smooth crop path: prevents jitter while enabling clean, responsive speaker transitions."""
    if len(path) <= 1:
        return path

    smoothed = []
    cur_fx = path[0]["focus_x"]
    cur_fy = path[0]["focus_y"]

    for pt in path:
        target_fx = pt["focus_x"]
        target_fy = pt["focus_y"]

        dist = abs(target_fx - cur_fx)
        # Adaptive responsive transition: fast on speaker shifts, smooth on small drift
        if dist > 0.15:
            alpha = 0.55  # Fast 2-3 frame transition on speaker changes
        elif dist > 0.05:
            alpha = 0.40  # Moderate tracking movement
        else:
            alpha = 0.25  # Smooth small jitter within a shot

        cur_fx = alpha * target_fx + (1.0 - alpha) * cur_fx
        cur_fy = alpha * target_fy + (1.0 - alpha) * cur_fy

        item = dict(pt)
        item["focus_x"] = round(cur_fx, 4)
        item["focus_y"] = round(cur_fy, 4)
        smoothed.append(item)

    return smoothed


def render_reel(
    video_path: str | Path,
    analysis: Dict,
    output_path: Optional[Path] = None,
    debug_overlay: bool = False,
) -> Path:
    """Render 9:16 vertical reel from active-speaker crop path.

    If debug_overlay=True, overlays active speaker indicators, track boxes, and crop window.
    """
    video_path = Path(video_path)
    output_path = output_path or OUTPUTS_DIR / f"{video_path.stem}_reel.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_path = analysis["crop_path"]
    src_fps = analysis["metadata"]["fps"]
    src_w = analysis["metadata"]["resolution"]["w"]
    src_h = analysis["metadata"]["resolution"]["h"]

    # Build frame-indexed lookups
    focus_map = {p["frame"]: (p["focus_x"], p["focus_y"], p.get("active_speaker_track_id"), p.get("speaker_confidence", 0.0)) for p in crop_path}

    cap = cv2.VideoCapture(str(video_path))
    tmp_raw = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    target_size = (src_w, src_h) if debug_overlay else (REEL_WIDTH, REEL_HEIGHT)
    out = cv2.VideoWriter(str(tmp_raw), fourcc, src_fps, target_size)

    frame_idx = 0
    last_fx, last_fy = 0.5, 0.5
    last_spk, last_conf = None, 0.0

    print(f"Rendering {'debug reel' if debug_overlay else '9:16 reel'}: {target_size[0]}×{target_size[1]} @ {src_fps:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in focus_map:
            last_fx, last_fy, last_spk, last_conf = focus_map[frame_idx]

        x1, y1, x2, y2 = compute_crop_box(src_w, src_h, 9, 16, last_fx, last_fy)

        if debug_overlay:
            # Draw 9:16 crop window overlay on full frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw status banner
            speaker_text = f"Speaker: #{last_spk} (conf: {last_conf:.2f})" if last_spk is not None else "Speaker: Group/Ambiguous"
            cv2.rectangle(frame, (10, 10), (550, 70), (0, 0, 0), -1)
            cv2.putText(frame, speaker_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx} | Crop: [{x1}:{x2}]", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out.write(frame)
        else:
            # Crop and resize to 9:16 reel
            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, (REEL_WIDTH, REEL_HEIGHT), interpolation=cv2.INTER_AREA)
            out.write(resized)

        frame_idx += 1

    cap.release()
    out.release()

    # Re-encode with FFmpeg (mux audio if available)
    wav_path = OUTPUTS_DIR / video_path.stem / f"{video_path.stem}.wav"
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(tmp_raw)]
    if wav_path.exists():
        ffmpeg_cmd.extend(["-i", str(wav_path), "-c:a", "aac", "-b:a", "128k"])
    ffmpeg_cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        str(output_path)
    ])

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: FFmpeg mux returned error ({result.stderr}). Re-trying video-only encode...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_raw), "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(output_path)],
            capture_output=True,
        )

    tmp_raw.unlink(missing_ok=True)
    print(f"✓ Reel saved: {output_path}")
    return output_path


def extract_still(
    video_path: str | Path,
    analysis: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Extract a representative still frame from the peak speech segment."""
    video_path = Path(video_path)
    output_path = output_path or OUTPUTS_DIR / f"{video_path.stem}_still.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_path = analysis["crop_path"]

    # Pick frame with highest speaker confidence, or middle frame
    best_frame_entry = max(crop_path, key=lambda p: p.get("speaker_confidence", 0.0))
    target_frame = best_frame_entry["frame"]
    fx = best_frame_entry["focus_x"]
    fy = best_frame_entry["focus_y"]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        # Fallback to middle frame
        cap = cv2.VideoCapture(str(video_path))
        mid = analysis["metadata"]["total_frames"] // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        fx, fy = 0.5, 0.5

    src_w = analysis["metadata"]["resolution"]["w"]
    src_h = analysis["metadata"]["resolution"]["h"]
    x1, y1, x2, y2 = compute_crop_box(src_w, src_h, 9, 16, fx, fy)
    crop = frame[y1:y2, x1:x2]

    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"✓ Still frame saved: {output_path}")
    return output_path
