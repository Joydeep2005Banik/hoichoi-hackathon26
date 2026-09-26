"""Video pipeline — MVP with tracking, smoothing, and caching.

Flow:
  1. Analyze at low resolution (~4-5 FPS sampling)
  2. Detect faces on sampled frames
  3. Track subjects across frames (handle detection gaps)
  4. Select primary subject (longest/most stable track)
  5. Compute smoothed crop path
  6. Render 9:16 reel at full resolution
  7. Extract representative still frame
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
import cv2
import numpy as np

from src.detection import detect_faces
from src.tracking import SimpleTracker, Track
from src.utils import PROXY_MAX_DIM, REEL_WIDTH, REEL_HEIGHT, REEL_FPS, OUTPUTS_DIR
from src.utils.geometry import compute_crop_box


def analyse_video(
    video_path: str | Path,
    sample_fps: float = 4.5,
    cache_path: Optional[Path] = None,
) -> Dict:
    """Analyze video: detect + track subjects, compute crop path.

    Returns analysis dict with tracks, crop path, and metadata.
    Cached to JSON for fast reruns.
    """
    video_path = Path(video_path)

    # Check cache (validate schema to prevent stale legacy cache issues)
    if cache_path and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text())
            if isinstance(cached_data, dict) and "metadata" in cached_data and "crop_path" in cached_data:
                return cached_data
            else:
                print(f"Warning: Cache at {cache_path} is using an outdated schema. Re-analyzing...")
        except Exception as e:
            print(f"Warning: Failed to load cache at {cache_path} ({e}). Re-analyzing...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Proxy resolution
    scale = min(1.0, PROXY_MAX_DIM / max(src_w, src_h))
    proxy_w = int(src_w * scale)
    proxy_h = int(src_h * scale)

    # Sample interval (aim for ~4-5 FPS analysis)
    sample_interval = max(1, int(fps / sample_fps))

    print(f"Analyzing {video_path.name}: {src_w}×{src_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Proxy: {proxy_w}×{proxy_h}, sampling every {sample_interval} frames (~{fps/sample_interval:.1f} FPS)")

    # Track subjects
    tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
    detections_log = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Detect on proxy
            proxy = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
            boxes_proxy = detect_faces(proxy)

            # Scale boxes back to source resolution
            boxes_src = [
                (int(b[0] / scale), int(b[1] / scale),
                 int(b[2] / scale), int(b[3] / scale))
                for b in boxes_proxy
            ]

            # Update tracker
            tracks = tracker.update(boxes_src, frame_idx)

            detections_log.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "n_detections": len(boxes_src),
                "n_tracks": len(tracks),
            })

        frame_idx += 1

    cap.release()

    # Select primary subject
    primary_track = tracker.get_primary_track()

    if primary_track is None:
        # Fallback: center crop
        print("No tracks found, using center crop")
        crop_path = [{
            "frame": 0,
            "time": 0.0,
            "focus_x": 0.5,
            "focus_y": 0.5,
            "track_id": None,
        }]
    else:
        print(f"Primary track: {primary_track.id} ({len(primary_track.boxes)} detections)")

        # Build crop path from primary track
        raw_path = []
        for frame, box in zip(primary_track.frames, primary_track.boxes):
            cx = (box[0] + box[2]) / 2 / src_w
            cy = (box[1] + box[3]) / 2 / src_h
            raw_path.append({
                "frame": frame,
                "time": frame / fps,
                "focus_x": cx,
                "focus_y": cy,
                "track_id": primary_track.id,
            })

        # Temporal smoothing (moving average)
        crop_path = _smooth_crop_path(raw_path, window=5)

    # Analysis result
    analysis = {
        "source": str(video_path),
        "metadata": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": total_frames / fps,
            "resolution": {"w": src_w, "h": src_h},
            "sample_interval": sample_interval,
            "sample_fps": fps / sample_interval,
        },
        "detections_summary": detections_log,
        "primary_track_id": primary_track.id if primary_track else None,
        "crop_path": crop_path,
    }

    # Cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(analysis, indent=2))
        print(f"Cached analysis to {cache_path.name}")

    return analysis


def _smooth_crop_path(path: List[Dict], window: int = 5) -> List[Dict]:
    """Apply moving average smoothing to crop path."""
    if len(path) <= 1:
        return path

    smoothed = []
    for i, point in enumerate(path):
        # Window around current point
        start = max(0, i - window // 2)
        end = min(len(path), i + window // 2 + 1)

        fx_avg = sum(p["focus_x"] for p in path[start:end]) / (end - start)
        fy_avg = sum(p["focus_y"] for p in path[start:end]) / (end - start)

        smoothed.append({
            "frame": point["frame"],
            "time": point["time"],
            "focus_x": fx_avg,
            "focus_y": fy_avg,
            "track_id": point["track_id"],
        })

    return smoothed


def render_reel(
    video_path: str | Path,
    analysis: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Render 9:16 vertical reel from analysis crop path.

    Reads source video at full resolution and applies crop path.
    """
    video_path = Path(video_path)
    output_path = output_path or OUTPUTS_DIR / f"{video_path.stem}_reel.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_path = analysis["crop_path"]
    src_fps = analysis["metadata"]["fps"]
    src_w = analysis["metadata"]["resolution"]["w"]
    src_h = analysis["metadata"]["resolution"]["h"]

    # Build frame lookup
    focus_map = {p["frame"]: (p["focus_x"], p["focus_y"]) for p in crop_path}

    # Open source video
    cap = cv2.VideoCapture(str(video_path))

    # Temporary raw output (before FFmpeg re-encode)
    tmp_raw = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_raw), fourcc, src_fps, (REEL_WIDTH, REEL_HEIGHT))

    frame_idx = 0
    last_fx, last_fy = 0.5, 0.5

    print(f"Rendering reel: {REEL_WIDTH}×{REEL_HEIGHT} @ {src_fps:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get focus for this frame
        if frame_idx in focus_map:
            last_fx, last_fy = focus_map[frame_idx]

        # Compute 9:16 crop box
        x1, y1, x2, y2 = compute_crop_box(src_w, src_h, 9, 16, last_fx, last_fy)
        crop = frame[y1:y2, x1:x2]

        # Resize to target reel dimensions
        resized = cv2.resize(crop, (REEL_WIDTH, REEL_HEIGHT), interpolation=cv2.INTER_AREA)
        out.write(resized)

        frame_idx += 1

        if frame_idx % 250 == 0:
            print(f"  Processed {frame_idx}/{analysis['metadata']['total_frames']} frames")

    cap.release()
    out.release()

    print(f"Re-encoding with FFmpeg...")
    # Re-encode with libx264 for compatibility
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp_raw),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            str(output_path)
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError("FFmpeg encoding failed")

    # Cleanup
    tmp_raw.unlink(missing_ok=True)

    print(f"✓ Reel saved: {output_path}")
    return output_path


def extract_still(
    video_path: str | Path,
    analysis: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Extract a representative still frame from the middle of the video."""
    video_path = Path(video_path)
    output_path = output_path or OUTPUTS_DIR / f"{video_path.stem}_still.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose middle frame
    total_frames = analysis["metadata"]["total_frames"]
    mid_frame = total_frames // 2

    # Find crop path entry nearest to middle
    crop_path = analysis["crop_path"]
    mid_crop = min(crop_path, key=lambda p: abs(p["frame"] - mid_frame))

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_crop["frame"])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {mid_crop['frame']}")

    # Apply crop
    src_w = analysis["metadata"]["resolution"]["w"]
    src_h = analysis["metadata"]["resolution"]["h"]
    fx = mid_crop["focus_x"]
    fy = mid_crop["focus_y"]

    x1, y1, x2, y2 = compute_crop_box(src_w, src_h, 9, 16, fx, fy)
    crop = frame[y1:y2, x1:x2]

    # Save as JPEG
    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"✓ Still frame saved: {output_path}")
    return output_path
