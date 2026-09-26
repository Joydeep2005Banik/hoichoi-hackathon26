"""Video pipeline — produce a 9:16 speaker-tracking reel."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
import cv2
import numpy as np

from src.detection import detect_faces, faces_center
from src.utils import PROXY_MAX_DIM, REEL_WIDTH, REEL_HEIGHT, REEL_FPS, OUTPUTS_DIR
from src.utils.geometry import compute_crop_box


def _probe_video(path: str) -> Dict:
    """Use ffprobe to get video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def analyse_video(
    video_path: str | Path,
    sample_interval: int = 10,
    cache_path: Optional[Path] = None,
) -> List[Dict]:
    """Sample frames, detect faces, return per-sample metadata.

    Results are cached to JSON for re-use.
    """
    video_path = Path(video_path)
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Proxy scale
    scale = min(1.0, PROXY_MAX_DIM / max(w, h))

    samples = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            proxy = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            ph, pw = proxy.shape[:2]
            boxes = detect_faces(proxy)
            fx, fy = faces_center(boxes, pw, ph)
            samples.append({
                "frame": frame_idx,
                "time": frame_idx / fps,
                "focus_x": fx,
                "focus_y": fy,
                "n_faces": len(boxes),
            })
        frame_idx += 1

    cap.release()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(samples, indent=2))

    return samples


def _smooth_focus(samples: List[Dict], alpha: float = 0.3) -> List[Dict]:
    """Exponential moving average on focus_x/y to avoid jumpy crops."""
    if not samples:
        return samples
    sx, sy = samples[0]["focus_x"], samples[0]["focus_y"]
    for s in samples:
        sx = alpha * s["focus_x"] + (1 - alpha) * sx
        sy = alpha * s["focus_y"] + (1 - alpha) * sy
        s["smooth_x"] = sx
        s["smooth_y"] = sy
    return samples


def render_reel(
    video_path: str | Path,
    samples: List[Dict],
    output_path: Optional[Path] = None,
    sample_interval: int = 10,
) -> Path:
    """Read source video and write a 9:16 reel with dynamic crop."""
    video_path = Path(video_path)
    output_path = output_path or OUTPUTS_DIR / f"{video_path.stem}_reel.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _smooth_focus(samples)

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or REEL_FPS

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (REEL_WIDTH, REEL_HEIGHT))

    # Build a lookup: frame_idx -> smoothed focus
    focus_map = {s["frame"]: (s.get("smooth_x", s["focus_x"]),
                               s.get("smooth_y", s["focus_y"])) for s in samples}

    frame_idx = 0
    last_fx, last_fy = 0.5, 0.5
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in focus_map:
            last_fx, last_fy = focus_map[frame_idx]

        x1, y1, x2, y2 = compute_crop_box(w, h, 9, 16, last_fx, last_fy)
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (REEL_WIDTH, REEL_HEIGHT), interpolation=cv2.INTER_AREA)
        out.write(resized)
        frame_idx += 1

    cap.release()
    out.release()

    # Re-encode with ffmpeg for broad compatibility
    tmp = output_path.with_suffix(".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(output_path), "-c:v", "libx264",
         "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
         str(tmp)],
        capture_output=True,
    )
    if tmp.exists():
        tmp.replace(output_path)

    return output_path
