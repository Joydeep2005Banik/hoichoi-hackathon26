# Hoichoi Hackathon P4 — Creative Reformatting Engine

## Goal
Given a master image or video, produce subject-aware crops in multiple aspect ratios.
Video output includes a dynamic-crop 9:16 reel that follows the active speaker.

## Deliverables (MVP)
| Input | Output |
|---|---|
| 1 image | 16:9 · 1:1 · 4:5 · 9:16 variants (subject-aware crop) |
| 1 video (≤ 3:10, 1080p, 25 fps) | 9:16 speaker-tracking reel + best-frame still |

## Constraints
- **Runtime**: 8 GB RAM, CPU-only laptop; lightweight cloud deploy.
- **Stack**: Python · OpenCV · NumPy · FFmpeg · Streamlit.
- **Models**: CPU-friendly only (MediaPipe Face/Pose, Haar cascades). No PyTorch/YOLO unless proven necessary. No training.
- **Proxy workflow**: analyse at low resolution, cache detection JSON, apply crops at full resolution.
- **Validation**: every generated asset passes machine-readable checks (resolution, aspect-ratio tolerance, file integrity) before surfacing in the UI.
- **Git hygiene**: no raw video in repo; use `.gitignore`.

## Architecture
```
app.py                  ← Streamlit entry point
src/
  image_pipeline/       ← single-image multi-crop
  video_pipeline/       ← video → reel (dynamic crop)
  detection/            ← face / subject detection
  tracking/             ← cross-frame subject tracking
  audio/                ← speaker-activity detection (RMS energy)
  validation/           ← output QA checks
  media/                ← I/O helpers (read, write, ffmpeg wrappers)
  utils/                ← geometry, config, logging
assets/                 ← sample inputs (small only)
models/                 ← downloaded model weights (gitignored)
outputs/                ← generated crops/reels (gitignored)
tests/                  ← pytest suite
```

## Key Design Decisions
1. **Proxy analysis** — detect at ≤ 480p, map ROI back to source res.
2. **Detection cascade** — MediaPipe Face → Haar fallback → saliency center.
3. **Speaker tracking** — per-frame face positions + audio RMS energy → assign "active" speaker per segment; smooth crop window across frames.
4. **Multi-person rule** — never lock crop to frame-0 positions; re-evaluate every N frames.
5. **Validation contract** — `validate_asset(path) → {ok: bool, checks: [...]}` runs before any file enters the output library.
