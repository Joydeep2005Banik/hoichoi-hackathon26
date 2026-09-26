# Creative Reformatting Engine
**Hoichoi Hackathon 2026 · Problem 4**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://hoichoi-hackathon26-problem-4.streamlit.app/)

A media pipeline that transforms master images and videos into platform-ready multi-aspect-ratio assets with subject-aware smart cropping and active-speaker-aware vertical reels.

---

## Problem Statement

**Problem 4 — Creative Reformatting Engine**  
*Track: Media Pipeline / Computer Vision*

A media pipeline that takes one master image and/or video for a title and produces every platform-ready ratio (16:9, 1:1, 9:16, 4:5) with subject-aware smart crop, a still pulled cleanly from video, and a subject-tracked, active-speaker-aware vertical reel — every derived asset validated against a machine-readable platform spec before it enters the library.

---

## Solution Overview

This implementation delivers subject-aware reformatting for master images and videos:

### Image Processing
- **Input**: One master image (JPG, PNG, WebP)
- **Output**: Subject-aware crops for **16:9**, **1:1**, **4:5**, and **9:16** aspect ratios
- **Intelligence**: Detects faces and subjects, computes weighted focus points, and generates crops that preserve subject visibility rather than applying fixed center crops

### Video Processing
- **Input**: One master video (MP4, MOV, AVI, MKV, WebM up to 400 MB)
- **Output**: Dynamic **9:16 vertical reel** with active-speaker tracking + representative still frame
- **Intelligence**: Lightweight face detection, multi-person tracking, audio-based speech activity detection, mouth motion analysis, and temporal crop path smoothing produce a reel that follows the active speaker across dialogue turns

### Validation
Every generated asset passes machine-readable validation checks (resolution, aspect ratio tolerance, subject visibility, safe margins, metadata consistency) before surfacing in the UI.

---

## Key Features

- **Subject-Aware Image Cropping**: Detects faces and subjects via lightweight CPU cascade (YuNet → Haar → saliency) and computes mathematically exact crops anchored on detected subjects
- **Multiple Aspect Ratios**: Generates 16:9 (landscape), 1:1 (square), 4:5 (portrait), and 9:16 (vertical) variants from a single input
- **Multi-Person Video Tracking**: IoU-based tracking maintains subject identity across detection gaps and occlusions
- **Active-Speaker Estimation**: Combines audio VAD (voice activity detection via RMS energy), mouth-region motion tracking, and temporal hysteresis to estimate the active speaker per sample segment
- **Dynamic Crop Path**: Smoothed focus trajectories follow the primary speaker or group center, with adaptive alpha blending for responsive transitions on speaker changes and smooth jitter suppression within a shot
- **Temporal Smoothing**: Prevents camera jitter while enabling clean speaker transitions via exponential moving average with distance-adaptive alpha
- **Machine-Readable Validation**: Structured PASS/FAIL checks with detailed metrics for dimension integrity, aspect ratio tolerance, subject visibility, safe margins, and metadata consistency
- **Analysis & Debug View**: Speaker timeline visualization, dialogue segment breakdown, crop path curves, and optional debug overlay reel with bounding boxes and confidence scores
- **Downloadable Outputs**: One-click download for all generated media assets and JSON sidecars (speaker segments, crop path, validation reports)

---

## System Architecture

### Image Pipeline
```
Master Image
    ↓
Proxy Analysis (≤480p)
    ↓
Subject Detection (YuNet → Haar → Saliency)
    ↓
Weighted Focus Computation
    ↓
Multi-Ratio Crop Generation (16:9, 1:1, 4:5, 9:16)
    ↓
Validation & QA Checks
    ↓
Gallery + Download
```

### Video Pipeline
```
Master Video
    ↓
Audio Extraction & VAD (RMS Energy)
    ↓
Proxy Analysis (~4.5 FPS sampling at ≤480p)
    ↓
Face Detection (YuNet)
    ↓
IoU-based Tracking + Mouth Motion Extraction
    ↓
Active Speaker Estimation (audio + visual cues + temporal hysteresis)
    ↓
Crop Path Computation (target active speaker or group center)
    ↓
Temporal Smoothing (adaptive alpha EMA)
    ↓
Full-Resolution Rendering (FFmpeg H.264 + AAC)
    ↓
Validation & QA Checks
    ↓
9:16 Reel + Still + Analysis View + Download
```

---

## Image Processing Pipeline

1. **Proxy Detection**: Resizes input to ≤480p proxy resolution for lightweight face/subject detection
2. **Detection Cascade**:
   - **YuNet Face Detection** (primary): CNN-based face detector optimized for CPU inference
   - **Haar Cascade** (fallback): Classical face detection when YuNet produces no detections
   - **Saliency Maps** (fallback): Spectral residual saliency when no faces are detected
3. **Subject Scoring**: Assigns importance scores based on detection confidence, face category priority, and spatial position
4. **Weighted Focus**: Computes normalized focus point `(focus_x, focus_y)` as the weighted centroid of detected subjects
5. **Subject-Aware Crop**: For each target aspect ratio, computes the largest inscribed crop centered on the focus point, then nudges the crop window to maximize subject visibility
6. **Validation**: Runs dimension checks, aspect ratio tolerance, subject visibility ratio, safe margin compliance, and metadata consistency before marking asset as ready

---

## Video Processing Pipeline

1. **Audio Extraction**: Uses FFmpeg to extract mono WAV audio stream for voice activity detection
2. **Voice Activity Detection (VAD)**: Computes RMS energy over 100ms windows; speech segments are identified by RMS threshold
3. **Proxy Sampling**: Analyzes video at reduced resolution (≤480p) and reduced temporal rate (~4-5 FPS sampling) to minimize computational load
4. **Face Detection**: Applies YuNet face detection on each sampled proxy frame
5. **IoU-Based Tracking**: Maintains persistent track IDs across frames using Intersection-over-Union matching with configurable IOU threshold (0.3) and maximum age (30 frames)
6. **Mouth Motion Tracking**: Extracts mouth region from each tracked face and computes inter-frame pixel differences to estimate mouth activity
7. **Active Speaker Estimation**:
   - Combines **audio speech score** (from VAD) and **visual mouth motion** into a composite speaker score per track
   - Applies **temporal hysteresis** (hold frames = 8) to prevent flickering on ambiguous or rapid speaker switches
   - Outputs active speaker track ID and confidence per sample
8. **Crop Path Generation**:
   - If a clear active speaker is identified, crop focuses on that speaker's face center
   - If multiple speakers are detected without a clear winner, crop focuses on the group bounding box center (excluding faces in the lower 22% of the frame to avoid subtitle occlusion)
   - If no faces are detected, crop holds at frame center (0.5, 0.5)
9. **Temporal Smoothing**:
   - Applies exponential moving average (EMA) to focus coordinates
   - **Adaptive alpha**: fast transitions (α=0.55) on large horizontal shifts (>15%), moderate (α=0.40) on medium shifts (5-15%), smooth (α=0.25) on small jitter (<5%)
   - Prevents crop jitter within a shot while enabling clean, responsive speaker transitions
10. **Full-Resolution Rendering**:
    - Reads source video at full resolution (1920×1080 typical)
    - Applies smoothed crop path frame-by-frame to extract 9:16 vertical windows
    - Resizes to 1080×1920 target resolution (INTER_AREA interpolation)
    - Re-encodes with **FFmpeg**: H.264 (libx264, preset=fast, CRF=23, yuv420p, faststart) + AAC audio (128k, 44.1kHz)
11. **Representative Still Extraction**: Picks the frame with highest speaker confidence (or middle frame if no speaker) and extracts 9:16 crop as JPEG
12. **Validation**: Validates 9:16 reel for correct resolution (1080×1920), aspect ratio tolerance, frame integrity, and playback compatibility
13. **Sidecar Generation**: Produces machine-readable JSON files:
    - `speaker_segments.json`: Consolidated dialogue segments with speaker track IDs, time ranges, and average confidence
    - `crop_path.json`: Per-sample focus coordinates, active speaker IDs, confidence scores, and face counts

---

## Active Speaker Estimation

This implementation uses a **lightweight heuristic approach** to estimate the active speaker, **not a trained audio-visual speaker identification model**.

### How It Works
- **Audio Signal**: Extracts RMS energy from the audio stream; speech segments are identified when RMS exceeds a threshold (typically 0.4)
- **Visual Signal**: Tracks mouth-region motion for each detected face by computing inter-frame pixel differences in the mouth bounding box
- **Composite Score**: Combines audio speech activity with per-face visual mouth motion to produce a speaker score for each tracked subject
- **Temporal Hysteresis**: Holds the current speaker for a minimum number of frames (hold_frames=8) before switching to a new speaker, preventing flickering during ambiguous or rapid turn-taking

### Limitations
- Does not perform speaker diarization or voice identification
- Relies on visual face detection and mouth motion rather than learned audio-visual embeddings
- May struggle with off-screen speakers, rapid dialogue overlap, or crowded multi-person scenes where faces are small or partially occluded
- Confidence drops during ambiguous segments (e.g., simultaneous speech, laughter, or group reaction shots)

---

## Validation

Every generated asset undergoes structured machine-readable validation before entering the output library:

### Image Asset Validation
- **Dimension Check**: Verifies exact pixel dimensions match target resolution
- **Aspect Ratio Tolerance**: Ensures crop aspect ratio is within ±0.5% of target ratio
- **Subject Visibility**: Confirms detected faces have ≥60% overlap with crop window
- **Safe Margins**: Checks that subject bounding boxes maintain ≥5% margin from crop edges (warning-level, does not fail asset)
- **Metadata Consistency**: Validates crop box coordinates, focus point, and detected subject records are structurally sound

### Video Asset Validation
- **Resolution Check**: Confirms 1080×1920 (9:16) dimensions
- **Frame Count**: Verifies frame count matches expected duration × FPS
- **Stream Integrity**: Uses FFmpeg to decode entire video and detect any corruption or decode errors
- **Aspect Ratio Tolerance**: Ensures 9:16 ratio is within ±1% tolerance

### Validation Output
Each asset produces a structured JSON validation report:
```json
{
  "status": "PASS" | "FAIL",
  "checks": [
    {"name": "dimension_check", "passed": true, "detail": "1080×1920"},
    {"name": "aspect_ratio", "passed": true, "detail": "0.5625 (within tolerance)"}
  ],
  "warnings": [
    {"message": "subject_near_edge", "detail": "Face at edge margin"}
  ]
}
```

---

## Tech Stack

- **Python 3.14**: Core runtime
- **Streamlit**: Interactive web UI and deployment
- **OpenCV (opencv-python-headless)**: Image/video I/O, frame manipulation, classical CV algorithms
- **NumPy**: Numerical operations, array processing
- **MediaPipe**: YuNet face detection model (CPU-optimized)
- **Pillow**: Image format handling
- **FFmpeg**: Video transcoding, audio extraction, H.264/AAC encoding
- **pytest**: Test suite (76 unit/integration tests)

---

## Repository Structure

```
hoichoi-hackathon26/
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (FFmpeg)
├── pytest.ini                  # Pytest configuration
├── PROJECT_SPEC.md             # Detailed technical specification
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Streamlit server config (maxUploadSize=400)
├── src/
│   ├── image_pipeline/         # Image multi-crop generation
│   ├── video_pipeline/         # Video analysis & reel rendering
│   ├── detection/              # Subject/face detection (YuNet, Haar, saliency)
│   ├── tracking/               # IoU-based tracking, speaker estimation
│   ├── audio/                  # Audio extraction, VAD, RMS energy
│   ├── validation/             # Asset QA checks
│   ├── media/                  # I/O helpers, FFmpeg wrappers
│   └── utils/                  # Geometry, config, constants
├── tests/                      # Pytest suite (unit + integration tests)
├── assets/                     # Sample inputs (images, small videos)
├── models/                     # YuNet face detection model weights (gitignored)
└── outputs/                    # Generated crops/reels (gitignored)
```

---

## Run Locally

### Prerequisites
- Python 3.9+ (tested on Python 3.14)
- FFmpeg (for video transcoding)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Joydeep2005Banik/hoichoi-hackathon26.git
   cd hoichoi-hackathon26
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies**:
   
   FFmpeg is required for video processing. Install via your package manager:
   
   - **Fedora/RHEL**:
     ```bash
     sudo dnf install ffmpeg
     ```
   
   - **Ubuntu/Debian**:
     ```bash
     sudo apt install ffmpeg
     ```
   
   - **macOS** (Homebrew):
     ```bash
     brew install ffmpeg
     ```
   
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**:
   - Streamlit will print a local URL (typically `http://localhost:8501`)
   - Open the URL in your browser

### Usage

**For Images**:
1. Upload a master image (JPG, PNG, WebP)
2. Click **"⚡ Generate Multi-Crop Assets"**
3. View generated 16:9, 1:1, 4:5, and 9:16 variants in the gallery
4. Inspect validation results (PASS/FAIL badges and detailed checks)
5. Download individual crops or JSON metadata

**For Videos**:
1. Upload a master video (MP4, MOV, AVI, MKV, WebM up to 400 MB)
2. Optionally adjust analysis sample rate (default 4.5 FPS) or enable debug overlay mode
3. Click **"⚡ Generate 9:16 Active-Speaker Reel"**
4. Wait for analysis (proxy detection, tracking, speaker estimation) and rendering
5. View the 9:16 vertical reel and representative still frame
6. Inspect analysis:
   - **Speaker Segments & Dialogue Turns**: Timeline table showing which speaker was active during each segment
   - **Crop Path & Tracking Curve**: Line chart of horizontal/vertical focus coordinates over time
   - **Machine-Readable Sidecars**: Download `speaker_segments.json` and `crop_path.json`
7. Download the 9:16 reel MP4, still JPEG, and analysis JSON files

---

## Deployment / Live Demo

**Live Application**: [https://hoichoi-hackathon26-problem-4.streamlit.app/](https://hoichoi-hackathon26-problem-4.streamlit.app/)

The application is deployed on **Streamlit Community Cloud** and can be opened directly in any modern browser. No local installation is required to use the live demo.

To deploy your own instance:
1. Fork this repository
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Create a new app pointing to your fork
4. Streamlit will automatically detect `requirements.txt`, `packages.txt`, and `.streamlit/config.toml`

---

## Limitations

- **CPU-Intensive Processing**: Video analysis and rendering run on CPU without GPU acceleration, resulting in processing times proportional to video duration and resolution
- **Heuristic Speaker Estimation**: Uses lightweight audio-visual cues (RMS energy + mouth motion) rather than learned speaker embeddings; may struggle with off-screen speakers, simultaneous speech, or highly overlapping dialogue
- **Ambiguity in Crowded Scenes**: Multi-person scenes with rapid speaker switching or small/occluded faces may produce lower-confidence speaker estimates, causing the crop to fall back to group center or hold the last valid speaker
- **Processing Time**: Analysis and rendering time scales with video length and resolution (e.g., ~2-5 minutes for a 3-minute 1080p video on typical cloud CPU)
- **Maximum Upload Size**: Configured for videos up to 400 MB; larger files require local processing or deployment with increased upload limits

---

## Future Improvements

The following enhancements are planned for future iterations:

- **GPU Acceleration**: Offload face detection and mouth motion computation to GPU for faster processing
- **Learned Speaker Models**: Replace heuristic scoring with audio-visual speaker embeddings (e.g., TalkNet, ASD) for robust multi-speaker diarization
- **Scene Change Detection**: Add keyframe-based scene boundary detection to reset speaker tracking across cuts
- **Multi-Language Support**: Extend VAD to handle non-English audio and multi-language dialogue
- **Batch Processing**: Support bulk upload and processing of multiple videos in parallel
- **Advanced Crop Strategies**: Add rule-based crop modes (e.g., "always follow leftmost speaker", "group shot only") selectable in the UI

---


