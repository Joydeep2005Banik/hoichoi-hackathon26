"""Audio analysis — speaker-activity detection via RMS energy."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import json
import struct
import wave


def extract_audio(video_path: str | Path, out_wav: Optional[Path] = None) -> Path:
    """Extract mono 16 kHz WAV from video using ffmpeg."""
    video_path = Path(video_path)
    out_wav = out_wav or video_path.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)],
        capture_output=True,
    )
    return out_wav


def compute_rms_energy(
    wav_path: str | Path,
    window_ms: int = 200,
) -> List[Dict]:
    """Compute RMS energy per window. Returns list of {time, rms}."""
    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = struct.unpack(f"<{n_frames}h", raw)
    window_size = int(sr * window_ms / 1000)

    results = []
    for i in range(0, len(samples), window_size):
        chunk = samples[i : i + window_size]
        if not chunk:
            break
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        results.append({"time": i / sr, "rms": rms})

    return results
