"""Audio analysis — speaker-activity and speech energy detection.

Provides lightweight VAD (Voice Activity Detection) via RMS energy thresholds
and short-time energy variance without heavy ML dependencies.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import struct
import wave
import numpy as np


def extract_audio(video_path: str | Path, out_wav: Optional[Path] = None) -> Optional[Path]:
    """Extract mono 16 kHz WAV from video using ffmpeg.

    Returns None if video has no audio stream.
    """
    video_path = Path(video_path)
    out_wav = out_wav or video_path.with_suffix(".wav")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        return None
    return out_wav


def compute_rms_energy(
    wav_path: str | Path,
    window_ms: int = 100,
) -> List[Dict]:
    """Compute RMS energy per window. Returns list of {time, rms, is_speech}."""
    wav_path = Path(wav_path)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return []

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                return []
            raw = wf.readframes(n_frames)

        samples = struct.unpack(f"<{n_frames}h", raw)
        window_size = int(sr * window_ms / 1000)

        results = []
        all_rms = []
        for i in range(0, len(samples), window_size):
            chunk = samples[i : i + window_size]
            if not chunk:
                break
            rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
            all_rms.append(rms)
            results.append({"time": i / sr, "rms": rms})

        if not all_rms:
            return []

        # Adaptive speech threshold based on energy distribution
        median_rms = np.median(all_rms)
        std_rms = np.std(all_rms)
        speech_threshold = max(300.0, median_rms + 0.5 * std_rms)

        for r in results:
            r["is_speech"] = bool(r["rms"] >= speech_threshold)
            r["speech_score"] = float(min(1.0, r["rms"] / (speech_threshold * 2.0 + 1e-6)))

        return results
    except Exception as e:
        print(f"Warning: Failed to compute RMS energy: {e}")
        return []


def get_speech_score_at_time(audio_energy: List[Dict], t: float) -> float:
    """Lookup speech probability/score at a specific timestamp in seconds."""
    if not audio_energy:
        return 0.5  # Neutral if no audio
    # Binary/linear search
    idx = int(t / 0.1)  # 100ms windows
    if 0 <= idx < len(audio_energy):
        return audio_energy[idx].get("speech_score", 0.5)
    return 0.0
