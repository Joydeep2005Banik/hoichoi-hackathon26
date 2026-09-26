"""Cross-frame subject tracking & active-speaker estimation.

Tracks faces across sampled frames and estimates active speakers using:
- Mouth-region optical motion / temporal pixel variance
- Audio speech activity (VAD / RMS energy)
- Face scale and centrality
- Temporal hysteresis to avoid erratic speaker switches
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def extract_mouth_region(frame_gray: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Extract and normalize the mouth region (lower 35% of face box)."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    if h <= 10 or w <= 10:
        return None

    # Lower 35% of face, center 60% horizontally
    mouth_y1 = int(y1 + 0.65 * h)
    mouth_y2 = y2
    mouth_x1 = int(x1 + 0.20 * w)
    mouth_x2 = int(x2 - 0.20 * w)

    # Clip to frame bounds
    fh, fw = frame_gray.shape[:2]
    mouth_y1 = max(0, min(fh - 1, mouth_y1))
    mouth_y2 = max(0, min(fh, mouth_y2))
    mouth_x1 = max(0, min(fw - 1, mouth_x1))
    mouth_x2 = max(0, min(fw, mouth_x2))

    if mouth_y2 <= mouth_y1 or mouth_x2 <= mouth_x1:
        return None

    crop = frame_gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    # Resize to fixed standard patch for consistent difference metrics
    return cv2.resize(crop, (40, 24), interpolation=cv2.INTER_AREA)


class Track:
    """A tracked subject across frames with motion and speaker metrics."""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int], frame: int, mouth_patch: Optional[np.ndarray] = None):
        self.id = track_id
        self.boxes: List[Tuple[int, int, int, int]] = [box]
        self.frames: List[int] = [frame]
        self.last_seen: int = frame
        self.missed_frames: int = 0
        self.last_mouth_patch: Optional[np.ndarray] = mouth_patch
        self.mouth_motions: List[float] = [0.0]
        self.speaker_scores: List[float] = [0.0]

    def update(self, box: Tuple[int, int, int, int], frame: int, mouth_patch: Optional[np.ndarray] = None):
        """Update track with new detection and compute mouth motion."""
        motion = 0.0
        if mouth_patch is not None and self.last_mouth_patch is not None:
            # Normalized absolute frame difference in mouth region
            diff = cv2.absdiff(mouth_patch, self.last_mouth_patch)
            motion = float(np.mean(diff))
            self.last_mouth_patch = mouth_patch
        elif mouth_patch is not None:
            self.last_mouth_patch = mouth_patch

        self.mouth_motions.append(motion)
        self.boxes.append(box)
        self.frames.append(frame)
        self.last_seen = frame
        self.missed_frames = 0

    def mark_missed(self, frame: int):
        """Mark this track as not detected in current frame."""
        self.missed_frames += 1
        if len(self.boxes) >= 2:
            dx = self.boxes[-1][0] - self.boxes[-2][0]
            dy = self.boxes[-1][1] - self.boxes[-2][1]
            last_box = self.boxes[-1]
            predicted = (
                last_box[0] + dx, last_box[1] + dy,
                last_box[2] + dx, last_box[3] + dy
            )
            self.boxes.append(predicted)
        else:
            self.boxes.append(self.boxes[-1])
        self.frames.append(frame)
        self.mouth_motions.append(0.0)

    @property
    def current_box(self) -> Tuple[int, int, int, int]:
        return self.boxes[-1]

    @property
    def center(self) -> Tuple[float, float]:
        box = self.current_box
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @property
    def recent_mouth_motion(self) -> float:
        """Average mouth motion over the last 5 samples."""
        recent = self.mouth_motions[-5:]
        return sum(recent) / len(recent) if recent else 0.0


class SpeakerEstimator:
    """Estimates active speaker across tracked subjects with hysteresis."""

    def __init__(self, switch_threshold: float = 0.15, hold_frames: int = 8):
        self.active_track_id: Optional[int] = None
        self.active_confidence: float = 0.0
        self.frames_since_switch: int = 0
        self.switch_threshold = switch_threshold  # Hysteresis margin required to switch
        self.hold_frames = hold_frames  # Minimum frames to hold before switching unless strong evidence

    def score_track(self, track: Track, audio_speech_score: float, img_w: int, img_h: int) -> float:
        """Compute raw speaker score for a track."""
        box = track.current_box
        cx, cy = track.center

        # Penalize detections in lower 25% of frame (hands/knees/clothing artifacts)
        if cy > 0.75 * img_h:
            return 0.02

        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        area_ratio = area / (img_w * img_h)
        if area_ratio < 0.002:  # < 0.2% frame area
            return 0.05

        motion_val = min(1.0, track.recent_mouth_motion / 12.0)
        size_boost = min(0.3, area_ratio * 6.0)
        dist_from_center = abs(cx / img_w - 0.5)
        center_boost = max(0.0, 0.2 * (1.0 - 2.0 * dist_from_center))

        if audio_speech_score > 0.35:
            score = (0.50 * motion_val) + (0.25 * audio_speech_score) + size_boost + center_boost
        else:
            score = (0.20 * motion_val) + size_boost + center_boost

        return round(score, 3)

    def step(
        self,
        active_tracks: List[Track],
        audio_speech_score: float,
        frame_idx: int,
        img_w: int,
        img_h: int,
    ) -> Tuple[Optional[int], float, Dict[int, float]]:
        """Compute speaker score for each track and return (active_track_id, confidence, all_scores)."""
        if not active_tracks:
            return None, 0.0, {}

        # Filter to tracks currently observed in this frame with plausible vertical position
        valid_tracks = [t for t in active_tracks if t.missed_frames == 0 and t.center[1] <= 0.78 * img_h]
        if not valid_tracks:
            valid_tracks = [t for t in active_tracks if t.missed_frames <= 3 and t.center[1] <= 0.78 * img_h]
            if not valid_tracks:
                return None, 0.0, {}

        scores = {t.id: self.score_track(t, audio_speech_score, img_w, img_h) for t in valid_tracks}

        if len(valid_tracks) == 1:
            track = valid_tracks[0]
            conf = 0.90 if audio_speech_score > 0.3 else 0.70
            self.active_track_id = track.id
            self.active_confidence = conf
            return track.id, conf, scores

        best_track_id = max(scores, key=scores.get)
        best_score = scores[best_track_id]

        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

        self.frames_since_switch += 1

        if self.active_track_id is None or self.active_track_id not in [t.id for t in valid_tracks]:
            self.active_track_id = best_track_id
            self.active_confidence = round(min(1.0, best_score * 1.2), 2)
            self.frames_since_switch = 0
        else:
            current_score = scores.get(self.active_track_id, 0.0)
            should_switch = (
                (best_score > current_score + self.switch_threshold) or
                (self.frames_since_switch > self.hold_frames and best_score > current_score + 0.08)
            )
            if should_switch and best_track_id != self.active_track_id:
                self.active_track_id = best_track_id
                self.active_confidence = round(min(1.0, 0.5 + margin), 2)
                self.frames_since_switch = 0
            else:
                self.active_confidence = round(min(1.0, 0.6 + (0.3 if audio_speech_score > 0.4 else 0.0)), 2)

        return self.active_track_id, self.active_confidence, scores


class SimpleTracker:
    """IoU-based multi-object tracker with mouth patch extraction."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 0

    def update(
        self,
        detections: List[Tuple[int, int, int, int]],
        frame_idx: int,
        frame_gray: Optional[np.ndarray] = None,
        mouth_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Track]:
        """Update tracks with new detections and extract mouth patches."""
        # Extract mouth patch using mouth_boxes (proxy-scaled) if supplied, else detections
        box_source = mouth_boxes if mouth_boxes is not None else detections
        patches = []
        for box in box_source:
            if frame_gray is not None:
                patches.append(extract_mouth_region(frame_gray, box))
            else:
                patches.append(None)

        if not self.tracks:
            for det, patch in zip(detections, patches):
                self.tracks.append(Track(self.next_id, det, frame_idx, patch))
                self.next_id += 1
            return self.tracks

        matched_tracks = set()
        matched_dets = set()

        for det_idx, (det, patch) in enumerate(zip(detections, patches)):
            best_iou = self.iou_threshold
            best_track_idx = -1

            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                iou = compute_iou(det, track.current_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx

            if best_track_idx >= 0:
                self.tracks[best_track_idx].update(det, frame_idx, patch)
                matched_tracks.add(best_track_idx)
                matched_dets.add(det_idx)

        # Mark unmatched tracks as missed
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_missed(frame_idx)

        # Create new tracks for unmatched detections
        for det_idx, (det, patch) in enumerate(zip(detections, patches)):
            if det_idx not in matched_dets:
                self.tracks.append(Track(self.next_id, det, frame_idx, patch))
                self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.missed_frames < self.max_age]

        return self.tracks

    def get_primary_track(self) -> Optional[Track]:
        """Return the most stable track."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: len(t.boxes) - t.missed_frames * 2)

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None
