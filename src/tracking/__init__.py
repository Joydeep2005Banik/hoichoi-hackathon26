"""Cross-frame subject tracking for video pipeline.

Simple tracking via IoU (Intersection over Union) matching between consecutive
detections. Handles temporary detection failures by propagating last known positions.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
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


class Track:
    """A tracked subject across frames."""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int], frame: int):
        self.id = track_id
        self.boxes = [box]
        self.frames = [frame]
        self.last_seen = frame
        self.missed_frames = 0

    def update(self, box: Tuple[int, int, int, int], frame: int):
        """Update track with new detection."""
        self.boxes.append(box)
        self.frames.append(frame)
        self.last_seen = frame
        self.missed_frames = 0

    def mark_missed(self, frame: int):
        """Mark this track as not detected in current frame."""
        self.missed_frames += 1
        # Extrapolate position
        if len(self.boxes) >= 2:
            # Simple velocity continuation
            dx = self.boxes[-1][0] - self.boxes[-2][0]
            dy = self.boxes[-1][1] - self.boxes[-2][1]
            last_box = self.boxes[-1]
            predicted = (
                last_box[0] + dx, last_box[1] + dy,
                last_box[2] + dx, last_box[3] + dy
            )
            self.boxes.append(predicted)
        else:
            # No velocity, repeat last box
            self.boxes.append(self.boxes[-1])
        self.frames.append(frame)

    @property
    def current_box(self) -> Tuple[int, int, int, int]:
        return self.boxes[-1]

    @property
    def center(self) -> Tuple[float, float]:
        box = self.current_box
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class SimpleTracker:
    """Simple IoU-based multi-object tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.next_id = 0

    def update(self, detections: List[Tuple[int, int, int, int]], frame: int) -> List[Track]:
        """Update tracks with new detections at given frame."""
        # Match detections to existing tracks
        if not self.tracks:
            # First frame or all tracks expired
            for det in detections:
                self.tracks.append(Track(self.next_id, det, frame))
                self.next_id += 1
            return self.tracks

        # Compute IoU matrix
        matched_tracks = set()
        matched_dets = set()

        for det_idx, det in enumerate(detections):
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
                self.tracks[best_track_idx].update(det, frame)
                matched_tracks.add(best_track_idx)
                matched_dets.add(det_idx)

        # Mark unmatched tracks as missed
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_missed(frame)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks.append(Track(self.next_id, det, frame))
                self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.missed_frames < self.max_age]

        return self.tracks

    def get_primary_track(self) -> Optional[Track]:
        """Return the most stable/longest track (primary subject)."""
        if not self.tracks:
            return None
        # Prefer tracks with most detections and least missed frames
        return max(self.tracks, key=lambda t: len(t.boxes) - t.missed_frames * 2)
