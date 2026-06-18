"""
tracking.py - ROS-free obstacle tracking skeleton for F1TENTH.

This module is intentionally ROS-free. It should take detector outputs
(DetectedObstacle) and produce stable tracks over time.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from perception.detect import DetectedObstacle


@dataclass
class TrackedObstacle:
    """Tracked obstacle with persistent ID and estimated state."""
    track_id: int = -1
    cx: float = 0.0
    cy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    width: float = 0.1
    height: float = 0.1
    hits: int = 1
    coasted: int = 0
    latest_detection: Optional[DetectedObstacle] = None


class _Track:
    """Internal track state skeleton (e.g., Kalman filter state)."""

    _next_id: int = 0

    def __init__(self, det: DetectedObstacle) -> None:
        self.track_id = _Track._next_id
        _Track._next_id += 1
        self.cx = det.cx
        self.cy = det.cy
        self.vx = 0.0
        self.vy = 0.0
        self.hits = 1
        self.coasted = 0
        self.latest_det = det

    def predict(self, dt: float) -> None:
        """TODO: Predict next state with motion model."""
        _ = dt

    def update(self, det: DetectedObstacle) -> None:
        """TODO: Update track with associated detection."""
        self.cx = det.cx
        self.cy = det.cy
        self.latest_det = det
        self.hits += 1
        self.coasted = 0

    def to_tracked_obstacle(self) -> TrackedObstacle:
        return TrackedObstacle(
            track_id=self.track_id,
            cx=self.cx,
            cy=self.cy,
            vx=self.vx,
            vy=self.vy,
            width=self.latest_det.width,
            height=self.latest_det.height,
            hits=self.hits,
            coasted=self.coasted,
            latest_detection=self.latest_det,
        )


class ObstacleTracker:
    """Multi-object tracking skeleton (NN/Hungarian + filter)."""

    def __init__(
        self,
        max_coasting: int = 5,
        association_threshold: float = 1.0,
        min_hits: int = 1,
    ) -> None:
        self.max_coasting = max_coasting
        self.assoc_th = association_threshold
        self.min_hits = min_hits
        self._tracks: Dict[int, _Track] = {}

    def update(
        self,
        detections: List[DetectedObstacle],
        dt: float = 0.05,
    ) -> List[TrackedObstacle]:
        """Update tracks with current detections.

        TODO:
        1. Predict all existing tracks using dt.
        2. Associate detections <-> tracks.
        3. Update matched tracks.
        4. Coast unmatched tracks.
        5. Delete stale tracks.
        6. Create new tracks for unmatched detections.
        7. Return confirmed tracks (hits >= min_hits).
        """
        _ = dt

        # Minimal placeholder behavior: create one track per detection each frame.
        self._tracks.clear()
        for det in detections:
            t = _Track(det)
            self._tracks[t.track_id] = t

        return [
            t.to_tracked_obstacle()
            for t in self._tracks.values()
            if t.hits >= self.min_hits
        ]

    def reset(self) -> None:
        """Clear all active tracks."""
        self._tracks.clear()

    def _associate(
        self,
        detections: List[DetectedObstacle],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """TODO: Implement data association (nearest-neighbour / Hungarian)."""
        _ = detections
        return [], [], list(self._tracks.keys())
