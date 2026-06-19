"""
detect.py - ROS-free LiDAR obstacle detection skeleton for F1TENTH.

This module is intentionally ROS-free so it can be implemented and tested
without ROS 2 runtime dependencies.
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Point2D:
    x: float = 0.0
    y: float = 0.0


@dataclass
class DetectedObstacle:
    """Detected obstacle representation in the LiDAR(sensor) frame."""
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0
    num_points: int = 0
    points: List[Point2D] = field(default_factory=list)
    id: int = -1


class LidarDetector:
    """Jump-distance LiDAR detector skeleton."""

    def __init__(
        self,
        cluster_threshold: float = 0.3,
        min_points: int = 3,
        max_range: float = 10.0,
        min_range: float = 0.05,
        max_size: float = 1.0,
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.min_points = min_points
        self.max_range = max_range
        self.min_range = min_range
        # clusters larger than this are walls / track boundaries, not obstacles
        self.max_size = max_size

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[DetectedObstacle]:
        """Return detected obstacles for one scan.

        Jump-distance clustering: polar->Cartesian, split on range jumps,
        keep compact clusters (drops the long wall/boundary clusters).
        """
        points = self._scan_to_cartesian(ranges, angle_min, angle_increment)
        clusters = self._cluster(points)

        obstacles: List[DetectedObstacle] = []
        oid = 0
        for cl in clusters:
            if len(cl) < self.min_points:
                continue
            obs = self._cluster_to_obstacle(cl, oid)
            if max(obs.width, obs.height) > self.max_size:
                continue                      # wall / boundary, not an obstacle
            obstacles.append(obs)
            oid += 1
        return obstacles

    def _scan_to_cartesian(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[Point2D]:
        """Valid LiDAR ranges -> ordered Point2D list (sensor frame)."""
        pts: List[Point2D] = []
        for i, r in enumerate(ranges):
            r = float(r)
            if not math.isfinite(r) or r < self.min_range or r > self.max_range:
                continue
            ang = angle_min + i * angle_increment
            pts.append(Point2D(r * math.cos(ang), r * math.sin(ang)))
        return pts

    def _cluster(self, points: List[Point2D]) -> List[List[Point2D]]:
        """Split ordered points into clusters at jump-distance discontinuities."""
        if not points:
            return []
        clusters: List[List[Point2D]] = []
        cur: List[Point2D] = [points[0]]
        for p in points[1:]:
            if math.hypot(p.x - cur[-1].x, p.y - cur[-1].y) > self.cluster_threshold:
                clusters.append(cur)
                cur = [p]
            else:
                cur.append(p)
        clusters.append(cur)
        return clusters

    def _cluster_to_obstacle(
        self,
        cluster: List[Point2D],
        idx: int,
    ) -> DetectedObstacle:
        """Centroid + axis-aligned bounding box of one cluster."""
        xs = [p.x for p in cluster]
        ys = [p.y for p in cluster]
        n = len(cluster)
        return DetectedObstacle(
            cx=sum(xs) / n,
            cy=sum(ys) / n,
            width=max(xs) - min(xs),
            height=max(ys) - min(ys),
            num_points=n,
            points=list(cluster),
            id=idx,
        )


class DBSCANDetector:
    """DBSCAN-based LiDAR detector skeleton (optional implementation)."""

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 3,
        max_range: float = 10.0,
        min_range: float = 0.05,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.max_range = max_range
        self.min_range = min_range

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[DetectedObstacle]:
        """TODO: Implement DBSCAN-based clustering on Cartesian points."""
        _ = (ranges, angle_min, angle_increment)
        return []
