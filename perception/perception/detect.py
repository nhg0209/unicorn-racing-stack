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
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.min_points = min_points
        self.max_range = max_range
        self.min_range = min_range

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[DetectedObstacle]:
        """Return detected obstacles for one scan.

        TODO:
        1. Filter invalid ranges and convert polar -> Cartesian points.
        2. Cluster adjacent points using jump-distance threshold.
        3. Fit centroid / bounding box for each cluster.
        4. Return list[DetectedObstacle].
        """
        _ = (ranges, angle_min, angle_increment)
        return []

    def _scan_to_cartesian(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[Point2D]:
        """TODO: Convert valid LiDAR ranges to Point2D list."""
        _ = (ranges, angle_min, angle_increment)
        return []

    def _cluster(self, points: List[Point2D]) -> List[List[Point2D]]:
        """TODO: Split ordered points into clusters using jump distance."""
        _ = points
        return []

    def _cluster_to_obstacle(
        self,
        cluster: List[Point2D],
        idx: int,
    ) -> DetectedObstacle:
        """TODO: Compute centroid and AABB from one cluster."""
        _ = (cluster, idx)
        return DetectedObstacle()


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
