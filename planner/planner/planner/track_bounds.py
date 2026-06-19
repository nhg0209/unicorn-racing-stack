#!/usr/bin/env python3
"""
TrackBounds utility for F1TENTH

Loads wall boundary CSVs (boundary_right.csv, boundary_left.csv) and computes
d_right / d_left for any set of (x, y) waypoints using nearest-point distance.

Usage:
    tb = TrackBounds('/path/to/map_dir')
    if tb.is_valid():
        d_right, d_left = tb.compute_distances(x_array, y_array)
"""

import numpy as np
import csv
import os


class TrackBounds:

    def __init__(self, map_dir: str):
        self._right = self._load_boundary(os.path.join(map_dir, 'boundary_right.csv'))
        self._left = self._load_boundary(os.path.join(map_dir, 'boundary_left.csv'))

    def is_valid(self) -> bool:
        return self._right is not None and self._left is not None

    def compute_distances(self, x: np.ndarray, y: np.ndarray):
        """
        Compute minimum distance from each waypoint to the right and left walls.

        Args:
            x: (N,) array of waypoint x coordinates
            y: (N,) array of waypoint y coordinates

        Returns:
            (d_right, d_left): each (N,) array of distances in meters
        """
        if not self.is_valid():
            return np.zeros_like(x), np.zeros_like(x)

        pts = np.column_stack([x, y])  # (N, 2)

        d_right = self._min_distances(pts, self._right)
        d_left = self._min_distances(pts, self._left)

        return d_right, d_left

    @staticmethod
    def _min_distances(waypoints: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """
        Compute minimum Euclidean distance from each waypoint to boundary points.
        Uses chunked computation to avoid excessive memory with large boundaries.

        Args:
            waypoints: (N, 2) array
            boundary: (M, 2) array

        Returns:
            (N,) array of minimum distances
        """
        n = len(waypoints)
        m = len(boundary)

        # If boundary is small enough, vectorize directly
        if m <= 5000:
            # (N, 1, 2) - (1, M, 2) → (N, M, 2) → (N, M) → (N,)
            diff = waypoints[:, np.newaxis, :] - boundary[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            return np.min(dists, axis=1)

        # For large boundaries, process in chunks to limit memory
        chunk_size = 5000
        min_dists = np.full(n, np.inf)
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            chunk = boundary[start:end]
            diff = waypoints[:, np.newaxis, :] - chunk[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            chunk_min = np.min(dists, axis=1)
            min_dists = np.minimum(min_dists, chunk_min)

        return min_dists

    @staticmethod
    def _load_boundary(csv_path: str):
        """Load boundary CSV (x_m, y_m). Returns Nx2 array or None."""
        if not os.path.exists(csv_path):
            return None

        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row['x_m']), float(row['y_m'])])

        if len(rows) == 0:
            return None

        return np.array(rows)
