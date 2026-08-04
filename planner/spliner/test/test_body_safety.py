#!/usr/bin/env python3
"""Does the body-safety image actually separate "the point is on a free cell" from "the CAR fits
here", and does it cost the planner a line it could otherwise drive?

Runs the REAL _free_mask / _path_body_unsafe against the REAL ifac map: its global_waypoints.json
AND its ifac.png occupancy grid, with a GridFilter stand-in that reproduces GridFilter's own pixel
convention and erosion exactly (same approach as lane_change_planner/test/test_grid_corridor.py).

Two questions, and they pull in opposite directions:

  * SAFETY -- a path hugging the k=3 corridor edge is a car centre 0.05 m from the wall, which puts
    0.10 m of a 0.30 m car inside it. The k=7 image must reject exactly those.
  * FEASIBILITY -- the floor must not reject the racing line or collapse the narrow sections the
    squeeze pass exists for, or it trades a wall strike for a TRAILING standstill.

Run (after sourcing the workspace):
  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_body_safety.py
"""
import json
import types
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from frenet_conversion.frenet_converter import FrenetConverter

REPO = Path(__file__).resolve().parents[3]
MOD = REPO / "planner/spliner/spliner/static_avoidance_node.py"

HALF_CAR = 0.15          # width_car / 2, the reserve the body image must deliver
RESOLUTION = 0.05        # [m/cell] on the shipped maps


class Grid:
    """GridFilter stand-in: same 255=free convention, same erosion, same world_to_pixel."""

    def __init__(self, mapdir, name, kernel_size):
        meta = yaml.safe_load((mapdir / f"{name}.yaml").read_text())
        self.resolution = meta["resolution"]
        self.origin = (meta["origin"][0], meta["origin"][1])
        img = np.array(Image.open(mapdir / meta["image"]).convert("L"))
        occ = img < int(255 * (1.0 - meta["occupied_thresh"]))
        base = np.where(occ, 0, 255).astype(np.uint8)
        # ROS OccupancyGrid row 0 is the BOTTOM of the map; a .png row 0 is the TOP.
        base = np.flipud(base)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.eroded_image = cv2.erode(base, k)

    def is_point_inside(self, x, y):
        """Verbatim GridFilter.is_point_inside — _path_off_track calls it per point."""
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return self.eroded_image[py, px] == 255


_CACHE = {}


def fixture(mapname="ifac"):
    """The node's real methods, bound to a bare object carrying only what they read."""
    if mapname in _CACHE:
        return _CACHE[mapname]
    san = types.ModuleType("san")
    san.__dict__["__file__"] = str(MOD)
    exec(compile(MOD.read_text(), str(MOD), "exec"), san.__dict__)

    class Fake:
        pass

    for n in ("_free_mask", "_path_body_unsafe", "_path_off_track", "_grid_corridor"):
        setattr(Fake, n, getattr(san.ObstacleSpliner, n))

    mapdir = REPO / "stack_master/maps" / mapname
    wp = json.load(open(mapdir / "global_waypoints.json"))["global_traj_wpnts_iqp"]["wpnts"]
    xy = np.array([[w["x_m"], w["y_m"]] for w in wp])
    s_m = np.array([w["s_m"] for w in wp])

    f = Fake()
    f.map_filter = Grid(mapdir, mapname, 3)      # SAMPLING image: reserves one cell (0.05 m)
    f.body_filter = Grid(mapdir, mapname, 7)     # BODY image: reserves 3 cells (0.15 m = half car)
    f.converter = FrenetConverter(xy[:, 0], xy[:, 1])
    f.gb_max_s = float(s_m[-1] + (s_m[1] - s_m[0]))
    f.grid_scan_max, f.grid_scan_step, f.wall_margin = 3.0, 0.05, 0.0
    _CACHE[mapname] = (f, xy, s_m)
    return _CACHE[mapname]


def test_body_kernel_reserves_half_a_car():
    # The floor is only worth anything if the erosion actually covers half the car at the maps'
    # resolution. floor(7/2) = 3 cells = 0.15 m; anything less and a "safe" point is not one.
    assert (7 // 2) * RESOLUTION >= HALF_CAR - 1e-9, "k=7 must reserve at least half a car"
    assert (3 // 2) * RESOLUTION < HALF_CAR, "…and the sampling kernel must NOT, or it is redundant"
    print(f"PASS body kernel reserves {(7 // 2) * RESOLUTION:.2f} m >= half car {HALF_CAR:.2f} m")


def test_raceline_is_body_free():
    # FEASIBILITY. A floor that rejects the racing line rejects everything, and the planner would
    # concede TRAILING everywhere instead of avoiding.
    for mapname in ("ifac", "f"):
        f, xy, _s = fixture(mapname)
        free = f._free_mask(xy, f.body_filter)
        assert free is not None and free.all(), \
            f"{mapname}: {int((~free).sum())}/{len(free)} raceline points are not body-free"
        assert not f._path_body_unsafe(xy), f"{mapname}: the raceline itself must never be vetoed"
        print(f"PASS {mapname}: raceline body-free at {free.sum()}/{len(free)} stations")


def test_wall_hugging_path_is_rejected_but_reads_free_to_the_sampling_image():
    # SAFETY, and the exact gap this closes: a path along the k=3 corridor edge is a car CENTRE one
    # cell from the wall. Every point of it is "free" to the sampling image -- which is all
    # do_spline used to check under trust_grid_bounds, because the waypoint corridor test is
    # skipped there -- and 0.10 m of the car is inside the wall.
    f, xy, s_m = fixture("ifac")
    d_scan = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    hugging = []
    for s in s_m[::7]:
        resp = f.converter.get_cartesian(np.full(d_scan.shape, s % f.gb_max_s), d_scan)
        pts = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
        free3 = f._free_mask(pts, f.map_filter)
        i0 = int(np.argmin(np.abs(d_scan)))
        if not free3[i0]:
            continue
        hi = i0
        while hi < free3.size - 1 and free3[hi + 1]:
            hi += 1
        hugging.append(pts[hi])                  # the outermost point the SAMPLING image allows
    hugging = np.asarray(hugging)
    assert len(hugging) > 20, "the harness must produce a usable number of edge points"
    free3 = f._free_mask(hugging, f.map_filter)
    free7 = f._free_mask(hugging, f.body_filter)
    assert free3.all(), "by construction every edge point is free to the sampling image"
    caught = int((~free7).sum())
    assert caught > 0.8 * len(hugging), \
        f"the body image must reject the corridor edge, caught only {caught}/{len(hugging)}"
    assert f._path_body_unsafe(hugging), "a path along the corridor edge must be vetoed"
    assert not f._path_off_track(hugging), \
        "…and the sampling-image check must NOT catch it — that is the hole being closed"
    print(f"PASS corridor-edge path: {caught}/{len(hugging)} points body-unsafe, "
          f"0 caught by the sampling image")


def test_floor_leaves_room_to_avoid():
    # FEASIBILITY, quantified: how much lateral room does a car CENTRE still have on the tightest
    # map? The narrowest body-free span bounds every avoidance the planner can offer there.
    f, xy, s_m = fixture("ifac")
    d_scan = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    spans = []
    for s in s_m:
        resp = f.converter.get_cartesian(np.full(d_scan.shape, s % f.gb_max_s), d_scan)
        pts = (resp.T if resp.ndim == 2 else resp).reshape(-1, 2)
        free = f._free_mask(pts, f.body_filter)
        i0 = int(np.argmin(np.abs(d_scan)))
        if not free[i0]:
            spans.append(0.0)
            continue
        lo = hi = i0
        while lo > 0 and free[lo - 1]:
            lo -= 1
        while hi < free.size - 1 and free[hi + 1]:
            hi += 1
        spans.append(float(d_scan[hi] - d_scan[lo]))
    spans = np.asarray(spans)
    assert spans.min() >= 0.44, \
        f"the tightest body-free span is {spans.min():.2f} m — the floor has closed the track"
    print(f"PASS ifac body-free span: min {spans.min():.2f} med {np.median(spans):.2f} "
          f"max {spans.max():.2f} m")


def test_no_map_does_not_veto():
    # A bench or bag run without /map must not have every path rejected: the sampling-side checks
    # are all there is then, exactly as _path_off_track already assumes.
    f, xy, _s = fixture("ifac")
    bare = types.SimpleNamespace(
        body_filter=types.SimpleNamespace(eroded_image=None),
        _free_mask=f._free_mask)
    assert san_path_body_unsafe(bare, xy) is False, "no map must mean no veto"
    bare2 = types.SimpleNamespace(_free_mask=f._free_mask)      # attribute missing entirely
    assert san_path_body_unsafe(bare2, xy) is False
    print("PASS a missing map yields no body veto")


def san_path_body_unsafe(obj, xy):
    f, _xy, _s = fixture("ifac")
    return type(f)._path_body_unsafe(obj, xy)


if __name__ == "__main__":
    test_body_kernel_reserves_half_a_car()
    test_raceline_is_body_free()
    test_wall_hugging_path_is_rejected_but_reads_free_to_the_sampling_image()
    test_floor_leaves_room_to_avoid()
    test_no_map_does_not_veto()
    print("ALL PASS")
