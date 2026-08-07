"""Which track boundary is on the driver's LEFT, decided by geometry alone.

ONE FUNCTION, TWO CALLERS, AND THAT IS THE POINT. global_planner_node uses this to LABEL the
bounds when a map is generated; stack_master/scripts/check_track_bounds.py uses it to CHECK the
labels afterwards. While the two carried separate logic, the checker could be right and the
generator wrong, which is exactly what shipped: map f (402 stations vs 0) and map ifac_0807 (461
vs 1) both have d_right/d_left exchanged, and the checker has been reporting it correctly to
nobody, because it was wired to no gate.

WHAT WAS WRONG WITH THE OLD DECISION. The generator picked the outer boundary by CONTOUR LENGTH
and then chose a side by comparing the outer contour's local traversal direction against the start
pose. Contour winding comes from cv2, not from the direction the car drives, and the centerline
gets flipped twice before this is decided -- once automatically when its direction disagrees with
the start pose, once more for reverse_mapping -- with neither flip correcting the labels. The only
thing that ever put them right again was dist_to_bounds swapping its return when reverse_mapping
was set, i.e. correctness rested on an operator setting a flag, and a double correction was one
mistake away.

The decision here needs no flag and no winding convention: take the tangent of the FINAL
centerline, look along its left normal, and see which contour lies that way. Flip the centerline
and the answer flips with it, because the answer is a property of the geometry.
"""
import math
import os

import numpy as np


def truth_left_right(pts, i, cont_a, cont_b):
    """(left_dist, right_dist) at pts[i] from two contours, via the tangent's left normal.

    Returns None where the tangent is degenerate. cont_a/cont_b are unordered: which one is outer
    and which inner does not matter and is never asked.
    """
    n = len(pts)
    ax, ay = pts[(i + 3) % n]
    bx, by = pts[(i - 3) % n]
    tx, ty = ax - bx, ay - by
    m = math.hypot(tx, ty)
    if m < 1e-9:
        return None
    tx, ty = tx / m, ty / m
    lx, ly = -ty, tx                       # left normal (left of travel)
    px, py = pts[i]

    def nearest(cont):
        d = cont - np.array([px, py])
        dist = np.hypot(d[:, 0], d[:, 1])
        j = int(np.argmin(dist))
        return float(d[j, 0] * lx + d[j, 1] * ly), float(dist[j])
    s_a, d_a = nearest(cont_a)
    s_b, d_b = nearest(cont_b)
    return (d_a if s_a > 0 else d_b), (d_a if s_a < 0 else d_b)


def score(pts, left_vals, right_vals, cont_a, cont_b, step_target=120):
    """How many sampled stations match the STORED labelling, and how many match the swap.

    (n_ok, n_swapped, n_ambiguous). Ambiguous means the two readings are within 5 cm of each
    other -- a station where the track is symmetric enough that the labelling cannot be judged.
    """
    n = len(pts)
    step = max(1, n // step_target)
    n_ok = n_swap = n_amb = 0
    for i in range(0, n, step):
        t = truth_left_right(pts, i, cont_a, cont_b)
        if t is None:
            continue
        t_l, t_r = t
        e_ok = abs(left_vals[i] - t_l) + abs(right_vals[i] - t_r)
        e_sw = abs(left_vals[i] - t_r) + abs(right_vals[i] - t_l)
        if abs(e_ok - e_sw) < 0.05:
            n_amb += 1
        elif e_ok < e_sw:
            n_ok += 1
        else:
            n_swap += 1
    return n_ok, n_swap, n_amb


def assign_sides(pts, cont_a, cont_b, step_target=120):
    """Return (bound_right, bound_left) chosen from the two contours by the centerline's geometry.

    `pts` must be the FINAL centerline -- after every flip -- in the same frame as the contours.
    Sampled and voted rather than decided at one station: a single tangent near a hairpin can read
    the wrong way, and on a closed ring the majority is unambiguous.

    Raises ValueError if the vote is not decisive, which on a real ring means the inputs are not
    what this function needs (wrong frame, one contour, an open path) -- failing is better than
    labelling a map by coin toss.
    """
    n = len(pts)
    step = max(1, n // step_target)
    a_left = b_left = 0
    for i in range(0, n, step):
        t = truth_left_right(pts, i, cont_a, cont_b)
        if t is None:
            continue
        # re-derive which contour supplied the left reading
        left_d, _right_d = t
        d_a = float(np.min(np.hypot(cont_a[:, 0] - pts[i][0], cont_a[:, 1] - pts[i][1])))
        if abs(left_d - d_a) < 1e-9:
            a_left += 1
        else:
            b_left += 1
    total = a_left + b_left
    if total == 0 or abs(a_left - b_left) <= 0.1 * total:
        raise ValueError(
            f"track side assignment is not decisive ({a_left} vs {b_left} of {total} samples) -- "
            f"check that the centerline and the contours are in the same frame")
    return (cont_b, cont_a) if a_left > b_left else (cont_a, cont_b)


def load_contours(map_dir, name):
    """The two free-space contours of a map, in metres, unordered. Raises if it is not a ring.

    Here rather than in the checker because the RUNTIME nodes need it too: a map whose labels are
    exchanged makes every corridor mirror-imaged, and the cheapest place to say so out loud is
    when the node loads the map.
    """
    import cv2
    import yaml
    info = yaml.safe_load(open(os.path.join(map_dir, f'{name}.yaml')))
    img = cv2.imread(os.path.join(map_dir, info['image']), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read the map image for {name}")
    res = float(info['resolution'])
    ox, oy = float(info['origin'][0]), float(info['origin'][1])
    height = img.shape[0]
    cnts, _ = cv2.findContours((img > 250).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=len, reverse=True)
    if len(cnts) < 2:
        raise IOError(f"{name}: expected 2 free-space contours, found {len(cnts)}")

    def to_m(c):
        return np.stack([c[:, 0, 0] * res + ox, (height - 1 - c[:, 0, 1]) * res + oy], 1)
    return to_m(cnts[0]), to_m(cnts[1])


def verdict(pts, d_right, d_left, map_dir, name):
    """('ok' | 'SWAPPED', n_ok, n_swapped, n_ambiguous) for a published line against its map.

    Never raises: a startup diagnostic that can take a node down is worse than the defect it
    reports. Returns ('unknown', 0, 0, 0) if the map cannot be read as a ring.
    """
    try:
        a, b = load_contours(map_dir, name)
        n_ok, n_sw, n_amb = score(np.asarray(pts, float), np.asarray(d_left, float),
                                  np.asarray(d_right, float), a, b)
    except Exception:
        return "unknown", 0, 0, 0
    return ("SWAPPED" if n_sw > n_ok else "ok"), n_ok, n_sw, n_amb
