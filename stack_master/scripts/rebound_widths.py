#!/usr/bin/env python3
"""Re-measure d_left/d_right against the true wall image, leaving the racing line untouched.

Why this exists
---------------
The generator picks its map image by probing <map>.yaml, <map>.png, <map>.pgm in that order, so
an image edited to shape the global line also becomes the source of the published track bounds.
Narrowing a corridor in GIMP is a legitimate way to force the line where you want it -- and it is
wrong for everything downstream. The reactive planner samples its avoidance candidates between
d_left and d_right, the static re-optimiser fits its humps inside them, and the state machine
judges whether a gap is passable with them. All three then plan inside a corridor the car does
not physically have, and the car refuses room that is really there.

Geometry and bounds belong on separate sources. This script leaves every x/y exactly as the
optimiser produced it and re-measures only the widths, on the untouched wall image.

How the widths are measured
---------------------------
The generator's own measurement, ported from global_planner_node.py: vertical flip -> 9x9
MORPH_OPEN -> watershed seeded on the centerline -> RETR_CCOMP, keeping contours that have a
parent or a child -> track_bounds.assign_sides for the left/right labelling -> interp_track at
0.1 m -> distance to the nearest bound point. Note there is no thresholding anywhere in that
chain: cv2.imread gives raw grey and every non-zero pixel is free space, so on a .pgm the 205
"unknown" cells count as free. Treating them as walls instead moves the measured bound by tens of
centimetres, which is why this is a port and not a reimplementation.

Because it is a port, it can be checked rather than trusted. Before writing anything the same
code runs against every candidate image, and the one that generated the file must reproduce the
stored numbers to within --tol. If none does, the port has drifted from the generator and the
script refuses to write.

Usage
-----
    rebound_widths.py <map>                  # dry run: report only
    rebound_widths.py <map> --apply          # rewrite (backs up to *.bak_rebound)
    rebound_widths.py <map> --truth foo.pgm  # name the truth image explicitly

Truth image defaults to <map>_origin.png if you kept one, else <map>.pgm.
Afterwards run:  check_track_bounds.py <map>
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.segmentation import watershed

HERE = Path(__file__).resolve()


def _find_repo():
    """Locate the source tree that holds gb_optimizer.

    This measurement is a port of the generator's, so it imports the generator's own
    track_bounds rather than duplicating it -- which means finding the source tree even when
    the script runs from an install prefix. Walking up from __file__ covers a source checkout
    and a --symlink-install (resolve() lands back in the tree); STACK_MASTER_MAPS_ROOT covers
    a plain copied install, where the installed path shares no ancestor with the sources.
    """
    marker = Path("planner/gb_optimizer/gb_optimizer/track_bounds.py")
    starts = [HERE]
    env_maps = os.environ.get("STACK_MASTER_MAPS_ROOT")
    if env_maps:
        starts.append(Path(env_maps).resolve())
    for start in starts:
        for cand in (start, *start.parents):
            if (cand / marker).exists():
                return cand
    sys.exit("cannot locate the unicorn-racing-stack source tree (gb_optimizer not found). "
             "Set STACK_MASTER_MAPS_ROOT, or run this from the checkout.")


REPO = _find_repo()
MAPS = Path(os.environ.get("STACK_MASTER_MAPS_ROOT") or REPO / "stack_master" / "maps")
sys.path.insert(0, str(REPO / "planner/gb_optimizer"))
sys.path.insert(0, str(REPO / "planner/gb_optimizer/gb_optimizer"))
from gb_optimizer import track_bounds as TB                                       # noqa: E402
from global_racetrajectory_optimization.helper_funcs_glob.src.interp_track \
    import interp_track                                                           # noqa: E402

KERNEL = np.ones((9, 9), np.uint8)
WPNT_KEYS = ("global_traj_wpnts_iqp", "global_traj_wpnts_sp", "centerline_waypoints")


def bounds_of(map_dir, image, res, ox, oy, cent_m):
    """Right/left bounds in metres, as global_planner_node.extract_track_bounds computes them.

    Returns (right_int, left_int, right_raw, left_raw) -- interpolated for measuring, raw for
    the RViz markers.
    """
    img = cv2.imread(str(Path(map_dir) / image), 0)
    if img is None:
        raise IOError(f"{image}: unreadable")
    opening = cv2.morphologyEx(cv2.flip(img, 0), cv2.MORPH_OPEN, KERNEL, iterations=1)

    cent_px = np.stack([(cent_m[:, 0] - ox) / res, (cent_m[:, 1] - oy) / res], 1)
    cent_img = np.zeros(opening.shape, np.uint8)
    cv2.drawContours(cent_img, [cent_px.astype(int)], 0, 255, 2, cv2.LINE_8)
    _, markers = cv2.connectedComponents(cent_img)
    labels = watershed(-cv2.distanceTransform(opening, cv2.DIST_L2, 5), markers, mask=opening)

    closed = []
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(opening.shape, np.uint8)
        mask[labels == label] = 255
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # generator: opened = (no child and no parent); it keeps the ones that are NOT opened
        closed += [c for i, c in enumerate(cnts)
                   if not (hier[0][i][2] < 0 and hier[0][i][3] < 0)]
    if len(closed) != 2:
        raise IOError(f"{image}: {len(closed)} closed contours, expected 2 "
                      f"(stray lidar beams or a gap in the wall?)")

    def to_m(c):
        p = c.reshape(-1, 2)
        return np.stack([p[:, 0] * res + ox, p[:, 1] * res + oy], 1)

    right, left = TB.assign_sides(cent_m, to_m(max(closed, key=len)), to_m(min(closed, key=len)))
    interp = lambda b: interp_track(
        reftrack=np.column_stack((b, np.zeros((b.shape[0], 2)))), stepsize_approx=0.1)[:, :2]
    return interp(right), interp(left), right, left


def widths(pts, right, left):
    """dist_to_bounds: distance from each waypoint to the nearest point on each bound."""
    near = lambda b: np.min(np.linalg.norm(pts[:, None, :] - b[None, :, :], axis=2), axis=1)
    return near(right), near(left)


def arrays_of(doc):
    for k in WPNT_KEYS:
        if k in doc and doc[k].get("wpnts"):
            yield k, doc[k]["wpnts"]


def rebuild_markers(doc, right_raw, left_raw):
    """Repoint the RViz bound spheres at the same wall the planners now use.

    The two bounds are drawn in different colours, and which colour belongs to which side is
    decided here by geometry rather than by assuming the generator's ordering: each colour
    group in the existing array is matched to whichever new bound its points sit closer to.
    One marker of that group then serves as the template for that side, so colour, scale and
    type all survive. A single-colour array is carried over as-is.
    """
    tb = doc.get("trackbounds_markers")
    if not tb or not tb.get("markers"):
        return 0
    old = tb["markers"]
    groups = {}
    for m in old:
        c = m.get("color", {})
        key = (round(c.get("r", 0), 3), round(c.get("g", 0), 3),
               round(c.get("b", 0), 3), round(c.get("a", 1), 3))
        groups.setdefault(key, []).append(m)

    def nearest(pts, bound):
        return float(np.mean(np.min(np.linalg.norm(
            pts[:, None, :] - bound[None, :, :], axis=2), axis=1)))

    tpl_right = tpl_left = old[0]
    if len(groups) >= 2:
        scored = []
        for key, ms in groups.items():
            pts = np.array([[m["pose"]["position"]["x"], m["pose"]["position"]["y"]] for m in ms])
            scored.append((nearest(pts, right_raw) - nearest(pts, left_raw), ms[0]))
        scored.sort(key=lambda z: z[0])          # most right-leaning first
        tpl_right, tpl_left = scored[0][1], scored[-1][1]

    out = []
    for tpl, pts in ((tpl_right, right_raw), (tpl_left, left_raw)):
        for x, y in pts:
            m = json.loads(json.dumps(tpl))
            m["pose"]["position"]["x"], m["pose"]["position"]["y"] = float(x), float(y)
            out.append(m)
    for i, m in enumerate(out):
        m["id"] = i
    tb["markers"] = out
    return len(out)


def wait_for_fresh(md, timeout, settle=2.0):
    """Block until the generator has written its output and stopped touching it.

    The planner node does not exit after writing -- it keeps publishing -- so there is no
    process exit to hang a launch event handler on. Freshness is judged against this call's
    own start time so a previous run's files never satisfy the wait, and the size/mtime must
    then hold still, because the two files are written separately and a read landing between
    them would measure a half-updated map.
    """
    targets = [md / "global_waypoints.json", md / "centerline.csv"]
    t0 = time.time()
    deadline = t0 + timeout
    last, stable = None, 0
    print(f"waiting up to {timeout:.0f}s for the generator to write {md.name}/ ...", flush=True)
    while time.time() < deadline:
        if all(t.exists() and t.stat().st_mtime >= t0 for t in targets):
            sig = tuple((t.stat().st_mtime, t.stat().st_size) for t in targets)
            stable = stable + 1 if sig == last else 0
            last = sig
            if stable * 1.0 >= settle:
                print(f"  generator output settled after {time.time()-t0:.0f}s\n", flush=True)
                return True
        time.sleep(1.0)
    print(f"  no fresh output after {timeout:.0f}s", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="map name under stack_master/maps/, or a path")
    ap.add_argument("--truth", default=None, help="image to measure against")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max mean error, in m, when reproducing the stored widths")
    ap.add_argument("--wait", type=float, default=0.0, metavar="SEC",
                    help="wait up to SEC for the generator to write a fresh "
                         "global_waypoints.json before measuring (launch use)")
    ap.add_argument("--optional", action="store_true",
                    help="exit 0 instead of erroring when there is no truth image "
                         "or nothing to re-base (launch use)")
    a = ap.parse_args()

    md = Path(a.map) if os.sep in a.map else MAPS / a.map
    md = md.resolve()
    name = md.name
    if a.wait > 0 and not wait_for_fresh(md, a.wait):
        sys.exit(0 if a.optional else "timed out waiting for the generator's output")
    meta = yaml.safe_load((md / f"{name}.yaml").read_text())
    res, ox, oy = float(meta["resolution"]), float(meta["origin"][0]), float(meta["origin"][1])

    cands = [p.name for p in (md / f"{name}_origin.png", md / f"{name}.png", md / f"{name}.pgm")
             if p.exists()]
    truth = a.truth or next((c for c in (f"{name}_origin.png", f"{name}.pgm") if c in cands), None)
    if truth is None:
        msg = f"no truth image among {cands} -- keep a {name}_origin.png or a {name}.pgm"
        if a.optional:
            print(msg + " -- skipping")
            return
        sys.exit(msg)
    print(f"map {name}\ncandidates {cands}\ntruth      {truth}\n")

    doc = json.load(open(md / "global_waypoints.json"))
    # The centerline only seeds the watershed; the csv is the generator's own copy of it, but a
    # map that never shipped one can fall back to the same points inside the json.
    cl_path = md / "centerline.csv"
    cl = np.genfromtxt(cl_path, delimiter=",", skip_header=1) if cl_path.exists() else None
    if cl is not None:
        cent = cl[:, :2]
    elif doc.get("centerline_waypoints", {}).get("wpnts"):
        cent = np.array([[w["x_m"], w["y_m"]] for w in doc["centerline_waypoints"]["wpnts"]])
        print(f"  (no centerline.csv; seeding from centerline_waypoints)")
    else:
        sys.exit(f"{name}: no centerline.csv and no centerline_waypoints. Stopping.")

    key0, w0 = next(arrays_of(doc))
    pts0 = np.array([[w["x_m"], w["y_m"]] for w in w0])
    dr0 = np.array([w["d_right"] for w in w0])
    dl0 = np.array([w["d_left"] for w in w0])

    print(f"{'image':>26s} {'|dr err|':>10s} {'|dl err|':>10s}   ({key0}, n={len(w0)})")
    best, best_err, sides = None, float("inf"), {}
    for c in cands:
        try:
            R, L, Rr, Lr = bounds_of(md, c, res, ox, oy, cent)
        except IOError as e:
            print(f"{c:>26s}   {e}")
            continue
        sides[c] = (R, L, Rr, Lr)
        r, l = widths(pts0, R, L)
        er, el = float(np.mean(np.abs(r - dr0))), float(np.mean(np.abs(l - dl0)))
        print(f"{c:>26s} {er:10.4f} {el:10.4f}")
        if er + el < best_err:
            best, best_err = c, er + el
    if best is None:
        sys.exit("no candidate image yielded track bounds. Stopping.")
    print(f"\ngenerated from: {best}  (mean error {best_err:.4f} m)")
    if best_err > a.tol:
        sys.exit(f"cannot reproduce the stored widths within {a.tol} m -- this port has drifted "
                 f"from global_planner_node. Stopping without writing.")
    print("  -> measurement matches the generator\n")
    if truth not in sides:
        sys.exit(f"no bounds from the truth image {truth}. Stopping.")
    if truth == best:
        print(f"  {truth} is already the generation source; nothing to re-base.\n")
        if a.optional:
            return

    R, L, Rr, Lr = sides[truth]
    shrunk = 0
    for key, wl in arrays_of(doc):
        pts = np.array([[w["x_m"], w["y_m"]] for w in wl])
        nr, nl = widths(pts, R, L)
        old = np.array([w["d_right"] for w in wl]) + np.array([w["d_left"] for w in wl])
        new = nr + nl
        shrunk = max(shrunk, int((new < old - 0.05).sum()))
        print(f"  {key:24s} width p50 {np.percentile(old, 50):.3f} -> {np.percentile(new, 50):.3f}"
              f"   min {old.min():.3f} -> {new.min():.3f}"
              f"   changed >10cm: {int((abs(new - old) > 0.10).sum())}/{len(wl)}")
        if (nr <= 0).any() or (nl <= 0).any():
            sys.exit("a waypoint measured outside the truth corridor. Stopping.")
        if a.apply:
            for w, r_, l_ in zip(wl, nr, nl):
                w["d_right"], w["d_left"] = float(r_), float(l_)
    cr, clf = widths(cent, R, L)
    if cl is not None:
        print(f"  {'centerline.csv':24s} width p50 {np.percentile(cl[:, 2] + cl[:, 3], 50):.3f} -> "
              f"{np.percentile(cr + clf, 50):.3f}   min {(cl[:, 2] + cl[:, 3]).min():.3f} -> "
              f"{(cr + clf).min():.3f}")
    if shrunk:
        print(f"\n  note: {shrunk} stations get NARROWER on {truth}. That is the safe direction, "
              f"but check for noise blobs inside the track.")

    if not a.apply:
        print("\ndry run -- nothing written. Re-run with --apply.")
        return

    if cl is not None:
        shutil.copy2(cl_path, str(cl_path) + ".bak_rebound")
        np.savetxt(cl_path, np.column_stack([cent[:, 0], cent[:, 1], cr, clf]), delimiter=",",
                   header="x_m,y_m,w_tr_right_m,w_tr_left_m", comments="", fmt="%.6f")
    n_mark = rebuild_markers(doc, Rr, Lr)
    gp = md / "global_waypoints.json"
    shutil.copy2(gp, str(gp) + ".bak_rebound")
    json.dump(doc, open(gp, "w"))
    print(f"\nwritten against {truth}  ({n_mark} bound markers repointed). backups: *.bak_rebound")
    print(f"  verify: check_track_bounds.py {name}")


if __name__ == "__main__":
    main()
