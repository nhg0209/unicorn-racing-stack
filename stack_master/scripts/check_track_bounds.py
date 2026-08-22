#!/usr/bin/env python3
"""Check (and optionally repair) the left/right labelling of a map's track bounds.

gb_optimizer decides which of the two track contours is "left" and which is "right" with ONE
global decision taken from the start pose (global_planner_node.extract_track_bounds), plus an
optional second flip from reverse_mapping (dist_to_bounds). When that comes out inverted, the map
ships with d_left/d_right exchanged for the WHOLE lap -- geometry is fine, only the labels are
mirrored. Everything downstream that reads a corridor then believes the roomy side is the side the
wall is on: the static-avoidance planner samples into the wall, and static_reopt solves its QP on a
mirrored corridor.

Ground truth here is the map image itself: the two free-space contours are extracted directly and
each waypoint is asked which contour lies to its left and which to its right, using the left normal
of the local tangent (the same left-positive convention the stack uses for d).

    python3 check_track_bounds.py f              # report only
    python3 check_track_bounds.py f --fix        # swap the labels back (writes .bak first)

--fix IS NOT A FULL REPAIR. The exchanged widths do not just sit in the json: write_centerline
feeds them to the trajectory optimizer, so the RACELINE ITSELF was optimized inside a mirrored
corridor. Relabelling makes avoidance work again -- the corridor the planners read is finally the
right way round -- but the line through it is still the product of the wrong one. A map that
reports SWAPPED should be REGENERATED once the generator is fixed (global_planner_node now assigns
sides from the centerline's own geometry, planner/gb_optimizer/gb_optimizer/track_bounds.py);
--fix is the stopgap for a map you cannot regenerate right now.
    python3 check_track_bounds.py --all          # report on every map

--fix swaps d_left<->d_right in every waypoint array of global_waypoints.json and the
w_tr_right_m/w_tr_left_m columns of centerline.csv. It refuses unless the swap is decisive, so it
cannot silently mangle a map that is already correct. It is a label fix, not a substitute for
regenerating the map with the correct start pose / reverse_mapping.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import yaml

# The SAME geometry global_planner_node labels the bounds with. Shared on purpose: while the
# generator and this checker carried separate logic, the checker could be right and the generator
# wrong -- which is exactly what shipped on maps f and ifac_0807.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                '..', 'planner', 'gb_optimizer'))
from gb_optimizer.track_bounds import (load_contours, score,  # noqa: E402,F401
                                       truth_left_right)

MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps')
WPNT_KEYS = ('centerline_waypoints', 'global_traj_wpnts_iqp', 'global_traj_wpnts_sp')


def read_centerline(path):
    with open(path) as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith('#')]
    header = None
    try:
        float(rows[0][0])
    except ValueError:
        header = rows[0]
        rows = rows[1:]
    return header, rows


def check_map(name, do_fix=False):
    map_dir = os.path.join(MAPS_DIR, name)
    outer, inner = load_contours(map_dir, name)
    print(f'\n=== {name} ===')
    verdicts = {}

    json_path = os.path.join(map_dir, 'global_waypoints.json')
    gw = json.load(open(json_path)) if os.path.exists(json_path) else None
    if gw is not None:
        for key in WPNT_KEYS:
            if key not in gw or 'wpnts' not in gw.get(key, {}):
                continue
            wp = gw[key]['wpnts']
            pts = np.array([[w['x_m'], w['y_m']] for w in wp])
            lv = [w['d_left'] for w in wp]
            rv = [w['d_right'] for w in wp]
            ok, sw, amb = score(pts, lv, rv, outer, inner)
            verdicts[key] = (ok, sw, amb)
            flag = 'SWAPPED' if sw > ok else ('ok' if ok > sw else 'unclear')
            print(f'  {key:26s} correct {ok:4d}  swapped {sw:4d}  ambiguous {amb:3d}   -> {flag}')
    else:
        print('  global_waypoints.json: MISSING')

    csv_path = os.path.join(map_dir, 'centerline.csv')
    if os.path.exists(csv_path):
        header, rows = read_centerline(csv_path)
        pts = np.array([[float(r[0]), float(r[1])] for r in rows])
        rv = [float(r[2]) for r in rows]        # w_tr_right_m
        lv = [float(r[3]) for r in rows]        # w_tr_left_m
        ok, sw, amb = score(pts, lv, rv, outer, inner)
        verdicts['centerline.csv'] = (ok, sw, amb)
        flag = 'SWAPPED' if sw > ok else ('ok' if ok > sw else 'unclear')
        print(f'  {"centerline.csv":26s} correct {ok:4d}  swapped {sw:4d}  ambiguous {amb:3d}   -> {flag}')
    else:
        print('  centerline.csv: MISSING')

    tot_ok = sum(v[0] for v in verdicts.values())
    tot_sw = sum(v[1] for v in verdicts.values())
    swapped = tot_sw > 4 * max(tot_ok, 1)      # decisive: a real swap is near-total, not marginal
    if not do_fix:
        if swapped:
            print(f'  => VERDICT: bounds are SWAPPED ({tot_sw} vs {tot_ok}). Re-run with --fix, or '
                  f'regenerate with the correct start pose / reverse_mapping.')
        elif tot_sw > tot_ok:
            print(f'  => VERDICT: looks swapped but NOT decisively ({tot_sw} vs {tot_ok}); inspect '
                  f'before touching anything.')
        else:
            print('  => VERDICT: labelling is correct.')
        return swapped

    if not swapped:
        print('  --fix refused: the swap is not decisive, nothing written.')
        return False

    if gw is not None:
        shutil.copyfile(json_path, json_path + '.bak')
        for key in WPNT_KEYS:
            if key not in gw or 'wpnts' not in gw.get(key, {}):
                continue
            for w in gw[key]['wpnts']:
                w['d_left'], w['d_right'] = w['d_right'], w['d_left']
        with open(json_path, 'w') as fh:
            json.dump(gw, fh)
        print(f'  fixed {json_path}  (backup: global_waypoints.json.bak)')

    if os.path.exists(csv_path):
        shutil.copyfile(csv_path, csv_path + '.bak')
        header, rows = read_centerline(csv_path)
        with open(csv_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            if header:
                w.writerow(header)
            for r in rows:
                r = list(r)
                r[2], r[3] = r[3], r[2]
                w.writerow(r)
        print(f'  fixed {csv_path}  (backup: centerline.csv.bak)')

    print('  NOTE: trackbounds_markers in the json are raw points and stay geometrically correct; '
          'only the left/right labels were mirrored.')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map', nargs='?', help='map name under stack_master/maps/')
    ap.add_argument('--all', action='store_true', help='check every map')
    ap.add_argument('--fix', action='store_true', help='swap the labels back (writes .bak first)')
    a = ap.parse_args()
    if a.all:
        names = sorted(d for d in os.listdir(MAPS_DIR)
                       if os.path.isdir(os.path.join(MAPS_DIR, d)))
    elif a.map:
        names = [a.map]
    else:
        ap.error('give a map name or --all')
    if a.all and a.fix:
        ap.error('--fix takes one map at a time')
    bad = 0
    for n in names:
        try:
            if check_map(n, do_fix=a.fix):
                bad += 1
        except Exception as exc:                  # a map without a clean ring is not a failure here
            print(f'\n=== {n} ===\n  skipped: {exc}')
    return 1 if (bad and not a.fix) else 0


if __name__ == '__main__':
    sys.exit(main())
