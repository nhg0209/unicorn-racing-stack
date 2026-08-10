#!/usr/bin/env python3
"""What the two avoidance planners were each doing when the car hit the wall.

Reads a `ros2 launch` console log and lines up, on one clock, the things that have to be compared
by hand otherwise: every lane-change engage (with the target's speed), every static-planner side
selection (with its corridor and keep-out), every SM path adoption, every saturated-steering cycle
and every crash event. Then it reports the two questions the round asked:

  P0  did the two planners ever claim the same obstacle in opposite directions, how big was the
      resulting reference step, and what did the steering do right after each SM planner swap
  P1  how many distinct tracker ids appeared, against how many boxes actually exist, and what
      happened immediately before each new id

WHAT THIS CANNOT ANSWER, and it is the reason a second run is needed: the console log carries a
tracker id only where some OTHER node happened to name one (a planner target, an SM free-check
block, a static-promotion message). There is no per-cycle tracker output in it at all, so per-id
lifetimes and the id-creation RATE are lower bounds, not measurements. /tracking/diag (diag_dynamic,
now with the `stat` array) is what closes that.

  ~/miniforge3/envs/unicorn/bin/python3 stack_master/scripts/analyze_run_log.py ~/run_0810_2102.log
"""
import argparse
import re
import sys
from collections import Counter, defaultdict

# [node-24] [INFO] [1786363382.981333609] [planner_change]: [LaneChange] IDLE -> ENTRY ...
LINE = re.compile(r"^\[([a-z_]+)-\d+\]\s+\[(\w+)\]\s+\[(\d+\.\d+)\]\s+\[([^\]]+)\]:\s*(.*)$")

ENGAGE = re.compile(
    r"IDLE -> (\w+) \(engage\): target id=(\d+) gap=([-\d.]+) m side=(\w+) "
    r"offset=([-\d.]+) m meet_in=([-\d.]+) m v_pass=([-\d.]+) v_opp=([-\d.]+)")
AVOID = re.compile(
    r"avoid (LEFT|RIGHT|RACELINE) d_end=([+\-\d.]+).*?corridor d=\[([-\d.]+),([-\d.]+)\] "
    r"\((\w+)\) obs keep-out d=\[([-\d.]+),([-\d.]+)\]")
ADOPT = re.compile(r"adopting fresh (\S+) path")
FEASIBLE = re.compile(r"static_feasible (\w+) -> (\w+) @ s=([\d.]+)")
IDMENT = re.compile(r"\bid=(\d+)\b")
CONFIRMED = re.compile(r"CONFIRMED static obstacle @\(([-\d.]+),([-\d.]+)\) r=([\d.]+)")
NOFEAS = re.compile(r"NO feasible candidate \((\d+) sampled\)")
PHASE = re.compile(r"\[LaneChange\] (\w+) -> (\w+)")


def rel(t):
    """The clock the crash report quotes: seconds mod 1000, so 1786363382.981 -> 382.981."""
    return t % 1000.0


def parse(path):
    ev = []
    for raw in open(path, errors="replace"):
        m = LINE.match(raw.strip())
        if not m:
            continue
        node, lvl, ts, logger, msg = m.groups()
        ev.append(dict(node=node, lvl=lvl, t=float(ts), logger=logger, msg=msg))
    return ev


def near(ev, t, pred, window, forward):
    """Events matching `pred` within `window` s before (forward=False) / after (forward=True) t."""
    out = []
    for e in ev:
        dt = e["t"] - t
        if (0 <= dt <= window) if forward else (-window <= dt < 0):
            if pred(e):
                out.append(e)
    return out


def p0(ev):
    engages, avoids, adopts = [], [], []
    for e in ev:
        m = ENGAGE.search(e["msg"])
        if m:
            engages.append(dict(t=e["t"], phase=m.group(1), id=int(m.group(2)),
                                gap=float(m.group(3)), side=m.group(4).lower(),
                                offset=float(m.group(5)), v_pass=float(m.group(7)),
                                v_opp=float(m.group(8))))
        m = AVOID.search(e["msg"])
        if m:
            avoids.append(dict(t=e["t"], side=m.group(1).lower(), d_end=float(m.group(2)),
                               cor=(float(m.group(3)), float(m.group(4))), src=m.group(5),
                               keep=(float(m.group(6)), float(m.group(7)))))
        m = ADOPT.search(e["msg"])
        if m:
            adopts.append(dict(t=e["t"], planner=m.group(1)))
    clipped = [e["t"] for e in ev if "steering angle clipped" in e["msg"]]
    crashes = [(e["t"], e["msg"][:60]) for e in ev
               if "iTTC wall check" in e["msg"] or "respawn" in e["msg"]
               or "collision" in e["msg"]]

    print("=" * 100)
    print("P0  the two planners, on one clock")
    print("=" * 100)
    print(f"\n--- lane_change engages: {len(engages)} ---")
    if engages:
        print(f"  {'t':>9}{'id':>5}{'gap':>7}{'side':>7}{'offset':>8}{'v_opp':>7}{'v_pass':>8}"
              f"   nearest static selection before it")
        for g in engages:
            prev = [a for a in avoids if a["t"] <= g["t"]]
            s = prev[-1] if prev else None
            if s is None:
                tag = "(none logged)"
            else:
                lane_d = g["offset"] if g["side"] == "left" else -g["offset"]
                step = abs(s["d_end"] - lane_d)
                agree = "SAME side" if (s["d_end"] > 0) == (lane_d > 0) else "OPPOSITE"
                tag = (f"{rel(s['t']):.3f} {s['side']:<5} d_end={s['d_end']:+.2f} "
                       f"-> {agree}, reference step {step:.2f} m")
            print(f"  {rel(g['t']):9.3f}{g['id']:5d}{g['gap']:7.1f}{g['side']:>7}"
                  f"{g['offset']:8.2f}{g['v_opp']:7.2f}{g['v_pass']:8.1f}   {tag}")
        vs = sorted(g["v_opp"] for g in engages)
        print(f"\n  v_opp of every engaged target: {vs}")
        print(f"    max {max(vs):.2f} m/s. The opponent vehicle was NOT spawned in this run, so "
              f"every one of these is a static box.")
        print(f"    engage_min_vs_mps = 0.35 refuses all of them; the closing-speed gate cannot, "
              f"because closing = ego - v_opp")
        print(f"    is LARGEST at v_opp = 0.")

    print(f"\n--- SM path adoptions: {len(adopts)} ---")
    by = Counter(a["planner"] for a in adopts)
    for k, v in by.most_common():
        print(f"  {k:<32} {v}")
    print(f"\n  {'t':>9}  {'planner adopted':<32}{'|steering| clipped within 2 s after':>36}")
    for a in adopts:
        n = len(near(ev, a["t"], lambda e: "steering angle clipped" in e["msg"], 2.0, True))
        flag = "   <-- saturated" if n >= 5 else ""
        print(f"  {rel(a['t']):9.3f}  {a['planner']:<32}{n:>36}{flag}")

    print(f"\n--- saturated steering: {len(clipped)} cycles ---")
    runs, cur = [], []
    for t in clipped:
        if cur and t - cur[-1] > 0.5:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    for r in sorted(runs, key=len, reverse=True)[:5]:
        print(f"  {len(r):3d} cycles  {rel(r[0]):.3f} -> {rel(r[-1]):.3f}  "
              f"({r[-1]-r[0]:.3f} s)")

    print(f"\n--- crash events: {len(crashes)} ---")
    for t, m in crashes:
        print(f"  {rel(t):9.3f}  {m}")
    print(f"\n--- static planner: {len(avoids)} selections, "
          f"{sum(1 for e in ev if NOFEAS.search(e['msg']))} 'NO feasible candidate' ---")
    sides = Counter(a["side"] for a in avoids)
    print(f"  sides chosen: {dict(sides)}")
    print(f"  NOTE the selection line is throttled (2.0 s by default, now avoid_log_throttle_s):")
    print(f"  {len(avoids)} lines over {ev[-1]['t']-ev[0]['t']:.0f} s of a 20 Hz planner is ~1 in "
          f"{max(1, int((ev[-1]['t']-ev[0]['t'])*20/max(len(avoids),1)))} cycles, so these are "
          f"SAMPLES, not the whole story.")


def p1(ev):
    print("\n" + "=" * 100)
    print("P1  tracker id churn (measurement only -- nothing is being tuned)")
    print("=" * 100)
    first, last, whonamed = {}, {}, defaultdict(Counter)
    for e in ev:
        for sid in IDMENT.findall(e["msg"]):
            i = int(sid)
            first.setdefault(i, e["t"])
            last[i] = e["t"]
            whonamed[i][e["logger"]] += 1
    boxes = [(e["t"], CONFIRMED.search(e["msg"]).groups()) for e in ev if CONFIRMED.search(e["msg"])]
    span = ev[-1]["t"] - ev[0]["t"]
    print(f"\n  distinct tracker ids NAMED anywhere in the log : {len(first)}  {sorted(first)}")
    print(f"  boxes CONFIRMED by static_obstacle_layer        : {len(boxes)}")
    for t, (x, y, r) in boxes:
        print(f"      {rel(t):9.3f}  @({x},{y}) r={r}")
    print(f"  log span                                        : {span:.1f} s")
    print(f"\n  ids per confirmed box: {len(first)/max(len(boxes),1):.1f}x  "
          f"-- a LOWER BOUND, see the header: only ids some other node happened to print are here.")
    print(f"\n  {'id':>5}{'first':>10}{'last':>10}{'lifetime':>10}   named by")
    for i in sorted(first):
        print(f"  {i:5d}{rel(first[i]):10.3f}{rel(last[i]):10.3f}"
              f"{last[i]-first[i]:10.2f}   " +
              ", ".join(f"{k}({v})" for k, v in whonamed[i].most_common()))
    print(f"\n  what happened in the 1.0 s before each id was first named:")
    for i in sorted(first):
        before = near(ev, first[i], lambda e: True, 1.0, False)
        interesting = [e for e in before
                       if PHASE.search(e["msg"]) or "re-projected" in e["msg"]
                       or "global line changed" in e["msg"] or NOFEAS.search(e["msg"])
                       or "static_feasible" in e["msg"] or "respawn" in e["msg"]]
        print(f"    id={i:<3} " + (
            "; ".join(f"{rel(e['t']):.3f} {e['logger']}: {e['msg'][:64]}"
                      for e in interesting[-2:]) or "(nothing notable logged)"))
    print(f"\n  static->dynamic reclassification messages the static planner printed:")
    for e in ev:
        if "reads ~0 speed" in e["msg"] or "treating dynamic-flagged" in e["msg"]:
            print(f"    {rel(e['t']):9.3f}  {e['msg'][e['msg'].find(']') + 2:][:96]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    a = ap.parse_args()
    ev = parse(a.log)
    if not ev:
        raise SystemExit(f"no parseable ros2 log lines in {a.log}")
    print(f"{len(ev)} log lines, {ev[-1]['t']-ev[0]['t']:.1f} s, "
          f"{len(set(e['node'] for e in ev))} nodes")
    p0(ev)
    p1(ev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
