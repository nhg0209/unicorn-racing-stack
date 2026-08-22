#!/usr/bin/env python3
"""Does the track diagnostic stay silent when off, and never throw when on?

publish_diag runs inside timer_callback, so an exception in it takes the tracker down with it --
and it is reached on every cycle, including cycles where no new detect array arrived and where a
track has no KF yet. This drives it against real Opponent_state/ObstacleSD objects (a real
filterpy EKF, not a stub) and checks the contract the report needs from it: RAW x[0], not
x[0] % track_length, and `matched` reported as unknown rather than as last cycle's answer on a
predict-only cycle.

  ~/miniforge3/envs/unicorn/bin/python3 perception/scripts/test_diag_publish.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "perception" / "scripts" / "multi_tracking.py"

src = MOD.read_text()
ns = {"__name__": "mt", "__file__": str(MOD)}
try:
    exec(compile(src, str(MOD), "exec"), ns)
except Exception as exc:                      # rclpy/msg imports are fine, node construction is not
    if not ns.get("StaticDynamic"):
        raise SystemExit(f"could not load multi_tracking: {exc}")
StaticDynamic = ns["StaticDynamic"]
Opponent_state = ns["Opponent_state"]
ObstacleSD = ns["ObstacleSD"]

RATE = 40.0
TRACK_LEN = 38.4


class _Log:
    def info(self, *a, **k): pass
    warn = warning = debug = info

    def error(self, *a, **k): raise AssertionError(a)


class _Clock:
    def now(self):
        return type("T", (), {"nanoseconds": 1_000_000_000})()


class _Pub:
    def __init__(self): self.sent = []

    def publish(self, msg): self.sent.append(msg.data)


def configure():
    Opponent_state.rate, Opponent_state.dt = RATE, 1.0 / RATE
    Opponent_state.process_var_vs = Opponent_state.process_var_vd = 0.8
    Opponent_state.measurment_var_s = Opponent_state.measurment_var_d = 0.005
    Opponent_state.measurment_var_vs = Opponent_state.measurment_var_vd = 0.2
    Opponent_state.track_length = TRACK_LEN
    ObstacleSD.ttl = 5


def track(tid, s, vs, static_flag, initialised, matched=True):
    configure()                              # tracks are built before node(), so configure here
    o = ObstacleSD(id=tid, s_meas=s, d_meas=0.0, lap=0, size=0.4, isVisible=True, t_meas=0.0)
    o.staticFlag = static_flag
    o.nb_meas = 12
    o.matched = matched
    o.dynamic_state.isInitialised = initialised
    if initialised:
        o.dynamic_state.id = tid
        o.dynamic_state.ttl = 40
        o.dynamic_state.dynamic_kf.x = [s, vs, 0.0, 0.0]
    return o


def node(tracks, on=True, fresh=True):
    configure()
    n = StaticDynamic.__new__(StaticDynamic)
    n.get_logger = lambda: _Log()
    n.get_clock = lambda: _Clock()
    n.diag_pub = _Pub()
    n.diag_dynamic = on
    n._diag_k, n._diag_fresh, n._diag_last_s = 0, fresh, {}
    n.track_length = TRACK_LEN
    n.tracked_obstacles = tracks
    n.meas_obstacles = [None, None]
    return n


def test_a_off_by_default_publishes_nothing():
    n = node([track(1, 5.0, 3.0, False, True)], on=False)
    n.publish_diag()
    assert n.diag_pub.sent == [], "the diagnostic published while disabled"
    # and it is off in the shipped yaml
    import yaml
    y = yaml.safe_load((REPO / "stack_master" / "config" / "opponent_tracker_params.yaml").read_text())
    assert y["tracking"]["ros__parameters"]["diag_dynamic"] is False, "shipped enabled"


def test_b2_the_static_tracks_that_become_keep_outs_are_reported_too():
    """The static planner turns CONFIRMED-STATIC tracks into keep-outs, so tracing a swerve back to
    the track that produced the box needs those tracks -- with the position publishObstacles
    actually publishes for them, which is obs.mean and not the KF."""
    conf = track(2, 10.0, 0.0, True, False)
    conf.mean = [10.25, -0.37]
    unc = track(3, 12.0, 0.0, None, False)
    n = node([track(1, 5.0, 3.0, False, True), conf, unc])
    n.publish_diag()
    rec = json.loads(n.diag_pub.sent[-1])
    assert [r["id"] for r in rec["dyn"]] == [1]
    assert [r["id"] for r in rec["stat"]] == [2, 3], f"static side missing: {rec['stat']}"
    r = rec["stat"][0]
    # mean_d is what the keep-out is built from; match it against the planner's own
    # `obs keep-out d=[lo,hi]` log line at the same instant
    assert r["mean_s"] == 10.25 and r["mean_d"] == -0.37, r
    for k in ("sf", "nb", "m", "vis", "size"):
        assert k in r, f"{k} missing -- it is what says ghost or real"
    assert r["sf"] is True and rec["stat"][1]["sf"] is None
    assert r["s"] is None and r["vs"] is None, "a static track has no KF state to report"


def test_b_only_dynamic_tracks_and_the_raw_s_is_reported():
    # 42.5 is PAST the track length: publish_Marker and /tracking/obstacles both show 4.1 there,
    # which is the whole reason this instrument exists.
    tracks = [track(1, 42.5, 3.0, False, True),
              track(2, 10.0, 0.0, True, False),        # confirmed static: not a dynamic track
              track(3, 12.0, 0.0, None, False)]        # unclassified: not a dynamic track
    n = node(tracks)
    n.publish_diag()
    rec = json.loads(n.diag_pub.sent[-1])
    assert [r["id"] for r in rec["dyn"]] == [1], f"wrong tracks reported: {rec['dyn']}"
    assert [r["id"] for r in rec["stat"]] == [2, 3], f"static side: {rec['stat']}"
    r = rec["dyn"][0]
    assert r["s"] == 42.5, f"s was wrapped to {r['s']}; the raw state is the point"
    assert r["s"] % rec["L"] != r["s"], "pick a test s that actually exceeds the track length"
    for k in ("id", "dyn_id", "s", "ds", "vs", "d", "Pss", "m", "sf", "nb", "ttl", "dttl",
              "init", "avs", "mean_s", "mean_d", "size", "vis"):
        assert k in r, f"{k} missing from the record"
    assert rec["k"] == 1 and rec["fresh"] is True and rec["nm"] == 2 and rec["ntrk"] == 3


def test_c_ds_measures_the_per_cycle_advance():
    o = track(1, 5.0, 3.0, False, True)
    n = node([o])
    n.publish_diag()
    assert json.loads(n.diag_pub.sent[-1])["dyn"][0]["ds"] is None, "no previous sample to diff"
    o.dynamic_state.dynamic_kf.x[0] = 5.075        # one cycle of 3 m/s at 40 Hz
    n.publish_diag()
    r = json.loads(n.diag_pub.sent[-1])["dyn"][0]
    assert abs(r["ds"] - 0.075) < 1e-9, r["ds"]
    assert r["s"] == 5.075


def test_d_a_predict_only_cycle_reports_matched_as_unknown():
    # update() returns early when no new detect array arrived, so `matched` still holds the
    # previous cycle's answer. Reporting it as if it were this cycle's would be a lie in exactly
    # the log the "does an unmatched track keep predicting" question is asked from.
    n = node([track(1, 5.0, 3.0, False, True, matched=True)], fresh=False)
    n.publish_diag()
    assert json.loads(n.diag_pub.sent[-1])["dyn"][0]["m"] is None
    n = node([track(1, 5.0, 3.0, False, True, matched=False)], fresh=True)
    n.publish_diag()
    assert json.loads(n.diag_pub.sent[-1])["dyn"][0]["m"] is False


def test_e_a_dynamic_track_without_a_kf_does_not_throw():
    # staticFlag False and isInitialised False is a real state (ttl death, vs gate rejection)
    o = track(1, 5.0, 0.0, False, False)
    n = node([o])
    n.publish_diag()
    r = json.loads(n.diag_pub.sent[-1])["dyn"][0]
    assert r["init"] is False and r["dyn_id"] is None and r["dttl"] is None
    # filterpy's untouched x is (4, 1), so float(x[0]) raises rather than returning a number:
    # the state is meaningless here and must be reported as such, not read out shape-first
    for k in ("s", "ds", "vs", "d", "Pss"):
        assert r[k] is None, f"{k} reported {r[k]} off an uninitialised filter"
    n.track_length = None                    # before the first global path
    n.publish_diag()
    assert json.loads(n.diag_pub.sent[-1])["L"] is None


def test_f_dead_tracks_stop_being_remembered():
    o = track(1, 5.0, 3.0, False, True)
    n = node([o])
    n.publish_diag()
    assert set(n._diag_last_s) == {1}
    n.tracked_obstacles = []
    n.publish_diag()
    assert n._diag_last_s == {}, "the ds bookkeeping grows for every id the tracker ever made"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
