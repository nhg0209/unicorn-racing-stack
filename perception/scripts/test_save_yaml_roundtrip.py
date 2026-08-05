#!/usr/bin/env python3
"""Does pressing "save params" in rqt keep the tracking keys the save routine does not know about?

save_yaml assigned a freshly built dict to data['tracking']['ros__parameters'], which deletes
every key that dict does not list. It did not list static_net_floor_m, livox_max_view_dist,
measure or from_bag, so one save dropped them from the yaml and the node silently fell back to its
code defaults on the next run -- and static_net_floor_m is the static/dynamic classification
floor, i.e. a race-day behaviour change nobody asked for and nothing reports.

  ~/miniforge3/envs/unicorn/bin/python3 perception/scripts/test_save_yaml_roundtrip.py
"""
import sys
import tempfile
import types
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MOD = REPO / "perception" / "scripts" / "multi_tracking.py"
SHIPPED = REPO / "stack_master" / "config" / "opponent_tracker_params.yaml"

src = MOD.read_text()
ns = {"__name__": "mt", "__file__": str(MOD)}
try:
    exec(compile(src, str(MOD), "exec"), ns)
except Exception as exc:                      # rclpy imports are fine, node construction is not
    if not ns.get("StaticDynamic"):
        raise SystemExit(f"could not load multi_tracking: {exc}")
StaticDynamic = ns["StaticDynamic"]
Opponent_state = ns["Opponent_state"]

# every tracking key the shipped yaml carries but the save dict does not build
SAVE_BLIND = ("static_net_floor_m", "livox_max_view_dist", "measure", "from_bag")


class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    warning = warn
    def error(self, *a, **k): raise AssertionError(a)
    def debug(self, *a, **k): pass


def node(path):
    # the class-level EKF tunables the save routine reads (set from the yaml at node startup)
    for k, v in (("P_vs", 0.2), ("P_d", 0.2), ("P_vd", 0.2), ("measurment_var_s", 0.005),
                 ("measurment_var_d", 0.005), ("measurment_var_vs", 0.2),
                 ("measurment_var_vd", 0.2), ("process_var_vs", 0.8), ("process_var_vd", 0.8)):
        setattr(Opponent_state, k, v)
    n = StaticDynamic.__new__(StaticDynamic)
    n.get_logger = lambda: _Log()
    n.save_yaml_path = str(path)
    n.rate = 40
    n.max_dist, n.var_pub = 1.5, 0.5
    n.aggro_multiplier = 2.0
    n.dist_deletion, n.dist_infront = 7.0, 8.0
    n.max_std, n.min_std, n.min_nb_meas = 0.2, 0.16, 6
    n.noMemoryMode, n.debug_mode, n.publish_static = False, False, True
    n.ratio_to_glob_path = 0.6
    n.ttl_dynamic, n.ttl_static = 40, 5
    n.vs_reset = 0.1
    n.demote_disp_m, n.demote_time_s = 0.5, 0.4
    n.static_speed_max_mps, n.demote_speed_mps = 0.4, 1.0
    n.near_dynamic_suppress_m = 0.5
    return n


def test_a_save_keeps_the_keys_the_save_routine_does_not_know_about():
    shipped = yaml.safe_load(SHIPPED.read_text())
    block = shipped["tracking"]["ros__parameters"]
    missing = [k for k in SAVE_BLIND if k not in block]
    assert not missing, f"the shipped yaml no longer carries {missing}; update this test"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "opponent_tracker_params.yaml"
        p.write_text(SHIPPED.read_text())
        n = node(p)
        n.save_yaml()
        after = yaml.safe_load(p.read_text())
        blk = after["tracking"]["ros__parameters"]
        lost = [k for k in SAVE_BLIND if k not in blk]
        assert not lost, (
            f"saving deleted {lost} from the yaml -- the node falls back to its code defaults on "
            f"the next run, silently")
        for k in SAVE_BLIND:
            assert blk[k] == block[k], f"{k} changed on save: {block[k]} -> {blk[k]}"
        # ...and the detect block, which this node does not own, is untouched
        assert after.get("detect") == shipped.get("detect"), "the detect block was rewritten"
        # ...and the keys it DOES own are written
        assert blk["min_nb_meas"] == 6 and blk["save_params"] is False
    print(f"PASS a save keeps {', '.join(SAVE_BLIND)} and leaves the detect block alone")


if __name__ == "__main__":
    test_a_save_keeps_the_keys_the_save_routine_does_not_know_about()
    print("ALL PASS")
