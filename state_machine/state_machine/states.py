from __future__ import annotations

from typing import List, TYPE_CHECKING

from f110_msgs.msg import Wpnt

if TYPE_CHECKING:
    from state_machine.state_machine_node import StateMachine

"""
Here we define the behaviour in the different states.
Every function should be fairly concise, and output an array of f110_msgs.Wpnt
"""


def GlobalTracking(state_machine: "StateMachine") -> List[Wpnt]:
    # Index with the LIVE spacing (wpnt_dist, refreshed from each /global_waypoints_scaled), not
    # the static 0.1 param: the obstacle-aware swapped line keeps the point COUNT but its humps
    # lengthen the lap, so its spacing is ~0.103 — dividing by 0.1 detached the local window
    # from the car by up to ~1 m near the lap end (broken-looking local path, far lookahead).
    s = int(state_machine.cur_s / state_machine.wpnt_dist + 0.5)
    # ...then RE-ANCHOR the index by POSITION. cur_s and this line can be in different frames for
    # up to half a second: on a static-reopt swap the frenet converter re-takes /global_waypoints
    # immediately while this window comes from /global_waypoints_scaled, which sector_tuner only
    # re-publishes on its 0.5 s timer. During that gap `s` indexes the new arc length with the old
    # station numbering (or vice versa) and the window slides ALONG the track away from the car.
    # Measured on the real car (bag ot_speed_0731_2034, t=14.94-15.24): the window sat 0.55 m
    # AHEAD of the car — longitudinally, lateral error was only 0.018 m — for 0.4 s, which is past
    # the controller's AEB_thres (0.5 m), so AEB clamped the command to 2.0 m/s mid-lap and then
    # stepped +1.24 m/s on release. Nearest-by-position cannot drift like that.
    s = state_machine.anchor_gb_index(s)
    return [state_machine.cur_gb_wpnts.list[(s + i) % state_machine.num_glb_wpnts]
            for i in range(state_machine.n_loc_wpnts)]


def Overtaking(state_machine: "StateMachine") -> List[Wpnt]:
    if state_machine.ot_planner == "spliner" or state_machine.ot_planner == "predictive_spliner":
        return state_machine.get_splini_wpts()
    else:
        s = state_machine.cur_id_ot
        return [
            state_machine.overtake_wpnts[(s + i) % state_machine.num_ot_points]
            for i in range(state_machine.n_loc_wpnts)
        ]


def RECOVERY(state_machine: "StateMachine"):
    return state_machine.get_recovery_wpts()


def START(state_machine: "StateMachine"):
    return state_machine.get_start_wpts()


def FTGOnly(state_machine: "StateMachine"):
    """No waypoints are generated in this follow the gap only state, all the
    control inputs are generated in the control node."""
    return []
