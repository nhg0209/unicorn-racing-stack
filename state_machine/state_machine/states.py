from __future__ import annotations

from typing import List, TYPE_CHECKING

from f110_msgs.msg import Wpnt

if TYPE_CHECKING:
    from state_machine.state_machine import StateMachine

"""
Here we define the behaviour in the different states.
Every function should be fairly concise, and output an array of f110_msgs.Wpnt
"""


def GlobalTracking(state_machine: "StateMachine") -> List[Wpnt]:
    # Start the local window at the waypoint whose arc-length is closest to the
    # car's s. The old `cur_s / waypoints_dist` index assumes a uniformly spaced
    # raceline (ROS1 used 0.1 m); a non-uniform raceline made the window start
    # several metres AHEAD of the car, so Pure Pursuit cut corners into the wall.
    wl = state_machine.cur_gb_wpnts.list
    n = state_machine.num_glb_wpnts
    cur_s = state_machine.cur_s
    s = min(range(n), key=lambda i: abs(wl[i].s_m - cur_s))
    return [wl[(s + i) % n] for i in range(state_machine.n_loc_wpnts)]


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
