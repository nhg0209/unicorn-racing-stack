from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from state_machine.states_types import StateType

if TYPE_CHECKING:
    from state_machine.state_machine_node import StateMachine

"""
Transitions should loosely follow the following template (basically a match-case)

if (logic sum of bools obtained by methods of state_machine):
    return StateType.<DESIRED STATE>
elif (e.g. state_machine.obstacles are near):
    return StateType.<ANOTHER DESIRED STATE>
...

NOTE: ideally put the most common cases on top of the match-case

NOTE 2: notice that, when implementing new states, if an attribute/condition in the
    StateMachine is not available, your IDE will tell you, but only if you have a smart
    enough IDE.

NOTE 3: transitions must not have side effects on the state machine!
    i.e. any attribute of the state machine should not be modified in the transitions.
    (overtaking_ttl_elapsed_sec is now updated by the node in _update_overtake_ttl, not here.
     StartTransition still flips cur_start_wpnts.is_init - a remaining ROS1 carry-over.)
"""


def GlobalTrackingTransition(state_machine: "StateMachine", close_to_raceline=None) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.GB_TRACK`"""
    if close_to_raceline is None:
        close_to_raceline = state_machine._check_close_to_raceline()

    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        return ObstacleTransition(state_machine, close_to_raceline)


def RecoveryTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.RECOVERY`"""
    recovery_sustainability = state_machine._check_sustainability(
        state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts
    )
    close_to_raceline = (
        state_machine._check_close_to_raceline(state_machine.recovery_exit_d_m)
        * state_machine._check_close_to_raceline_heading(20)
    )

    if recovery_sustainability and not close_to_raceline:
        return StateType.RECOVERY, StateType.RECOVERY

    return GlobalTrackingTransition(state_machine, close_to_raceline)


def TrailingTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.TRAILING`"""
    close_to_raceline = (
        state_machine._check_close_to_raceline(state_machine.recovery_exit_d_m)
        * state_machine._check_close_to_raceline_heading(20)
    )
    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        if state_machine._check_ftg():
            return StateType.FTGONLY, StateType.FTGONLY
        return ObstacleTransition(state_machine, close_to_raceline)


def OvertakingTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.OVERTAKE` (pure: the ttl clock is updated by the node
    in `_update_overtake_ttl`, not here)."""
    ot_sustainability = state_machine._check_overtaking_mode_sustainability()
    enemy_in_front = state_machine._check_enemy_in_front()
    # Stay in OVERTAKE while the path is still sustainable AND either an enemy is directly ahead or
    # the ttl latch still has budget (keeps overtaking briefly after the enemy clears -> anti-chatter).
    # The latch is wall-clock: overtaking_ttl_sec of no-enemy time, not overtaking_ttl_sec * rate
    # cycles, so a slow loop shortens neither the budget nor the anti-chatter it buys.
    if ot_sustainability and (
        enemy_in_front or state_machine.overtaking_ttl_elapsed_sec < state_machine.overtaking_ttl_sec
    ):
        return StateType.OVERTAKE, StateType.OVERTAKE
    close_to_raceline = (
        state_machine._check_close_to_raceline(state_machine.recovery_exit_d_m)
        * state_machine._check_close_to_raceline_heading(20)
    )
    return GlobalTrackingTransition(state_machine, close_to_raceline)


def StartTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.START`"""
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)

    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        close_to_raceline = (
            state_machine._check_close_to_raceline(state_machine.recovery_exit_d_m)
            * state_machine._check_close_to_raceline_heading(20)
        )
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine, close_to_raceline)


def FTGOnlyTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.FTGONLY`"""
    close_to_raceline = (
        state_machine._check_close_to_raceline(state_machine.recovery_exit_d_m)
        * state_machine._check_close_to_raceline_heading(20)
    )
    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
            return StateType.GB_TRACK, StateType.GB_TRACK

        # Same RECOVERY entry gate as the other transitions (this one had no lateral gate at all,
        # so FTGONLY handed over to the recovery spline whenever a path happened to be available).
        if state_machine._check_line_lost():
            recovery_availability = state_machine._check_latest_wpnts(
                state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts
            )
            if recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts):
                return StateType.RECOVERY, StateType.RECOVERY

        if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
            return StateType.OVERTAKE, StateType.OVERTAKE
        else:
            return StateType.FTGONLY, StateType.FTGONLY


##################################################################################################################
##################################################################################################################


def NonObstacleTransition(state_machine: "StateMachine", close_to_raceline) -> Tuple[StateType, StateType]:
    if close_to_raceline:
        return StateType.GB_TRACK, StateType.GB_TRACK

    # Same entry gate as ObstacleTransition: only reach for the recovery spline once the raceline
    # is actually lost, not merely because the caller's exit threshold was missed. LOSTLINE below
    # is resolved back to GB_TRACK by the node in the same cycle, so the in-between band
    # (recovery_exit_d_m <= |d| < recovery_entry_d_m) keeps global-tracking.
    if state_machine._check_line_lost():
        if state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
            if state_machine._check_on_spline(state_machine.cur_recovery_wpnts):
                return StateType.RECOVERY, StateType.RECOVERY

    # DELIBERATE, AND IT COSTS SOMETHING -- read this before using the state string to debug.
    # The pair is (reported, driven) = (LOSTLINE, GB_TRACK), and the node resolves cur_state back
    # to GB_TRACK at the end of the SAME cycle, before it publishes. So:
    #   * LOSTLINE NEVER APPEARS on /state_machine. Confirmed on the car: bags
    #     rosbag2_2026_08_19-22_38_39 and -22_42_37, 13590 state messages, zero LOSTLINE. Do not
    #     go looking for it in a bag to find out whether the car lost the line -- it is not there,
    #     and its absence is not evidence.
    #   * it is still a committed transition while it lasts, so _commit_state stamps
    #     _last_transition_time. Sitting in the in-between band (recovery_exit_d_m <= |d| <
    #     recovery_entry_d_m) therefore re-arms min_dwell_sec every min_dwell_sec, which can hold
    #     off a genuinely wanted dwell-gated switch (RECOVERY) by up to that long.
    # Left as is on purpose: collapsing it to a plain GB_TRACK return would change the dwell
    # bookkeeping, and that is a behaviour change, not a notation fix.
    return StateType.LOSTLINE, StateType.GB_TRACK


def ObstacleTransition(state_machine: "StateMachine", close_to_raceline) -> Tuple[StateType, StateType]:
    if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
        return StateType.GB_TRACK, StateType.GB_TRACK

    # RECOVERY entry is judged on _check_line_lost() (recovery_entry_d_m), NOT on the inverse of
    # close_to_raceline: that flag carries the tight per-state exit hysteresis, so keying entry off
    # it made the bar tighter while trailing than while global-tracking. See _check_line_lost.
    if state_machine._check_line_lost():
        recovery_availability = state_machine._check_latest_wpnts(
            state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts
        )
        if recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts):
            return StateType.RECOVERY, StateType.RECOVERY

    if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
        return StateType.OVERTAKE, StateType.OVERTAKE

    # Dropping a static OVERTAKE while the car is still OUT on the avoidance hump used to snap the
    # reference straight back to the raw raceline — which, for a static obstacle, is the line that
    # runs INTO it. The controller steered toward the obstacle while the gap PID braked for it.
    # Hold the avoidance geometry until the car is genuinely back near the line, so the drop is a
    # deceleration and not a swerve toward the thing being avoided. The predicate is conservative
    # (off the line AND still on the cached path) and side-effect free.
    if state_machine._hold_static_avoidance_reference():
        return StateType.TRAILING, StateType.OVERTAKE

    # Otherwise TRAILING follows the global raceline. The recovery spline is for rejoining the line
    # when we are off it AND the path is free (the RECOVERY return above) — it is not a trailing
    # reference. The old `elif recovery_availability -> TRAILING, RECOVERY` branch was the common
    # case, not the exception: the callers gate close_to_raceline at recovery_exit_d_m, and while
    # that was hardcoded at 0.05 m -- below normal tracking error -- trailing latched onto the
    # recovery line and followed it away from the raceline, keeping cur_d large and the branch
    # self-sustaining.
    return StateType.TRAILING, StateType.GB_TRACK
