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
    (overtaking_ttl_count is now updated by the node in _update_overtake_ttl, not here.
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
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)

    if recovery_sustainability and not close_to_raceline:
        return StateType.RECOVERY, StateType.RECOVERY

    return GlobalTrackingTransition(state_machine, close_to_raceline)


def TrailingTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.TRAILING`"""
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        if state_machine._check_ftg():
            return StateType.FTGONLY, StateType.FTGONLY
        return ObstacleTransition(state_machine, close_to_raceline)


def OvertakingTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.OVERTAKE` (pure: the ttl counter is updated by the node
    in `_update_overtake_ttl`, not here)."""
    ot_sustainability = state_machine._check_overtaking_mode_sustainability()
    enemy_in_front = state_machine._check_enemy_in_front()
    # Stay in OVERTAKE while the path is still sustainable AND either an enemy is directly ahead or
    # the ttl latch still has budget (keeps overtaking briefly after the enemy clears -> anti-chatter).
    if ot_sustainability and (
        enemy_in_front or state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold
    ):
        return StateType.OVERTAKE, StateType.OVERTAKE
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
    return GlobalTrackingTransition(state_machine, close_to_raceline)


def StartTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.START`"""
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)

    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        close_to_raceline = (
            state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
        )
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine, close_to_raceline)


def FTGOnlyTransition(state_machine: "StateMachine") -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.FTGONLY`"""
    close_to_raceline = state_machine._check_close_to_raceline(0.05) * state_machine._check_close_to_raceline_heading(20)
    if len(state_machine.cur_obstacles_in_interest) == 0:
        return NonObstacleTransition(state_machine, close_to_raceline)
    else:
        if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
            return StateType.GB_TRACK, StateType.GB_TRACK

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

    if state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
        if state_machine._check_on_spline(state_machine.cur_recovery_wpnts):
            return StateType.RECOVERY, StateType.RECOVERY

    return StateType.LOSTLINE, StateType.GB_TRACK


def ObstacleTransition(state_machine: "StateMachine", close_to_raceline) -> Tuple[StateType, StateType]:
    recovery_availability = False
    if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
        return StateType.GB_TRACK, StateType.GB_TRACK

    if not close_to_raceline:
        recovery_availability = state_machine._check_latest_wpnts(
            state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts
        )
        if recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts):
            return StateType.RECOVERY, StateType.RECOVERY

    if state_machine._check_overtaking_mode() or state_machine._check_static_overtaking_mode():
        return StateType.OVERTAKE, StateType.OVERTAKE
    else:
        if close_to_raceline:
            return StateType.TRAILING, StateType.GB_TRACK
        elif recovery_availability:
            return StateType.TRAILING, StateType.RECOVERY
        elif state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
            return StateType.TRAILING, StateType.GB_TRACK
        else:
            return StateType.TRAILING, StateType.GB_TRACK
