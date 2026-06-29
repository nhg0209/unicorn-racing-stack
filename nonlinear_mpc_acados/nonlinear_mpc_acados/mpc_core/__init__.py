"""mpc_core — ROS-free MPCC implementations (acados + IPOPT backends).

Original source: unicorn-racing-stack/evo_mpcc/nonlinear_mpc_casadi/scripts/
Ported here as pure Python so the same MPC class can be driven from any
runtime (rclpy, plain script, unit tests). ROS coupling lives in the
wrapper layer (ros2_ws/src/nonlinear_mpc_acados).

Usage:
    from mpc_core.acados_kinematic import MPC as AcadosMPC
    mpc = AcadosMPC(cost_type=None, system_model=None, logger=my_logger)
    mpc.set_initial_params(param_dict, vheid_dict, is_ot=False)
    mpc.set_track_data(...)
    mpc.setup_MPC()
    u_first, traj, u_seq, cost = mpc.solve(initial_state, obstacles)

Boundary visualization: instead of publishing RViz markers internally, set
`mpc.boundary_hook = callable(shifted_points)` to receive the corridor
points each cycle. `shifted_points` is a list of `(x, y)` tuples.
"""
from ._ros_compat import NullLogger, monotonic_now, yaw_to_quat

__all__ = ["NullLogger", "monotonic_now", "yaw_to_quat"]
