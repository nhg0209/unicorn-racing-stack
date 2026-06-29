#!/usr/bin/env python3
from casadi import *
import numpy as np

from ._ros_compat import NullLogger, yaw_to_quat

g_ = 9.81

class MPC:
    def __init__(self, cost_type, system_model, logger=None):
        self._log = logger if logger is not None else NullLogger("[MPC-ipopt]")
        self.dT = None
        self.N = None
        self.L = None
        self.v_max = None
        self.v_min = None
        self.theta_max = None
        self.theta_min = None
        self.s_min, self.s_max = None, None
        self.p_min = None
        self.p_max = None
        self.x_min, self.x_max, self.y_min, self.y_max, self.psi_min, self.psi_max = None, None, None, None, None, None
        self.n_states, self.n_controls, self.T_V = None, None, None
        self.f = None
        self.U = None
        self.P = None
        self.X = None
        self.obj = 0
        self.X0 = None  # initial estimate for the states solution
        self.u0 = None  # initial estimate for the controls solution
        self.g = []
        self.Q = None
        self.R = None
        self.opts = {}
        self.param = {}
        self.lbg, self.ubg = None, None
        self.lbx, self.ubx = None, None
        self.nlp = None
        self.solver = None
        self.X_OBST = 3
        self.inf = 1e6
        self.SIDE_DECISION = 1
        self.is_ot = False

        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l = None
        self.WARM_START = False
        self.INTEGRATION_MODE = "Euler"  # RK4 and RK3 method are the other two choices
        self.p_initial = 2.5  # projected centerline vel can set to desired value for initial estimation
        self.boundary_hook = None  # callable(right_pts, left_pts) or None
        # Debug values populated each solve()
        self.dbg_n_obs_input = 0
        self.dbg_sel_dmin = float('inf')
        self.dbg_sel_x = float('inf')
        self.dbg_sel_y = float('inf')
        self.dbg_side_pref = 0.0
        self.dbg_solver_status = ""

    def setup_MPC(self):
        self.init_system_model()
        self.init_constraints()
        self.compute_optimization_cost()
        self.init_ipopt_solver()
        self.init_mpc_start_conditions()

    def init_system_model(self):
        # States
        x = MX.sym('x')
        y = MX.sym('y')
        psi = MX.sym('psi')
        s = MX.sym('s')

        # Controls
        v = MX.sym('v')
        theta = MX.sym('theta')
        p = MX.sym('p')

        states = vertcat(x, y, psi, s)
        controls = vertcat(v, theta, p)
        self.n_states = states.size1()
        self.n_controls = controls.size1()
        self.T_V = self.n_states + self.n_controls
        rhs = vertcat(v * cos(psi), v * sin(psi), (v / self.L) * tan(theta), p)  # dynamic equations of the states
        
        self.f = Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = MX.sym('U', self.n_controls, self.N)
        # Slack variables for the soft-relaxed hard obstacle constraint.
        # One per prediction step. Lower-bounded at 0 in lbx; appears as a
        # quadratic penalty in the cost so the optimizer keeps it ≈ 0 when
        # avoidance is feasible, and only inflates it when truly blocked.
        self.SLACK = MX.sym('SLACK', self.N, 1)

        # Live-tunable weights live at the tail of P (5 extra slots).
        # Order: [q_cte, q_lag, q_d_delta, R_safe, M_slack].
        # ROS node fills these per cycle via solve() so they can change
        # without an NLP rebuild — letting rqt_reconfigure tweak them live.
        self._n_live = 5
        self.P = MX.sym('P', self.n_states + 2 * self.N + self.X_OBST
                                       + self.SIDE_DECISION + self._n_live)
        self._live_base = (self.n_states + 2 * self.N + self.X_OBST
                                       + self.SIDE_DECISION)
        self._idx_q_cte     = self._live_base + 0
        self._idx_q_lag     = self._live_base + 1
        self._idx_q_d_delta = self._live_base + 2
        self._idx_R_safe    = self._live_base + 3
        self._idx_M_slack   = self._live_base + 4

        self.X = MX.sym('X', self.n_states, (self.N + 1))

        self.Q = MX.zeros(2, 2)
        self.Q[0, 0] = self.param['mpc_w_cte']  # cross track error
        self.Q[1, 1] = self.param['mpc_w_lag']  # lag error

        self.S = MX.zeros(3, 3)
        self.S[0, 0] = self.param['mpc_w_accel']  # change in velocity i.e, acceleration
        self.S[1, 1] = self.param['mpc_w_delta_d']  # change in steering angle. weighing matrices (change in controls)
        self.S[2, 2] = self.param['mpc_w_delta_vp'] # change in vp i.e, vp acceleration

        self.mpc_vp_project = self.param['mpc_vp_project']
        self.obj = 0  # Objective function
        self.g = []  # constraints vector

    def set_initial_params(self, param, vheid, is_ot):
        '''Set initial parameters related to MPC'''
        self.vheid = vheid
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        self.L = param['L']
        self.theta_max, self.v_max = param['theta_max'], param['v_max']
        self.p_initial = self.v_max
        self.theta_min = -self.theta_max
        # Forbid reverse / near-zero forward — MPCC formulation is rotation-
        # invariant in (x,y) when v_min < 0, which lets IPOPT pick degenerate
        # circular detours around obstacles. Keeping v strictly positive
        # forces forward motion so the only feasible avoidance is a true
        # lateral detour.
        self.v_min = 0.5
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.INTEGRATION_MODE = param['INTEGRATION_MODE']
        self.mpc_v_track = param['mpc_v_track']
        self.Vbias_max = param["Vbias_max"]
        self.is_ot = is_ot

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y, element_arc_lengths, original_arc_length_total, ref_v):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total
        self.ref_v = ref_v

    def compute_optimization_cost(self):
        st = self.X[:, 0]  # initial state
        self.g = vertcat(self.g, st - self.P[0:self.n_states])  # initial condition constraints

        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            ################## get ref msg ##################
            ref_v = self.ref_v(st_next[3])
            ################## get ref msg ##################

            dx, dy = self.center_lut_dx(st_next[3]), self.center_lut_dy(st_next[3])
            t_angle = atan2(dy, dx)
            ref_x, ref_y = self.center_lut_x(st_next[3]), self.center_lut_y(st_next[3])
            # Contouring error
            e_c = (sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)) / 0.5
            # Lag error
            e_l = (-cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)) / 0.5

            # Distance to currently-tracked obstacle (used by both the barrier
            # cost AND the adaptive lane-tracking weight just below).
            obs_x_idx = self.n_states + 2 * self.N + 1
            pos = st_next[0:2]
            obs = self.P[obs_x_idx: obs_x_idx + 2]
            diff_0 = pos - obs
            dist2_0 = mtimes(diff_0.T, diff_0)
            # Tuned barrier shape: zero beyond 3 m (no premature slowdown),
            # ramps up cleanly inside 2 m, dominates inside 1 m.
            sigma_obs = 1.0

            # Adaptive lane-tracking weight: lane error penalty drops to 5%
            # at obstacle center so the MPC can really commit to a detour
            # without paying full q_cte. With sigma=1.0:
            #   attenuation = 1 - 0.95 * exp(-d^2 / 2)
            #     d=∞ -> 1.00   (normal)
            #     d=2 -> 0.87
            #     d=1 -> 0.42   (lane cost ~halved)
            #     d=0.5 -> 0.16
            #     d=0   -> 0.05 (lane cost essentially OFF — detour free)
            attenuation = 1.0 - 0.95 * exp(-dist2_0 / (2.0 * sigma_obs * sigma_obs))
            # LIVE: q_cte, q_lag pulled from P (rqt-tunable without rebuild)
            self.obj = self.obj + attenuation * (
                self.P[self._idx_q_cte] * e_c * e_c
                + self.P[self._idx_q_lag] * e_l * e_l
            )

            # ---- Side-preference cost (very soft hint) ----
            # Multiplied by |side_pref| so when decide_side_pref returns 0
            # (no pref) this cost vanishes entirely — was forcing e_c=0
            # (=line tracking) on top of obstacle slack, which made the
            # NLP infeasible whenever a detour required leaving the line.
            side_pref_idx = self.n_states + 2 * self.N + self.X_OBST + self.SIDE_DECISION - 1
            side_pref_p = self.P[side_pref_idx]
            sigma_side = 0.6
            proximity_side = exp(-dist2_0 / (2.0 * sigma_side * sigma_side))
            D_DETOUR = 0.30
            W_SIDE   = 3.0
            side_target = side_pref_p * D_DETOUR
            self.obj = self.obj + W_SIDE * fabs(side_pref_p) * proximity_side * (e_c - side_target) ** 2

            # Heading cost — Liniger MPCC's q_μ. Penalize (psi - tangent_ref).
            # Wrap-safe form 4·sin²((Δψ)/2) ≈ Δψ² for small angles, bounded
            # for any wrap. Mild weight so it complements contour/lag cost
            # rather than fights it. Pulls car heading onto the line so
            # the MPC can't get stuck with yaw rotated 60° off raceline.
            yaw_err = st_next[2] - t_angle
            self.obj = self.obj + self.param.get('q_mu', 5.0) * 4.0 * sin(yaw_err / 2.0) ** 2

            self.obj = self.obj - self.mpc_vp_project * (con[2] / self.p_max) * self.dT

            # Per-step reference velocity tracking — always on (was OT-only off).
            self.obj = self.obj + (((con[0] - ref_v) / self.Vbias_max) ** 2) * self.mpc_v_track

            # NOTE: soft Gaussian barrier removed (caused premature slowdown
            # because the soft cost makes "stay away by slowing down so the
            # horizon never reaches the obstacle" cheaper than "detour
            # laterally"). Slack-relaxed hard constraint below provides
            # all the avoidance behavior needed.

            # Lateral-acceleration penalty (a_lat = v² tan(δ)/L) — also
            # opt-in via w_alat. Default off because it competes with
            # obstacle avoidance.
            a_lat_max = self.param.get('a_lat_max', 8.0)
            w_alat = self.param.get('w_alat', 0.0)
            if w_alat > 0:
                a_lat = con[0] * con[0] * tan(con[1]) / self.L
                excess = sqrt((fabs(a_lat) - a_lat_max) ** 2 + 1e-3) + (fabs(a_lat) - a_lat_max)
                self.obj = self.obj + w_alat * 0.25 * excess ** 2

            # delta u — LIVE: q_d_delta from P (steer rate weight). q_dv (accel
            # smoothness, S[0,0]) and S[2,2] still come from the cached self.S.
            if k < self.N - 1:
                con_next = self.U[:, k + 1]
                dv_v   = con_next[0] - con[0]
                dv_th  = con_next[1] - con[1]
                dv_p   = con_next[2] - con[2]
                self.obj += (self.S[0, 0] * dv_v * dv_v
                             + self.P[self._idx_q_d_delta] * dv_th * dv_th
                             + self.S[2, 2] * dv_p * dv_p)

            k1 = self.f(st, con)
            if self.INTEGRATION_MODE == "Euler":
                st_next_euler = st + (self.dT * k1)
            elif self.INTEGRATION_MODE == "RK3":
                k2 = self.f(st + self.dT / 2 * k1, con)
                k3 = self.f(st + self.dT * (2 * k2 - k1), con)
                st_next_euler = st + self.dT / 6 * (k1 + 4 * k2 + k3)
            elif self.INTEGRATION_MODE == "RK4":
                k2 = self.f(st + self.dT / 2 * k1, con)
                k3 = self.f(st + self.dT / 2 * k2, con)
                k4 = self.f(st + self.dT * k3, con)
                st_next_euler = st + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            self.g = vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = vertcat(self.g, self.P[self.n_states + 2 * k] * st_next[0] - self.P[self.n_states + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes

            # Slack-relaxed HARD obstacle constraint (Brito et al. 2019).
            # dist^2 - R_safe^2 + slack[k] >= 0, slack[k] >= 0.
            # Cost penalty M * slack[k]^2 keeps slack ≈ 0 in normal traffic,
            # so the constraint behaves like a true hard "stay outside R_safe"
            # circle. Only when truly infeasible does slack inflate to keep
            # IPOPT alive — avoidance is then "as close to safe as possible".
            obs_x_idx = self.n_states + 2 * self.N + 1
            obs_pos = self.P[obs_x_idx: obs_x_idx + 2]
            d2 = (st_next[0] - obs_pos[0]) ** 2 + (st_next[1] - obs_pos[1]) ** 2
            # LIVE: R_safe & M_slack from P (rqt-tunable without rebuild)
            R_safe_p = self.P[self._idx_R_safe]
            self.g = vertcat(self.g, d2 - R_safe_p * R_safe_p + self.SLACK[k])
            self.obj = self.obj + self.P[self._idx_M_slack] * self.SLACK[k] ** 2

            # Path-velocity coupling: p (arc-length growth rate) must equal v
            # (vehicle longitudinal speed). EVO-MPCC's original formulation
            # leaves them independent, which lets IPOPT find degenerate
            # solutions where (x,y) loops while s grows linearly — the
            # "circular detour" we observed. Enforcing p=v ties s growth to
            # actual displacement, so loops increase contour/lag error.
            self.g = vertcat(self.g, con[2] - con[0])

    def init_ipopt_solver(self):
        # Optimization variables: States + Controls + Slack (per-step obstacle slack)
        OPT_variables = vertcat(reshape(self.X, self.n_states * (self.N + 1), 1),
                                reshape(self.U, self.n_controls * self.N, 1),
                                self.SLACK)
        self.opts["ipopt"] = {}
        self.opts["ipopt"]["max_iter"] = 80
        self.opts["ipopt"]["max_cpu_time"] = float(self.param.get('ipopt_max_cpu_time', 0.10))
        self.opts["ipopt"]["print_level"] = 0
        self.opts["ipopt"]["sb"] = "yes"
        self.opts["verbose"] = False
        # JIT + EXPAND: SX expansion + native C compile. Biggest IPOPT speedup.
        self.opts["expand"] = True
        self.opts["jit"] = True
        self.opts["compiler"] = "shell"
        self.opts["jit_options"] = {"flags": ["-O3", "-march=native", "-pipe"], "verbose": False}
        self.opts["print_time"] = 0
        # Tighter tolerance — was 1e-3 (too loose), causing the solver to
        # accept unconverged steps that triggered v↓ in corners. 5e-4 lets
        # IPOPT finish convergence without ballooning solve time too much.
        self.opts["ipopt"]["acceptable_tol"] = 5e-4
        self.opts["ipopt"]["acceptable_obj_change_tol"] = 1e-4
        self.opts["ipopt"]["acceptable_iter"] = 3
        self.opts["ipopt"]["fixed_variable_treatment"] = "make_parameter"
        self.opts["ipopt"]["mu_strategy"] = "adaptive"
        self.opts["ipopt"]["warm_start_init_point"] = "yes"
        self.opts["ipopt"]["warm_start_bound_push"] = 1e-9
        self.opts["ipopt"]["warm_start_mult_bound_push"] = 1e-9
        # Linear solver: try MUMPS (default ma27 is OK; ma57 better but
        # often not built). Override via 'ipopt_linear_solver' ROS param.
        self.opts["ipopt"]["linear_solver"] = self.param.get('ipopt_linear_solver', 'mumps')
        # Nonlinear problem formulation with solver initialization
        self.nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        # g layout (interleaved per-step):
        #   indices 0..3                : initial state (4 entries, =0)
        #   per k in 0..N-1: 7 entries
        #     offset 0..3 within step: dynamics       -> [0, 0]
        #     offset 4 within step: corridor          -> set in solve()
        #     offset 5 within step: obstacle slack    -> [0, +inf)
        #     offset 6 within step: p=v coupling      -> [0, 0]
        n_per_step = self.n_states + 3  # 4 dyn + 1 corridor + 1 obstacle + 1 p=v
        n_g = self.n_states + n_per_step * self.N
        self.lbg = np.zeros((n_g, 1))
        self.ubg = np.zeros((n_g, 1))
        # Set obstacle slack constraint upper bound to +inf (per step).
        for k in range(self.N):
            obs_idx = self.n_states + n_per_step * k + (self.n_states + 1)
            self.ubg[obs_idx, 0] = np.inf
        # x bounds: states + controls + slack
        n_x = self.n_states * (self.N + 1) + self.n_controls * self.N + self.N
        self.lbx = np.zeros((n_x, 1))
        self.ubx = np.zeros((n_x, 1))

        for k in range(self.N + 1):
            self.lbx[self.n_states * k:self.n_states * (k + 1), 0] = np.array(
                [[self.x_min, self.y_min, self.psi_min, self.s_min]])
            self.ubx[self.n_states * k:self.n_states * (k + 1), 0] = np.array(
                [[self.x_max, self.y_max, self.psi_max, self.s_max]])
        state_count = self.n_states * (self.N + 1)
        # Upper and lower bounds for the control optimization variables
        for k in range(self.N):
            self.lbx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_min, self.theta_min, self.p_min]])
            self.ubx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_max, self.theta_max, self.p_max]])
            state_count += self.n_controls
        # Slack bounds: [0, +inf) per step.
        slack_start = self.n_states * (self.N + 1) + self.n_controls * self.N
        for k in range(self.N):
            self.lbx[slack_start + k, 0] = 0.0
            self.ubx[slack_start + k, 0] = np.inf
        
    def init_mpc_start_conditions(self):
        self.u0 = np.zeros((self.N, self.n_controls))
        self.X0 = np.zeros((self.N + 1, self.n_states))

    def get_angle_at_centerline(self, s):
        dx, dy = self.center_lut_dx(s), self.center_lut_dy(s)
        return np.arctan2(dy, dx)

    def get_point_at_centerline(self, s):
        return self.center_lut_x(s), self.center_lut_y(s)

    def get_path_constraints_points(self, prev_soln):
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        for k in range(1, self.N + 1):
            right_points[k - 1, :] = np.array([self.right_lut_x(prev_soln[k, 3]),
                                      self.right_lut_y(prev_soln[k, 3])], dtype=object).squeeze()  # Right boundary
            left_points[k - 1, :] = np.array([self.left_lut_x(prev_soln[k, 3]),
                                     self.left_lut_y(prev_soln[k, 3])], dtype=object).squeeze()  # Left boundary

        return right_points, left_points

    def construct_warm_start_soln(self, initial_state):
        # CRITICAL: initial_state is passed by reference from solve(); do NOT
        # mutate index 2 (yaw). Earlier the code overwrote initial_state[2]
        # with the raceline tangent, which made the MPC believe the car was
        # always aligned with the raceline regardless of actual heading. The
        # leaked yaw then went into p[0:n_states] as the initial-condition
        # constraint, making every published control assume an aligned car.
        if initial_state[3] >= self.arc_lengths_orig_l:
            initial_state[3] -= self.arc_lengths_orig_l
        # Seed X0[0] with the TRUE current pose. Subsequent warm-start steps
        # can ride along the raceline tangent (reasonable initial guess for
        # the optimizer); the equality constraint X[:,0] == P[0:4] then ties
        # the actual solution to the real pose.
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            init_speed = self.p_initial/10
            s_next = self.X0[k - 1, 3] + init_speed * self.dT
            psi_next = self.get_angle_at_centerline(s_next)
            x_next, y_next = self.get_point_at_centerline(s_next)
            phi_dot = (psi_next - self.X0[k - 1, 2]) / self.dT
            theta_init = atan2((phi_dot * self.vheid["l_wb"]), init_speed)
            self.X0[k, :] = np.array([x_next, y_next, psi_next, s_next], dtype=object)
            
    def filter_estimate(self, initial_arc_pos):
        if (self.X0[0, 3] >= self.arc_lengths_orig_l) and ((initial_arc_pos >= self.arc_lengths_orig_l) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.arc_lengths_orig_l
        # every time of the vehicle pos should be adjust
        if initial_arc_pos >= self.arc_lengths_orig_l:
            initial_arc_pos -= self.arc_lengths_orig_l
        return initial_arc_pos

    def solve(self, initial_state, obstacles):
        p = np.zeros(self.n_states + 2 * self.N + self.X_OBST
                              + self.SIDE_DECISION + self._n_live)
        # Fill LIVE-tunable trailing slots from current self.* attributes.
        # Defaults seeded by node from yaml; rqt updates them in-place.
        p[self._idx_q_cte]     = float(getattr(self, 'q_cte_live',     1.0))
        p[self._idx_q_lag]     = float(getattr(self, 'q_lag_live',     300.0))
        p[self._idx_q_d_delta] = float(getattr(self, 'q_d_delta_live', 80.0))
        p[self._idx_R_safe]    = float(getattr(self, 'R_safe_live',    0.7))
        p[self._idx_M_slack]   = float(getattr(self, 'M_slack_live',   1.0e4))
        
        delta_yaw = self.X0[1, 2] - initial_state[2]
        if abs(delta_yaw) >= np.pi:
            new_val_ceil = initial_state[2] + np.ceil(delta_yaw / (2 * np.pi)) * (2 * np.pi)
            new_val_floor = initial_state[2] + np.floor(delta_yaw / (2 * np.pi)) * (2 * np.pi)
            if abs(new_val_ceil - self.X0[1, 2]) < abs(new_val_floor - self.X0[1, 2]):
                initial_state[2] = new_val_ceil
            else:
                initial_state[2] = new_val_floor
        if not self.WARM_START:
            self.X0 = np.zeros((self.N + 1, self.n_states))
            self._log.info("Warm start started")
            self.construct_warm_start_soln(initial_state)
            self._log.info("Warm start accomplished")

        # Stuck-detection. The MPC is "stuck" if its commanded speed is
        # well below the local reference speed — typically because IPOPT
        # is trapped in a local minimum where slowing down is cheaper
        # than committing to a detour. Re-seed warm start with ref_v so
        # IPOPT explores the high-v region next iteration.
        try:
            ref_v_now = float(self.ref_v(initial_state[3] if self.n_states == 4 else initial_state[6]))
        except Exception:
            ref_v_now = self.v_max
        # Stuck reseed: only fire when truly stalled (15 % of ref_v); use the
        # ACTUAL car speed as the seed so IPOPT doesn't try to leap from 0
        # straight to ref_v=5 — that big gap is what triggers the cascade
        # of infeasibles after a corner cut.
        v_actual_now = float(initial_state[3] if self.n_states == 4 else initial_state[6])
        if self.WARM_START and abs(float(self.u0[0, 0])) < 0.15 * ref_v_now:
            self.X0 = np.zeros((self.N + 1, self.n_states))
            self.construct_warm_start_soln(initial_state)
            seed_v = max(0.5, min(v_actual_now * 1.2, ref_v_now * 0.6))
            self.u0[:, 0] = seed_v
            if self.n_controls >= 3:
                self.u0[:, 2] = seed_v

        initial_state[3] = self.filter_estimate(initial_state[3])
        p[0:self.n_states] = initial_state  # initial condition of the robot posture
        right_points, left_points = self.get_path_constraints_points(self.X0)
        # print(f"right_points: {right_points}, left_points: {left_points}")
        select_front_obstacle_result = self.select_front_obstacle(initial_state[0], initial_state[1], initial_state[2], obstacles, D_max=12.0, D_min_trig=10.0) # should be [dmin, x_obs, y_obs]
        # Stash for /mpc_debug
        self.dbg_n_obs_input = len(obstacles) if obstacles is not None else 0
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = select_front_obstacle_result
        p[self.n_states + 2 * self.N:self.n_states + 2 * self.N + self.X_OBST] = np.array(select_front_obstacle_result)
        obs_choosen = p[self.n_states + 2 * self.N + 1: self.n_states + 2 * self.N + self.X_OBST] # should be [x_obs, y_obs]
        side_pref = self.decide_side_pref(obs_choosen, left_points, right_points)
        self.dbg_side_pref = float(side_pref)
        p[self.n_states + 2 * self.N + self.X_OBST + self.SIDE_DECISION - 1] = side_pref

        # ---- Avoidance prior on warm-start X0 ----
        # First time the car meets an obstacle, the warm-start X0 holds a
        # straight-ahead (no-detour) trajectory. Seeding it with a lateral
        # offset in the side_pref direction within ~3 m of the obstacle
        # gives IPOPT a feasible avoidance trajectory to start from. Without
        # this, lap-1 obstacle avoidance often fails because the solver
        # can't jump from straight X0 to a detour solution in one outer
        # iteration. Subsequent cycles already have a curved X0.
        sel_dmin = float(select_front_obstacle_result[0])
        if self.n_states == 4 and side_pref != 0 and sel_dmin < 4.0:
            ox, oy = obs_choosen[0], obs_choosen[1]
            # Smaller offset (0.3 m, was 0.6 m) — large jumps were dynamically
            # infeasible at 5 m/s and caused IPOPT to declare infeasible.
            D_DETOUR_PRIOR = 0.30
            for k in range(1, self.N + 1):
                xk, yk, psik = self.X0[k, 0], self.X0[k, 1], self.X0[k, 2]
                d = ((xk - ox) ** 2 + (yk - oy) ** 2) ** 0.5
                w = np.exp(-(d * d) / (2.0 * 1.0 * 1.0))
                nx, ny = -np.sin(psik), np.cos(psik)
                self.X0[k, 0] += side_pref * D_DETOUR_PRIOR * w * nx
                self.X0[k, 1] += side_pref * D_DETOUR_PRIOR * w * ny

        self.publish_boundary_markers(right_points, left_points)

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            delta_x_path = right_points[k, 0] - left_points[k, 0]
            delta_y_path = right_points[k, 1] - left_points[k, 1]
            p[self.n_states + 2 * k:self.n_states + 2 * k + 2] = [-delta_x_path, delta_y_path]
            up_bound = max(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                           -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            low_bound = min(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                            -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            # Corridor entry index in g: 4 (initial) + 7k + 4 (offset within step)
            corridor_idx = self.n_states + (self.n_states + 3) * k + self.n_states
            self.lbg[corridor_idx, 0] = low_bound
            self.ubg[corridor_idx, 0] = up_bound
        
        # Initial guess includes zero slack (assume currently feasible)
        slack_init = np.zeros((self.N, 1))
        x_init = vertcat(reshape(self.X0.T, self.n_states * (self.N + 1), 1),
                         reshape(self.u0.T, self.n_controls * self.N, 1),
                         slack_init)

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        opti_value = sol['f'].full().item()
        try:
            self.dbg_solver_status = self.solver.stats().get('return_status', '?')
        except Exception:
            self.dbg_solver_status = '?'

        # Extract state, control, and slack from solution.
        n_X = self.n_states * (self.N + 1)
        n_U = self.n_controls * self.N
        self.X0 = reshape(sol['x'][0:n_X], self.n_states, self.N + 1).T
        u = reshape(sol['x'][n_X:n_X + n_U], self.n_controls, self.N).T
        # Slack solution available at sol['x'][n_X+n_U: ] for diagnostics if needed.
        con_first = u[0, :].T
        trajectory = self.X0.full()  # size is (N+1, n_states)
        inputs = u.full()
        self.X0 = vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = vertcat(u[1:, :], u[u.size1() - 1, :])

        return con_first, trajectory, inputs, opti_value

    def heading(self, yaw):
        """yaw → (x, y, z, w) tuple. Caller wraps into geometry_msgs/Quaternion."""
        return yaw_to_quat(yaw)

    def publish_boundary_markers(self, right_points, left_points):
        """Forward boundary points to an injected hook (no-op if not set).

        ROS-agnostic core: marker assembly + publish lives in the wrapper.
        Hook signature: hook(right_points: ndarray, left_points: ndarray).
        """
        if self.boundary_hook is None:
            return
        self.boundary_hook(right_points, left_points)

    def decide_side_pref(self, obstacle_pos, left_points, right_points, margin=0.1):
        """
        Decide which side (left or right) the vehicle should overtake from,
        based on the obstacle's Euclidean distance to both lane boundaries.

        Parameters
        ----------
        obstacle_pos : tuple (x, y)
            The obstacle center position in global coordinates.
        left_points : np.ndarray, shape = (N, 2)
            Array of points representing the left lane boundary.
        right_points : np.ndarray, shape = (N, 2)
            Array of points representing the right lane boundary.
        margin : float
            If the absolute difference between left/right average distances 
            is smaller than this threshold, no preference is returned (0).

        Returns
        -------
        side_pref : int
            +1 → Prefer overtaking on the left side  
            -1 → Prefer overtaking on the right side  
            0 → Nearly equal spacing (keep current direction or no preference)
        """

        x_o, y_o = obstacle_pos

        if x_o > 1e5 and y_o > 1e5:
            # No obstacle detected
            return 0

        # Compute Euclidean distance from obstacle to each boundary point
        dist_L = np.sqrt((left_points[:, 0] - x_o)**2 + (left_points[:, 1] - y_o)**2)
        dist_R = np.sqrt((right_points[:, 0] - x_o)**2 + (right_points[:, 1] - y_o)**2)

        # Take the smallest two distances (closest boundary points) to smooth noise
        top2_L = np.sort(dist_L)[:2] if len(dist_L) >= 2 else dist_L
        top2_R = np.sort(dist_R)[:2] if len(dist_R) >= 2 else dist_R
        mean_L = float(np.mean(top2_L))
        mean_R = float(np.mean(top2_R))

        # ---- USER RULES ----
        # 1) If one side is too narrow to fit the car (W_car + 6 cm safety)
        #    → must go the OTHER way.
        # 2) Otherwise, if widths are roughly equal (within `margin`)
        #    → default to RIGHT (-1).
        # 3) Otherwise pick the wider side.
        W_CAR_SAFE = 0.21   # car_half_width(~0.075) + clearance ~0.13 ≈ 0.21
        left_blocked  = mean_L < W_CAR_SAFE
        right_blocked = mean_R < W_CAR_SAFE
        if left_blocked and not right_blocked:
            return -1   # only right viable
        if right_blocked and not left_blocked:
            return +1   # only left viable
        if left_blocked and right_blocked:
            return -1   # both bad — default right (user pref)
        diff = mean_L - mean_R
        if abs(diff) <= margin:
            return -1   # ties go RIGHT (user pref)
        return +1 if diff > 0 else -1


    def select_front_obstacle(self, curr_x, curr_y, curr_yaw,
                          obstacles, D_max=5.0, D_min_trig=2.0):
        """
        Select the nearest obstacle in front of the vehicle within a given range.

        Parameters
        ----------
        curr_x, curr_y, curr_yaw : float
            Current vehicle position and heading (in world coordinates).
        obstacles : list of (x_o, y_o)
            List of obstacle centers (and optionally radius if available).
        D_max : float
            Maximum distance threshold. Obstacles farther than this are ignored.
        D_min_trig : float
            "Trigger distance" threshold. Obstacles closer than this are considered
            immediate and returned directly.

        Returns
        -------
        tuple :
            (d_min, x_o_min, y_o_min)
                If a valid obstacle is found ahead of the vehicle.  
            (self.inf, self.inf, self.inf)
                If no valid obstacle exists within range or field of view.
        """

        pos = np.array([curr_x, curr_y])
        t = np.array([np.cos(curr_yaw), np.sin(curr_yaw)])  # Vehicle forward direction

        candidates = []

        for (x_o, y_o) in obstacles:
            obs = np.array([x_o, y_o])
            v = obs - pos
            dist = np.linalg.norm(v)

            # 1. Distance filter: ignore obstacles that are too far
            if dist > D_max:
                continue

            # 2. Only consider obstacles in front of the vehicle (dot product > 0)
            if np.dot(t, v) <= 0.0:
                continue

            candidates.append((dist, x_o, y_o))

        if len(candidates) == 0:
            # No obstacle found
            return self.inf, self.inf, self.inf

        # Sort by distance (ascending)
        candidates.sort(key=lambda c: c[0])
        d_min, x_min, y_min = candidates[0]

        # If the nearest obstacle is within the trigger distance, return it;
        # otherwise, treat as no obstacle.
        if d_min < D_min_trig:
            return d_min, x_min, y_min
        else:
            return self.inf, self.inf, self.inf

