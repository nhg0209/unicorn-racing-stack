#!/usr/bin/env python3
"""
Trajectory Optimizer Node — 2-stage IQP -> SP (ported from IFAC2026 global_planner).

Reads maps/<map>/centerline.csv (+ boundary_{right,left}.csv) produced by
centerline_extractor and writes maps/<map>/global_waypoints.json in the unicorn
gb_optimizer schema (same keys readwrite_global_waypoints.read_global_waypoints
expects), so the existing global_trajectory_publisher republishes it unchanged.

Vehicle + algorithm parameters come from stack_master/config/<racecar_version>/
racecar_f110.ini and .../veh_dyn_info/ (CAR by default, per project setup).

ROS parameters:
  map_name          str   map folder under stack_master/maps/
  racecar_version   str   config folder under stack_master/config/ (default CAR)
  safety_width_iqp  float vehicle safety width for IQP [m]  (overrides ini width_opt)
  safety_width_sp   float vehicle safety width for SP  [m]  (overrides ini width_opt)
  enable_check_traj bool  run post-optimisation sanity checks
  enable_mintime    bool  also run opt_mintime (CasADi; off by default)
  safety_width_mintime float

Output json (consumed by global_trajectory_publisher):
  global_traj_wpnts_iqp / _sp, centerline_waypoints, trackbounds_markers,
  est_lap_time, map_info_str (+ matching marker arrays).

2-pass note: the optimised raceline's d_left/d_right are filled from the REAL
wall boundary CSVs (boundary_{right,left}.csv, extracted from the original map),
NOT the _modi virtual walls — so downstream avoidance/frenet see the true track.
"""

import configparser
import csv
import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from .tph import prep_track  # noqa: F401  (local flat copy — registers tph alias)
from . import tph

_HERE = os.path.dirname(os.path.realpath(__file__))
# opt_mintime_traj lives in gb_optimizer's vendored TUM lib (CasADi; lazy-imported).
_TUM_LIB_PATH = os.path.join(_HERE, 'global_racetrajectory_optimization')
if _TUM_LIB_PATH not in sys.path:
    sys.path.insert(0, _TUM_LIB_PATH)


PARAMS = {
    'map_name':             '',
    'racecar_version':      'CAR',     # config folder under stack_master/config/
    # safety widths default to <0 = "use racecar_f110.ini optim_opts_*.width_opt".
    # Pass a value > 0 (launch arg / CLI) to override the ini per-run.
    'safety_width_iqp':     -1.0,
    'safety_width_sp':      -1.0,
    'enable_check_traj':    True,
    'enable_mintime':       False,
    'safety_width_mintime': 0.70,
}

_DEFAULT_IMP_OPTS = {
    'flip_imp_track': False,
    'set_new_start': False,
    'new_start': [0.0, 0.0],
    'min_track_width': 0.8,
    'num_laps': 1,
}


def _resolve_map_dir(map_name: str) -> str:
    maps_dir = os.path.join(get_package_share_directory('stack_master'), 'maps', map_name)
    for probe in (map_name + '.yaml', map_name + '.png', map_name + '.pgm'):
        p = os.path.join(maps_dir, probe)
        if os.path.islink(p) or os.path.exists(p):
            return os.path.dirname(os.path.realpath(p))
    return maps_dir


class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        if not self.map_name:
            self.get_logger().error('map_name parameter is required!')
            return

        self.map_dir = _resolve_map_dir(self.map_name)
        self.get_logger().info(f'map_dir: {self.map_dir}')

        # ini + veh_dyn come from stack_master/config/<racecar_version>/ (CAR default)
        cfg_dir = os.path.join(
            get_package_share_directory('stack_master'), 'config', self.racecar_version)
        self._ini_path = os.path.join(cfg_dir, 'racecar_f110.ini')
        self._veh_dyn_dir = os.path.join(cfg_dir, 'veh_dyn_info')
        self.get_logger().info(f'config: {self._ini_path}')

        self.pars = self._load_pars()
        # safety widths: ini optim_opts_*.width_opt is the source; ROS param overrides (> 0)
        if self.safety_width_iqp <= 0:
            self.safety_width_iqp = float(self.pars['optim_opts_mincurv']['width_opt'])
        if self.safety_width_sp <= 0:
            self.safety_width_sp = float(self.pars['optim_opts_sp']['width_opt'])
        self.get_logger().info(
            f'safety_width_iqp={self.safety_width_iqp} (ini width_opt unless overridden)  '
            f'safety_width_sp={self.safety_width_sp}  racecar_version={self.racecar_version}')

        try:
            self.run()
        except Exception as exc:
            import traceback
            self.get_logger().error(f'Optimization failed: {exc}\n{traceback.format_exc()}')

    # ─────────────────────────────────────────────────────────────────────────
    # ini loading
    # ─────────────────────────────────────────────────────────────────────────
    def _load_pars(self) -> dict:
        parser = configparser.ConfigParser()
        if not parser.read(self._ini_path):
            raise FileNotFoundError(f'racecar_f110.ini not found: {self._ini_path}')

        g = 'GENERAL_OPTIONS'
        o = 'OPTIMIZATION_OPTIONS'

        def _get(sec, key, default=None):
            try:
                return json.loads(parser.get(sec, key))
            except (configparser.NoOptionError, configparser.NoSectionError):
                return default

        pars: dict = {}
        pars['ggv_file']             = _get(g, 'ggv_file', 'ggv.csv')
        pars['ax_max_machines_file'] = _get(g, 'ax_max_machines_file', 'ax_max_machines.csv')
        pars['stepsize_opts']        = _get(g, 'stepsize_opts')
        pars['reg_smooth_opts']      = _get(g, 'reg_smooth_opts')
        pars['veh_params']           = _get(g, 'veh_params')
        pars['vel_calc_opts']        = _get(g, 'vel_calc_opts')
        pars['curv_calc_opts']       = _get(g, 'curv_calc_opts')
        # imp_opts is not present in unicorn's racecar_f110.ini -> sensible default
        pars['imp_opts']             = _get(g, 'imp_opts', dict(_DEFAULT_IMP_OPTS))
        pars['optim_opts_mincurv']   = _get(o, 'optim_opts_mincurv')
        pars['optim_opts_sp']        = _get(o, 'optim_opts_shortest_path')
        pars['optim_opts_mintime']   = _get(o, 'optim_opts_mintime', {})
        pars['vehicle_params_mintime'] = _get(o, 'vehicle_params_mintime', {})
        pars['tire_params_mintime']  = _get(o, 'tire_params_mintime', {})
        pars['pwr_params_mintime']   = _get(o, 'pwr_params_mintime', {})

        # ini optim_opts_*.width_opt is the safety-width source; ROS params override only when > 0.
        if self.safety_width_iqp > 0:
            pars['optim_opts_mincurv']['width_opt'] = self.safety_width_iqp
        if self.safety_width_sp > 0:
            pars['optim_opts_sp']['width_opt'] = self.safety_width_sp
        if pars['optim_opts_mintime']:
            if self.safety_width_mintime > 0:
                pars['optim_opts_mintime']['width_opt'] = self.safety_width_mintime
            vp = pars['vehicle_params_mintime']
            if 'wheelbase_front' in vp and 'wheelbase_rear' in vp:
                vp['wheelbase'] = vp['wheelbase_front'] + vp['wheelbase_rear']

        # Load GGV + ax_max_machines from stack_master/config/<version>/veh_dyn_info/
        pars['ggv'], pars['ax_max_machines'] = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=os.path.join(self._veh_dyn_dir, pars['ggv_file']),
            ax_max_machines_import_path=os.path.join(self._veh_dyn_dir, pars['ax_max_machines_file']),
        )

        return pars

    # ─────────────────────────────────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        csv_path = os.path.join(self.map_dir, 'centerline.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'Centerline CSV not found: {csv_path}\nRun centerline_extractor first.')

        reftrack_imp = self._load_centerline(csv_path)
        self.get_logger().info(f'Centerline loaded: {len(reftrack_imp)} points')

        bound_r, bound_l = self._load_boundaries()
        if bound_r is not None:
            self.get_logger().info(
                f'Boundaries: right={len(bound_r)}, left={len(bound_l)} points')

        self.get_logger().info('=== Preparing track ===')
        reftrack_interp, normvec_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
            tph.prep_track.prep_track(
                reftrack_imp=reftrack_imp,
                reg_smooth_opts=self.pars['reg_smooth_opts'],
                stepsize_opts=self.pars['stepsize_opts'],
                debug=False,
                min_width=self.pars['imp_opts']['min_track_width'],
            )
        self.get_logger().info(f'prep_track done: {len(reftrack_interp)} points')

        spline_lengths_interp = tph.calc_spline_lengths.calc_spline_lengths(
            coeffs_x=coeffs_x_interp, coeffs_y=coeffs_y_interp)
        psi_interp, kappa_interp, dkappa_interp = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_interp,
            coeffs_y=coeffs_y_interp,
            ind_spls=np.arange(coeffs_x_interp.shape[0]),
            t_spls=np.zeros(coeffs_x_interp.shape[0]),
            calc_curv=True,
            calc_dcurv=True,
        )

        self.get_logger().info('=== Running mincurv_iqp ===')
        traj_iqp, lap_iqp, reftrack_iqp, normvec_iqp = self._run_iqp(
            reftrack_interp, normvec_interp, a_interp,
            spline_lengths_interp, psi_interp, kappa_interp, dkappa_interp)
        if self.enable_check_traj and bound_r is not None:
            self._run_check('IQP', traj_iqp, bound_r, bound_l, self.safety_width_iqp)

        self.get_logger().info('=== Running shortest_path ===')
        traj_sp, lap_sp = self._run_sp(reftrack_iqp, normvec_iqp)

        iqp_json = os.path.join(self.map_dir, 'global_waypoints.json')
        sp_json = os.path.join(self.map_dir, 'shortest_path.json')
        self._save_json(traj_iqp, traj_sp, lap_iqp,
                        reftrack_interp, psi_interp, kappa_interp, spline_lengths_interp,
                        bound_r, bound_l)
        self.get_logger().info(f'json saved: {iqp_json}')
        self.get_logger().info(f'json saved: {sp_json}')

        if self.enable_check_traj and bound_r is not None:
            self._run_check('SP', traj_sp, bound_r, bound_l, self.safety_width_sp)

        if self.enable_mintime and self.pars['optim_opts_mintime']:
            self.get_logger().info('=== Prepping IQP track for mintime ===')
            reftrack_mt, normvec_mt, a_mt, coeffs_x_mt, coeffs_y_mt = \
                tph.prep_track.prep_track(
                    reftrack_imp=reftrack_iqp,
                    reg_smooth_opts=self.pars['reg_smooth_opts'],
                    stepsize_opts=self.pars['stepsize_opts'],
                    debug=False,
                    min_width=self.pars['imp_opts']['min_track_width'],
                )
            self.get_logger().info('=== Running opt_mintime ===')
            traj_mt, lap_mt = self._run_mintime(
                reftrack_mt, normvec_mt, a_mt, coeffs_x_mt, coeffs_y_mt)
            self._save_json(traj_mt, traj_sp, lap_mt,
                            reftrack_interp, psi_interp, kappa_interp, spline_lengths_interp,
                            bound_r, bound_l)
            if self.enable_check_traj and bound_r is not None:
                self._run_check('MinTime', traj_mt, bound_r, bound_l, self.safety_width_mintime)

        self.get_logger().info('=== Summary ===')
        self._log_stats('IQP', traj_iqp, lap_iqp)
        self._log_stats('SP ', traj_sp, lap_sp)
        if self.enable_mintime and self.pars['optim_opts_mintime']:
            self._log_stats('MinTime', traj_mt, lap_mt)
        self.get_logger().info('=== Done ===')

    # ─────────────────────────────────────────────────────────────────────────
    # IQP optimisation
    # ─────────────────────────────────────────────────────────────────────────
    def _run_iqp(self, reftrack_interp, normvec_interp, a_interp,
                 spline_lengths, psi, kappa, dkappa):
        t0 = time.perf_counter()

        alpha_opt, reftrack_iqp, normvec_iqp = tph.iqp_handler.iqp_handler(
            reftrack=reftrack_interp,
            normvectors=normvec_interp,
            A=a_interp,
            spline_len=spline_lengths,
            psi=psi,
            kappa=kappa,
            dkappa=dkappa,
            kappa_bound=self.pars['veh_params']['curvlim'],
            w_veh=self.safety_width_iqp,
            print_debug=False,
            plot_debug=False,
            stepsize_interp=self.pars['stepsize_opts']['stepsize_reg'],
            iters_min=self.pars['optim_opts_mincurv']['iqp_iters_min'],
            curv_error_allowed=self.pars['optim_opts_mincurv']['iqp_curverror_allowed'],
        )[0:3]

        traj, lap = self._build_trajectory(reftrack_iqp, normvec_iqp, alpha_opt)
        self.get_logger().info(
            f'[IQP] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap, reftrack_iqp, normvec_iqp

    # ─────────────────────────────────────────────────────────────────────────
    # SP optimisation
    # ─────────────────────────────────────────────────────────────────────────
    def _run_sp(self, reftrack_interp, normvec_interp):
        t0 = time.perf_counter()

        alpha_opt = tph.opt_shortest_path.opt_shortest_path(
            reftrack=reftrack_interp,
            normvectors=normvec_interp,
            w_veh=self.safety_width_sp,
            print_debug=False,
        )

        traj, lap = self._build_trajectory(reftrack_interp, normvec_interp, alpha_opt)
        self.get_logger().info(
            f'[SP ] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap

    # ─────────────────────────────────────────────────────────────────────────
    # opt_mintime — minimum lap time (CasADi/IPOPT, optional)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_mintime(self, reftrack_interp, normvec_interp, a_interp,
                     coeffs_x_interp, coeffs_y_interp):
        t0 = time.perf_counter()

        import opt_mintime_traj  # lazy: CasADi + sklearn only needed here

        export_path = os.path.join(self.map_dir, 'mintime_export')
        os.makedirs(export_path, exist_ok=True)

        pars_mt = dict(self.pars)
        pars_mt['optim_opts'] = dict(self.pars['optim_opts_mintime'])
        pars_mt['optim_opts']['var_friction'] = None
        pars_mt['optim_opts']['warm_start'] = False

        if pars_mt['optim_opts'].get('reopt_mintime_solution', False):
            opts = pars_mt['optim_opts']
            opts['width_opt'] = (opts['width_opt']
                                 + (opts['w_tr_reopt'] - opts['w_veh_reopt'])
                                 + opts['w_add_spl_regr'])

        alpha_opt, v_opt, reftrack_out, a_interp_out, normvec_out = \
            opt_mintime_traj.src.opt_mintime.opt_mintime(
                reftrack=reftrack_interp,
                coeffs_x=coeffs_x_interp,
                coeffs_y=coeffs_y_interp,
                normvectors=normvec_interp,
                pars=pars_mt,
                tpamap_path='',
                tpadata_path='',
                export_path=export_path,
                print_debug=True,
                plot_debug=False,
            )

        ref = reftrack_out if reftrack_out is not None else reftrack_interp
        norm = normvec_out if normvec_out is not None else normvec_interp

        if pars_mt['optim_opts'].get('reopt_mintime_solution', False):
            raceline_mt = ref[:, :2] + np.expand_dims(alpha_opt, 1) * norm
            w_tr_right_mt = ref[:, 2] - alpha_opt
            w_tr_left_mt = ref[:, 3] + alpha_opt
            racetrack_mt = np.column_stack((raceline_mt, w_tr_right_mt, w_tr_left_mt))

            ref_reopt, norm_reopt, a_reopt = tph.prep_track.prep_track(
                reftrack_imp=racetrack_mt,
                reg_smooth_opts=self.pars['reg_smooth_opts'],
                stepsize_opts=self.pars['stepsize_opts'],
                debug=False,
                min_width=self.pars['imp_opts']['min_track_width'],
            )[:3]

            w_tr_tmp = 0.5 * pars_mt['optim_opts']['w_tr_reopt'] * np.ones(ref_reopt.shape[0])
            racetrack_reopt = np.column_stack((ref_reopt[:, :2], w_tr_tmp, w_tr_tmp))

            alpha_opt = tph.opt_min_curv.opt_min_curv(
                reftrack=racetrack_reopt,
                normvectors=norm_reopt,
                A=a_reopt,
                kappa_bound=self.pars['veh_params']['curvlim'],
                w_veh=pars_mt['optim_opts']['w_veh_reopt'],
                print_debug=False,
                plot_debug=False,
            )[0]
            ref, norm = ref_reopt, norm_reopt

        raceline_interp, _, coeffs_x_opt, coeffs_y_opt, \
            spline_inds_opt, t_vals_opt, s_points_opt, \
            spline_lengths_opt, el_lengths_opt = \
            tph.create_raceline.create_raceline(
                refline=ref[:, :2],
                normvectors=norm,
                alpha=alpha_opt,
                stepsize_interp=self.pars['stepsize_opts']['stepsize_interp_after_opt'],
            )

        psi_vel, kappa = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_opt, coeffs_y=coeffs_y_opt,
            ind_spls=spline_inds_opt, t_spls=t_vals_opt)

        if pars_mt['optim_opts'].get('reopt_mintime_solution', False):
            vx_profile = tph.calc_vel_profile.calc_vel_profile(
                ggv=self.pars['ggv'], ax_max_machines=self.pars['ax_max_machines'],
                v_max=self.pars['veh_params']['v_max'], kappa=kappa,
                el_lengths=el_lengths_opt, closed=True,
                filt_window=self.pars['vel_calc_opts']['vel_profile_conv_filt_window'],
                dyn_model_exp=self.pars['vel_calc_opts']['dyn_model_exp'],
                drag_coeff=self.pars['veh_params']['dragcoeff'],
                m_veh=self.pars['veh_params']['mass'])
        else:
            s_splines = np.cumsum(spline_lengths_opt)
            s_splines = np.insert(s_splines, 0, 0.0)
            vx_profile = np.interp(s_points_opt, s_splines[:-1], v_opt)
            vx_profile = np.minimum(vx_profile, self.pars['veh_params']['v_max'])

        vx_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_cl, el_lengths=el_lengths_opt, eq_length_output=False)
        t_profile = tph.calc_t_profile.calc_t_profile(
            vx_profile=vx_profile, ax_profile=ax_profile, el_lengths=el_lengths_opt)

        traj = np.column_stack([
            s_points_opt, raceline_interp[:, 0], raceline_interp[:, 1],
            psi_vel, kappa, vx_profile, ax_profile])
        lap = float(t_profile[-1])
        self.get_logger().info(
            f'[MinTime] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap

    # ─────────────────────────────────────────────────────────────────────────
    # Shared post-optimisation pipeline (create_raceline → vel profile)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_trajectory(self, reftrack, normvec, alpha_opt):
        pars = self.pars

        raceline_interp, _, coeffs_x_opt, coeffs_y_opt, \
            spline_inds_opt, t_vals_opt, s_points_opt, \
            _, el_lengths_opt = \
            tph.create_raceline.create_raceline(
                refline=reftrack[:, :2],
                normvectors=normvec,
                alpha=alpha_opt,
                stepsize_interp=pars['stepsize_opts']['stepsize_interp_after_opt'],
            )

        psi_vel, kappa = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_opt, coeffs_y=coeffs_y_opt,
            ind_spls=spline_inds_opt, t_spls=t_vals_opt)

        vx_profile = tph.calc_vel_profile.calc_vel_profile(
            ggv=pars['ggv'], ax_max_machines=pars['ax_max_machines'],
            v_max=pars['veh_params']['v_max'], kappa=kappa,
            el_lengths=el_lengths_opt, closed=True,
            filt_window=pars['vel_calc_opts']['vel_profile_conv_filt_window'],
            dyn_model_exp=pars['vel_calc_opts']['dyn_model_exp'],
            drag_coeff=pars['veh_params']['dragcoeff'],
            m_veh=pars['veh_params']['mass'])

        vx_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_cl, el_lengths=el_lengths_opt, eq_length_output=False)
        t_profile = tph.calc_t_profile.calc_t_profile(
            vx_profile=vx_profile, ax_profile=ax_profile, el_lengths=el_lengths_opt)

        # columns: [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
        traj = np.column_stack([
            s_points_opt, raceline_interp[:, 0], raceline_interp[:, 1],
            psi_vel, kappa, vx_profile, ax_profile])
        return traj, float(t_profile[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # CSV I/O
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_centerline(csv_path: str) -> np.ndarray:
        """Load centerline.csv → (N, 4) reftrack_imp [x, y, w_right, w_left]."""
        rows = []
        with open(csv_path, 'r') as f:
            for r in csv.DictReader(f):
                rows.append([
                    float(r['x_m']), float(r['y_m']),
                    float(r.get('w_tr_right_m', 1.0)),
                    float(r.get('w_tr_left_m', 1.0)),
                ])
        if len(rows) < 10:
            raise ValueError(f'Centerline too short: {len(rows)} points')
        return np.array(rows)

    def _load_boundaries(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r_path = os.path.join(self.map_dir, 'boundary_right.csv')
        l_path = os.path.join(self.map_dir, 'boundary_left.csv')
        if not (os.path.exists(r_path) and os.path.exists(l_path)):
            self.get_logger().warn('[CheckTraj] boundary CSVs not found — validation disabled')
            return None, None
        return (np.loadtxt(r_path, delimiter=',', skiprows=1),
                np.loadtxt(l_path, delimiter=',', skiprows=1))

    @staticmethod
    def _dists_to_bounds(raceline_xy: np.ndarray,
                         bound_r: Optional[np.ndarray],
                         bound_l: Optional[np.ndarray]):
        """Per-raceline-point distance to the REAL right/left walls (-> d_right/d_left)."""
        if bound_r is None or bound_l is None:
            n = len(raceline_xy)
            return np.zeros(n), np.zeros(n)
        d_right = np.array([np.min(np.linalg.norm(bound_r - pt, axis=1)) for pt in raceline_xy])
        d_left = np.array([np.min(np.linalg.norm(bound_l - pt, axis=1)) for pt in raceline_xy])
        return d_right, d_left

    # ─────────────────────────────────────────────────────────────────────────
    # JSON output (global_waypoints.json — unicorn gb_optimizer schema)
    # ─────────────────────────────────────────────────────────────────────────
    def _save_json(self, traj_iqp, traj_sp, lap_iqp,
                   reftrack_interp, psi_interp, kappa_interp, spline_lengths,
                   bound_r=None, bound_l=None):
        def _psi_ros(p):
            return (float(p) + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi

        def _header():
            return {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': 'map'}

        def _traj_to_wpnts(traj):
            # 2-pass: d_right/d_left = distance to the REAL walls (not _modi virtual walls)
            d_right, d_left = self._dists_to_bounds(traj[:, 1:3], bound_r, bound_l)
            return {
                'header': _header(),
                'wpnts': [
                    {
                        'id': i, 's_m': float(r[0]), 'd_m': 0.0,
                        'x_m': float(r[1]), 'y_m': float(r[2]),
                        'd_right': float(d_right[i]), 'd_left': float(d_left[i]),
                        'psi_rad': _psi_ros(r[3]),
                        'kappa_radpm': float(r[4]),
                        'vx_mps': float(r[5]), 'ax_mps2': float(r[6]),
                    }
                    for i, r in enumerate(traj)
                ],
            }

        def _vel_markers(traj, ns, col_r, col_g, col_b):
            markers = []
            for i, r in enumerate(traj):
                h = max(float(r[5]) * 0.1317, 0.01)
                markers.append({
                    'header': _header(), 'ns': ns, 'id': i, 'type': 3, 'action': 0,
                    'pose': {
                        'position': {'x': float(r[1]), 'y': float(r[2]), 'z': h / 2.0},
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                    },
                    'scale': {'x': 0.1, 'y': 0.1, 'z': h},
                    'color': {'r': col_r, 'g': col_g, 'b': col_b, 'a': 0.8},
                    'lifetime': {'sec': 0, 'nanosec': 0}, 'frame_locked': False,
                    'points': [], 'colors': [], 'text': '',
                    'mesh_resource': '', 'mesh_use_embedded_materials': False,
                })
            return {'markers': markers}

        def _sphere_markers(xys, ns, col_r, col_g, col_b, scale=0.1):
            return [
                {
                    'header': _header(), 'ns': ns, 'id': i, 'type': 2, 'action': 0,
                    'pose': {
                        'position': {'x': float(xy[0]), 'y': float(xy[1]), 'z': 0.0},
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                    },
                    'scale': {'x': scale, 'y': scale, 'z': scale},
                    'color': {'r': col_r, 'g': col_g, 'b': col_b, 'a': 0.8},
                    'lifetime': {'sec': 0, 'nanosec': 0}, 'frame_locked': False,
                    'points': [], 'colors': [], 'text': '',
                    'mesh_resource': '', 'mesh_use_embedded_materials': False,
                }
                for i, xy in enumerate(xys)
            ]

        # Centerline waypoints from prep_track reftrack (d_right/d_left = _modi widths)
        s_vals = np.concatenate([[0.0], np.cumsum(spline_lengths)])
        cl_wpnts = [
            {
                'id': i, 's_m': float(s_vals[i]), 'd_m': 0.0,
                'x_m': float(row[0]), 'y_m': float(row[1]),
                'd_right': float(row[2]), 'd_left': float(row[3]),
                'psi_rad': _psi_ros(psi_interp[i]),
                'kappa_radpm': float(kappa_interp[i]),
                'vx_mps': 0.0, 'ax_mps2': 0.0,
            }
            for i, row in enumerate(reftrack_interp)
        ]

        # Trackbounds markers from REAL boundary CSVs
        tb_markers = []
        mid = 0
        for side_path, (cr, cg, cb) in [
            (os.path.join(self.map_dir, 'boundary_right.csv'), (0.0, 1.0, 0.0)),
            (os.path.join(self.map_dir, 'boundary_left.csv'), (1.0, 0.0, 1.0)),
        ]:
            if os.path.exists(side_path):
                pts = np.loadtxt(side_path, delimiter=',', skiprows=1)
                for pt in pts:
                    m = _sphere_markers([[pt[0], pt[1]]], 'trackbounds', cr, cg, cb)[0]
                    m['id'] = mid
                    tb_markers.append(m)
                    mid += 1

        data = {
            'map_info_str': {'data': self.map_name},
            'est_lap_time': {'data': float(lap_iqp)},
            'centerline_markers': {
                'markers': _sphere_markers(
                    [[w['x_m'], w['y_m']] for w in cl_wpnts],
                    'centerline', 0.0, 1.0, 1.0, 0.05)
            },
            'centerline_waypoints': {'header': _header(), 'wpnts': cl_wpnts},
            'global_traj_markers_iqp': _vel_markers(traj_iqp, 'iqp', 1.0, 0.0, 0.0),
            'global_traj_wpnts_iqp': _traj_to_wpnts(traj_iqp),
            'global_traj_markers_sp': _vel_markers(traj_sp, 'sp', 0.0, 0.0, 1.0),
            'global_traj_wpnts_sp': _traj_to_wpnts(traj_sp),
            'trackbounds_markers': {'markers': tb_markers},
        }

        json_path = os.path.join(self.map_dir, 'global_waypoints.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        self.get_logger().info(f'JSON saved: {json_path}')

    def _run_check(self, label, traj, bound_r, bound_l, safety_width):
        raceline = traj[:, 1:3]
        kappa = traj[:, 4]
        vx = traj[:, 5]

        veh_half = self.pars['veh_params']['width'] / 2
        safety_half = safety_width / 2
        curvlim = self.pars['veh_params']['curvlim']
        v_max = self.pars['veh_params']['v_max']
        a_lat_max = float(self.pars['ggv'][:, 2].min())

        errors, warnings = [], []
        for side, bound in (('RIGHT', bound_r), ('LEFT', bound_l)):
            dist = np.array([np.min(np.linalg.norm(bound - pt, axis=1)) for pt in raceline])
            n_hit = int((dist < veh_half).sum())
            if n_hit:
                errors.append(f'{side} wall hit: {n_hit} pts (min={dist.min():.3f}m < {veh_half:.2f}m)')
            n_close = int(((dist >= veh_half) & (dist < safety_half)).sum())
            if n_close:
                warnings.append(f'Low margin to {side}: {n_close} pts < {safety_half:.2f}m')

        n_curv = int((np.abs(kappa) > curvlim).sum())
        if n_curv:
            warnings.append(f'Curvature limit exceeded: {n_curv} pts (max={np.abs(kappa).max():.3f})')
        n_vel = int((vx > v_max + 0.1).sum())
        if n_vel:
            warnings.append(f'Velocity limit exceeded: {n_vel} pts (max={vx.max():.2f})')
        a_lat = vx**2 * np.abs(kappa)
        n_alat = int((a_lat > a_lat_max * 1.05).sum())
        if n_alat:
            warnings.append(f'Lateral accel exceeded: {n_alat} pts (max={a_lat.max():.2f})')

        min_r = np.array([np.min(np.linalg.norm(bound_r - pt, axis=1)) for pt in raceline])
        min_l = np.array([np.min(np.linalg.norm(bound_l - pt, axis=1)) for pt in raceline])
        self.get_logger().info(
            f'[CheckTraj {label}] min_r={min_r.min():.3f}m  min_l={min_l.min():.3f}m  '
            f'max_κ={np.abs(kappa).max():.3f}  max_v={vx.max():.2f}  max_a_lat={a_lat.max():.2f}')
        for e in errors:
            self.get_logger().error(f'  [CheckTraj {label}] ERROR: {e}')
        for w in warnings:
            self.get_logger().warn(f'  [CheckTraj {label}] WARN: {w}')
        if not errors and not warnings:
            self.get_logger().info(f'  [CheckTraj {label}] OK')

    def _log_stats(self, label, traj, lap_time):
        vx = traj[:, 5]
        self.get_logger().info(
            f'[{label}] length={traj[-1, 0]:.2f}m  lap≈{lap_time:.2f}s  '
            f'v_max={vx.max():.2f}  v_min={vx.min():.2f}  v_avg={vx.mean():.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
