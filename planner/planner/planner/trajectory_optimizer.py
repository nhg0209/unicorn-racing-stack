#!/usr/bin/env python3

import os
import csv
import numpy as np

import rclpy
from rclpy.node import Node


def _find_maps_dir(map_name):
    """Robustly locate stack_master/maps/<map_name>.

    The packages were regrouped (planner/planner/planner/...), so the old
    fixed dirname() chain no longer reaches the repo root. Ascend from this
    file's real path (symlink-install resolves it to the source tree) until a
    'stack_master/maps/<map_name>' directory is found; fall back to the
    installed stack_master share directory.
    """
    here = os.path.dirname(os.path.realpath(__file__))
    for _ in range(8):
        cand = os.path.join(here, 'stack_master', 'maps', map_name)
        if os.path.isdir(cand):
            return cand
        here = os.path.dirname(here)
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('stack_master'), 'maps', map_name)
    except Exception:
        return os.path.join('stack_master', 'maps', map_name)


class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        # ---- Parameters --------------------------------------------------
        self.declare_parameter('map_name', '')
        self.declare_parameter('input_csv', 'centerline.csv')
        self.declare_parameter('output_csv', 'global_waypoints.csv')
        self.declare_parameter('safety_margin', 0.20)   # [m]   clearance from each wall
        self.declare_parameter('v_max',         6.0)    # [m/s] vehicle speed cap
        self.declare_parameter('a_lat_max',     6.0)    # [m/s^2] lateral grip limit
        self.declare_parameter('a_long_max',    4.0)    # [m/s^2] longitudinal accel limit
        self.declare_parameter('target_ds',     0.25)   # [m]   uniform ds for QP input/output

        map_name      = self.get_parameter('map_name').value
        input_csv     = self.get_parameter('input_csv').value
        output_csv    = self.get_parameter('output_csv').value
        safety_margin = self.get_parameter('safety_margin').value
        v_max         = self.get_parameter('v_max').value
        a_lat_max     = self.get_parameter('a_lat_max').value
        a_long_max    = self.get_parameter('a_long_max').value
        target_ds     = self.get_parameter('target_ds').value

        if not map_name:
            self.get_logger().error('[TrajectoryOptimizer] map_name parameter is required!')
            return

        # ---- I/O paths ---------------------------------------------------
        map_dir  = _find_maps_dir(map_name)
        in_path  = os.path.join(map_dir, input_csv)
        out_path = os.path.join(map_dir, output_csv)

        if not os.path.exists(in_path):
            self.get_logger().error(f'[TrajectoryOptimizer] input not found: {in_path}')
            return

        # ---- Load + optimize + save -------------------------------------
        self.get_logger().info(f'[TrajectoryOptimizer] loading: {in_path}')
        x_c, y_c, w_r, w_l = self._load_centerline(in_path)

        # Real track walls (preferred over the constant centerline half-widths,
        # which do not reflect the actual corridor at tight corners).
        bound_l = self._load_xy(os.path.join(map_dir, 'boundary_left.csv'))
        bound_r = self._load_xy(os.path.join(map_dir, 'boundary_right.csv'))
        if bound_l is not None and bound_r is not None:
            self.get_logger().info(
                f'[TrajectoryOptimizer] using real walls '
                f'(L={len(bound_l)}, R={len(bound_r)} pts) for lateral bounds')

        self.get_logger().info(
            f'[TrajectoryOptimizer] optimizing on {len(x_c)} centerline points '
            f'(margin={safety_margin}, v_max={v_max}, a_lat={a_lat_max}, '
            f'a_long={a_long_max}, target_ds={target_ds})'
        )
        x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new = self._optimize(
            x_c, y_c, w_r, w_l,
            safety_margin=safety_margin,
            v_max=v_max,
            a_lat_max=a_lat_max,
            a_long_max=a_long_max,
            target_ds=target_ds,
            bound_l=bound_l,
            bound_r=bound_r,
        )

        self._save_global_waypoints(out_path, x_opt, y_opt, w_r_new, w_l_new, psi, kappa, vx)
        self.get_logger().info(
            f'[TrajectoryOptimizer] saved {len(x_opt)} pts → {out_path} '
            f'(v_min={vx.min():.2f}, v_max={vx.max():.2f} m/s, '
            f'|kappa|max={np.max(np.abs(kappa)):.3f})'
        )

    # ======================================================================
    #                       STUDENT IMPLEMENTATION
    # ======================================================================
    @staticmethod
    def _optimize(x_c, y_c, w_r, w_l,
                  safety_margin, v_max, a_lat_max, a_long_max, target_ds,
                  bound_l=None, bound_r=None):
        """
        Minimum-curvature trajectory optimization.

        Parameters
        ----------
        x_c, y_c   : (N,) centerline coordinates (closed loop, no duplicate end)
        w_r, w_l   : (N,) track half-widths to the right / left walls
        safety_margin : [m] keep this far from each wall
        v_max, a_lat_max, a_long_max : vehicle limits
        target_ds  : [m] desired arc-length spacing for the optimized output

        Returns
        -------
        x_opt, y_opt, psi, kappa, vx : (M,) arrays for the optimized raceline
        w_r_new, w_l_new             : (M,) remaining clearance to the original walls
        """

        # TODO: Minimum-Curvature Path Optimization
        #
        # ┌─ Step 1. (optional) resample the centerline to uniform ds
        # │   x_r, y_r, w_r_r, w_l_r = _resample_uniform(x_c, y_c, w_r, w_l, target_ds)
        # │
        # ├─ Step 2. compute the unit normal vector n_i at every point.
        # │   tangent t_i via centered difference → normalize →
        # │   normal n_i = (-t_y, t_x) (pointing left)
        # │
        # ├─ Step 3. formulate the optimization problem (see lecture notes).
        # │   raceline parametrization:  r_i = p_i + alpha_i * n_i
        # │   objective :  minimize total curvature, e.g. min Sum kappa_i^2
        # │   bound     :  alpha_i in [ -w_r_i + safety_margin, +w_l_i - safety_margin ]
        # │   You may approximate curvature however you like (finite differences,
        # │   spline derivatives, etc.) and choose a different objective if you
        # │   prefer (shortest path, min lap time, etc. — see lecture).
        # │
        # ├─ Step 4. solve it.
        # │   Any optimization tool is fair game — convex QP solvers (osqp, cvxpy,
        # │   quadprog), general nonlinear minimizers (scipy.optimize.minimize),
        # │   evolution strategies (CMA-ES), or a ready-made package such as
        # │   trajectory_planning_helpers. Pick whichever fits your formulation.
        # │
        # ├─ Step 5. recover the optimal line coordinates from alpha.
        # │   x_opt = x_r + nx * alpha,   y_opt = y_r + ny * alpha
        # │
        # ├─ Step 6. (recommended) dedupe + cubic-spline resample to uniform ds
        # │   prevents clustered points on the inside of corners → clean psi, kappa.
        # │
        # ├─ Step 7. compute psi (heading) and kappa (curvature) via _geom()
        # │   kappa = (x' y'' − y' x'') / (x'^2 + y'^2)^(3/2)         (slide 45)
        # │
        # ├─ Step 8. speed profile via _speed_profile()                (slide 47)
        # │   - cornering limit:  v_max = sqrt(a_y / kappa)
        # │   - forward pass (accel limit) + backward pass (brake limit)
        # │   - pointwise min(cornering, accel, brake)
        # │
        # └─ Step 9. compute remaining clearance to the original walls (w_r_new, w_l_new)
        #             and return it together with the optimized line.

        # Step 1: resample centerline to uniform arc-length spacing
        x_r, y_r, w_r_r, w_l_r = TrajectoryOptimizer._resample_uniform(
            x_c, y_c, w_r, w_l, target_ds)
        N = len(x_r)

        # Step 2: unit (left-pointing) normals via centered-difference tangent
        tx = np.roll(x_r, -1) - np.roll(x_r, 1)
        ty = np.roll(y_r, -1) - np.roll(y_r, 1)
        tnorm = np.hypot(tx, ty)
        tnorm[tnorm < 1e-9] = 1e-9
        tx /= tnorm
        ty /= tnorm
        nx, ny = -ty, tx

        # Lateral bounds for the shift alpha. Keep the car (treated as a disc of
        # radius CAR_R) clear of each wall by safety_margin. Prefer the real walls
        # (boundary CSVs) projected onto the normal; fall back to the constant
        # centerline half-widths when boundaries are unavailable.
        CAR_R = 0.30                       # f1tenth half-extent [m] (conservative)
        clearance = safety_margin + CAR_R
        if bound_l is not None and bound_r is not None:
            # signed offset of the nearest wall point along the (left) normal
            def _wall_proj(wall):
                proj = np.empty(N)
                for i in range(N):
                    dx = wall[:, 0] - x_r[i]
                    dy = wall[:, 1] - y_r[i]
                    d2 = dx * dx + dy * dy
                    j = int(np.argmin(d2))
                    proj[i] = dx[j] * nx[i] + dy[j] * ny[i]
                return proj
            left_proj = _wall_proj(bound_l)    # typically > 0 (wall to the left)
            right_proj = _wall_proj(bound_r)   # typically < 0 (wall to the right)
            ub = np.maximum(left_proj, right_proj) - clearance
            lb = np.minimum(left_proj, right_proj) + clearance
        else:
            lb = -w_r_r + clearance
            ub = w_l_r - clearance
        # Narrow track guard: if margins overlap, pin alpha to the corridor centre
        mid = 0.5 * (lb + ub)
        bad = lb > ub
        lb = np.where(bad, mid, lb)
        ub = np.where(bad, mid, ub)

        # Steps 3-5: minimise discrete curvature  Sum |r_{i+1} - 2 r_i + r_{i-1}|^2
        # with r_i = p_i + alpha_i n_i. Quadratic in alpha -> linear normal equations.
        idx = np.arange(N)
        im1 = np.roll(idx, 1)
        ip1 = np.roll(idx, -1)
        Ax = np.zeros((N, N))
        Ay = np.zeros((N, N))
        Ax[idx, im1] += nx[im1]
        Ay[idx, im1] += ny[im1]
        Ax[idx, idx] += -2.0 * nx
        Ay[idx, idx] += -2.0 * ny
        Ax[idx, ip1] += nx[ip1]
        Ay[idx, ip1] += ny[ip1]
        bx = np.roll(x_r, -1) - 2.0 * x_r + np.roll(x_r, 1)
        by = np.roll(y_r, -1) - 2.0 * y_r + np.roll(y_r, 1)
        A = np.vstack([Ax, Ay])
        b = np.concatenate([bx, by])
        AtA = A.T @ A + 1e-6 * np.eye(N)
        alpha = np.linalg.solve(AtA, -(A.T @ b))
        # Projected clipping (a couple of passes so bound-active points settle)
        for _ in range(3):
            alpha = np.clip(alpha, lb, ub)

        # Step 5: recover optimized line
        x_opt = x_r + nx * alpha
        y_opt = y_r + ny * alpha

        # Step 6: light periodic smoothing to remove residual kinks, then RE-CLIP
        # back into the corridor (smoothing cuts corners and would otherwise push
        # the apex into the inside wall).
        for _ in range(3):
            x_opt = 0.25 * np.roll(x_opt, 1) + 0.5 * x_opt + 0.25 * np.roll(x_opt, -1)
            y_opt = 0.25 * np.roll(y_opt, 1) + 0.5 * y_opt + 0.25 * np.roll(y_opt, -1)
            alpha = np.clip((x_opt - x_r) * nx + (y_opt - y_r) * ny, lb, ub)
            x_opt = x_r + nx * alpha
            y_opt = y_r + ny * alpha

        # Step 7: heading + curvature of the optimized line
        psi, kappa = TrajectoryOptimizer._geom(x_opt, y_opt)

        # Step 8: point-mass speed profile (cornering + accel/brake passes)
        vx = TrajectoryOptimizer._speed_profile(
            x_opt, y_opt, kappa, v_max, a_lat_max, a_long_max)

        # Step 9: remaining clearance to the original walls after the shift
        w_r_new = np.maximum(w_r_r + alpha, 0.0)
        w_l_new = np.maximum(w_l_r - alpha, 0.0)

        return x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new

    # ======================================================================
    #                            HELPERS
    # ======================================================================
    @staticmethod
    def _load_centerline(path):
        """centerline.csv → (x, y, w_tr_right, w_tr_left) numpy arrays."""
        xs, ys, wrs, wls = [], [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row['x_m']))
                ys.append(float(row['y_m']))
                wrs.append(float(row['w_tr_right_m']))
                wls.append(float(row['w_tr_left_m']))
        x = np.asarray(xs); y = np.asarray(ys)
        wr = np.asarray(wrs); wl = np.asarray(wls)
        # drop duplicate closing point if present
        if len(x) > 1 and np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-3:
            x, y, wr, wl = x[:-1], y[:-1], wr[:-1], wl[:-1]
        return x, y, wr, wl

    @staticmethod
    def _load_xy(path):
        """Load an x_m,y_m CSV (e.g. a track boundary) → (M,2) array, or None."""
        if not os.path.exists(path):
            return None
        xs, ys = [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'x_m' in row and 'y_m' in row:
                    xs.append(float(row['x_m']))
                    ys.append(float(row['y_m']))
        if len(xs) < 2:
            return None
        return np.column_stack([xs, ys])

    @staticmethod
    def _resample_uniform(x, y, w_r, w_l, target_ds):
        """Linear-interp resample of a closed loop onto uniform arc-length spacing."""
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        L = s[-1]
        N_new = max(20, int(round(L / target_ds)))
        s_new = np.linspace(0.0, L, N_new, endpoint=False)
        x_p  = np.concatenate((x,   [x[0]]))
        y_p  = np.concatenate((y,   [y[0]]))
        wr_p = np.concatenate((w_r, [w_r[0]]))
        wl_p = np.concatenate((w_l, [w_l[0]]))
        return (np.interp(s_new, s, x_p),
                np.interp(s_new, s, y_p),
                np.interp(s_new, s, wr_p),
                np.interp(s_new, s, wl_p))

    @staticmethod
    def _geom(x, y):
        """Heading psi and signed curvature kappa via centered differences (closed loop)."""
        dx  = (np.roll(x, -1) - np.roll(x, 1)) * 0.5
        dy  = (np.roll(y, -1) - np.roll(y, 1)) * 0.5
        ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
        psi = np.arctan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        denom[denom < 1e-9] = 1e-9
        kappa = (dx * ddy - dy * ddx) / denom
        return psi, kappa

    @staticmethod
    def _speed_profile(x, y, kappa, v_max, a_lat_max, a_long_max):
        """Point-mass speed profile: cornering limit + fwd/bwd accel smoothing."""
        N = len(x)
        ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        ds[ds < 1e-6] = 1e-6
        v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6)))
        # backward pass: braking limit
        for _ in range(2):
            for i in range(N):
                j = (i - 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[j])
                v[j] = min(v[j], v_cap)
        # forward pass: acceleration limit
        for _ in range(2):
            for i in range(N):
                j = (i + 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[i])
                v[j] = min(v[j], v_cap)
        return v

    @staticmethod
    def _save_global_waypoints(path, x, y, w_r, w_l, psi, kappa, vx):
        header = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m',
                  'psi_rad', 'kappa_radpm', 'vx_mps']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(x)):
                w.writerow([f'{x[i]:.6f}', f'{y[i]:.6f}',
                            f'{w_r[i]:.4f}', f'{w_l[i]:.4f}',
                            f'{psi[i]:.6f}', f'{kappa[i]:.6f}',
                            f'{vx[i]:.4f}'])


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
