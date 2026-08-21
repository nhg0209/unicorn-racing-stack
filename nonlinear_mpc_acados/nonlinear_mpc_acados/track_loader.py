"""Track data loader — CSV waypoints → CasADi spline LUTs.

Ported from `Nonlinear_MPC_node.py::preprocess_track_data` (ROS1).
ROS-free: pure numpy + casadi. The ROS2 wrapper hands the resulting
`TrackData` straight into `mpc.set_track_data(...)`.

Per-track CSV layout (all in one directory `<track_dir>/`):
    track<NAME>_centerline_waypoints.csv   x, y, ref_v   (ref_v scaled by vel_scale)
    track<NAME>_center_derivates.csv       dx, dy
    track<NAME>_right_waypoints.csv        x, y
    track<NAME>_left_waypoints.csv         x, y

Loop closure: the centerline / boundaries are duplicated by `extend_part`
so the spline LUT covers `s ∈ [0, L * (1 + 1/extend_part)]`. This gives
the MPC's prediction horizon something to query at the lap-rollover seam.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from casadi import interpolant


@dataclass
class TrackData:
    """All artefacts produced by `load_track`. One per track, immutable per run."""
    center_lane: np.ndarray              # (N_total, 2) — extended for loop overlap
    element_arc_lengths: np.ndarray      # (N_total,) cumulative s on extended grid
    element_arc_lengths_orig: np.ndarray # (N_orig,)  cumulative s on raw centerline
    center_lut_x: Any
    center_lut_y: Any
    center_lut_dx: Any
    center_lut_dy: Any
    right_lut_x: Any
    right_lut_y: Any
    left_lut_x: Any
    left_lut_y: Any
    lut_ref_v: Any
    ref_v: np.ndarray = field(default=None)  # raw speed profile (only from build_track_from_wpnts)
    center_point_angles: np.ndarray = field(default=None)
    # Raw (un-inflated, un-extended) boundary lanes for RViz viz publishers
    # /center_path /right_path /left_path. center_lane (above) is extended
    # for spline lookup; these are the original ROS1 layout (N_orig × 2).
    raw_center_lane: np.ndarray = field(default=None)
    raw_right_lane: np.ndarray = field(default=None)
    raw_left_lane: np.ndarray = field(default=None)
    # 2026-08-03: 각 스테이션에 실제 적용된 벽마진 W_eff (N_orig,).
    # 코리도 LUT 벽은 실벽에서 이만큼 안쪽이다. refv 학습의 벽여유 게이트가
    # LUT 벽 기준으로 재면 마진이 이중계상되어(특히 직선, 0.18) 상향이
    # 과도하게 동결된다 — 실벽 여유 = LUT 여유 + wall_margin(s).
    wall_margin: np.ndarray = field(default=None)


def _read_csv_xy(path: str, with_v: bool = False, vel_scale: float = 1.0):
    """CSV → np.ndarray. `with_v=True` returns (xy[N,2], v[N]) for centerline file."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"track CSV not found: {path}")
    with open(path) as f:
        rows = [tuple(line) for line in csv.reader(f, delimiter=',') if line]
    if with_v:
        xy = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)
        v = np.array([float(r[2]) * vel_scale for r in rows], dtype=float)
        return xy, v
    return np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)


def _arc_lengths(waypoints: np.ndarray) -> np.ndarray:
    d = np.diff(waypoints, axis=0)
    consecutive = np.sqrt(np.sum(d ** 2, axis=1))
    return np.insert(np.cumsum(consecutive), 0, 0.0)


def _inflate_boundary(center_lane: np.ndarray, side_lane: np.ndarray,
                      inflation_factor: float) -> np.ndarray:
    """Pull each side waypoint inward by inflation_factor * 0.8 m."""
    inflated = side_lane.copy()
    for idx in range(len(center_lane)):
        v = inflated[idx, :] - center_lane[idx, :]
        d = np.linalg.norm(v)
        if d < 1e-9:
            continue
        unit = v / d
        inflated[idx, :] -= unit * inflation_factor * 0.8
    return inflated


def _bspline_xy(label_x: str, label_y: str, pts: np.ndarray, arc_lengths: np.ndarray):
    return (interpolant(label_x, 'bspline', [arc_lengths], pts[:, 0]),
            interpolant(label_y, 'bspline', [arc_lengths], pts[:, 1]))


def _bspline_v(label: str, v_col: np.ndarray, arc_lengths: np.ndarray):
    return interpolant(label, 'bspline', [arc_lengths], v_col[:, 0])


def _build_track(center_lane: np.ndarray, center_deriv: np.ndarray,
                 right_lane: np.ndarray, left_lane: np.ndarray, ref_v: np.ndarray,
                 inflation_factor: float, extend_part: int) -> TrackData:
    """Shared geometry → spline LUT pipeline. CSV and wpnt-msg loaders both
    funnel through here so the resulting `TrackData` is identical-shape."""
    # Stash raw (pre-inflation) boundaries for RViz viz publishers.
    raw_center_lane = center_lane.copy()
    raw_right_lane = right_lane.copy()
    raw_left_lane = left_lane.copy()
    right_lane = _inflate_boundary(center_lane, right_lane, inflation_factor)
    left_lane = _inflate_boundary(center_lane, left_lane, inflation_factor)

    n_orig = center_lane.shape[0]
    overlap = max(1, int(n_orig / extend_part))
    cl_ext = np.vstack((center_lane, center_lane[1:overlap, :]))
    rl_ext = np.vstack((right_lane, right_lane[1:overlap, :]))
    ll_ext = np.vstack((left_lane, left_lane[1:overlap, :]))
    cd_ext = np.vstack((center_deriv, center_deriv[1:overlap, :]))

    end_v = max(1, int(ref_v.shape[0] / extend_part))
    ref_v_ext = np.vstack((ref_v[:, np.newaxis], ref_v[1:end_v, np.newaxis]))

    s_orig = _arc_lengths(center_lane)
    s_ext = _arc_lengths(cl_ext)

    center_x_lut, center_y_lut = _bspline_xy('lut_center_x', 'lut_center_y', cl_ext, s_ext)
    center_dx_lut, center_dy_lut = _bspline_xy('lut_center_dx', 'lut_center_dy', cd_ext, s_ext)
    right_x_lut, right_y_lut = _bspline_xy('lut_right_x', 'lut_right_y', rl_ext, s_ext)
    left_x_lut, left_y_lut = _bspline_xy('lut_left_x', 'lut_left_y', ll_ext, s_ext)
    ref_v_lut = _bspline_v('lut_ref_v', ref_v_ext, s_ext)

    return TrackData(
        center_lane=cl_ext,
        element_arc_lengths=s_ext,
        element_arc_lengths_orig=s_orig,
        center_lut_x=center_x_lut, center_lut_y=center_y_lut,
        center_lut_dx=center_dx_lut, center_lut_dy=center_dy_lut,
        right_lut_x=right_x_lut, right_lut_y=right_y_lut,
        left_lut_x=left_x_lut, left_lut_y=left_y_lut,
        lut_ref_v=ref_v_lut,
        center_point_angles=np.arctan2(center_deriv[:, 1], center_deriv[:, 0]),
        raw_center_lane=raw_center_lane,
        raw_right_lane=raw_right_lane,
        raw_left_lane=raw_left_lane,
    )


def load_track(track_dir: str, track_name: str,
               vel_scale: float = 1.0,
               inflation_factor: float = 1.2,
               extend_part: int = 2) -> TrackData:
    """Load a track from `<track_dir>/track<track_name>/track<track_name>_*.csv`."""
    sub = os.path.join(track_dir, f"track{track_name}")
    base = os.path.join(sub, f"track{track_name}")

    center_lane, ref_v = _read_csv_xy(base + '_centerline_waypoints.csv',
                                      with_v=True, vel_scale=vel_scale)
    center_deriv = _read_csv_xy(base + '_center_derivates.csv')
    right_lane = _read_csv_xy(base + '_right_waypoints.csv')
    left_lane = _read_csv_xy(base + '_left_waypoints.csv')

    return _build_track(center_lane, center_deriv, right_lane, left_lane, ref_v,
                        inflation_factor=inflation_factor, extend_part=extend_part)


def corridor_speed_cap(width: float, v_full: float, v_floor: float = 0.0,
                       w_tight: float = 1.0, w_wide: float = 1.6) -> float:
    """Speed cap from corridor width (complements the curvature cap).

    A narrow-but-gently-curving section has low |κ|, so √(a_lat/|κ|) lets the
    car run ~v_max and clip the wall. This caps speed by available width:
    wide corridor → `v_full`; narrow → ramped linearly down to `v_floor`
    between `w_wide` and `w_tight`. `v_floor<=0` (or w_wide<=w_tight) disables
    it (returns `v_full`), so it is opt-in and a no-op by default.
    """
    if v_floor <= 0.0 or w_wide <= w_tight:
        return v_full
    if width >= w_wide:
        return v_full
    if width <= w_tight:
        return v_floor
    frac = (width - w_tight) / (w_wide - w_tight)
    return v_floor + (v_full - v_floor) * frac


# Corridor bounds the car CENTER, but d_left/d_right are distances to the track
# walls. Subtract this margin (car half-width ~0.16 + buffer) on each side so the
# car body doesn't clip the wall — without it the raceline (which hugs the inside
# edge at apexes) leaves zero room and the car wedges (STUCK storm).
# 2026-06-02: 0.25→0.18 (final 좁은 트랙: d=0.36m서 0.25면 corridor 0.11m→QP
# infeasible→fails폭증. car 반폭~0.15 기준 0.18이 최소안전). Both wall-source
# paths (ref-derived and centerline-derived) use this single value.
#
# 2026-08-03: 0.18 → 0.0. **이 여유는 R_car 와 중복이었다.**
#   여기서 벽 LUT 를 0.18 안쪽으로 당기고, acados_kinematic.py:1283-1286 이
#   다시 R_CAR(0.30) 을 뺀다 → 총 잠식 0.48 m. 두 상수 모두 근거로 "차 반폭"
#   을 든다(위 주석 "car half-width ~0.16", ddrx_unified_params.yaml "차 반지름").
#   실제 반폭은 0.1016 m (차폭 0.2032).
#   ifac 맵 366 스테이션 재현 계산 (scratchpad/corridor_check.py):
#     0.18+0.30 → 코리도 역전(상한<하한, 해 없음) 28곳, 최소폭 -0.108 m,
#                 레이스라인이 자기 코리도 밖 227곳(62%)
#     0.00+0.30 → 역전 0곳, 최소폭 +0.235 m, 라인 밖 24곳
#   역전 구간에서는 위아래 제약이 동시에 위반되고 s_top+s_bot 이 기하 상수라
#   선형항 zl 의 횡방향 기울기가 0 이다 — 슬랙/존 벌점을 아무리 올려도 옆으로
#   미는 힘이 안 생긴다. 2026-06-02·07-29 의 "벌점이 셀수록 반대벽 클립",
#   그리고 존 4개로 트랙 58% 를 덮어야 했던 것이 전부 이 증상이었다.
#   위 0602 이력("0.25면 corridor 0.11m→QP infeasible")도 R_car 가 뒤에서
#   0.30 을 더 뺀다는 걸 모른 채 이 상수만 깎아 증상을 덮은 것이다.
#   **그러나 실측은 이 결론을 기각했다 (2026-08-03 심, 130 s / 12랩):**
#     0.18 → 주행 중 STUCK 0회(기준선) / 랩편차 0.154 s
#     0.00 → 주행 중 STUCK **14회**, 랩 9.81~12.33 (편차 2.52 s), 최소 벽마진 0.109
#   기하상 역전은 사라지는데 주행은 크게 악화했다. 즉 0.18 은 "중복된 차 반폭"
#   이 아니라 **다른 오차를 흡수하고 있었다.** 유력 후보: _walls_from_centerline
#   이 각 레이스라인 스테이션에 대해 *최근접 센터라인 점*의 벽을 쓰는 재샘플
#   오차(라인이 센터라인에서 크게 벗어나는 시케인에서 최대) + 문서화된
#   계획-추종 오차 0.03~0.13 m.
#   → 0.18 유지. 줄이려면 재샘플 오차부터 없애고(법선 투영으로 정확히 계산)
#     그 다음에 단계적으로. 0 으로 한 번에 가는 건 확인된 실패다.
_WALL_MARGIN = 0.22   # 0820: 벽스침 대응 0.18→0.22 (협소는 α식 자동축소)

# 2026-08-03 A안: 마진을 **트랙 폭에 비례**하게. 전 구간 0 은 실패했지만(위),
# 넓은 곳은 0.18 유지 · 좁은 곳만 축소하는 건 다른 얘기다.
#
# 문제의 산수 (ifac 시케인, 폭 ~1.0 m):
#     가용 반폭 = 1.0/2 − _WALL_MARGIN 0.18 − R_car 0.30 = 0.02 m
#   → 코리도가 사실상 사라지고 s21.8~23.7 은 상한<하한(역전, 해 없음).
#   실측(2026-08-03): 레이스라인 중심→실벽 최단거리 최소 0.180 m, 차체 여유
#   최소 +0.078 m — **물리 공간은 있는데 마진 산수가 다 먹었다.**
#   레이스라인의 62%(227/366)가 자기 코리도(벽−0.48) 밖이라 q_cte 와 코리도가
#   상시 충돌하고, 균형이 무너지는 스텁/시케인에서 벽을 친다.
#
#   W_eff(s) = min(_WALL_MARGIN, α · 가용반폭_R_car제외(s))
#   여기서 가용반폭_R_car제외 = (d_left + d_right)/2 − R_car.
#   넓은 구간은 α·(큰 값) > 0.18 이라 그대로 0.18, 좁은 구간만 자동 축소된다.
#   α=0.5 → 시케인에서 0.18 → 약 0.10.
# 0 으로 두면 기능 off (= 전 구간 _WALL_MARGIN 고정, 종전 동작).
_WALL_MARGIN_WIDTH_ALPHA = 0.5
_WALL_MARGIN_R_CAR = 0.27   # 0821 실측 차체 0.40×0.35 (ddrx R_car_live 와 일치)


def _adaptive_wall_margin(d_left, d_right):
    """폭 비례 마진. d_left/d_right 는 원본(마진 미적용) 배열 또는 스칼라."""
    if _WALL_MARGIN_WIDTH_ALPHA <= 0.0:
        return _WALL_MARGIN
    avail = (np.asarray(d_left, dtype=float) + np.asarray(d_right, dtype=float)) / 2.0 \
        - _WALL_MARGIN_R_CAR
    return np.minimum(_WALL_MARGIN, np.maximum(0.0, _WALL_MARGIN_WIDTH_ALPHA * avail))


def _walls_from_centerline(ref_xy: np.ndarray, wall_wpnts) -> tuple:
    """Corridor walls for each reference station, sourced from the CENTERLINE's
    (reliable) d_left/d_right rather than the reference line's own.

    A wall's absolute position is fully determined by the centerline:
    `centerline_point + (d − _WALL_MARGIN)·centerline_normal` — independent of the
    reference (raceline) line. In raceline mode the raceline wpnts carry unreliable
    d_left/d_right, so we resample the true centerline walls onto the raceline
    stations: for each reference point take the nearest centerline wpnt's walls.
    The returned lanes are index-aligned to `ref_xy`, so the LUTs built from them
    share the reference's arc-length frame (what the MPC corridor samples at `sk`).

    Returns (left_lane, right_lane, wall_margin), lanes shape (len(ref_xy), 2),
    wall_margin shape (len(ref_xy),) — 각 스테이션에 적용된 W_eff.
    """
    cx = np.array([float(w.x_m) for w in wall_wpnts], dtype=float)
    cy = np.array([float(w.y_m) for w in wall_wpnts], dtype=float)
    cpsi = np.array([
        (w.psi_centerline_rad if abs(getattr(w, 'psi_centerline_rad', 0.0)) > 1e-9
         else w.psi_rad)
        for w in wall_wpnts], dtype=float)
    _dl0 = np.array([float(w.d_left) for w in wall_wpnts])
    _dr0 = np.array([float(w.d_right) for w in wall_wpnts])
    _wm = _adaptive_wall_margin(_dl0, _dr0)      # 0803: 폭 비례 (스칼라 폴백 가능)
    # 2026-08-06 좌벽 국소 마진 존 — 아킬레스 돌출부 밴드(런타임 chord s13.4~16.0,
    # d_left 0.53~0.67 로 좁은 쪽). 라인 수술 없이 스침·쐐기를 코리도로 완화:
    # 밴드 좌벽만 +0.06 m (라인 xy 앵커 캡슐, 0.9 m 풀 / 1.8 m 페이드 램프 —
    # 옛 코스트 존의 경계 풍선효과 회피). 실측 근거: lapA 무장애물 42랩에서
    # 밴드 실벽 min 0.180 m(헐 여유 ~3 cm) + 밴드 내 STUCK 텔레포트, 추월
    # 쐐기 STUCK 좌표(4.13,4.91)도 이 밴드. d_right 무변(여유 0.73~1.43).
    # wall_margin 배열엔 미반영 — wall_alarm 실벽 환산이 좌측에서 extra 만큼
    # 보수적이 될 뿐(안전 방향).
    # 2026-08-13 hall_0813: MPCC 지름길이 s6~10 좌벽 5cm 까지 접근(32랩 전랩 d+0.31~0.37,
    # 최협 dL0.51 에서 +0.37 랩만 접촉 = 여유 -1.5cm). 라인·튜닝 무변경, 좌벽만 +0.10.
    # ⚠ 앵커는 hall_0813 map 좌표 — 다른 맵에서 켜지려면 재앵커 필수 (구 ifac 앵커: (6.182,3.718)-(3.839,5.033))
    _ZA = np.array([5.713, 5.035]); _ZB = np.array([7.670, 8.612])
    _ZEXTRA, _ZR_FULL, _ZR_ZERO = 0.10, 0.9, 1.8
    _ab = _ZB - _ZA
    _t = np.clip(((np.column_stack((cx, cy)) - _ZA) @ _ab) / (_ab @ _ab), 0.0, 1.0)
    _dseg = np.hypot(cx - (_ZA[0] + _t * _ab[0]), cy - (_ZA[1] + _t * _ab[1]))
    _extra_l = _ZEXTRA * np.clip((_ZR_ZERO - _dseg) / (_ZR_ZERO - _ZR_FULL), 0.0, 1.0)
    d_l = np.maximum(0.05, _dl0 - _wm - _extra_l)
    d_r = np.maximum(0.05, _dr0 - _wm)
    cs, cc = np.sin(cpsi), np.cos(cpsi)
    # right normal = (+sin, −cos); left normal = (−sin, +cos)
    left_wall = np.column_stack((cx - d_l * cs, cy + d_l * cc))
    right_wall = np.column_stack((cx + d_r * cs, cy - d_r * cc))

    ref_xy = np.asarray(ref_xy, dtype=float)
    left_lane = np.empty_like(ref_xy)
    right_lane = np.empty_like(ref_xy)
    _wm_arr = np.broadcast_to(np.asarray(_wm, dtype=float), _dl0.shape)
    wall_margin = np.empty(ref_xy.shape[0])
    for i in range(ref_xy.shape[0]):
        j = int(np.argmin((cx - ref_xy[i, 0]) ** 2 + (cy - ref_xy[i, 1]) ** 2))
        left_lane[i] = left_wall[j]
        right_lane[i] = right_wall[j]
        wall_margin[i] = _wm_arr[j]
    return left_lane, right_lane, wall_margin


def build_track_from_wpnts(wpnts, vel_scale: float = 1.0,
                           wall_wpnts=None,
                           inflation_factor: float = 1.2,
                           extend_part: int = 2,
                           default_v: float = 5.0,
                           corridor_half_width: float = 0.0,
                           a_lat_max: float = 6.0,
                           a_long_max: float = 3.0,
                           corridor_v_floor: float = 0.0,
                           corridor_v_tight: float = 1.0,
                           corridor_v_wide: float = 1.6) -> TrackData:
    """Same TrackData as `load_track`, but built from a list of `f110_msgs/Wpnt`.

    Designed for race-stack `/centerline_waypoints` ingestion: wpnt fields
    `x_m, y_m, psi_centerline_rad, d_left, d_right, vx_mps` carry exactly the
    info the ROS1 CSV pipeline (centerline + derivative + L/R boundaries +
    ref_v) provides. `default_v` is used as fallback when wpnt.vx_mps is ≤ 0
    (centerline-only wpnts may not carry a velocity profile).
    """
    n = len(wpnts)
    if n < 4:
        raise ValueError(f"need ≥4 centerline wpnts, got {n}")

    center_lane = np.empty((n, 2), dtype=float)
    center_deriv = np.empty((n, 2), dtype=float)
    right_lane = np.empty((n, 2), dtype=float)
    left_lane = np.empty((n, 2), dtype=float)
    ref_v = np.empty(n, dtype=float)
    wall_margin_arr = np.zeros(n, dtype=float)   # 0803: 스테이션별 적용 W_eff

    # `/centerline_waypoints` carries the centerline tangent in `psi_rad`
    # (the wpnts ARE the centerline). `psi_centerline_rad` is only filled
    # on raceline wpnts as a side-channel for d_left/d_right corridor
    # conversion — here it would be 0 and collapse all boundaries onto the
    # ±y axis. Fall back to psi_rad when psi_centerline_rad is unset.
    # ref_v: prefer wpnt.vx_mps; else κ-aware cap √(a_lat_max/|κ|), then
    # default_v cap. centerline-only wpnts ship vx=0, so this branch is
    # what's actually exercised under `/centerline_waypoints`.
    # corridor_half_width > 0: fixed-width MPC corridor (centerline ± half).
    #   mpc lateral search space cap, 좌우 대칭. 코너에서 raw d_left/d_right
    #   가 망가지지 않게 함. inflation 은 우회됨 (사용자 설정값이 곧 cap).
    use_fixed_corridor = corridor_half_width > 1e-3
    for i, w in enumerate(wpnts):
        x, y = w.x_m, w.y_m
        psi_c = getattr(w, 'psi_centerline_rad', 0.0)  # optional field; absent on car's Wpnt
        psi = psi_c if abs(psi_c) > 1e-9 else w.psi_rad
        c, s = np.cos(psi), np.sin(psi)
        center_lane[i] = (x, y)
        center_deriv[i] = (c, s)
        if use_fixed_corridor:
            d_r = d_l = corridor_half_width
            wall_margin_arr[i] = 0.0
        else:
            # Walls inset by _WALL_MARGIN (module const) so the car body clears
            # the wall — corridor bounds the car CENTER. See _WALL_MARGIN docstring.
            _wm2 = float(_adaptive_wall_margin(float(w.d_left), float(w.d_right)))
            d_r = max(0.05, float(w.d_right) - _wm2)
            d_l = max(0.05, float(w.d_left) - _wm2)
            wall_margin_arr[i] = _wm2
        # right normal = (+sin(psi), -cos(psi)); left normal = (-sin(psi), +cos(psi))
        # NOTE: d_left/d_right are measured along the CENTERLINE normal, so this
        # projection is exact ONLY when psi == centerline tangent. Line 220
        # prefers psi_centerline_rad (the centerline tangent) for exactly this
        # reason. The psi_rad fallback (wpnts lacking centerline psi, e.g. a raw
        # raceline) would place the boundary off by 1/cos(Δψ) at points where the
        # wpnt heading differs from the centerline — feed centerline psi to avoid.
        right_lane[i] = (x + d_r * s,  y - d_r * c)
        left_lane[i]  = (x - d_l * s,  y + d_l * c)
        # Speed reference: cap to BOTH our top speed (default_v = v_max) AND the
        # κ-aware lateral-grip limit √(a_lat_max/|κ|). The IQP raceline ships
        # vx_mps optimized for a higher-grip/faster car (7-13 m/s here); used
        # raw (previous behaviour) it never slowed for apexes at our v=5 → the
        # car entered geometric apexes too hot → wedge (STUCK storm in raceline
        # mode). Taking the min slows it at high-curvature points per OUR limits
        # while still following the raceline GEOMETRY. Because the raceline is
        # min-curvature, its |κ| at corners is lower than the centerline's, so
        # the κ-cap permits MORE corner speed than centerline → faster lap.
        kappa = abs(float(w.kappa_radpm))
        v_kappa = np.sqrt(a_lat_max / kappa) if kappa > 1e-3 else default_v
        v_msg = float(w.vx_mps) * vel_scale
        v_src = v_msg if v_msg > 1e-3 else default_v
        # Corridor-width cap: the κ-cap is blind to width, so a narrow-but-
        # straight section (low |κ|) otherwise runs at v_max and clips the wall
        # (final-map s≈60 crash). d_r/d_l here are already wall-margin-subtracted,
        # so (d_r+d_l) is the usable corridor the car center must stay within.
        v_corr = corridor_speed_cap(d_r + d_l, default_v, corridor_v_floor,
                                    corridor_v_tight, corridor_v_wide)
        ref_v[i] = float(np.clip(min(v_src, v_kappa, default_v, v_corr),
                                 1.0, default_v))

    # ── Forward-backward velocity profile (TUM calc_vel_profile style) ──
    # The pointwise κ-cap above sets the APEX speed but does NOT tell the car to
    # brake BEFORE the apex. At v=5 with a real braking limit the car then
    # entered apexes too hot (late braking) → wedge in raceline mode. Propagate
    # the speed limits along the loop: a backward pass enforces "slow enough to
    # brake into the next point", a forward pass enforces "don't exceed what
    # acceleration out of the previous point allows". Result: a feasible profile
    # that decelerates AHEAD of corners. Loop-closed (the track is a circuit).
    # Applied only in the raceline/raw-corridor path (use_fixed_corridor=False);
    # C1 (2026-06-17): the BACKWARD (braking) pass now runs in fixed-corridor
    # mode too. Horizon is short (N=20·dT=0.04 = 0.8s ≈ 3m); without pre-corner
    # deceleration baked into ref_v the solver can't brake in time at high speed
    # (6→2.5 needs ~5m > horizon), which forced the external brake_anticip hack.
    # The pass only ever LOWERS ref_v (min), uses a_long = |A_MIN_DYN| (solver-
    # matched), so it can never make corner entry hotter — strictly safer.
    # The FORWARD (accel) pass stays raceline-only: the car accelerates far
    # faster than a_long(=3.0) (dyn_a_max=7.5), so capping exit ramp at 3.0
    # would needlessly throttle corner exits.
    if n >= 4:
        a_long = float(a_long_max)
        ds = np.empty(n, dtype=float)
        for i in range(n):
            j = (i + 1) % n
            ds[i] = float(np.hypot(center_lane[j, 0] - center_lane[i, 0],
                                   center_lane[j, 1] - center_lane[i, 1]))
        ref_v = velocity_feasibility_pass(
            ref_v, ds, a_brake=a_long,
            a_accel=None if use_fixed_corridor else a_long)
        ref_v = np.clip(ref_v, 1.0, default_v)

    # Wall-source split: in raceline mode the reference line (raceline) carries
    # unreliable d_left/d_right, so override the ref-derived walls above with the
    # centerline's true walls, resampled onto the raceline stations. center_lane
    # stays the raceline → LUTs keep the raceline arc-length frame. Skipped for
    # fixed-width corridor (walls are the user-set cap there).
    if wall_wpnts is not None and not use_fixed_corridor:
        left_lane, right_lane, wall_margin_arr = _walls_from_centerline(
            center_lane, wall_wpnts)

    # Fixed-width corridor already represents the mpc cap → skip inflation
    # (otherwise the boundary would be pulled inside the user-set width).
    eff_inflation = 0.0 if use_fixed_corridor else inflation_factor
    td = _build_track(center_lane, center_deriv, right_lane, left_lane, ref_v,
                      inflation_factor=eff_inflation, extend_part=extend_part)
    td.ref_v = ref_v.copy()  # stash raw (post-brake-profile) array for tests & diagnostics
    td.wall_margin = wall_margin_arr.copy()   # 0803: 실벽 여유 복원용 (refv 학습 게이트)
    return td


def velocity_feasibility_pass(ref_v, ds, a_brake, a_accel=None, n_sweeps=2):
    """TUM calc_vel_profile 스타일 forward-backward pass (원형 트랙).

    backward(제동)는 항상, forward(가속)는 a_accel 지정 시만. min-only —
    기존 프로파일을 절대 올리지 않으므로 strictly safer. build_track_from_wpnts
    의 inline 구현을 추출한 것 (online 학습기 speed_profile_learner 와 공유).
    """
    v = np.asarray(ref_v, float).copy()
    n = v.size
    if n < 2:
        return v
    ds = np.asarray(ds, float)
    for _ in range(n_sweeps):   # 2 wrap sweeps to converge across the start seam
        for i in range(n - 1, -1, -1):          # backward: braking
            j = (i + 1) % n
            v_brake = float(np.sqrt(v[j] ** 2 + 2.0 * a_brake * ds[i]))
            if v_brake < v[i]:
                v[i] = v_brake
        if a_accel is not None:                  # forward: accel
            for i in range(n):
                k = (i - 1) % n
                v_acc = float(np.sqrt(v[k] ** 2 + 2.0 * a_accel * ds[k]))
                if v_acc < v[i]:
                    v[i] = v_acc
    return v


def find_current_arc_length(track: TrackData, car_pos: np.ndarray,
                            arc_min_dist_tol: float = 1.0):
    """Project car (x,y) onto extended centerline; returns (current_s, nearest_idx).

    Uses dot-product projection on the segment between nearest waypoint and
    its neighbour. Wraps `current_s` modulo the original (unextended) arc
    length so MPC's solver-internal lap counter stays consistent.
    """
    cl = track.center_lane
    s_arr = track.element_arc_lengths
    L_orig = float(track.element_arc_lengths_orig[-1])

    distances = np.linalg.norm(cl - car_pos, axis=1)
    nearest = int(np.argmin(distances))
    min_dist = float(distances[nearest])

    if min_dist > arc_min_dist_tol:
        if nearest == 0:
            next_idx, prev_idx = 1, cl.shape[0] - 1
        elif nearest == cl.shape[0] - 1:
            next_idx, prev_idx = 0, cl.shape[0] - 2
        else:
            next_idx, prev_idx = nearest + 1, nearest - 1
        seg_back = cl[prev_idx] - cl[nearest]
        if np.dot(car_pos - cl[nearest], seg_back) > 0:
            actual = prev_idx
        else:
            actual = nearest
            nearest = next_idx
        seg_fwd = cl[nearest] - cl[actual]
        seg_norm = np.linalg.norm(seg_fwd)
        if seg_norm > 1e-9:
            projection = np.dot(car_pos - cl[actual], seg_fwd) / seg_norm
        else:
            projection = 0.0
        current_s = float(s_arr[actual]) + projection
    else:
        current_s = float(s_arr[nearest])

    # NOTE: do NOT force current_s=0 when nearest==0. When the car is just
    # BEHIND the start/finish seam the projection legitimately yields s≈L
    # (actual=last waypoint); clobbering it to 0 injected a full-lap
    # discontinuity at the line every lap. The wrap below handles s≥L_orig,
    # and the exactly-at-start case already reads s_arr[0]=0 via the else branch.
    if current_s >= L_orig:
        current_s = current_s % L_orig
    return current_s, nearest


def learned_speeds_for_wpnts(wpnts, learned_s, learned_v, L, tol=0.3):
    """각 wpnt 의 누적 arc-length s 에서, tol[m] 안에 학습 bin 이 있으면 그 ref_v,
    없으면 기존 wpnt.vx_mps 를 유지(부분 override). 안전 캡은 build_track_from_wpnts
    가 이후 그대로 적용한다."""
    import numpy as _np
    out = _np.array([float(getattr(w, 'vx_mps', 0.0)) for w in wpnts], dtype=float)
    learned_s = _np.asarray(learned_s, float); learned_v = _np.asarray(learned_v, float)
    if learned_s.size == 0:
        return out
    xy = _np.array([(float(w.x_m), float(w.y_m)) for w in wpnts])
    seg = _np.sqrt((_np.diff(xy, axis=0) ** 2).sum(axis=1))
    s_w = _np.insert(_np.cumsum(seg), 0, 0.0)
    for i, sw in enumerate(s_w):
        d = _np.abs(((learned_s - (sw % L) + L / 2.0) % L) - L / 2.0)  # 원형거리
        j = int(_np.argmin(d))
        if d[j] <= tol:
            out[i] = float(learned_v[j])
    return out
