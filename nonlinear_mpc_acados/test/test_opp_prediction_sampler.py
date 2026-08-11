import numpy as np
from nonlinear_mpc_acados.mpc_core.opp_prediction_sampler import (
    stage_times, sample_opp_xy, project_s)


def test_project_s_nearest_vertex():
    s = np.array([0.0, 1.0, 2.0, 3.0]); x = s.copy(); y = np.zeros(4)
    assert project_s(1.9, 0.5, x, y, s) == 2.0


def test_project_s_offline_point():
    s = np.array([0.0, 1.0, 2.0]); x = s.copy(); y = np.zeros(3)
    # point far in +y above s=1 vertex still projects to nearest vertex s=1
    assert project_s(1.0, 5.0, x, y, s) == 1.0


def test_stage_times_uniform():
    assert np.allclose(stage_times(0.1, 5), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_stage_times_nonuniform():
    assert np.allclose(stage_times(0.1, 3, dt_vec=[0.1, 0.2, 0.4]), [0.0, 0.1, 0.3, 0.7])


def test_sample_advances_along_path():
    s = np.linspace(0, 10, 11); x = s.copy(); y = np.zeros_like(s)
    out = sample_opp_xy(s, x, y, s0=2.0, v=1.0, times=np.array([0.0, 1.0, 2.0]), track_len=10.0)
    assert np.allclose(out[:, 0], [2.0, 3.0, 4.0], atol=1e-6)
    assert np.allclose(out[:, 1], 0.0, atol=1e-6)


def test_sample_wraps_closed_track():
    s = np.linspace(0, 10, 11); x = s.copy(); y = np.zeros_like(s)
    out = sample_opp_xy(s, x, y, s0=9.0, v=1.0, times=np.array([2.0]), track_len=10.0)
    assert np.allclose(out[0, 0], 1.0, atol=1e-6)


def test_sample_unsorted_input():
    s = np.array([5.0, 0.0, 10.0]); x = np.array([5.0, 0.0, 10.0]); y = np.zeros(3)
    out = sample_opp_xy(s, x, y, s0=0.0, v=1.0, times=np.array([5.0]), track_len=10.0)
    assert np.allclose(out[0, 0], 5.0, atol=1e-6)
