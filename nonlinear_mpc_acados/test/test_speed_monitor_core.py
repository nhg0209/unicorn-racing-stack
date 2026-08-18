"""speed_monitor_core 순수함수 단위테스트 (ROS/matplotlib 불필요).

Run:
    cd nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_speed_monitor_core.py -v
"""
from __future__ import annotations
from nonlinear_mpc_acados.speed_monitor_core import extract_speeds, trim_window


class TestExtractSpeeds:
    def test_extracts_default_indices(self):
        # idx0=v_cmd, idx2=v_actual, idx8=ref_v (DBG_FIELDS 순서)
        data = [1.0, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]
        assert extract_speeds(data) == (1.0, 2.0, 8.0)

    def test_short_array_returns_none(self):
        assert extract_speeds([1.0, 0.1, 2.0]) is None      # len 3 < 9

    def test_boundary_length_ok(self):
        data = [0.0] * 9
        data[0], data[2], data[8] = 3.0, 4.0, 5.0
        assert extract_speeds(data) == (3.0, 4.0, 5.0)      # len == max_idx+1

    def test_custom_indices(self):
        assert extract_speeds([9.0, 8.0, 7.0], i_cmd=0, i_actual=1, i_ref=2) == (9.0, 8.0, 7.0)


class TestTrimWindow:
    def test_empty_returns_zero(self):
        assert trim_window([], 10.0, 5.0) == 0

    def test_all_recent_zero(self):
        assert trim_window([8.0, 9.0, 10.0], 10.0, 5.0) == 0   # cutoff=5, 전부 ≥5

    def test_drops_old(self):
        # cutoff = 10-5 = 5 → 5 미만(1,2) 버림 → 2
        assert trim_window([1.0, 2.0, 6.0, 9.0], 10.0, 5.0) == 2

    def test_boundary_not_old(self):
        # t == cutoff(5.0) 은 오래된 것 아님(strict <)
        assert trim_window([5.0, 6.0], 10.0, 5.0) == 0

    def test_all_old(self):
        assert trim_window([1.0, 2.0], 100.0, 5.0) == 2
