"""Codegen path isolation — switching kinematic<->dynamic (또는 dyn_mu) 가
stale acados codegen 디렉터리를 재사용하면 안 된다.

고정 export dir 시절 use_dynamic 을 토글하면 acados 가 이전 codegen 을 그대로
재사용해 조용히 틀린 모델로 돌았다. export dir + json 을 OCP 구조에 키잉해서
설정마다 별도 codegen 을 갖게 한 것이 그 수정이다.

2026-08-18: LMPC(joint nx=18) 는 제거된 기능이라 해당 테스트 2개 삭제.
캐시 위치는 2026-07-16 부터 /tmp 가 아니라 ~/.acados_codegen 이다.

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_codegen_paths -v
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.acados_kinematic import codegen_paths


class TestCodegenPaths(unittest.TestCase):
    def test_kinematic_and_dynamic_differ(self):
        # 2026-06-10 unified layout: kinematic is ALSO 8-state (f_kin) — the
        # dyn/kin tag alone must keep the codegens apart at identical nx.
        kin = codegen_paths(use_dynamic=False, nx_solver=8)
        dyn = codegen_paths(use_dynamic=True, nx_solver=8)
        self.assertNotEqual(kin[0], dyn[0], "export dir must differ kin vs dyn")
        self.assertNotEqual(kin[1], dyn[1], "json must differ kin vs dyn")

    def test_deterministic_for_same_config(self):
        a = codegen_paths(use_dynamic=True, nx_solver=8)
        b = codegen_paths(use_dynamic=True, nx_solver=8)
        self.assertEqual(a, b, "same config must map to the same paths (warm reuse)")

    def test_returns_dir_and_json_pair(self):
        export_dir, json_path = codegen_paths(use_dynamic=True, nx_solver=8)
        self.assertIn("/.acados_codegen/", export_dir, "export dir under ~/.acados_codegen")
        self.assertTrue(json_path.endswith(".json"), "json path ends in .json")

    def test_mu_keys_the_codegen(self):
        # dyn_mu 는 tanh 타이어/ellipse 에 codegen-time 으로 박힌다 —
        # mu 0.6 vs 1.0489 가 같은 dir 을 재사용하면 stale 모델.
        lo = codegen_paths(use_dynamic=True, nx_solver=8,
                           dyn_mu=0.6)
        hi = codegen_paths(use_dynamic=True, nx_solver=8,
                           dyn_mu=1.0489)
        self.assertNotEqual(lo[0], hi[0], "mu must key the export dir")
        self.assertNotEqual(lo[1], hi[1], "mu must key the json")

    def test_mu_default_keeps_legacy_tag(self):
        # mu 미지정(레거시 호출) → 기존 태그 그대로 (경로 호환).
        legacy = codegen_paths(use_dynamic=True, nx_solver=8)
        self.assertTrue(legacy[0].endswith("/acados_codegen_evompcc_dyn8"))


if __name__ == "__main__":
    unittest.main()
