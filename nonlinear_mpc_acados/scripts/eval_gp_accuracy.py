#!/usr/bin/env python3
"""eval_gp_accuracy.py — 2단계 측정.

같은 CSV 집합에서 residual 추출 소스(ekf vs finitediff)를 공정 비교:
  (B) residual 타깃 노이즈: 각 출력 std + lag-1 자기상관(ac1).
  (A) held-out GP 적합도: CSV 단위 train/test 분할 후 per-output RMSE + 평균 NLL.
      (torch/gpytorch/l4acados 필요. 없으면 A 스킵.)

Usage:
    cd nonlinear_mpc_acados && PYTHONPATH=.:$HOME/l4acados/src \
        python3 scripts/eval_gp_accuracy.py
    python3 scripts/eval_gp_accuracy.py --csv-glob '~/mpc_logs/mpc_*.csv' --no-gp
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_residuals as er


def noise_metrics(residuals: np.ndarray) -> dict:
    """per-output std 와 lag-1 자기상관(ac1). residuals: [N,3]."""
    if residuals.shape[0] < 3:
        return {"std": [float("nan")] * 3, "ac1": [float("nan")] * 3}
    std = residuals.std(axis=0)
    ac1 = []
    for j in range(residuals.shape[1]):
        a = residuals[:-1, j]
        b = residuals[1:, j]
        a0, b0 = a - a.mean(), b - b.mean()
        denom = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
        ac1.append(float((a0 * b0).sum() / denom) if denom > 1e-12 else float("nan"))
    return {"std": [float(v) for v in std], "ac1": ac1}


def collect_residuals(csv_paths, source: str) -> np.ndarray:
    """여러 CSV 의 residual([N,3])을 이어붙여 반환."""
    chunks = []
    for f in csv_paths:
        _, y = er.process_csv(Path(f), source=source)
        if len(y) > 0:
            chunks.append(y)
    if not chunks:
        return np.empty((0, 3))
    return np.concatenate(chunks, axis=0)


def _print_noise_table(by_source: dict):
    names = ["d_vx", "d_vy", "d_r"]
    print("\n=== (B) residual 타깃 노이즈 ===")
    print(f"{'output':8s} | {'ekf.std':>9s} {'fd.std':>9s} | {'ekf.ac1':>9s} {'fd.ac1':>9s}")
    for j, nm in enumerate(names):
        e, f = by_source["ekf"], by_source["finitediff"]
        print(f"{nm:8s} | {e['std'][j]:9.4f} {f['std'][j]:9.4f} | "
              f"{e['ac1'][j]:9.3f} {f['ac1'][j]:9.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-glob", default=str(er.MPC_LOGS / "mpc_*.csv"))
    p.add_argument("--no-gp", action="store_true", help="(A) GP 평가 스킵, B만")
    p.add_argument("--inducing", type=int, default=200)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import glob
    csvs = sorted(glob.glob(str(Path(args.csv_glob).expanduser())))
    print(f"Found {len(csvs)} CSV(s)")
    if not csvs:
        print("CSV 없음. --csv-glob 확인."); return

    # (B) 노이즈
    by_source = {src: noise_metrics(collect_residuals(csvs, src))
                 for src in ("ekf", "finitediff")}
    _print_noise_table(by_source)

    # (A) held-out GP
    if args.no_gp:
        print("\n(A) GP 평가 스킵 (--no-gp)")
        return
    try:
        from gp_eval_held_out import eval_held_out_gp  # Task 5 에서 추가
    except Exception as e:
        print(f"\n(A) GP 평가 불가 ({e}). --no-gp 로 B만 보거나 Task 5 미구현.")
        return
    eval_held_out_gp(csvs, inducing=args.inducing, iters=args.iters, seed=args.seed)


if __name__ == "__main__":
    main()
