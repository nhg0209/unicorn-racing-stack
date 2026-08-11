#!/usr/bin/env python3
"""gp_eval_held_out.py — 2단계 (A) held-out GP 적합도.

CSV 단위 train/test 분할(시간 누수 방지) 후, train_gp_residual 과 동일한
inducing-point GP 를 train 으로 적합하고 test 에서 per-output RMSE + 평균 NLL 측정.
ekf vs finitediff 데이터셋을 같은 split·같은 하이퍼파라미터로 비교.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import gpytorch
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_residuals as er


def _dataset_per_source(csv_paths, source):
    """각 CSV 를 하나의 그룹으로 → (X_list, Y_list) CSV 단위."""
    X_list, Y_list = [], []
    for f in csv_paths:
        x, y = er.process_csv(Path(f), source=source)
        if len(x) > 0:
            X_list.append(x); Y_list.append(y)
    return X_list, Y_list


def _train_eval(Xtr, Ytr, Xte, Yte, inducing, iters):
    """정규화 → GP 적합(train) → test RMSE(real) + 평균 NLL. train_gp_residual 레시피."""
    Xtr = torch.tensor(Xtr).double(); Ytr = torch.tensor(Ytr).double()
    Xte = torch.tensor(Xte).double(); Yte = torch.tensor(Yte).double()
    Xm, Xs = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Ym, Ys = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
    Xtrn, Ytrn = (Xtr - Xm) / Xs, (Ytr - Ym) / Ys
    Xten, Yten = (Xte - Xm) / Xs, (Yte - Ym) / Ys

    nout = Ytrn.size(-1)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout).double()
    gp = BatchIndependentInducingPointGPModel(
        Xtrn, Ytrn, lik, inducing_points=min(inducing, Xtrn.size(0)),
        use_ard=True, residual_dimension=nout).double()

    gp.train(); lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
    opt = torch.optim.Adam(gp.parameters(), lr=0.05)
    for _ in range(iters):
        opt.zero_grad(); loss = -mll(gp(Xtrn), Ytrn).sum(); loss.backward(); opt.step()

    gp.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(gp(Xten))
        rmse_real = ((pred.mean - Yten) ** 2).mean(0).sqrt() * Ys
        # NLL 을 real-Y 공간으로 보정(+Σ log Ys): 소스마다 Ys 가 다르므로 정규화-공간
        # NLL 은 소스 간 절대 비교가 불가. 이 오프셋이 있어야 ekf vs finitediff 가 공정.
        nll = -pred.log_prob(Yten).item() / max(Yten.size(0), 1) + float(Ys.log().sum())
    return [float(v) for v in rmse_real], float(nll)


def eval_held_out_gp(csv_paths, inducing=200, iters=200, seed=42) -> dict:
    torch.manual_seed(seed)
    # CSV 단위 split: 두 소스가 같은 train/test CSV 를 쓰도록 인덱스 고정
    n = len(csv_paths)
    if n < 2:
        print("\n(A) GP 평가: CSV 가 2개 미만 → split 불가. 스킵.")
        return {}
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = max(1, n // 5)
    test_idx, train_idx = set(perm[:n_test].tolist()), set(perm[n_test:].tolist())
    train_csvs = [csv_paths[i] for i in sorted(train_idx)]
    test_csvs = [csv_paths[i] for i in sorted(test_idx)]
    print(f"\n=== (A) held-out GP (train {len(train_csvs)} CSV / test {len(test_csvs)} CSV) ===")

    out = {}
    print(f"{'source':10s} | {'rmse_vx':>9s} {'rmse_vy':>9s} {'rmse_r':>9s} | {'NLL':>9s}")
    for src in ("ekf", "finitediff"):
        Xtr_l, Ytr_l = _dataset_per_source(train_csvs, src)
        Xte_l, Yte_l = _dataset_per_source(test_csvs, src)
        if not Xtr_l or not Xte_l:
            print(f"{src:10s} | (데이터 부족 — skip)")
            continue
        Xtr, Ytr = np.concatenate(Xtr_l), np.concatenate(Ytr_l)
        Xte, Yte = np.concatenate(Xte_l), np.concatenate(Yte_l)
        torch.manual_seed(seed)  # 두 소스 동일 초기화 → 공정 비교
        rmse, nll = _train_eval(Xtr, Ytr, Xte, Yte, inducing, iters)
        out[src] = {"rmse": rmse, "nll": nll}
        print(f"{src:10s} | {rmse[0]:9.4f} {rmse[1]:9.4f} {rmse[2]:9.4f} | {nll:9.3f}")
    print("\n(해석: ekf 의 rmse·NLL 이 finitediff 보다 낮으면 EKF 타깃이 GP 정확도↑.\n"
          " 더 높으면 가정 재검토 — 결론 강제하지 않음."
          " ekf/finitediff 는 필터·범위가 달라 표본 행집합이 완전히 같지 않으니 작은 차이는 과해석 금지.)")
    return out
