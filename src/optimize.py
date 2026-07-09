# Regime-conditional mean-variance optimization (cvxpy).
#   maximize  mu_eff'w - gamma(regime) * w'Sigma w - tp * |w - w_prev|_1
#   s.t.      sum(w)=1,  0 <= w <= max_weight
# The regime enters via two knobs: gamma (risk aversion) and mu shrinkage.
# Full shrinkage in crisis turns it into pure minimum-variance.

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import OPTIMIZER_CONFIG, OptimizerConfig

TRADING_DAYS=252


def estimate_moments(returns_window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # annualized mu and Ledoit-Wolf-shrunk Sigma from a trailing window.
    # Caller must pass only rows up to the decision date.
    mu=returns_window.mean().to_numpy() * TRADING_DAYS
    sigma=LedoitWolf().fit(returns_window.to_numpy()).covariance_ * TRADING_DAYS
    return mu, sigma


def solve_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    regime: str,
    w_prev: np.ndarray,
    cfg: OptimizerConfig=OPTIMIZER_CONFIG,
) -> np.ndarray:
    # solve for target weights; fall back to w_prev if the solver fails so a
    # long backtest degrades gracefully instead of crashing.
    n=len(mu)
    gamma=cfg.risk_aversion[regime]
    mu_eff=(1.0 - cfg.mu_shrinkage[regime]) * mu

    w=cp.Variable(n)
    objective=(
        mu_eff @ w
        - gamma * cp.quad_form(w, cp.psd_wrap(sigma))
        - cfg.turnover_penalty * cp.norm1(w - w_prev)
    )
    problem=cp.Problem(
        cp.Maximize(objective),
        [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight],
    )
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return w_prev.copy()
    if w.value is None:
        return w_prev.copy()

    # clip solver noise (tiny negatives) and renormalize
    out=np.clip(np.asarray(w.value).ravel(), 0.0, None)
    return out / out.sum()
