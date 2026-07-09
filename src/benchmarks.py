# Benchmark portfolios: 60/40 and equal-weight. Both run through the same
# simulate_targets engine as the strategy (same cadence, drift and costs), so
# any outperformance comes from the regime signal, not gentler treatment.

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, simulate_targets

SIXTY_FORTY={"equity": 0.60, "gold": 0.0, "bonds": 0.40}
EQUAL_WEIGHT={"equity": 1 / 3, "gold": 1 / 3, "bonds": 1 / 3}


def run_fixed_mix(
    returns: pd.DataFrame,
    mix: dict[str, float],
    rebalance_dates: pd.DatetimeIndex,
    tc_bps: float,
) -> BacktestResult:
    # constant-mix benchmark rebalanced back to mix on the given dates
    row=pd.Series(mix).reindex(returns.columns).fillna(0.0)
    targets=pd.DataFrame([row] * len(rebalance_dates), index=rebalance_dates)
    return simulate_targets(returns, targets, tc_bps)


def run_benchmarks(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    tc_bps: float,
) -> dict[str, BacktestResult]:
    # both benchmarks on the strategy's rebalance calendar
    return {
        "60/40": run_fixed_mix(returns, SIXTY_FORTY, rebalance_dates, tc_bps),
        "equal_weight": run_fixed_mix(returns, EQUAL_WEIGHT, rebalance_dates, tc_bps),
    }
