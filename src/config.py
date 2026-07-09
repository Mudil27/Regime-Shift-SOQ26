# All tunables in one place. Values are fixed on principled grounds, not
# tuned against the holdout.

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
CACHE_DIR=PROJECT_ROOT / "data" / "cache"
RESULTS_DIR=PROJECT_ROOT / "results"

# INR-only instruments so regime signals aren't polluted by USDINR moves.
# asset class -> candidate tickers, first with usable history wins.
TICKERS: dict[str, list[str]]={
    "equity": ["^NSEI", "NIFTYBEES.NS"],
    "gold": ["GOLDBEES.NS", "SETFGOLD.NS"],
    # bond ETF coverage on yfinance is patchy, so several fallbacks.
    "bonds": ["LTGILTBEES.NS", "GILT5YBEES.NS", "EBBETF0430.NS", "LIQUIDBEES.NS"],
}

START_DATE="2007-01-01"  # covers the 2008 and 2020 stress
END_DATE: str | None=None  # None means today

RANDOM_STATE=42


@dataclass(frozen=True)
class HMMConfig:
    n_states: int=3          # bull / bear / crisis
    n_em_inits: int=5        # EM restarts, keep the best log-likelihood
    covariance_type: str="full"
    n_iter: int=200
    vol_window: int=21       # ~1 month realized-vol window


@dataclass(frozen=True)
class BacktestConfig:
    warmup_days: int=1260    # ~5y before the first trade
    refit_every: int=126     # refit the HMM every ~6 months
    rebalance_every: int=21  # monthly scheduled rebalance
    est_window: int=252      # trailing window for mu/Sigma
    # only switch acting regime once the filter is >=70% sure, else argmax
    # flapping at boundaries churns the portfolio.
    switch_confidence: float=0.70
    tc_bps: float=7.0        # cost per unit turnover, sane range 5-10
    holdout_frac: float=0.20 # last 20% is a locked holdout


@dataclass(frozen=True)
class OptimizerConfig:
    max_weight: float=0.75   # per-asset cap keeps some diversification
    # risk aversion per regime: bull takes risk, crisis is near min-variance.
    risk_aversion: dict[str, float]=field(
        default_factory=lambda: {"bull": 5.0, "bear": 15.0, "crisis": 50.0}
    )
    # shrink expected returns toward zero; crisis ignores mu entirely.
    mu_shrinkage: dict[str, float]=field(
        default_factory=lambda: {"bull": 0.0, "bear": 0.5, "crisis": 1.0}
    )
    # L1 turnover penalty, calibrated to the 7 bps cost so we only trade
    # when the expected gain beats the cost.
    turnover_penalty: float=0.01


HMM_CONFIG=HMMConfig()
BACKTEST_CONFIG=BacktestConfig()
OPTIMIZER_CONFIG=OptimizerConfig()
