# Walk-forward backtest engine with transaction costs.
# Timing (same for the benchmarks): a decision at t uses data up to the close
# of t, the new weights earn returns from t+1, and the rebalance cost is
# deducted from the t+1 return. The HMM is refit every refit_every days on an
# expanding window; between refits the last model is reused but only ever fed
# data up to the decision date (filtered decode in regime.py).

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BACKTEST_CONFIG, HMM_CONFIG, OPTIMIZER_CONFIG, BacktestConfig
from .features import build_features
from .optimize import estimate_moments, solve_weights
from .regime import filtered_posteriors, fit_hmm

logger=logging.getLogger(__name__)


def turnover_cost(turnover: float, tc_bps: float) -> float:
    # cost fraction for one rebalance: sum|w_new - w_old| * bps
    return turnover * tc_bps / 1e4


def drift_weights(w: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    # weights after one day of market drift (no trading)
    grown=w * (1.0 + asset_returns)
    return grown / grown.sum()


@dataclass
class BacktestResult:
    returns: pd.Series        # daily net portfolio returns
    weights: pd.DataFrame     # weights held at the start of each day
    regimes: pd.Series        # causal regime label per decision date
    turnover: pd.Series       # turnover at each rebalance date
    targets: pd.DataFrame     # optimizer targets at each rebalance date

    @property
    def annual_turnover(self) -> float:
        years=len(self.returns) / 252
        return float(self.turnover.sum() / years) if years > 0 else 0.0


def simulate_targets(
    returns: pd.DataFrame, targets: pd.DataFrame, tc_bps: float
) -> BacktestResult:
    # simulate a path of pre-decided target weights (used by benchmarks).
    # targets is indexed by rebalance dates; weights drift in between.
    assets=list(returns.columns)
    dates=returns.index[returns.index >= targets.index[0]]
    target_iter={d: targets.loc[d].to_numpy() for d in targets.index}

    w=target_iter[targets.index[0]]
    pending_cost=0.0
    daily_ret, held, turn={}, {}, {}

    for t in dates[1:]:  # first target date earns nothing itself
        r=returns.loc[t].to_numpy()
        held[t]=w
        daily_ret[t]=float(w @ r) - pending_cost
        pending_cost=0.0
        w=drift_weights(w, r)
        if t in target_iter:
            new_w=target_iter[t]
            tover=float(np.abs(new_w - w).sum())
            turn[t]=tover
            pending_cost=turnover_cost(tover, tc_bps)
            w=new_w

    return BacktestResult(
        returns=pd.Series(daily_ret, name="ret"),
        weights=pd.DataFrame(held, index=assets).T,
        regimes=pd.Series(dtype=object),
        turnover=pd.Series(turn, dtype=float),
        targets=targets,
    )


def run_walkforward(
    prices: pd.DataFrame,
    cfg: BacktestConfig=BACKTEST_CONFIG,
) -> BacktestResult:
    # full regime-switching strategy. Rebalances on a fixed cadence AND on
    # confident regime switches. The acting regime is the filtered argmax with
    # hysteresis: it flips only once the new regime clears switch_confidence,
    # which stops boundary flapping from triggering costly daily rebalances.
    returns=prices.pct_change().dropna()
    feats=build_features(prices["equity"])
    dates=feats.index
    assets=list(returns.columns)
    n_assets=len(assets)

    if len(dates) <= cfg.warmup_days:
        raise ValueError("not enough history for the warmup window")

    fitted=None
    posteriors: pd.DataFrame | None=None
    w: np.ndarray | None=None
    pending_cost=0.0
    acting: str | None=None

    daily_ret: dict[pd.Timestamp, float]={}
    held: dict[pd.Timestamp, np.ndarray]={}
    regimes: dict[pd.Timestamp, str]={}
    turn: dict[pd.Timestamp, float]={}
    tgts: dict[pd.Timestamp, np.ndarray]={}

    for i in range(cfg.warmup_days, len(dates)):
        t=dates[i]

        # 1) book today's return with the weights carried into today
        if w is not None:
            r=returns.loc[t].to_numpy()
            held[t]=w
            daily_ret[t]=float(w @ r) - pending_cost
            pending_cost=0.0
            w=drift_weights(w, r)

        # 2) periodic refit on data up to and including t
        if (i - cfg.warmup_days) % cfg.refit_every == 0:
            fitted=fit_hmm(feats.iloc[: i + 1])
            # one forward pass over the whole history stays causal: row t
            # depends only on rows <= t, so reading it later is fine.
            posteriors=filtered_posteriors(fitted, feats)
            logger.info(
                "refit at %s: state means (ret) %s",
                t.date(),
                np.round(fitted.model.means_[:, 0], 3),
            )

        # 3) causal acting regime for today (argmax with hysteresis)
        post_t=posteriors.loc[t]
        top=post_t.idxmax()
        if acting is None:
            acting=top
        elif top != acting and post_t[top] >= cfg.switch_confidence:
            acting=top
        regimes[t]=acting

        # 4) rebalance on schedule or on an acting-regime flip
        scheduled=(i - cfg.warmup_days) % cfg.rebalance_every == 0
        flipped=i > cfg.warmup_days and regimes.get(dates[i - 1]) != acting
        if scheduled or flipped:
            window=returns.loc[:t].tail(cfg.est_window)
            mu, sigma=estimate_moments(window)
            w_prev=w if w is not None else np.full(n_assets, 1.0 / n_assets)
            new_w=solve_weights(mu, sigma, acting, w_prev, OPTIMIZER_CONFIG)
            if w is not None:
                tover=float(np.abs(new_w - w).sum())
                turn[t]=tover
                pending_cost=turnover_cost(tover, cfg.tc_bps)
            tgts[t]=new_w
            w=new_w

    return BacktestResult(
        returns=pd.Series(daily_ret, name="ret"),
        weights=pd.DataFrame(held, index=assets).T,
        regimes=pd.Series(regimes, name="regime"),
        turnover=pd.Series(turn, dtype=float),
        targets=pd.DataFrame(tgts, index=assets).T,
    )
