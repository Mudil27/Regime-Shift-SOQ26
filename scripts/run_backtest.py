#!/usr/bin/env python
# Run the full walk-forward backtest and print the results tables.
# Usage: python scripts/run_backtest.py [--refresh] [--tc-bps 7] [--rebalance-every 21] [--data-dir DIR]

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import run_walkforward
from src.benchmarks import run_benchmarks
from src.config import BACKTEST_CONFIG, RESULTS_DIR
from src.data import chosen_tickers, load_prices, load_prices_from_dir
from src.metrics import regime_stats, save_all_plots, stats_table


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Run the RegimeShift walk-forward backtest.")
    p.add_argument("--refresh", action="store_true",
                   help="force re-download instead of using data/cache/")
    p.add_argument("--tc-bps", type=float, default=BACKTEST_CONFIG.tc_bps,
                   help="transaction cost in bps of turnover (default %(default)s)")
    p.add_argument("--rebalance-every", type=int,
                   default=BACKTEST_CONFIG.rebalance_every,
                   help="scheduled rebalance cadence in trading days "
                        "(default %(default)s)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="load prices from equity.csv/gold.csv/bonds.csv in "
                        "this directory instead of yfinance (for evaluating "
                        "on your own data)")
    return p.parse_args()


def fmt(table: pd.DataFrame) -> str:
    out=table.copy()
    for col in ("CAGR", "AnnVol", "MaxDD"):
        out[col]=out[col].map("{:+.2%}".format)
    for col in ("Sharpe", "Sortino", "Calmar"):
        out[col]=out[col].map("{:.2f}".format)
    if "AnnTurnover" in out:
        out["AnnTurnover"]=out["AnnTurnover"].map("{:.2f}".format)
    return out.to_string()


def report(title: str, named: dict[str, pd.Series],
           turnover: dict[str, float] | None=None) -> None:
    print(f"\n=== {title} ===")
    print(fmt(stats_table(named, turnover)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args=parse_args()

    cfg=dataclasses.replace(
        BACKTEST_CONFIG, tc_bps=args.tc_bps, rebalance_every=args.rebalance_every
    )

    print("Loading data...")
    if args.data_dir is not None:
        print(f"Using user-supplied CSVs from {args.data_dir}")
        prices=load_prices_from_dir(args.data_dir)
    else:
        print("Tickers:", chosen_tickers(refresh=args.refresh))
        prices=load_prices(refresh=args.refresh)
    returns=prices.pct_change().dropna()

    print("\nRunning walk-forward backtest "
          f"(tc={cfg.tc_bps}bps, rebalance every {cfg.rebalance_every}d)...")
    strat=run_walkforward(prices, cfg)

    scheduled=strat.targets.index  # benchmarks trade on the same calendar
    benches=run_benchmarks(returns, scheduled, cfg.tc_bps)

    named={"RegimeShift": strat.returns,
           **{k: v.returns for k, v in benches.items()}}
    turnover={"RegimeShift": strat.annual_turnover,
              **{k: v.annual_turnover for k, v in benches.items()}}

    # the last holdout_frac of the panel was never used in development;
    # the split date depends only on panel length, not on results.
    holdout_start=prices.index[int(len(prices) * (1 - cfg.holdout_frac))]
    dev={k: v.loc[v.index < holdout_start] for k, v in named.items()}
    holdout={k: v.loc[v.index >= holdout_start] for k, v in named.items()}

    report("Full walk-forward period", named, turnover)
    report(f"Development period (pre {holdout_start.date()})", dev)
    report(f"LOCKED HOLDOUT ({holdout_start.date()} onward, "
           "never used in development)", holdout)

    print("\n=== Strategy returns by detected regime ===")
    print(regime_stats(strat.returns, strat.regimes).round(4).to_string())
    print("\n=== Time spent per regime (days) ===")
    print(strat.regimes.value_counts().to_string())

    paths=save_all_plots(
        strat.returns,
        {k: v.returns for k, v in benches.items()},
        prices["equity"],
        strat.regimes,
        strat.weights,
        RESULTS_DIR,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_table(named, turnover).to_csv(RESULTS_DIR / "summary.csv")
    pd.DataFrame(named).to_csv(RESULTS_DIR / "daily_returns.csv")
    strat.weights.to_csv(RESULTS_DIR / "weights.csv")
    strat.regimes.to_csv(RESULTS_DIR / "regimes.csv")

    print("\nSaved plots:")
    for p in paths:
        print(" ", p)
    print("Saved tables: summary.csv, daily_returns.csv, weights.csv, regimes.csv")


if __name__ == "__main__":
    main()
