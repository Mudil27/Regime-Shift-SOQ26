# Data loading: yfinance download, on-disk cache, causal cleaning and
# alignment. Each asset class has a candidate list because yfinance coverage
# of Indian ETFs is patchy; we take the first candidate with enough history.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import CACHE_DIR, END_DATE, START_DATE, TICKERS

logger=logging.getLogger(__name__)

MIN_HISTORY_ROWS=750      # ~3y of closes to be a usable leg
MAX_FFILL_DAYS=5          # forward-fill at most a week of gaps (past-only)
BAD_PRINT_LOG_RET=0.35    # |daily log ret| above this is a vendor bad print


def _cache_path(ticker: str) -> Path:
    safe=ticker.replace("^", "_").replace(".", "_")
    return Path(CACHE_DIR) / f"{safe}.csv"


def download_ticker(ticker: str, refresh: bool=False) -> pd.Series:
    # adjusted-close series, served from the disk cache unless refresh=True
    path=_cache_path(ticker)
    if path.exists() and not refresh:
        s=pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name=ticker
        return s

    raw=yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # dividend/split-adjusted closes
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=ticker)

    close=raw["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns MultiIndex cols
        close=close.iloc[:, 0]
    close=close.dropna()
    close.name=ticker

    path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(path)
    return close


def _clean_bad_prints(s: pd.Series) -> pd.Series:
    # each price is judged only against the previous accepted price, so this
    # stays causal (future rows can't change how a past row is cleaned).
    values=s.to_numpy(copy=True)
    n_fixed=0
    for i in range(1, len(values)):
        if abs(np.log(values[i] / values[i - 1])) > BAD_PRINT_LOG_RET:
            values[i]=values[i - 1]
            n_fixed+=1
    if n_fixed:
        logger.warning("%s: replaced %d bad print(s)", s.name, n_fixed)
    return pd.Series(values, index=s.index, name=s.name)


def _pick_ticker(
    asset: str, candidates: list[str], refresh: bool
) -> tuple[str, pd.Series]:
    # the aligned panel is only as long as its shortest asset, so pick the
    # qualifying candidate with the earliest start. This is why LIQUIDBEES
    # (2009-) wins the bond leg over the gilt ETFs (2016+).
    best: pd.Series | None=None
    for ticker in candidates:
        s=download_ticker(ticker, refresh=refresh)
        if len(s) < MIN_HISTORY_ROWS:
            logger.warning(
                "%s: candidate %s rejected (%d rows < %d)",
                asset, ticker, len(s), MIN_HISTORY_ROWS,
            )
            continue
        if best is None or s.index[0] < best.index[0]:
            best=s
    if best is None:
        raise RuntimeError(
            f"No usable ticker for {asset!r}; tried {candidates}. "
            "Check connectivity or edit config.TICKERS."
        )
    logger.info(
        "%s -> %s (%d rows, %s to %s)",
        asset, best.name, len(best), best.index[0].date(), best.index[-1].date(),
    )
    return str(best.name), _clean_bad_prints(best).rename(asset)


def _align(series: list[pd.Series]) -> pd.DataFrame:
    # union of trading days, ffill short gaps, trim to the common span
    prices=pd.concat(series, axis=1).sort_index()
    prices=prices.ffill(limit=MAX_FFILL_DAYS)

    common_start=max(s.dropna().index[0] for s in series)
    prices=prices.loc[common_start:].dropna()
    prices.index.name="date"

    logger.info(
        "aligned panel: %d days, %s to %s",
        len(prices), prices.index[0].date(), prices.index[-1].date(),
    )
    return prices


def load_prices(refresh: bool=False) -> pd.DataFrame:
    # align all asset classes into one price frame; columns are asset names.
    return _align([
        _pick_ticker(asset, candidates, refresh)[1]
        for asset, candidates in TICKERS.items()
    ])


def load_prices_from_dir(data_dir: str | Path) -> pd.DataFrame:
    # load the panel from user CSVs instead of yfinance. One file per asset
    # class (equity.csv, gold.csv, bonds.csv); date column first, then a
    # close column ('Close' if present, else the first numeric column).
    series=[]
    for asset in TICKERS:
        path=Path(data_dir) / f"{asset}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} — provide one CSV per asset class named "
                f"{', '.join(a + '.csv' for a in TICKERS)}"
            )
        df=pd.read_csv(path)
        df[df.columns[0]]=pd.to_datetime(df[df.columns[0]])
        df=df.set_index(df.columns[0]).sort_index()
        col="Close" if "Close" in df.columns else \
            df.select_dtypes("number").columns[0]
        s=df[col].dropna().rename(asset)
        logger.info("%s <- %s (%d rows, %s to %s)", asset, path,
                    len(s), s.index[0].date(), s.index[-1].date())
        series.append(_clean_bad_prints(s))
    return _align(series)


def chosen_tickers(refresh: bool=False) -> dict[str, str]:
    # asset class -> ticker actually used (for logging)
    return {
        asset: _pick_ticker(asset, candidates, refresh)[0]
        for asset, candidates in TICKERS.items()
    }
