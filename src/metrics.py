# Performance stats and plots. Sharpe/Sortino use a zero risk-free rate — the
# cash-like bond leg sets an implicit floor and a constant rf shifts every
# strategy's Sharpe equally.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR

TRADING_DAYS=252

# plot palette (light mode). Series colors are categorical; regime shading
# uses status colors with a labeled legend.
SERIES_COLORS={0: "#2a78d6", 1: "#1baf7a", 2: "#eda100"}
REGIME_COLORS={"bull": "#0ca30c", "bear": "#fab219", "crisis": "#d03b3b"}
SURFACE="#fcfcfb"
INK="#0b0b0b"
INK_SECONDARY="#52514e"
MUTED="#898781"
GRID="#e1e0d9"
BASELINE="#c3c2b7"


def perf_stats(returns: pd.Series) -> dict[str, float]:
    # standard annualized stats from daily returns
    r=returns.dropna()
    if r.empty:
        return {k: np.nan for k in
                ("CAGR", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar")}

    wealth=(1 + r).cumprod()
    years=len(r) / TRADING_DAYS
    cagr=float(wealth.iloc[-1] ** (1 / years) - 1)
    vol=float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe=float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan

    downside=r[r < 0]
    sortino=(
        float(r.mean() * TRADING_DAYS / (downside.std() * np.sqrt(TRADING_DAYS)))
        if len(downside) > 1 and downside.std() > 0
        else np.nan
    )
    dd=wealth / wealth.cummax() - 1
    max_dd=float(dd.min())
    calmar=cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {"CAGR": cagr, "AnnVol": vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": max_dd, "Calmar": calmar}


def stats_table(named_returns: dict[str, pd.Series],
                annual_turnover: dict[str, float] | None=None) -> pd.DataFrame:
    # one row per portfolio, optionally with average annual turnover
    table=pd.DataFrame({name: perf_stats(r) for name, r in named_returns.items()}).T
    if annual_turnover:
        table["AnnTurnover"]=pd.Series(annual_turnover)
    return table


def regime_stats(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    # annualized return/vol conditioned on the regime
    aligned=pd.concat([returns.rename("ret"), regimes.rename("regime")],
                      axis=1).dropna()
    rows={}
    for label, grp in aligned.groupby("regime"):
        rows[label]={
            "days": len(grp),
            "share": len(grp) / len(aligned),
            "ann_ret": grp["ret"].mean() * TRADING_DAYS,
            "ann_vol": grp["ret"].std() * np.sqrt(TRADING_DAYS),
        }
    return pd.DataFrame(rows).T.sort_index()


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def _new_fig(title: str, subtitle: str | None=None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax=plt.subplots(figsize=(10, 5), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_SECONDARY, pad=10)
    return fig, ax


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path=out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_cumulative(named_returns: dict[str, pd.Series],
                    out_dir: Path=RESULTS_DIR) -> Path:
    # wealth of 1 unit invested, strategy vs benchmarks, log scale
    fig, ax=_new_fig("Cumulative growth of 1", "Net of transaction costs, log scale")
    ends: list[tuple[str, pd.Timestamp, float]]=[]
    for i, (name, r) in enumerate(named_returns.items()):
        wealth=(1 + r.dropna()).cumprod()
        color=SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(wealth.index, wealth, lw=2, color=color, label=name)
        ends.append((name, wealth.index[-1], float(wealth.iloc[-1])))
    ax.set_yscale("log")

    # end-of-line labels, dodged apart in log space so close finishes don't collide
    ymin, ymax=ax.get_ylim()
    min_gap=0.045 * (np.log10(ymax) - np.log10(ymin))
    ends.sort(key=lambda e: e[2])
    log_pos=[np.log10(v) for _, _, v in ends]
    for j in range(1, len(log_pos)):
        log_pos[j]=max(log_pos[j], log_pos[j - 1] + min_gap)
    for (name, x, _), ly in zip(ends, log_pos):
        ax.annotate(f" {name}", (x, 10 ** ly),
                    color=INK_SECONDARY, fontsize=9, va="center")

    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    ax.margins(x=0.10)
    return _save(fig, "cumulative_returns.png", out_dir)


def plot_drawdown(named_returns: dict[str, pd.Series],
                  out_dir: Path=RESULTS_DIR) -> Path:
    fig, ax=_new_fig("Drawdown", "Peak-to-trough decline")
    for i, (name, r) in enumerate(named_returns.items()):
        wealth=(1 + r.dropna()).cumprod()
        dd=wealth / wealth.cummax() - 1
        ax.plot(dd.index, dd, lw=2, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                label=name)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower left")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return _save(fig, "drawdowns.png", out_dir)


def plot_regimes(equity: pd.Series, regimes: pd.Series,
                 out_dir: Path=RESULTS_DIR) -> Path:
    # equity index with the causal regime label as background shading
    fig, ax=_new_fig("Equity index with detected regimes",
                     "Shading = filtered (real-time) regime label")
    reg=regimes.dropna()
    px=equity.loc[reg.index[0]:]

    # merge consecutive same-label days into one shaded span
    change=(reg != reg.shift(1)).cumsum()
    for _, block in reg.groupby(change):
        ax.axvspan(block.index[0], block.index[-1],
                   color=REGIME_COLORS[block.iloc[0]], alpha=0.14, lw=0)
    ax.plot(px.index, px, lw=1.5, color=INK_SECONDARY)

    handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
             for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), frameon=False,
              labelcolor=INK_SECONDARY, fontsize=9, loc="upper left")
    return _save(fig, "regimes.png", out_dir)


def plot_weights(weights: pd.DataFrame, out_dir: Path=RESULTS_DIR) -> Path:
    # weight evolution as a stacked area (weights sum to 1)
    fig, ax=_new_fig("Portfolio weights over time",
                     "Held weights, drifting between rebalances")
    w=weights.clip(lower=0)
    colors=[SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(w.shape[1])]
    ax.stackplot(w.index, w.T.to_numpy(), labels=w.columns, colors=colors,
                 alpha=0.85, edgecolor=SURFACE, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9,
              loc="upper left", ncol=3)
    return _save(fig, "weights.png", out_dir)


def save_all_plots(strategy_returns: pd.Series,
                   benchmark_returns: dict[str, pd.Series],
                   equity: pd.Series,
                   regimes: pd.Series,
                   weights: pd.DataFrame,
                   out_dir: Path=RESULTS_DIR) -> list[Path]:
    named={"RegimeShift": strategy_returns, **benchmark_returns}
    return [
        plot_cumulative(named, out_dir),
        plot_drawdown(named, out_dir),
        plot_regimes(equity, regimes, out_dir),
        plot_weights(weights, out_dir),
    ]
