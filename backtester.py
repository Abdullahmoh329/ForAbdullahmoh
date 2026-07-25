"""
Vectorized long/short backtester. Signal-in, equity-curve-and-metrics-out.
Deliberately simple and transparent -- every trade cost and assumption
is visible here, nothing is hidden inside a black-box library.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    initial_capital: float = config.INITIAL_CAPITAL,
    fee_bps: float = config.FEE_BPS,
) -> dict:
    """
    signal: integer series aligned to `close`, values in {-1, 0, 1}
            (short, flat, long). Signal on bar t is assumed executed at
            the CLOSE of bar t (i.e. position held is signal.shift(0) applied
            to return from t to t+1) -- avoids lookahead bias since the
            signal at t only uses information available through bar t.
    """
    df = pd.DataFrame({"close": close, "signal": signal}).dropna()
    if df.empty or len(df) < 5:
        return _empty_result()

    df["ret"] = df["close"].pct_change().fillna(0)
    df["position"] = df["signal"].shift(1).fillna(0)  # act on next bar's return
    df["trade"] = df["position"].diff().fillna(df["position"]).abs()  # position changes = trades
    fee = (fee_bps / 10000.0)

    df["strategy_ret"] = df["position"] * df["ret"] - df["trade"] * fee
    df["equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod()

    n_trades = int((df["trade"] > 0).sum())
    total_return = df["equity"].iloc[-1] / initial_capital - 1
    n_years = max(len(df) / 252, 1 / 252)
    cagr = (df["equity"].iloc[-1] / initial_capital) ** (1 / n_years) - 1 if df["equity"].iloc[-1] > 0 else -1

    daily_std = df["strategy_ret"].std()
    sharpe = (df["strategy_ret"].mean() / daily_std * np.sqrt(252)) if daily_std and daily_std > 0 else 0.0

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1
    max_dd = drawdown.min()

    wins = df.loc[(df["position"] != 0) & (df["strategy_ret"] > 0), "strategy_ret"]
    losses = df.loc[(df["position"] != 0) & (df["strategy_ret"] < 0), "strategy_ret"]
    active_bars = df.loc[df["position"] != 0, "strategy_ret"]
    win_rate = (len(wins) / len(active_bars)) if len(active_bars) > 0 else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else (np.inf if wins.sum() > 0 else 0.0)

    return {
        "equity_curve": df["equity"],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "n_bars": len(df),
    }


def _empty_result() -> dict:
    return {
        "equity_curve": pd.Series(dtype=float), "total_return_pct": 0.0, "cagr_pct": 0.0,
        "sharpe": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0,
        "profit_factor": 0.0, "n_trades": 0, "n_bars": 0,
    }


def buy_and_hold_baseline(close: pd.Series, initial_capital: float = config.INITIAL_CAPITAL) -> dict:
    signal = pd.Series(1, index=close.index)
    return run_backtest(close, signal, initial_capital, fee_bps=0.0)
