"""
All external data access lives here. Free-tier only: yfinance.
If yfinance changes its API or rate-limits you, this is the one file to fix.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
import config


def get_daily_history(ticker: str, days: int = config.PRICE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Daily OHLCV bars, cleaned, with a DatetimeIndex."""
    df = yf.Ticker(ticker).history(period=f"{days}d", interval=config.PRICE_INTERVAL, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}. Check the symbol.")
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_recent_intraday(ticker: str) -> pd.DataFrame:
    """Recent intraday bars, used to compute a live session VWAP."""
    try:
        df = yf.Ticker(ticker).history(
            period=f"{config.INTRADAY_LOOKBACK_DAYS}d",
            interval=config.INTRADAY_INTERVAL,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def get_news_headlines(ticker: str, max_items: int = config.NEWS_MAX_HEADLINES) -> list[str]:
    """Recent headlines for a ticker via yfinance's news feed."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    headlines = []
    for it in items[:max_items]:
        # yfinance news items nest the payload under "content" in newer versions
        content = it.get("content", it)
        title = content.get("title") or it.get("title")
        if title:
            headlines.append(title)
    return headlines


def get_options_snapshot(ticker: str, num_expiries: int = config.OPTIONS_NUM_EXPIRIES) -> pd.DataFrame:
    """
    Pulls the nearest N option expirations and returns a combined calls+puts
    dataframe with a 'type' column. This is EOD/delayed chain data from
    yfinance, not real-time order flow -- used only as a volume/OI proxy.
    """
    tk = yf.Ticker(ticker)
    try:
        expiries = tk.options[:num_expiries]
    except Exception:
        return pd.DataFrame()

    frames = []
    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        calls = chain.calls.copy()
        calls["type"] = "call"
        puts = chain.puts.copy()
        puts["type"] = "put"
        for d in (calls, puts):
            d["expiry"] = exp
        frames.append(calls)
        frames.append(puts)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    keep = ["contractSymbol", "strike", "volume", "openInterest", "impliedVolatility", "type", "expiry"]
    combined = combined[[c for c in keep if c in combined.columns]].fillna(0)
    return combined
