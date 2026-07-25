"""
Technical indicator library. Pure pandas/numpy -- no TA-Lib dependency,
so it installs cleanly anywhere.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import config


# ---------------------------------------------------------------- VWAP ----
def session_vwap(intraday_df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, reset each calendar day."""
    if intraday_df.empty:
        return pd.Series(dtype=float)
    df = intraday_df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tpv"] = typical * df["volume"]
    day = df.index.date
    cum_tpv = df.groupby(day)["tpv"].cumsum()
    cum_vol = df.groupby(day)["volume"].cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    vwap.name = "vwap"
    return vwap


def daily_vwap_proxy(daily_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling volume-weighted average close, used on daily bars when no
    intraday data is available. Not a true session VWAP -- a smoothed proxy.
    """
    typical = (daily_df["high"] + daily_df["low"] + daily_df["close"]) / 3.0
    tpv = typical * daily_df["volume"]
    vwap = tpv.rolling(window).sum() / daily_df["volume"].rolling(window).sum()
    vwap.name = "vwap_proxy"
    return vwap


# ----------------------------------------------------------------- RSI ----
def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out.name = "rsi"
    return out


# ---------------------------------------------------------------- MACD ----
def macd(close: pd.Series, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.rename("macd"), signal_line.rename("macd_signal"), hist.rename("macd_hist")


# ------------------------------------------------------------- Swings -----
def swing_points(series: pd.Series, order: int = config.SWING_LOOKBACK):
    """Indices of local maxima and minima (swing highs / lows)."""
    vals = series.values
    highs_idx = argrelextrema(vals, np.greater_equal, order=order)[0]
    lows_idx = argrelextrema(vals, np.less_equal, order=order)[0]
    highs_idx = np.array([i for i in highs_idx if 0 < i < len(vals) - 1])
    lows_idx = np.array([i for i in lows_idx if 0 < i < len(vals) - 1])
    return highs_idx, lows_idx


# --------------------------------------------------------- Divergence -----
def rsi_macd_divergence(df: pd.DataFrame, lookback: int = config.DIVERGENCE_LOOKBACK) -> pd.Series:
    """
    Flags bullish/bearish divergence between price and RSI+MACD histogram
    over a rolling lookback window, using swing highs/lows.

    Returns a Series of strings per bar: 'bullish', 'bearish', or '' (none).
    A divergence is only flagged when RSI and MACD histogram agree in
    direction -- this is what makes it "confirmed" rather than single-
    indicator noise.
    """
    close = df["close"]
    r = df["rsi"]
    h = df["macd_hist"]
    out = pd.Series("", index=df.index)

    highs_idx, lows_idx = swing_points(close, order=config.SWING_LOOKBACK)

    for i in range(lookback, len(df)):
        window_start = i - lookback
        recent_lows = [j for j in lows_idx if window_start <= j <= i]
        recent_highs = [j for j in highs_idx if window_start <= j <= i]

        # Bullish divergence: price makes a lower low, RSI & MACD hist make higher low
        if len(recent_lows) >= 2:
            j1, j2 = recent_lows[-2], recent_lows[-1]
            if close.iloc[j2] < close.iloc[j1] and r.iloc[j2] > r.iloc[j1] and h.iloc[j2] > h.iloc[j1]:
                out.iloc[i] = "bullish"

        # Bearish divergence: price makes a higher high, RSI & MACD hist make lower high
        if len(recent_highs) >= 2:
            j1, j2 = recent_highs[-2], recent_highs[-1]
            if close.iloc[j2] > close.iloc[j1] and r.iloc[j2] < r.iloc[j1] and h.iloc[j2] < h.iloc[j1]:
                out.iloc[i] = "bearish"

    out.name = "divergence"
    return out


# ------------------------------------------------------------------ ADX ---
def adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    adx_val.name = "adx"
    return adx_val


# ------------------------------------------------------------ Trendlines --
def trendline_signal(close: pd.Series, lookback: int = config.TRENDLINE_LOOKBACK) -> pd.DataFrame:
    """
    Fits a linear trendline through swing lows (support) and swing highs
    (resistance) within a rolling lookback window, then flags whether the
    latest close has broken above resistance or below support.
    """
    highs_idx, lows_idx = swing_points(close, order=config.SWING_LOOKBACK)
    n = len(close)
    slope_out = pd.Series(0.0, index=close.index)          # % per bar, normalized by price
    status_out = pd.Series("inside_channel", index=close.index)   # always populated
    direction_out = pd.Series("flat", index=close.index)          # always populated
    breakout_out = pd.Series("", index=close.index)               # kept for backward compat

    for i in range(lookback, n):
        window_lows = [j for j in lows_idx if i - lookback <= j <= i]
        window_highs = [j for j in highs_idx if i - lookback <= j <= i]

        support_level = resistance_level = None
        slopes = []

        if len(window_lows) >= 2:
            xs = np.array(window_lows)
            ys = close.iloc[window_lows].values
            m, b = np.polyfit(xs, ys, 1)
            support_level = m * i + b
            slopes.append(m)

        if len(window_highs) >= 2:
            xs = np.array(window_highs)
            ys = close.iloc[window_highs].values
            m, b = np.polyfit(xs, ys, 1)
            resistance_level = m * i + b
            slopes.append(m)

        px = close.iloc[i]
        if slopes:
            avg_slope = float(np.mean(slopes))
            slope_out.iloc[i] = (avg_slope / px * 100) if px else 0.0
            if slope_out.iloc[i] > 0.05:
                direction_out.iloc[i] = "up"
            elif slope_out.iloc[i] < -0.05:
                direction_out.iloc[i] = "down"

        if resistance_level is not None and px > resistance_level:
            status_out.iloc[i] = "above_resistance"
            breakout_out.iloc[i] = "breakout_up"
        elif support_level is not None and px < support_level:
            status_out.iloc[i] = "below_support"
            breakout_out.iloc[i] = "breakdown"
        else:
            status_out.iloc[i] = "inside_channel"

    return pd.DataFrame({
        "trend_slope": slope_out,
        "trend_status": status_out,
        "trend_direction": direction_out,
        "trend_break": breakout_out,
    })


# --------------------------------------------------------------- Patterns -
def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    range_ = (h - l).replace(0, np.nan)
    upper_wick = h - c.where(c > o, o)
    lower_wick = c.where(c < o, o) - l

    prev_o, prev_c = o.shift(1), c.shift(1)

    bullish_engulf = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    bearish_engulf = (c < o) & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)
    doji = (body / range_) < 0.1
    hammer = (lower_wick > 2 * body) & (upper_wick < body) & (c > o)
    shooting_star = (upper_wick > 2 * body) & (lower_wick < body) & (c < o)

    out = pd.DataFrame(index=df.index)
    out["bullish_engulfing"] = bullish_engulf.fillna(False)
    out["bearish_engulfing"] = bearish_engulf.fillna(False)
    out["doji"] = doji.fillna(False)
    out["hammer"] = hammer.fillna(False)
    out["shooting_star"] = shooting_star.fillna(False)
    return out


# ------------------------------------------------------------------ Gaps --
def gap_signal(df: pd.DataFrame, min_pct: float = 0.5) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close) / prev_close * 100
    out = pd.DataFrame(index=df.index)
    out["gap_pct"] = gap_pct
    out["gap_up"] = gap_pct > min_pct
    out["gap_down"] = gap_pct < -min_pct
    return out


# ---------------------------------------------------------- Volume z-score
def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    z = (volume - mean) / std.replace(0, np.nan)
    z.name = "vol_z"
    return z


def build_indicator_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Runs every indicator above and returns one combined dataframe."""
    df = daily_df.copy()
    df["rsi"] = rsi(df["close"])
    macd_line, macd_sig, macd_hist = macd(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, macd_sig, macd_hist
    df["adx"] = adx(df)
    df["vwap_proxy"] = daily_vwap_proxy(df)
    df["vwap_dev_pct"] = (df["close"] - df["vwap_proxy"]) / df["vwap_proxy"] * 100
    df["vol_z"] = volume_zscore(df["volume"])

    df["divergence"] = rsi_macd_divergence(df)

    trend = trendline_signal(df["close"])
    df = df.join(trend)

    patterns = candlestick_patterns(df)
    df = df.join(patterns)

    gaps = gap_signal(df)
    df = df.join(gaps)

    return df
