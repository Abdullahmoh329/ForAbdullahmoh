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


# =====================================================================
# Second wave of indicators. Individually, none of these is trustworthy
# on its own -- that's the point. They exist so the strategy search has
# a real universe of ~30-50 signals to screen and combine, instead of
# resting on 5-6 indicators repeated with different thresholds.
# =====================================================================

def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.rename("stoch_k"), d.rename("stoch_d")


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    wr = -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
    wr.name = "williams_r"
    return wr


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out = (typical - sma) / (0.015 * mad.replace(0, np.nan))
    out.name = "cci"
    return out


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"]
    direction = typical.diff()
    pos_flow = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    neg_flow = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    out = 100 - (100 / (1 + ratio))
    out.name = "mfi"
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    out.name = "atr"
    return out


def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = mid + n_std * std, mid - n_std * std
    position = (close - lower) / (upper - lower).replace(0, np.nan)  # 0=lower band, 1=upper band
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return position.rename("bb_position"), width.rename("bb_width")


def moving_average_signals(close: pd.Series) -> pd.DataFrame:
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out = pd.DataFrame(index=close.index)
    out["ema_fast_above_slow"] = (ema9 > ema21).astype(int)
    out["golden_cross"] = (sma50 > sma200).astype(int)
    out["price_above_sma50"] = (close > sma50).astype(int)
    return out


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    out = (close - close.shift(period)) / close.shift(period) * 100
    out.name = "roc"
    return out


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    out = close - close.shift(period)
    out.name = "momentum"
    return out


def obv_slope(df: pd.DataFrame, period: int = 10) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    obv = (direction * df["volume"]).cumsum()
    slope = obv.diff(period) / period
    # normalize by average volume so it's comparable across tickers
    norm = slope / df["volume"].rolling(period).mean().replace(0, np.nan)
    norm.name = "obv_slope"
    return norm


def donchian_breakout(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(period).max().shift(1)   # prior N bars, excludes today
    lower = df["low"].rolling(period).min().shift(1)
    out = pd.DataFrame(index=df.index)
    out["donchian_breakout_up"] = (df["close"] > upper).astype(int)
    out["donchian_breakout_down"] = (df["close"] < lower).astype(int)
    return out


def macd_crossover(macd_line: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
    above = macd_line > signal_line
    out = pd.DataFrame(index=macd_line.index)
    out["macd_cross_up"] = (above & ~above.shift(1).fillna(False)).astype(int)
    out["macd_cross_down"] = (~above & above.shift(1).fillna(True)).astype(int)
    return out


def adx_rising(adx_series: pd.Series, period: int = 5) -> pd.Series:
    out = (adx_series.diff(period) > 0).astype(int)
    out.name = "adx_rising"
    return out


def extended_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple 3-bar continuation/reversal patterns beyond single-candle ones."""
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["three_higher_highs"] = ((df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))).astype(int)
    out["three_lower_lows"] = ((df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))).astype(int)
    # simplified morning/evening star: big down/up day, small indecisive day, big reversal day
    body = (df["close"] - df["open"])
    day1, day2, day3 = body.shift(2), body.shift(1), body
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    small_middle = (day2.abs() / range_.shift(1)) < 0.3
    out["morning_star"] = ((day1 < 0) & small_middle & (day3 > 0) & (c > c.shift(2))).fillna(False).astype(int)
    out["evening_star"] = ((day1 > 0) & small_middle & (day3 < 0) & (c < c.shift(2))).fillna(False).astype(int)
    return out


def volume_spike(vol_z: pd.Series, threshold: float = 2.0) -> pd.Series:
    out = (vol_z > threshold).astype(int)
    out.name = "volume_spike"
    return out


def price_above_vwap(close: pd.Series, vwap: pd.Series) -> pd.Series:
    out = (close > vwap).astype(int)
    out.name = "price_above_vwap"
    return out


def build_indicator_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Runs every indicator above and returns one combined dataframe."""
    df = daily_df.copy()
    df["rsi"] = rsi(df["close"], period=config.RSI_PERIOD)
    df["rsi_7"] = rsi(df["close"], period=7)
    df["rsi_21"] = rsi(df["close"], period=21)
    macd_line, macd_sig, macd_hist = macd(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, macd_sig, macd_hist
    df["adx"] = adx(df)
    df["adx_rising"] = adx_rising(df["adx"])
    df["vwap_proxy"] = daily_vwap_proxy(df)
    df["vwap_dev_pct"] = (df["close"] - df["vwap_proxy"]) / df["vwap_proxy"] * 100
    df["price_above_vwap"] = price_above_vwap(df["close"], df["vwap_proxy"])
    df["vol_z"] = volume_zscore(df["volume"])
    df["volume_spike"] = volume_spike(df["vol_z"])

    df["divergence"] = rsi_macd_divergence(df)

    trend = trendline_signal(df["close"])
    df = df.join(trend)

    patterns = candlestick_patterns(df)
    df = df.join(patterns)
    df = df.join(extended_patterns(df))

    gaps = gap_signal(df)
    df = df.join(gaps)

    stoch_k, stoch_d = stochastic_oscillator(df)
    df["stoch_k"], df["stoch_d"] = stoch_k, stoch_d
    df["williams_r"] = williams_r(df)
    df["cci"] = cci(df)
    df["mfi"] = money_flow_index(df)
    df["atr_pct"] = atr(df) / df["close"] * 100
    bb_pos, bb_width = bollinger_bands(df["close"])
    df["bb_position"], df["bb_width"] = bb_pos, bb_width
    df = df.join(moving_average_signals(df["close"]))
    df["roc"] = rate_of_change(df["close"])
    df["momentum"] = momentum(df["close"])
    df["obv_slope"] = obv_slope(df)
    df = df.join(donchian_breakout(df))
    df = df.join(macd_crossover(macd_line, macd_sig))

    return df
