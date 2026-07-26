"""
Free-tier options "flow" proxy.

IMPORTANT LIMITATION: yfinance option chains are end-of-day / delayed
snapshots of volume and open interest -- NOT a real-time print of large
block trades or sweeps the way paid flow products (e.g. unusual options
scanners) provide. What we CAN legitimately derive from free data:

  - Put/Call volume ratio (sentiment lean)
  - Volume / Open-Interest ratio per strike (a same-day-positioning
    signal: high volume relative to existing OI suggests fresh
    positioning rather than closing old contracts)
  - Concentration of unusual activity by strike/expiry

Treat the output as a directional lean, not a precision signal.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import config
import data_fetch


def analyze_options_flow(ticker: str) -> dict:
    chain = data_fetch.get_options_snapshot(ticker)
    if chain.empty:
        return {
            "available": False, "put_call_ratio": None, "unusual_score": 0.0,
            "unusual_strikes": pd.DataFrame(), "call_volume": 0, "put_volume": 0,
            "chain": pd.DataFrame(),
        }

    call_vol = chain.loc[chain["type"] == "call", "volume"].sum()
    put_vol = chain.loc[chain["type"] == "put", "volume"].sum()
    put_call_ratio = (put_vol / call_vol) if call_vol > 0 else np.nan

    chain = chain.copy()
    chain["vol_oi_ratio"] = chain["volume"] / chain["openInterest"].replace(0, np.nan)
    unusual = chain[chain["vol_oi_ratio"] >= config.UNUSUAL_VOL_OI_RATIO].sort_values(
        "vol_oi_ratio", ascending=False
    )

    # Unusual score: net skew of unusual call volume vs unusual put volume, normalized -1..1
    unusual_call_vol = unusual.loc[unusual["type"] == "call", "volume"].sum()
    unusual_put_vol = unusual.loc[unusual["type"] == "put", "volume"].sum()
    total_unusual = unusual_call_vol + unusual_put_vol
    unusual_score = ((unusual_call_vol - unusual_put_vol) / total_unusual) if total_unusual > 0 else 0.0

    return {
        "available": True,
        "put_call_ratio": put_call_ratio,
        "unusual_score": unusual_score,   # +1 = skewed unusually bullish, -1 = unusually bearish
        "unusual_strikes": unusual.head(15)[["contractSymbol", "type", "strike", "expiry", "volume", "openInterest", "vol_oi_ratio"]],
        "call_volume": int(call_vol),
        "put_volume": int(put_vol),
        "chain": chain,
    }
