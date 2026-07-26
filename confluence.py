"""
Today's live multi-factor confluence read.

The backtested Strategy (strategy_engine.py) answers "has this specific
rule combination worked historically for this ticker?" using only
genuinely historical features. This module answers a different question:
"right now, today, how many independent signal categories agree on
direction?" -- and it's allowed to use today's sentiment and options-flow
snapshot, since it isn't claiming to be a historical backtest.

Each category casts one vote in [-1, +1] (bearish..bullish), so no single
indicator can dominate just by having a large raw magnitude. ADX doesn't
vote on direction -- it scales conviction, since a strong trend makes
whatever direction the other factors agree on more credible, and a weak
trend (chop) makes every directional vote less trustworthy.
"""
from __future__ import annotations
import pandas as pd


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_confluence(
    latest: pd.Series,
    backtest_signal: int,          # -1/0/1 from the discovered historical strategy
    options_unusual_score: float,  # -1..1, from options_flow.py
    sentiment_score: float,        # -1..1, from sentiment.py
) -> dict:
    votes = {}

    # --- Backtested technical strategy (its own multi-factor confluence already) ---
    votes["Backtested strategy"] = float(backtest_signal)

    # --- Trend ---
    trend_vote = 0.0
    if latest.get("trend_direction") == "up":
        trend_vote += 0.6
    elif latest.get("trend_direction") == "down":
        trend_vote -= 0.6
    if latest.get("trend_status") == "above_resistance":
        trend_vote += 0.4
    elif latest.get("trend_status") == "below_support":
        trend_vote -= 0.4
    votes["Trend"] = _clip(trend_vote)

    # --- Momentum: RSI level + MACD histogram + confirmed divergence ---
    momentum_vote = 0.0
    rsi_val = latest.get("rsi")
    if pd.notna(rsi_val):
        if rsi_val < 30:
            momentum_vote += 0.5
        elif rsi_val > 70:
            momentum_vote -= 0.5
    macd_hist = latest.get("macd_hist")
    if pd.notna(macd_hist):
        momentum_vote += 0.3 if macd_hist > 0 else -0.3
    if latest.get("divergence") == "bullish":
        momentum_vote += 0.6
    elif latest.get("divergence") == "bearish":
        momentum_vote -= 0.6
    votes["Momentum / divergence"] = _clip(momentum_vote)

    # --- Patterns + gaps ---
    pattern_vote = 0.0
    if latest.get("bullish_engulfing") or latest.get("hammer"):
        pattern_vote += 0.5
    if latest.get("bearish_engulfing") or latest.get("shooting_star"):
        pattern_vote -= 0.5
    if latest.get("gap_up"):
        pattern_vote += 0.3
    if latest.get("gap_down"):
        pattern_vote -= 0.3
    votes["Patterns / gaps"] = _clip(pattern_vote)

    # --- Options flow proxy ---
    votes["Options flow (proxy)"] = _clip(options_unusual_score or 0.0)

    # --- News sentiment ---
    votes["News sentiment"] = _clip(sentiment_score or 0.0)

    # --- Weighted combination ---
    weights = {
        "Backtested strategy": 2.5,
        "Trend": 1.5,
        "Momentum / divergence": 1.5,
        "Patterns / gaps": 1.0,
        "Options flow (proxy)": 1.2,
        "News sentiment": 0.8,
    }
    raw = sum(votes[k] * weights[k] for k in votes)
    max_possible = sum(weights.values())
    normalized = raw / max_possible  # -1..1

    # ADX scales conviction, doesn't vote on direction: a strong trend
    # makes agreement more meaningful, chop makes it less so.
    adx_val = latest.get("adx", 0) or 0
    if adx_val >= 30:
        conviction_mult = 1.15
    elif adx_val >= 20:
        conviction_mult = 1.0
    else:
        conviction_mult = 0.8

    score_100 = _clip(normalized * conviction_mult, -1, 1) * 100

    if score_100 >= 40:
        label = "Strong bullish confluence"
    elif score_100 >= 15:
        label = "Bullish lean"
    elif score_100 <= -40:
        label = "Strong bearish confluence"
    elif score_100 <= -15:
        label = "Bearish lean"
    else:
        label = "Mixed / no confluence"

    agree = sum(1 for v in votes.values() if v > 0.15)
    disagree = sum(1 for v in votes.values() if v < -0.15)
    n_factors = len(votes)

    return {
        "score": round(score_100, 1),
        "label": label,
        "votes": votes,             # category -> -1..1 vote, for a breakdown table
        "n_agree": agree,
        "n_disagree": disagree,
        "n_factors": n_factors,
        "adx_conviction_mult": conviction_mult,
    }
