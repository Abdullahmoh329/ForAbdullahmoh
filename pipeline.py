"""
Ties data_fetch -> indicators -> sentiment -> options_flow -> strategy_engine
-> backtester together into one call per ticker. This is what app.py drives.
"""
from __future__ import annotations
import pandas as pd

import data_fetch
import indicators
import sentiment
import options_flow
import strategy_engine
import backtester


def analyze_ticker(ticker: str) -> dict:
    daily = data_fetch.get_daily_history(ticker)
    ind_df = indicators.build_indicator_frame(daily)

    sent = sentiment.get_ticker_sentiment(ticker)
    opts = options_flow.analyze_options_flow(ticker)
    options_score = opts["unusual_score"] if opts["available"] else 0.0

    feat_df = strategy_engine.assemble_features(ind_df, sent["score"], options_score)
    importances = strategy_engine.feature_importance(feat_df)
    top_features = importances.index.tolist()

    strat = strategy_engine.generate_strategy(ticker, feat_df, top_features)
    baseline = backtester.buy_and_hold_baseline(feat_df["close"].dropna())

    return {
        "ticker": ticker,
        "price_df": daily,
        "indicator_df": ind_df,
        "feature_df": feat_df,
        "sentiment": sent,
        "options": opts,
        "feature_importance": importances,
        "strategy": strat,
        "baseline": baseline,
        "latest": ind_df.iloc[-1] if not ind_df.empty else None,
    }
