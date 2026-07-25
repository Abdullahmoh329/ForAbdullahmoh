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
import options_strategy


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
    reliability = strategy_engine.compute_reliability(strat)

    latest_row = ind_df.iloc[-1] if not ind_df.empty else None

    # Today's live signal: apply the discovered rules to the most recent bar
    current_signal = 0
    if latest_row is not None and (strat.long_rules or strat.short_rules):
        last_feat_row = feat_df.iloc[[-1]]
        sig_series = strat.build_signal(last_feat_row)
        current_signal = int(sig_series.iloc[0])

    options_chain_raw = data_fetch.get_options_snapshot(ticker)
    option_idea = options_strategy.suggest_options_idea(
        ticker=ticker,
        signal=current_signal,
        reliability=reliability,
        latest_indicators=latest_row if latest_row is not None else {},
        options_chain=options_chain_raw,
        sentiment_score=sent["score"],
    )

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
        "latest": latest_row,
        "reliability": reliability,
        "current_signal": current_signal,
        "option_idea": option_idea,
    }
