"""
Ties data_fetch -> indicators -> sentiment -> options_flow -> strategy_engine
-> confluence -> options_strategy -> backtester together into one call per
ticker. This is what app.py drives.
"""
from __future__ import annotations
import pandas as pd

import data_fetch
import indicators
import sentiment
import options_flow
import strategy_engine
import backtester
import confluence
import options_strategy
import ml_study


def analyze_ticker(ticker: str) -> dict:
    daily = data_fetch.get_daily_history(ticker)
    ind_df = indicators.build_indicator_frame(daily)

    sent = sentiment.get_ticker_sentiment(ticker)
    opts = options_flow.analyze_options_flow(ticker)
    options_score = opts["unusual_score"] if opts["available"] else 0.0

    # Historical technical feature frame -- used ONLY for the backtested
    # strategy search (sentiment/options columns are attached for display
    # but excluded from the search itself; see strategy_engine docstring).
    feat_df = strategy_engine.assemble_features(ind_df, sent["score"], options_score)
    importances = strategy_engine.feature_importance(feat_df)
    top_features = importances.index.tolist()

    strat = strategy_engine.generate_strategy(ticker, feat_df, top_features)
    baseline = backtester.buy_and_hold_baseline(feat_df["close"].dropna())

    ml_result = ml_study.run_walk_forward_study(feat_df, strategy_engine.FEATURE_COLUMNS)
    reliability = strategy_engine.compute_reliability(strat, ml_aggregate=ml_result)

    latest_row = ind_df.iloc[-1] if not ind_df.empty else None

    # Today's live backtested-rule signal
    backtest_signal = 0
    if latest_row is not None and (strat.long_rules or strat.short_rules):
        last_feat_row = feat_df.iloc[[-1]]
        sig_series = strat.build_signal(last_feat_row)
        backtest_signal = int(sig_series.iloc[0])

    # Today's live multi-factor confluence: combines the backtested signal
    # with trend, momentum/divergence, patterns, options flow, and
    # sentiment -- this is the "don't rely on one indicator" layer.
    conf = confluence.compute_confluence(
        latest=latest_row if latest_row is not None else pd.Series(dtype=float),
        backtest_signal=backtest_signal,
        options_unusual_score=options_score,
        sentiment_score=sent["score"],
    )

    option_idea = options_strategy.suggest_options_idea(
        ticker=ticker,
        confluence_result=conf,
        reliability=reliability,
        latest_indicators=latest_row if latest_row is not None else pd.Series(dtype=float),
        options_chain=opts.get("chain", pd.DataFrame()),
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
        "current_signal": backtest_signal,
        "confluence": conf,
        "option_idea": option_idea,
        "ml_study": ml_result,
    }
