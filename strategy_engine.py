"""
This is the "brain" of the app.

For each ticker it:
  1. Builds a feature frame (technicals + sentiment + options-flow proxy).
  2. Trains a Random Forest to predict forward N-day direction, purely to
     surface WHICH features matter most for that specific ticker (feature
     importance) -- this is what makes the resulting strategy "novel per
     ticker" rather than a single rule set applied to everything.
  3. Runs a randomized search over rule combinations built from the
     highest-importance features, evaluating each candidate with the
     backtester on a TRAIN window and validating the winner on a held-out
     TEST window (walk-forward, not just curve-fit).

The output is a human-readable rule set plus in-sample and out-of-sample
backtest metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier

import config
import backtester


FEATURE_COLUMNS = [
    "rsi", "macd_hist", "adx", "vwap_dev_pct", "vol_z", "trend_slope",
    "divergence_bull", "divergence_bear", "gap_up", "gap_down",
    "bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star",
    "trend_break_up", "trend_break_down", "sentiment", "options_unusual_score",
]


def assemble_features(indicator_df: pd.DataFrame, sentiment_score: float, options_score: float) -> pd.DataFrame:
    df = indicator_df.copy()
    df["divergence_bull"] = (df["divergence"] == "bullish").astype(int)
    df["divergence_bear"] = (df["divergence"] == "bearish").astype(int)
    df["gap_up"] = df["gap_up"].astype(int)
    df["gap_down"] = df["gap_down"].astype(int)
    df["bullish_engulfing"] = df["bullish_engulfing"].astype(int)
    df["bearish_engulfing"] = df["bearish_engulfing"].astype(int)
    df["hammer"] = df["hammer"].astype(int)
    df["shooting_star"] = df["shooting_star"].astype(int)
    df["trend_break_up"] = (df["trend_break"] == "breakout_up").astype(int)
    df["trend_break_down"] = (df["trend_break"] == "breakdown").astype(int)
    # sentiment/options are point-in-time snapshots (today's news/chain), applied uniformly
    # across history as the best available proxy for "current tone" -- flagged in the UI.
    df["sentiment"] = sentiment_score
    df["options_unusual_score"] = options_score

    df["forward_return"] = df["close"].shift(-config.FORWARD_RETURN_DAYS) / df["close"] - 1
    df["label_up"] = (df["forward_return"] > 0).astype(int)
    return df


def feature_importance(feat_df: pd.DataFrame) -> pd.Series:
    data = feat_df.dropna(subset=FEATURE_COLUMNS + ["label_up"])
    if len(data) < 50:
        return pd.Series(dtype=float)
    X = data[FEATURE_COLUMNS]
    y = data["label_up"]
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)


# ------------------------------------------------------------ Rule search -
@dataclass
class Rule:
    feature: str
    op: str          # ">" or "<"
    threshold: float

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.feature]
        return col > self.threshold if self.op == ">" else col < self.threshold

    def __str__(self):
        return f"{self.feature} {self.op} {self.threshold:.2f}"


@dataclass
class Strategy:
    ticker: str
    long_rules: list = field(default_factory=list)
    short_rules: list = field(default_factory=list)
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)

    def build_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if self.long_rules:
            long_mask = pd.Series(True, index=df.index)
            for r in self.long_rules:
                long_mask &= r.evaluate(df)
            signal[long_mask] = 1
        if self.short_rules and config.ALLOW_SHORT:
            short_mask = pd.Series(True, index=df.index)
            for r in self.short_rules:
                short_mask &= r.evaluate(df)
            signal[short_mask] = -1
        return signal

    def describe(self) -> str:
        long_desc = " AND ".join(str(r) for r in self.long_rules) if self.long_rules else "(none)"
        short_desc = " AND ".join(str(r) for r in self.short_rules) if self.short_rules else "(none)"
        return f"LONG when: {long_desc}\nSHORT when: {short_desc}"


BOOL_FEATURES = {
    "divergence_bull", "divergence_bear", "gap_up", "gap_down", "bullish_engulfing",
    "bearish_engulfing", "hammer", "shooting_star", "trend_break_up", "trend_break_down",
}


def _sample_threshold(rng: np.random.Generator, series: pd.Series, feature: str) -> tuple[str, float]:
    if feature in BOOL_FEATURES:
        return ">", 0.5
    lo, hi = series.quantile(0.1), series.quantile(0.9)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return ">", 0.0
    threshold = rng.uniform(lo, hi)
    op = rng.choice([">", "<"])
    return op, threshold


def generate_strategy(ticker: str, feat_df: pd.DataFrame, top_features: list[str], seed: int = config.RANDOM_STATE) -> Strategy:
    """
    Randomized search (a lightweight evolutionary-style search) over rule
    combinations restricted to this ticker's own top-importance features.
    Selection is based on TRAIN-window Sharpe; the winner is then scored
    out-of-sample on the TEST window to check it isn't just curve-fit.
    """
    rng = np.random.default_rng(seed)
    data = feat_df.dropna(subset=FEATURE_COLUMNS).copy()
    split_idx = int(len(data) * config.TRAIN_TEST_SPLIT)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    if len(train) < 60 or not top_features:
        return Strategy(ticker=ticker)

    candidate_pool = top_features[:6] if len(top_features) >= 3 else FEATURE_COLUMNS

    best_strategy, best_sharpe = None, -np.inf

    for _ in range(config.N_RANDOM_STRATEGIES):
        n_conditions = rng.integers(1, 3)  # 1-2 conditions combined -> keeps rules readable
        chosen = rng.choice(candidate_pool, size=min(n_conditions, len(candidate_pool)), replace=False)

        long_rules = []
        for feat in chosen:
            op, thresh = _sample_threshold(rng, train[feat], feat)
            long_rules.append(Rule(feature=feat, op=op, threshold=thresh))

        candidate = Strategy(ticker=ticker, long_rules=long_rules, short_rules=[])
        signal = candidate.build_signal(train)
        result = backtester.run_backtest(train["close"], signal)

        if result["n_trades"] < config.MIN_TRADES_FOR_VALID_STRATEGY:
            continue
        if result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_strategy = candidate
            best_strategy.train_metrics = result

    if best_strategy is None:
        return Strategy(ticker=ticker)

    test_signal = best_strategy.build_signal(test)
    best_strategy.test_metrics = backtester.run_backtest(test["close"], test_signal)
    return best_strategy
