"""
This is the "brain" of the app.

For each ticker it:
  1. Builds a feature frame of genuinely historical technicals (trend,
     momentum/divergence, ADX, patterns, gaps, volume). Sentiment and
     options-flow are deliberately EXCLUDED from this step -- see note
     below.
  2. Trains a Random Forest to predict forward N-day direction, purely to
     surface WHICH features matter most for that specific ticker (feature
     importance) -- this is what makes the resulting strategy "novel per
     ticker" rather than a single rule set applied to everything.
  3. Runs a randomized search over MULTI-FACTOR rule combinations --
     candidates must draw conditions from at least two different
     indicator categories (trend, momentum, volatility, volume, pattern)
     so the backtested strategy is a confluence of signals, never a
     single indicator. Selection is by TRAIN-window Sharpe; the winner is
     then scored out-of-sample on a held-out TEST window (walk-forward,
     not just curve-fit).

Why sentiment / options-flow aren't in the backtest:
  Both are TODAY's snapshot -- there's no free source of historical,
  point-in-time headlines or historical options chains, so if we assigned
  today's sentiment score to every historical bar it would just be a
  constant column: it can't teach the search anything about the past, and
  a constant condition can silently zero out an otherwise-good strategy
  (AND'd with "always true" or "always false" for the whole backtest).
  Instead they're combined into TODAY's decision through confluence.py,
  which sits alongside the backtested technical signal rather than
  polluting it. This is more honest than a backtest number that secretly
  wasn't testing what it claims to.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier

import config
import backtester


# Every feature usable in the BACKTESTED technical strategy, tagged by
# category. The rule search below enforces coverage across categories --
# this is what makes a discovered strategy a confluence, not a single
# indicator dressed up as one.
FEATURE_CATEGORIES = {
    "rsi": "momentum",
    "macd_hist": "momentum",
    "divergence_bull": "momentum",
    "divergence_bear": "momentum",
    "adx": "volatility",
    "trend_slope": "trend",
    "trend_up": "trend",
    "trend_down": "trend",
    "trend_break_up": "trend",
    "trend_break_down": "trend",
    "vwap_dev_pct": "volume",
    "vol_z": "volume",
    "gap_up": "pattern",
    "gap_down": "pattern",
    "bullish_engulfing": "pattern",
    "bearish_engulfing": "pattern",
    "hammer": "pattern",
    "shooting_star": "pattern",
}
FEATURE_COLUMNS = list(FEATURE_CATEGORIES.keys())

BOOL_FEATURES = {
    "divergence_bull", "divergence_bear", "gap_up", "gap_down", "bullish_engulfing",
    "bearish_engulfing", "hammer", "shooting_star", "trend_break_up", "trend_break_down",
    "trend_up", "trend_down",
}


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
    df["trend_break_up"] = (df["trend_status"] == "above_resistance").astype(int)
    df["trend_break_down"] = (df["trend_status"] == "below_support").astype(int)
    df["trend_up"] = (df["trend_direction"] == "up").astype(int)
    df["trend_down"] = (df["trend_direction"] == "down").astype(int)

    # Kept on the frame for display/confluence purposes, but NOT included
    # in FEATURE_COLUMNS / the backtest -- see module docstring.
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

    @property
    def category(self) -> str:
        return FEATURE_CATEGORIES.get(self.feature, "other")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.feature]
        return col > self.threshold if self.op == ">" else col < self.threshold

    def __str__(self):
        return f"[{self.category}] {self.feature} {self.op} {self.threshold:.2f}"


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

    def n_categories(self) -> int:
        cats = {r.category for r in self.long_rules} | {r.category for r in self.short_rules}
        return len(cats)

    def describe(self) -> str:
        long_desc = " AND ".join(str(r) for r in self.long_rules) if self.long_rules else "(none)"
        short_desc = " AND ".join(str(r) for r in self.short_rules) if self.short_rules else "(none)"
        return f"LONG when: {long_desc}\nSHORT when: {short_desc}"


def _sample_threshold(rng: np.random.Generator, series: pd.Series, feature: str) -> tuple[str, float]:
    if feature in BOOL_FEATURES:
        return ">", 0.5
    lo, hi = series.quantile(0.1), series.quantile(0.9)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return ">", 0.0
    threshold = rng.uniform(lo, hi)
    op = rng.choice([">", "<"])
    return op, threshold


def _sample_diverse_conditions(rng, candidate_pool, train, n_conditions):
    """
    Samples `n_conditions` features from candidate_pool, requiring at
    least MIN_STRATEGY_CATEGORIES distinct indicator categories among
    them. Returns None if it can't build a diverse-enough set from the
    pool (caller should skip that iteration).
    """
    pool = list(candidate_pool)
    rng.shuffle(pool)
    chosen = pool[:n_conditions]
    categories = {FEATURE_CATEGORIES.get(f, "other") for f in chosen}
    if len(categories) < min(config.MIN_STRATEGY_CATEGORIES, len(set(FEATURE_CATEGORIES.get(f, "other") for f in candidate_pool))):
        return None
    rules = []
    for feat in chosen:
        op, thresh = _sample_threshold(rng, train[feat], feat)
        rules.append(Rule(feature=feat, op=op, threshold=thresh))
    return rules


def generate_strategy(ticker: str, feat_df: pd.DataFrame, top_features: list[str], seed: int = config.RANDOM_STATE) -> Strategy:
    """
    Randomized search (a lightweight evolutionary-style search) over
    MULTI-FACTOR rule combinations restricted to this ticker's own
    top-importance features, with a hard requirement that each candidate
    draws from at least `config.MIN_STRATEGY_CATEGORIES` distinct
    indicator categories (trend / momentum / volatility / volume /
    pattern). This is what turns "RSI > 70" into "trend is up AND MACD
    shows bullish divergence AND ADX confirms a real trend" -- a
    confluence, not a single indicator.

    Selection is based on TRAIN-window Sharpe; the winner is then scored
    out-of-sample on the TEST window to check it isn't just curve-fit.
    """
    rng = np.random.default_rng(seed)
    data = feat_df.dropna(subset=FEATURE_COLUMNS).copy()
    split_idx = int(len(data) * config.TRAIN_TEST_SPLIT)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    if len(train) < 60 or not top_features:
        return Strategy(ticker=ticker)

    # Widen the pool beyond the raw top-N so the search actually has
    # multiple categories to draw from, not just the single best feature
    # repeated with different thresholds.
    candidate_pool = top_features[:10] if len(top_features) >= 4 else FEATURE_COLUMNS

    best_strategy, best_sharpe = None, -np.inf

    for _ in range(config.N_RANDOM_STRATEGIES):
        n_conditions = int(rng.integers(config.MIN_RULE_CONDITIONS, config.MAX_RULE_CONDITIONS + 1))
        long_rules = _sample_diverse_conditions(rng, candidate_pool, train, n_conditions)
        if long_rules is None:
            continue

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


def compute_reliability(strat: "Strategy") -> dict:
    """
    A single 0-100 confidence score for a discovered strategy, built ONLY
    from its own backtest numbers -- not a guarantee of future performance,
    just a way to compare "how much does the evidence support this rule"
    across tickers at a glance.

    Weighting:
      - 35 pts: out-of-sample Sharpe (the number that matters most)
      - 20 pts: out-of-sample win rate
      - 15 pts: out-of-sample profit factor
      - 15 pts: sample size (more OOS trades = more evidence)
      - 10 pts: train/test consistency (penalizes strategies that only
                worked in-sample -- the classic overfit tell)
      - 5 pts:  category diversity (rewards genuine confluence over a
                strategy that happened to only need 2 categories)
    """
    tm, vm = strat.train_metrics or {}, strat.test_metrics or {}
    if not vm or vm.get("n_trades", 0) == 0:
        return {"score": 0, "label": "No signal", "reason": "Strategy search found no valid out-of-sample trades.", "n_categories": 0}

    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    sharpe_pts = clip(vm.get("sharpe", 0) / 3.0, -1, 1) * 35
    winrate_pts = clip(vm.get("win_rate_pct", 0) / 100, 0, 1) * 20
    pf = vm.get("profit_factor", 0)
    pf = 3.0 if pf == float("inf") else pf
    pf_pts = clip(pf / 3.0, 0, 1) * 15
    sample_pts = clip(vm.get("n_trades", 0) / 30, 0, 1) * 15
    train_sharpe, test_sharpe = tm.get("sharpe", 0), vm.get("sharpe", 0)
    gap = abs(train_sharpe - test_sharpe)
    consistency_pts = clip(1 - gap / 3.0, 0, 1) * 10
    n_cats = strat.n_categories()
    diversity_pts = clip(n_cats / 3.0, 0, 1) * 5

    score = round(max(0, sharpe_pts + winrate_pts + pf_pts + sample_pts + consistency_pts + diversity_pts))

    if score >= 65:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"

    reasons = []
    if vm.get("n_trades", 0) < config.MIN_TRADES_FOR_VALID_STRATEGY:
        reasons.append("few out-of-sample trades")
    if gap > 1.5:
        reasons.append("large gap between in-sample and out-of-sample results (overfit risk)")
    if vm.get("sharpe", 0) < 0:
        reasons.append("negative out-of-sample Sharpe")
    reason = "; ".join(reasons) if reasons else f"confluence across {n_cats} indicator categories with consistent in-/out-of-sample performance"

    return {"score": score, "label": label, "reason": reason, "n_categories": n_cats}
