"""
Central configuration for the trading advisor app.
Edit WATCHLIST to your tickers. Everything else has sane defaults.
"""

# ---- Universe ----
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "SPY", "QQQ"]

# ---- Data ----
PRICE_LOOKBACK_DAYS = 730     # ~2 years of daily bars for backtest/training
PRICE_INTERVAL = "1d"         # yfinance interval
INTRADAY_INTERVAL = "5m"      # used only for session VWAP on the most recent day
INTRADAY_LOOKBACK_DAYS = 5

# ---- Indicators ----
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ADX_PERIOD = 14
SWING_LOOKBACK = 5            # bars on each side to define a swing high/low
TRENDLINE_LOOKBACK = 60       # bars used to fit trendlines
DIVERGENCE_LOOKBACK = 30      # bars scanned for RSI/MACD vs price divergence

# ---- Options flow proxy ----
OPTIONS_NUM_EXPIRIES = 3      # nearest N expiries to aggregate
UNUSUAL_VOL_OI_RATIO = 1.0    # volume/OI above this on a strike is flagged "unusual"

# ---- News / sentiment ----
NEWS_MAX_HEADLINES = 20

# ---- Strategy search (per-ticker "novel" strategy generation) ----
N_RANDOM_STRATEGIES = 1200     # candidates sampled per ticker (larger feature universe -> wider search)
FORWARD_RETURN_DAYS = 5       # label horizon: return over next N bars
TRAIN_TEST_SPLIT = 0.7        # fraction of history used for training/search
MIN_TRADES_FOR_VALID_STRATEGY = 8
MIN_RULE_CONDITIONS = 2       # every discovered strategy must combine at least this many conditions
MAX_RULE_CONDITIONS = 5       # ...and no more than this many, to stay readable
MIN_STRATEGY_CATEGORIES = 3   # ...spanning at least this many distinct indicator categories (confluence)

# ---- ML walk-forward study ----
ML_STUDY_FOLDS = 5            # walk-forward folds for the ensemble ML study
ML_MIN_ROWS_FOR_STUDY = 150   # need this many clean rows before the study runs at all

# ---- Backtest ----
INITIAL_CAPITAL = 10_000.0
FEE_BPS = 5.0                 # round-trip cost assumption, in basis points per trade
ALLOW_SHORT = True

# ---- Random Forest (feature importance / secondary signal) ----
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 5
RANDOM_STATE = 42
