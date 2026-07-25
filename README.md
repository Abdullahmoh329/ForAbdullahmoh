# Trading Idea Advisor

A personal research tool: for each ticker on your watchlist it pulls free-tier
market data, computes a broad technical/sentiment/options-flow-proxy feature
set, **searches for a strategy tailored to that specific ticker's own
history**, and backtests it in-sample and out-of-sample.

This is a research aid, not an autopilot. It does not place trades.

## Quick start

```bash
cd trading_advisor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Edit your watchlist either in the sidebar at runtime, or permanently in
`config.py` (`WATCHLIST = [...]`).

## How it works

```
data_fetch.py     -> OHLCV, intraday bars, news headlines, options chains (yfinance)
indicators.py     -> VWAP, RSI, MACD, RSI/MACD divergence, ADX, trendlines,
                      candlestick patterns, gaps, volume z-score
sentiment.py      -> VADER sentiment score over recent headlines (finance-tuned lexicon)
options_flow.py   -> put/call ratio + volume/open-interest "unusual activity" proxy
strategy_engine.py-> per-ticker Random Forest feature importance, then a
                      randomized rule-search (evolutionary-style) restricted
                      to that ticker's top features, selected by in-sample
                      Sharpe and validated out-of-sample
backtester.py     -> vectorized long/short backtest engine with fees, Sharpe,
                      CAGR, max drawdown, win rate, profit factor
pipeline.py       -> wires the above together for one ticker
app.py            -> Streamlit dashboard
```

### Why each ticker gets a different strategy

`strategy_engine.py` trains a Random Forest per ticker to rank which
features (RSI level, MACD divergence, ADX, VWAP deviation, trendline
breaks, gaps, sentiment, options skew, etc.) actually mattered for that
ticker's forward returns historically. It then randomly samples rule
combinations built **only from that ticker's top features** and keeps
whichever rule combination had the best in-sample Sharpe ratio, before
re-testing it on a held-out out-of-sample window. Two tickers with
different price behavior will end up with genuinely different rules,
not the same indicator stack with different tickers plugged in.

## Known limitations (read before trusting any number this app shows you)

- **Data is free-tier and delayed/EOD.** `yfinance` is unofficial and can
  break or rate-limit without notice. Options chain volume/OI is a
  same-day snapshot, not a live feed.
- **"Options flow" here is a proxy**, built from volume/open-interest
  ratios and put/call ratios on delayed chain data — it is *not* real-time
  block/sweep order flow like paid scanners provide.
- **Sentiment score is a snapshot**, applied uniformly across the
  backtest history (there's no free source of historical point-in-time
  headlines), so it does not meaningfully contribute to the *backtested*
  edge — treat it as a "current tone" indicator for today's decision only,
  not as validated backtest alpha.
- **Small watchlist, limited history.** With ~10-30 tickers and ~2 years
  of daily bars, the strategy search can still overfit. The train/test
  split and minimum-trade-count filter reduce this but don't eliminate it.
- **Backtest costs are simplified** (a flat basis-point fee, no slippage
  model, no partial fills, no realistic position sizing). Real execution
  will differ.
- **Past performance, including out-of-sample backtest performance, does
  not guarantee future results.** Paper trade any strategy this tool
  surfaces before risking real capital, and treat every output as one
  input to your own judgment — not a signal to act on automatically.

## Extending it

- Swap in a paid data provider (Polygon, Databento, Benzinga, Unusual
  Whales) by editing only `data_fetch.py` — nothing downstream needs to
  change if you keep the same column names.
- To add a new indicator, add a function to `indicators.py`, wire it
  into `build_indicator_frame`, and add its column name to
  `FEATURE_COLUMNS` in `strategy_engine.py`.
- To try a different strategy-search method (e.g. a real genetic
  algorithm, or Bayesian optimization), only `generate_strategy()` in
  `strategy_engine.py` needs to change — the backtester and UI are
  agnostic to how the strategy was found.
