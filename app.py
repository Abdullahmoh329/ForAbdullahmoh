import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
import pipeline

st.set_page_config(page_title="Trading Idea Advisor", layout="wide", page_icon="📊")

CUSTOM_CSS = """
<style>
:root {
    --bg: #0d1117;
    --panel: #151b23;
    --line: #2a323d;
    --accent: #4fd1c5;
    --accent-warn: #f0a35c;
    --up: #3fb950;
    --down: #f85149;
    --text: #e6edf3;
    --muted: #8b949e;
}
html, body, [class*="css"]  { color: var(--text); }
.stApp { background-color: var(--bg); }
h1, h2, h3 { font-family: 'IBM Plex Mono', 'Courier New', monospace; letter-spacing: -0.5px; }
.metric-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 14px 16px; margin-bottom: 8px;
}
.rule-box {
    background: var(--panel); border-left: 3px solid var(--accent);
    padding: 10px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    white-space: pre-wrap; border-radius: 4px;
}
.disclaimer {
    background: #241c0f; border: 1px solid #6b4e16; border-radius: 6px;
    padding: 12px 16px; font-size: 0.82rem; color: #f0c987;
}
.signal-card {
    border-radius: 10px; padding: 18px 20px; margin-bottom: 14px;
    border: 1px solid var(--line);
}
.signal-long { background: rgba(63,185,80,0.10); border-color: #3fb950; }
.signal-short { background: rgba(248,81,73,0.10); border-color: #f85149; }
.signal-flat { background: rgba(139,148,158,0.08); border-color: var(--line); }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.3px;
}
.badge-high { background: rgba(63,185,80,0.18); color: #3fb950; }
.badge-medium { background: rgba(240,163,92,0.18); color: #f0a35c; }
.badge-low { background: rgba(248,81,73,0.18); color: #f85149; }
.badge-none { background: rgba(139,148,158,0.18); color: #8b949e; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📊 Trading Idea Advisor")
st.caption("Per-ticker strategy discovery from technicals, sentiment, and an options-flow proxy — built on free-tier data.")

st.markdown(
    '<div class="disclaimer">⚠️ Educational tool, not financial advice. Free-tier data (delayed/EOD), '
    'a small watchlist, and a limited backtest window mean results can look better than live trading will feel. '
    'Options "flow" here is a volume/open-interest proxy from delayed chain data, not real-time order flow. '
    'Every reliability score and options idea is derived only from this app\'s own backtest — not a guarantee. '
    'Paper trade any strategy before risking capital.</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("Watchlist")
    default_text = ", ".join(config.WATCHLIST)
    tickers_input = st.text_area("Tickers (comma-separated)", value=default_text, height=80, key="tickers_input")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    run = st.button("Run analysis", type="primary", use_container_width=True, key="run_button")
    st.caption(f"Forward-return horizon: {config.FORWARD_RETURN_DAYS}d · Strategy candidates searched: {config.N_RANDOM_STRATEGIES}")

if "results" not in st.session_state:
    st.session_state.results = {}

if run:
    st.session_state.results = {}
    progress = st.progress(0.0, text="Starting...")
    for i, t in enumerate(tickers):
        progress.progress((i) / max(len(tickers), 1), text=f"Analyzing {t}...")
        try:
            st.session_state.results[t] = pipeline.analyze_ticker(t)
        except Exception as e:
            st.session_state.results[t] = {"error": str(e)}
        progress.progress((i + 1) / max(len(tickers), 1), text=f"Done: {t}")
    progress.empty()

results = st.session_state.results

if not results:
    st.info("Set your watchlist in the sidebar and click **Run analysis**.")
    st.stop()

BADGE_CLASS = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low", "No signal": "badge-none"}
SIGNAL_LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}
SIGNAL_CLASS = {1: "signal-long", -1: "signal-short", 0: "signal-flat"}
TREND_STATUS_LABEL = {
    "above_resistance": "Broke above resistance trendline",
    "below_support": "Broke below support trendline",
    "inside_channel": "Inside channel",
}
TREND_DIR_LABEL = {"up": "Uptrend", "down": "Downtrend", "flat": "Sideways"}

# ---- Overview table across the whole watchlist ----
st.subheader("Watchlist overview")
rows = []
for t, r in results.items():
    if "error" in r:
        rows.append({"Ticker": t, "Error": r["error"]})
        continue
    rel = r["reliability"]
    rows.append({
        "Ticker": t,
        "Signal": SIGNAL_LABEL.get(r["current_signal"], "FLAT"),
        "Reliability": f"{rel['label']} ({rel['score']})",
        "Last close": round(r["price_df"]["close"].iloc[-1], 2),
        "RSI": round(r["latest"]["rsi"], 1) if pd.notna(r["latest"]["rsi"]) else None,
        "ADX": round(r["latest"]["adx"], 1) if pd.notna(r["latest"]["adx"]) else None,
        "Trend": TREND_DIR_LABEL.get(r["latest"]["trend_direction"], "n/a"),
        "Sentiment": round(r["sentiment"]["score"], 2),
        "Options skew": round(r["options"]["unusual_score"], 2) if r["options"]["available"] else None,
        "Strategy OOS Sharpe": round(r["strategy"].test_metrics.get("sharpe", 0.0), 2) if r["strategy"].test_metrics else None,
        "Buy&Hold return %": round(r["baseline"]["total_return_pct"], 1),
    })
overview_df = pd.DataFrame(rows).set_index("Ticker")
st.dataframe(overview_df, use_container_width=True, key="overview_table")

st.divider()

# ---- Per-ticker deep dive ----
tabs = st.tabs(list(results.keys()))
for tab, t in zip(tabs, results.keys()):
    with tab:
        r = results[t]
        if "error" in r:
            st.error(f"Could not analyze {t}: {r['error']}")
            continue

        latest = r["latest"]
        strat = r["strategy"]
        rel = r["reliability"]
        sig = r["current_signal"]
        idea = r["option_idea"]

        # ---- Today's signal summary card ----
        st.markdown(
            f'<div class="signal-card {SIGNAL_CLASS.get(sig, "signal-flat")}">'
            f'<span style="font-size:1.4rem; font-weight:700;">{t} — {SIGNAL_LABEL.get(sig, "FLAT")}</span>'
            f'&nbsp;&nbsp;<span class="badge {BADGE_CLASS.get(rel["label"], "badge-none")}">Reliability: {rel["label"]} · {rel["score"]}/100</span>'
            f'<div style="margin-top:6px; color:var(--muted); font-size:0.85rem;">{rel["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close", f"${r['price_df']['close'].iloc[-1]:.2f}")
        col2.metric("RSI (14)", f"{latest['rsi']:.1f}" if pd.notna(latest["rsi"]) else "n/a")
        col3.metric("ADX (14)", f"{latest['adx']:.1f}" if pd.notna(latest["adx"]) else "n/a")
        trend_line = f"{TREND_DIR_LABEL.get(latest['trend_direction'], 'n/a')} · {TREND_STATUS_LABEL.get(latest['trend_status'], '')}"
        col4.metric("Trend", trend_line)

        # --- Price chart with VWAP proxy ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)
        df_plot = r["indicator_df"].tail(180)
        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"], name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["vwap_proxy"], name="VWAP (proxy)",
                                  line=dict(color="#4fd1c5", width=1.3)), row=1, col=1)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["volume"], name="Volume", marker_color="#3d4b5c"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["rsi"], name="RSI", line=dict(color="#f0a35c")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#8b949e", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#8b949e", row=3, col=1)
        fig.update_layout(height=560, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           margin=dict(l=10, r=10, t=10, b=10), showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{t}")

        left, right = st.columns(2)

        with left:
            st.markdown("##### Discovered strategy (search-optimized for this ticker)")
            if strat.long_rules or strat.short_rules:
                st.markdown(f'<div class="rule-box">{strat.describe()}</div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.caption("In-sample (train)")
                    tm = strat.train_metrics
                    st.write(f"Sharpe: **{tm.get('sharpe', 0):.2f}**  ·  Return: **{tm.get('total_return_pct', 0):.1f}%**  ·  "
                             f"Max DD: **{tm.get('max_drawdown_pct', 0):.1f}%**  ·  Trades: **{tm.get('n_trades', 0)}**")
                with m2:
                    st.caption("Out-of-sample (test)")
                    vm = strat.test_metrics
                    st.write(f"Sharpe: **{vm.get('sharpe', 0):.2f}**  ·  Return: **{vm.get('total_return_pct', 0):.1f}%**  ·  "
                             f"Max DD: **{vm.get('max_drawdown_pct', 0):.1f}%**  ·  Trades: **{vm.get('n_trades', 0)}**")
                st.caption(f"Buy & hold over same period: {r['baseline']['total_return_pct']:.1f}% return, "
                           f"Sharpe {r['baseline']['sharpe']:.2f}")
            else:
                st.warning("No strategy cleared the minimum trade/quality bar for this ticker. Try a longer lookback or a different ticker.")

            if not r["feature_importance"].empty:
                st.markdown("##### What drives this ticker (Random Forest importance)")
                imp_top = r["feature_importance"].head(8).sort_values()
                imp_fig = go.Figure(go.Bar(
                    x=imp_top.values, y=imp_top.index, orientation="h",
                    marker_color="#4fd1c5",
                ))
                imp_fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0d1117",
                                       plot_bgcolor="#0d1117", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(imp_fig, use_container_width=True, key=f"feat_imp_{t}")

        with right:
            st.markdown("##### News sentiment")
            sent = r["sentiment"]
            st.write(f"Aggregate score: **{sent['score']:.2f}** (−1 bearish → +1 bullish), from {sent['n_headlines']} headlines")
            for j, (h, s) in enumerate(sent["headlines"][:8]):
                tag = "🟢" if s > 0.15 else ("🔴" if s < -0.15 else "⚪")
                st.write(f"{tag} {h}  `{s:+.2f}`")

            st.markdown("##### Options flow proxy")
            opts = r["options"]
            if opts["available"]:
                st.write(f"Put/Call volume ratio: **{opts['put_call_ratio']:.2f}**  ·  "
                         f"Unusual-activity skew: **{opts['unusual_score']:+.2f}** (+bullish / −bearish)")
                st.caption(f"Call volume: {opts['call_volume']:,}  ·  Put volume: {opts['put_volume']:,}")
                if not opts["unusual_strikes"].empty:
                    st.dataframe(opts["unusual_strikes"], use_container_width=True, height=220, key=f"unusual_{t}")
            else:
                st.caption("No options chain data available for this ticker.")

        st.divider()
        st.markdown("##### 💡 Options idea (from this signal + backtest reliability)")
        st.markdown(f"**Structure: {idea['structure']}**")
        st.write(idea["rationale"])
        for w in idea["warnings"]:
            st.warning(w)
        if idea["candidates"] is not None and not idea["candidates"].empty:
            st.caption("Live candidate contracts near-the-money, ~14-60 days to expiry, sorted by volume:")
            st.dataframe(idea["candidates"], use_container_width=True, key=f"optideas_{t}")
        elif idea["structure"] != "No clear edge":
            st.caption("No matching liquid contracts found in the current chain snapshot for this structure/expiry window.")

        st.markdown("##### Strategy equity curve (out-of-sample)")
        if strat.test_metrics.get("equity_curve") is not None and len(strat.test_metrics["equity_curve"]) > 1:
            ec_fig = go.Figure()
            ec_fig.add_trace(go.Scatter(x=strat.test_metrics["equity_curve"].index,
                                         y=strat.test_metrics["equity_curve"], name="Strategy",
                                         line=dict(color="#4fd1c5")))
            ec_fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0d1117",
                                  plot_bgcolor="#0d1117", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(ec_fig, use_container_width=True, key=f"equity_{t}")
        else:
            st.caption("Not enough out-of-sample bars to plot an equity curve.")
