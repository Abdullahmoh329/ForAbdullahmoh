"""
Turns (technical signal + backtest reliability + live options chain) into a
plain-language options idea with concrete candidate contracts.

Deliberately conservative: it never sizes a position or tells you to
execute anything. It picks a *contract type and rough structure* that
matches the signal's direction and confidence, then points at real
contracts from the current chain snapshot that fit that structure. You
still decide strike/expiry/size -- treat this as a research starting
point, not an instruction.
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime


def _days_to_expiry(expiry_str: str) -> int:
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d")
        return (exp - datetime.now()).days
    except Exception:
        return -1


def suggest_options_idea(
    ticker: str,
    signal: int,               # -1 / 0 / 1 -- from the discovered strategy, latest bar
    reliability: dict,         # output of strategy_engine.compute_reliability
    latest_indicators: pd.Series,
    options_chain: pd.DataFrame,
    sentiment_score: float,
) -> dict:
    adx_val = latest_indicators.get("adx", 0) or 0
    trend_strong = adx_val >= 25
    rel_label = reliability.get("label", "No signal")

    if signal == 0 or rel_label == "No signal":
        structure = "No clear edge"
        rationale = (
            "The strategy search didn't find a reliable directional edge for this ticker right now. "
            "Sitting out, or using a neutral/income structure if you already hold the stock "
            "(e.g. a covered call), fits better than a directional options bet."
        )
        contract_type = None
    elif signal == 1:
        if rel_label == "High" and trend_strong:
            structure = "Long call (directional)"
            rationale = (
                f"The backtested strategy is bullish with {rel_label.lower()} reliability, and ADX "
                f"({adx_val:.0f}) shows a trending market -- conditions where a directional long call "
                "tends to have room to work rather than chop in a range."
            )
        elif rel_label in ("High", "Medium"):
            structure = "Bull call spread (defined-risk)"
            rationale = (
                f"Bullish signal with {rel_label.lower()} reliability but "
                f"{'a trending' if trend_strong else 'a non-trending'} market (ADX {adx_val:.0f}). "
                "A defined-risk spread caps cost and downside if the move stalls."
            )
        else:
            structure = "Wait, or small bull call spread"
            rationale = "Bullish lean but low reliability -- if trading it at all, keep size and risk small."
        contract_type = "call"
    else:
        if rel_label == "High" and trend_strong:
            structure = "Long put (directional)"
            rationale = (
                f"The backtested strategy is bearish with {rel_label.lower()} reliability, and ADX "
                f"({adx_val:.0f}) shows a trending market."
            )
        elif rel_label in ("High", "Medium"):
            structure = "Bear put spread (defined-risk)"
            rationale = (
                f"Bearish signal with {rel_label.lower()} reliability but "
                f"{'a trending' if trend_strong else 'a non-trending'} market (ADX {adx_val:.0f}). "
                "A defined-risk spread caps cost if the move doesn't follow through."
            )
        else:
            structure = "Wait, or small bear put spread"
            rationale = "Bearish lean but low reliability -- if trading it at all, keep size and risk small."
        contract_type = "put"

    warnings = []
    if signal == 1 and sentiment_score < -0.15:
        warnings.append("News sentiment is currently negative, which cuts against this bullish technical signal.")
    if signal == -1 and sentiment_score > 0.15:
        warnings.append("News sentiment is currently positive, which cuts against this bearish technical signal.")

    candidates = pd.DataFrame()
    if contract_type and not options_chain.empty:
        chain = options_chain[options_chain["type"] == contract_type].copy()
        chain["dte"] = chain["expiry"].apply(_days_to_expiry)
        chain = chain[(chain["dte"] >= 14) & (chain["dte"] <= 60)]
        if not chain.empty:
            target_expiry = chain.loc[chain["dte"].sub(30).abs().idxmin(), "expiry"]
            near_expiry = chain[chain["expiry"] == target_expiry]
            liquid = near_expiry[near_expiry["volume"] > 0].sort_values("volume", ascending=False)
            candidates = liquid.head(6)[["contractSymbol", "strike", "expiry", "volume", "openInterest", "impliedVolatility"]]

    return {
        "structure": structure,
        "rationale": rationale,
        "warnings": warnings,
        "candidates": candidates,
    }
