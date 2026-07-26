"""
Turns (multi-factor confluence + backtest reliability + live options
chain) into a plain-language options idea with concrete candidate
contracts.

Deliberately conservative: it never sizes a position or tells you to
execute anything. It picks a *contract type and rough structure* that
matches the confluence direction and strength, then points at real
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
    confluence_result: dict,   # output of confluence.compute_confluence
    reliability: dict,         # output of strategy_engine.compute_reliability
    latest_indicators: pd.Series,
    options_chain: pd.DataFrame,
    sentiment_score: float,
) -> dict:
    score = confluence_result.get("score", 0)
    label = confluence_result.get("label", "Mixed / no confluence")
    n_agree = confluence_result.get("n_agree", 0)
    n_disagree = confluence_result.get("n_disagree", 0)
    n_factors = confluence_result.get("n_factors", 6)
    adx_val = latest_indicators.get("adx", 0) or 0
    trend_strong = adx_val >= 25
    rel_label = reliability.get("label", "No signal")

    direction = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    strong_confluence = abs(score) >= 40 and n_disagree <= 1
    moderate_confluence = abs(score) >= 15

    if direction == "neutral" or not moderate_confluence:
        structure = "No clear edge"
        contract_type = None
        rationale = (
            f"Only {n_agree} of {n_factors} factors lean the same direction right now ({label}). "
            "That's not enough agreement across trend, momentum, patterns, options flow, and "
            "sentiment to justify a directional options bet. Sitting out, or a neutral/income "
            "structure if you already hold the stock, fits better."
        )
    elif direction == "bullish":
        contract_type = "call"
        if strong_confluence and trend_strong and rel_label in ("High", "Medium"):
            structure = "Long call (directional)"
        elif strong_confluence or (moderate_confluence and rel_label in ("High", "Medium")):
            structure = "Bull call spread (defined-risk)"
        else:
            structure = "Wait, or small bull call spread"
        rationale = (
            f"{n_agree} of {n_factors} factors agree bullish ({label}, score {score:+.0f}/100), "
            f"backtest reliability is {rel_label.lower()}, and ADX ({adx_val:.0f}) shows a "
            f"{'trending' if trend_strong else 'non-trending'} market."
        )
    else:
        contract_type = "put"
        if strong_confluence and trend_strong and rel_label in ("High", "Medium"):
            structure = "Long put (directional)"
        elif strong_confluence or (moderate_confluence and rel_label in ("High", "Medium")):
            structure = "Bear put spread (defined-risk)"
        else:
            structure = "Wait, or small bear put spread"
        rationale = (
            f"{n_agree} of {n_factors} factors agree bearish ({label}, score {score:+.0f}/100), "
            f"backtest reliability is {rel_label.lower()}, and ADX ({adx_val:.0f}) shows a "
            f"{'trending' if trend_strong else 'non-trending'} market."
        )

    warnings = []
    if moderate_confluence:
        if n_disagree >= 2:
            warnings.append(f"{n_disagree} of {n_factors} factors disagree with this direction -- confluence is not unanimous.")
        if direction == "bullish" and sentiment_score < -0.15:
            warnings.append("News sentiment is currently negative, cutting against the bullish read.")
        if direction == "bearish" and sentiment_score > 0.15:
            warnings.append("News sentiment is currently positive, cutting against the bearish read.")

    candidates = pd.DataFrame()
    if contract_type and options_chain is not None and not options_chain.empty:
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
