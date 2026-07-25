"""
Headline sentiment scoring. Uses VADER (rule-based, no API key, no
network call beyond the news fetch itself) since a full FinBERT model
is heavy for a free-tier personal app. Good enough to rank tone, not
meant to be a precision NLP model.
"""
from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import data_fetch

_analyzer = SentimentIntensityAnalyzer()

# A few finance-specific nudges VADER's general-purpose lexicon misses.
_FINANCE_LEXICON_ADJUST = {
    "beat": 2.0, "beats": 2.0, "miss": -2.0, "misses": -2.0, "downgrade": -2.5,
    "upgrade": 2.5, "bankruptcy": -3.5, "lawsuit": -1.5, "guidance": 0.0,
    "cut": -1.5, "raises": 1.5, "raised": 1.5, "layoffs": -2.0, "recall": -2.0,
    "buyback": 1.5, "delisted": -3.0, "investigation": -2.0, "hack": -2.0,
    "breach": -2.0, "fraud": -3.0, "surge": 2.0, "plunge": -2.5, "soar": 2.5,
    "slump": -2.0,
}
_analyzer.lexicon.update(_FINANCE_LEXICON_ADJUST)


def score_headline(text: str) -> float:
    """Compound sentiment score in [-1, 1]."""
    return _analyzer.polarity_scores(text)["compound"]


def get_ticker_sentiment(ticker: str) -> dict:
    """
    Fetches recent headlines for a ticker and returns an aggregate
    sentiment score plus the underlying headlines/scores for display.
    """
    headlines = data_fetch.get_news_headlines(ticker)
    if not headlines:
        return {"score": 0.0, "n_headlines": 0, "headlines": []}

    scored = [(h, score_headline(h)) for h in headlines]
    avg_score = sum(s for _, s in scored) / len(scored)
    return {
        "score": avg_score,          # -1 (very bearish) to +1 (very bullish)
        "n_headlines": len(scored),
        "headlines": scored,         # list of (headline, score)
    }
