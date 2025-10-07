"""Lightweight sentiment analysis helpers for text snippets."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _get_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """Return the VADER compound score and a coarse label."""

    if not text:
        return 0.0, "neutral"

    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = float(scores.get("compound", 0.0))

    if compound >= 0.2:
        label = "positive"
    elif compound <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    return compound, label

