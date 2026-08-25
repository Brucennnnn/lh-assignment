"""Is this a question the internal knowledge base could plausibly answer?

Two cheap, explainable checks instead of a classifier LLM call:

1. `precheck` - a deny-list for request types that are clearly not internal
   knowledge lookups (weather, sport, creative writing, market prices, chit-chat).
   These are caught before retrieval because some of them, e.g. "write me a poem
   about expense claims", would otherwise retrieve well.
2. `classify_by_score` - after retrieval, if nothing in the corpus is even
   loosely related, the question is out of scope regardless of its wording.
"""
import re

from . import config

OUT_OF_SCOPE_PATTERNS = [
    (r"\b(weather|temperature|forecast|rain|humidity)\b", "weather"),
    (r"\b(football|soccer|match|game|score|tournament|world cup|olympics)\b.*\b(won|win|result|score)\b|\bwho won\b", "sports"),
    (r"\b(write|compose|create|generate)\s+(me\s+)?(a|an|some)?\s*(poem|song|story|joke|haiku|rap|essay|novel)\b", "creative_writing"),
    (r"\btell\s+me\s+a\s+(joke|story)\b", "creative_writing"),
    (r"\b(stock price|share price|exchange rate|bitcoin|crypto price|market cap)\b", "market_data"),
    (r"\b(what|who|when)\s+(is|are|was)\s+the\s+(capital|president|prime minister|population)\b", "general_knowledge"),
    (r"\b(recipe|cook|restaurant near|movie|netflix|horoscope)\b", "personal_life"),
]

REJECTION_MESSAGE = "This question is outside the scope of the available company knowledge."


def precheck(query: str) -> str | None:
    """Return an out-of-scope reason, or None to continue to retrieval."""
    normalised = " ".join(query.lower().split())
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, normalised):
            return reason
    return None


def classify_by_score(top_score: float) -> str | None:
    if top_score < config.SCOPE_THRESHOLD:
        return "no_related_content"
    return None
