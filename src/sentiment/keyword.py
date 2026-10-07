"""
Keyword-based sentiment scorer for financial news text.

Provides a fast, dependency-free heuristic that counts finance-specific
positive and negative keyword occurrences and maps the balance to a
[-1, 1] sentiment score.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Finance-specific keyword dictionaries
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: List[str] = [
    "beat",
    "surged",
    "record",
    "upgraded",
    "acquisition",
    "growth",
    "profit",
    "raised guidance",
    "expansion",
    "outperform",
    "bull",
    "rally",
    "breakthrough",
    "exceeds",
    "strong",
    "gains",
    "momentum",
    "soared",
    "recovery",
    "milestone",
    "dividend",
    "buyback",
    "partnership",
    "innovation",
    "robust",
]

NEGATIVE_KEYWORDS: List[str] = [
    "missed",
    "fell",
    "downgrade",
    "layoffs",
    "bankruptcy",
    "loss",
    "recession",
    "inflation",
    "crisis",
    "decline",
    "bear",
    "crash",
    "warning",
    "below",
    "weak",
    "cut guidance",
    "investigation",
    "fraud",
    "plunged",
    "defaulted",
    "debt",
    "lawsuit",
    "volatile",
    "uncertainty",
    "headwinds",
]


class KeywordSentimentScorer:
    """Rule-based sentiment scorer that counts finance keyword matches.

    The scorer is case-insensitive and supports multi-word keywords
    (e.g. ``"raised guidance"``).  The final score is a signed ratio
    of positive vs. negative matches in the range [-1, 1].

    Parameters
    ----------
    positive_keywords:
        Override the default :data:`POSITIVE_KEYWORDS` list.
    negative_keywords:
        Override the default :data:`NEGATIVE_KEYWORDS` list.
    positive_threshold:
        Minimum score above which the label is ``"positive"``.
        Defaults to ``0.1``.
    negative_threshold:
        Maximum score below which the label is ``"negative"``.
        Defaults to ``-0.1``.

    Examples
    --------
    >>> scorer = KeywordSentimentScorer()
    >>> label, score = scorer.score("Apple beat estimates and surged to a record high.")
    >>> label
    'positive'
    >>> round(score, 2)
    1.0
    """

    def __init__(
        self,
        positive_keywords: List[str] | None = None,
        negative_keywords: List[str] | None = None,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
    ) -> None:
        self._positive = [kw.lower() for kw in (positive_keywords or POSITIVE_KEYWORDS)]
        self._negative = [kw.lower() for kw in (negative_keywords or NEGATIVE_KEYWORDS)]
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, text: str) -> Tuple[str, float]:
        """Score the sentiment of *text* using keyword counting.

        The score is computed as:

        .. code-block:: text

            score = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        Parameters
        ----------
        text:
            The financial news text to analyse.  May be a headline,
            article body, or any free-form string.

        Returns
        -------
        tuple[str, float]
            A ``(label, score)`` pair where:

            * ``label`` is one of ``"positive"``, ``"negative"``, or
              ``"neutral"``.
            * ``score`` is a float in the range [-1, 1].
        """
        if not text or not text.strip():
            return "neutral", 0.0

        normalised = text.lower()
        pos_count = self._count_matches(normalised, self._positive)
        neg_count = self._count_matches(normalised, self._negative)

        total = max(pos_count + neg_count, 1)
        raw_score = (pos_count - neg_count) / total

        # Clamp to [-1, 1] as a safety measure
        raw_score = max(-1.0, min(1.0, raw_score))

        label = self._label(raw_score)
        return label, raw_score

    def score_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Score a list of texts.

        Parameters
        ----------
        texts:
            Iterable of strings to score.

        Returns
        -------
        list of tuple[str, float]
            One ``(label, score)`` pair per input text, in the same order.
        """
        return [self.score(t) for t in texts]

    def count_matches(
        self, text: str
    ) -> Tuple[int, int, List[str], List[str]]:
        """Return detailed match information for *text*.

        Parameters
        ----------
        text:
            Input text to analyse.

        Returns
        -------
        tuple
            ``(pos_count, neg_count, matched_positive, matched_negative)``
            where ``matched_positive`` and ``matched_negative`` are the
            actual keyword strings that were found.
        """
        normalised = text.lower()
        matched_pos: List[str] = []
        matched_neg: List[str] = []

        for kw in self._positive:
            if self._keyword_in_text(kw, normalised):
                matched_pos.append(kw)

        for kw in self._negative:
            if self._keyword_in_text(kw, normalised):
                matched_neg.append(kw)

        return len(matched_pos), len(matched_neg), matched_pos, matched_neg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_matches(normalised_text: str, keywords: List[str]) -> int:
        """Count how many keywords appear in *normalised_text*."""
        count = 0
        for kw in keywords:
            if KeywordSentimentScorer._keyword_in_text(kw, normalised_text):
                count += 1
        return count

    @staticmethod
    def _keyword_in_text(keyword: str, normalised_text: str) -> bool:
        """Return True if *keyword* appears as a word (or phrase) in text.

        Uses a word-boundary regex so that ``"bull"`` does not match
        ``"bulletin"`` and ``"bear"`` does not match ``"bearer"``.
        Multi-word keywords are matched as literal substrings with word
        boundaries around the whole phrase.
        """
        # Escape the keyword for use in a regex pattern
        escaped = re.escape(keyword)
        # For multi-word phrases, boundary is applied around the whole phrase
        pattern = r"\b" + escaped + r"\b"
        return bool(re.search(pattern, normalised_text))

    def _label(self, score: float) -> str:
        """Map a numeric *score* to a sentiment label string.

        Parameters
        ----------
        score:
            Numeric score in [-1, 1].

        Returns
        -------
        str
            ``"positive"``, ``"negative"``, or ``"neutral"``.
        """
        if score > self.positive_threshold:
            return "positive"
        if score < self.negative_threshold:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KeywordSentimentScorer("
            f"positive_keywords={len(self._positive)}, "
            f"negative_keywords={len(self._negative)})"
        )
