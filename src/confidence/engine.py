"""
Confidence scoring engine.

Computes a composite confidence score (0–100) for a sentiment analysis
result by combining retrieval quality, source quality, and sentiment
agreement signals.  Each signal is normalised to [0, 1] before being
combined via a configurable weighted sum.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInput:
    """All signals required to compute a confidence score.

    Attributes
    ----------
    retrieval_similarity:
        Mean cosine similarity (or equivalent) of retrieved articles to the
        query.  Expected range [0, 1].
    reranking_score:
        Normalised score from a cross-encoder or reranker.  Range [0, 1].
    article_count:
        Number of articles that contributed to the sentiment analysis.
        Will be normalised as ``min(count / 10, 1.0)``.
    source_diversity:
        Fraction of unique sources among all articles, i.e.
        ``unique_sources / total_articles``.  Range [0, 1].
    source_quality:
        Mean quality score of the contributing sources.  Range [0, 1].
    sentiment_agreement:
        Agreement across individual article sentiments (1 = perfect
        agreement, 0 = maximal disagreement).  Range [0, 1].
    llm_confidence_label:
        Self-reported confidence label from the LLM provider.
        One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """

    retrieval_similarity: float
    reranking_score: float
    article_count: int
    source_diversity: float
    source_quality: float
    sentiment_agreement: float
    llm_confidence_label: str


@dataclass
class ConfidenceResult:
    """Output of a confidence engine computation.

    Attributes
    ----------
    final_score:
        Composite confidence score in the range [0, 100].
    confidence_label:
        Bucketed label: ``"HIGH"`` (≥80), ``"MEDIUM"`` (≥50), or
        ``"LOW"`` (<50).
    component_scores:
        Normalised [0, 1] score for each input signal after
        normalisation/clamping.
    weights:
        The effective weight applied to each component.
    """

    final_score: float
    confidence_label: str
    component_scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the result to a plain dictionary."""
        return {
            "final_score": self.final_score,
            "confidence_label": self.confidence_label,
            "component_scores": self.component_scores,
            "weights": self.weights,
        }


# ---------------------------------------------------------------------------
# Default weight constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "retrieval_similarity": 0.20,
    "reranking_score": 0.15,
    "article_count": 0.15,
    "source_diversity": 0.10,
    "source_quality": 0.10,
    "sentiment_agreement": 0.20,
    "llm_confidence": 0.10,
}

_LLM_CONFIDENCE_MAP: Dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ConfidenceEngine:
    """Computes a composite confidence score from multiple quality signals.

    Parameters
    ----------
    retrieval_similarity_weight:
        Weight for the retrieval similarity component.
    reranking_score_weight:
        Weight for the reranking score component.
    article_count_weight:
        Weight for the article count component.
    source_diversity_weight:
        Weight for the source diversity component.
    source_quality_weight:
        Weight for the source quality component.
    sentiment_agreement_weight:
        Weight for the sentiment agreement component.
    llm_confidence_weight:
        Weight for the LLM self-reported confidence component.

    Examples
    --------
    >>> engine = ConfidenceEngine()
    >>> inputs = ConfidenceInput(
    ...     retrieval_similarity=0.85,
    ...     reranking_score=0.78,
    ...     article_count=12,
    ...     source_diversity=0.60,
    ...     source_quality=0.80,
    ...     sentiment_agreement=0.90,
    ...     llm_confidence_label="HIGH",
    ... )
    >>> result = engine.compute(inputs)
    >>> result.confidence_label
    'HIGH'
    """

    def __init__(
        self,
        retrieval_similarity_weight: float = _DEFAULT_WEIGHTS["retrieval_similarity"],
        reranking_score_weight: float = _DEFAULT_WEIGHTS["reranking_score"],
        article_count_weight: float = _DEFAULT_WEIGHTS["article_count"],
        source_diversity_weight: float = _DEFAULT_WEIGHTS["source_diversity"],
        source_quality_weight: float = _DEFAULT_WEIGHTS["source_quality"],
        sentiment_agreement_weight: float = _DEFAULT_WEIGHTS["sentiment_agreement"],
        llm_confidence_weight: float = _DEFAULT_WEIGHTS["llm_confidence"],
    ) -> None:
        self.weights: Dict[str, float] = {
            "retrieval_similarity": retrieval_similarity_weight,
            "reranking_score": reranking_score_weight,
            "article_count": article_count_weight,
            "source_diversity": source_diversity_weight,
            "source_quality": source_quality_weight,
            "sentiment_agreement": sentiment_agreement_weight,
            "llm_confidence": llm_confidence_weight,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "ConfidenceEngine weights sum to %.4f, not 1.0. "
                "Normalising automatically.",
                total,
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, inputs: ConfidenceInput) -> ConfidenceResult:
        """Compute the composite confidence score from *inputs*.

        Normalisation rules applied before the weighted sum:

        * ``retrieval_similarity`` — already in [0, 1]; clamped.
        * ``reranking_score`` — already in [0, 1]; clamped.
        * ``article_count`` — ``min(count / 10, 1.0)``.
        * ``source_diversity`` — already in [0, 1]; clamped.
        * ``source_quality`` — already in [0, 1]; clamped.
        * ``sentiment_agreement`` — already in [0, 1]; clamped.
        * ``llm_confidence_label`` — ``HIGH→1.0``, ``MEDIUM→0.6``,
          ``LOW→0.3``.

        Parameters
        ----------
        inputs:
            Populated :class:`ConfidenceInput` instance.

        Returns
        -------
        ConfidenceResult
            Contains the final score [0–100], a label (HIGH/MEDIUM/LOW),
            the per-component normalised scores, and the effective weights.
        """
        # Normalise each component to [0, 1]
        component_scores: Dict[str, float] = {
            "retrieval_similarity": _clamp(inputs.retrieval_similarity),
            "reranking_score": _clamp(inputs.reranking_score),
            "article_count": min(inputs.article_count / 10.0, 1.0),
            "source_diversity": _clamp(inputs.source_diversity),
            "source_quality": _clamp(inputs.source_quality),
            "sentiment_agreement": _clamp(inputs.sentiment_agreement),
            "llm_confidence": self._normalise_llm_confidence(
                inputs.llm_confidence_label
            ),
        }

        # Weighted sum → scale to 0-100
        raw = sum(
            component_scores[k] * self.weights[k] for k in component_scores
        )
        final_score = round(_clamp(raw) * 100.0, 2)
        confidence_label = self._bucket(final_score)

        return ConfidenceResult(
            final_score=final_score,
            confidence_label=confidence_label,
            component_scores=component_scores,
            weights=dict(self.weights),
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sentiment_agreement(sentiment_scores: List[float]) -> float:
        """Compute the agreement level across a set of sentiment scores.

        Agreement is defined as ``1 - std(scores)``, clipped to [0, 1].
        A standard deviation of 0 (perfect agreement) gives score 1.0;
        the maximum possible standard deviation for scores in [-1, 1] is
        approximately 1.0 (e.g. half positive, half negative extremes).

        Parameters
        ----------
        sentiment_scores:
            List of numeric sentiment scores in [-1, 1].

        Returns
        -------
        float
            Agreement score in [0, 1].  Returns ``1.0`` for lists with
            fewer than two elements.
        """
        if len(sentiment_scores) < 2:
            return 1.0
        try:
            std = statistics.stdev(sentiment_scores)
        except statistics.StatisticsError:
            return 1.0
        agreement = 1.0 - std
        return float(_clamp(agreement))

    @staticmethod
    def compute_source_diversity(source_names: List[str]) -> float:
        """Compute source diversity as the fraction of unique sources.

        Parameters
        ----------
        source_names:
            List of source name strings (may contain duplicates).

        Returns
        -------
        float
            ``unique_count / total_count`` in [0, 1].  Returns ``0.0``
            for an empty list.
        """
        if not source_names:
            return 0.0
        unique = len(set(source_names))
        return float(unique / len(source_names))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_llm_confidence(label: str) -> float:
        """Convert a textual LLM confidence label to a numeric value.

        Parameters
        ----------
        label:
            One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``
            (case-insensitive).  Unknown labels map to ``0.3``.

        Returns
        -------
        float
            Numeric representation in [0, 1].
        """
        return _LLM_CONFIDENCE_MAP.get(label.upper().strip(), 0.3)

    @staticmethod
    def _bucket(score: float) -> str:
        """Map a 0–100 *score* to a HIGH/MEDIUM/LOW label.

        Parameters
        ----------
        score:
            Composite confidence score in [0, 100].

        Returns
        -------
        str
            ``"HIGH"`` if score ≥ 80, ``"MEDIUM"`` if score ≥ 50,
            otherwise ``"LOW"``.
        """
        if score >= 80:
            return "HIGH"
        if score >= 50:
            return "MEDIUM"
        return "LOW"

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConfidenceEngine(weights={self.weights})"


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))
