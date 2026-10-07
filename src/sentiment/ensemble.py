"""
Ensemble sentiment scorer.

Combines signals from FinBERT, an LLM provider, and the keyword-based
scorer into a single weighted aggregate score and label.  Any of the
three signal sources may be absent (``None``); the remaining weights are
redistributed proportionally so the total always sums to 1.0.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Stores the output of an ensemble sentiment scoring operation.

    Attributes
    ----------
    finbert_label:
        Sentiment label produced by FinBERT (positive/negative/neutral)
        or ``None`` if the FinBERT signal was unavailable.
    finbert_score:
        Numeric FinBERT sentiment score in [-1, 1] or ``None``.
    llm_label:
        Sentiment label produced by the LLM provider or ``None``.
    llm_score:
        Numeric LLM sentiment score in [-1, 1] or ``None``.
    keyword_label:
        Sentiment label from the keyword scorer or ``None``.
    keyword_score:
        Numeric keyword score in [-1, 1] or ``None``.
    ensemble_label:
        Final aggregated label: ``"positive"``, ``"negative"``, or
        ``"neutral"``.
    ensemble_score:
        Final weighted average score in [-1, 1].
    effective_weights:
        The actual weights used after redistribution for missing signals.
    """

    finbert_label: Optional[str]
    finbert_score: Optional[float]
    llm_label: Optional[str]
    llm_score: Optional[float]
    keyword_label: Optional[str]
    keyword_score: Optional[float]
    ensemble_label: str
    ensemble_score: float
    effective_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "finbert_label": self.finbert_label,
            "finbert_score": self.finbert_score,
            "llm_label": self.llm_label,
            "llm_score": self.llm_score,
            "keyword_label": self.keyword_label,
            "keyword_score": self.keyword_score,
            "ensemble_label": self.ensemble_label,
            "ensemble_score": self.ensemble_score,
            "effective_weights": self.effective_weights,
        }


# ---------------------------------------------------------------------------
# Signal type alias
# ---------------------------------------------------------------------------

# A signal is a (label, score) pair or None
Signal = Optional[Tuple[str, float]]


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------


class EnsembleScorer:
    """Weighted ensemble that combines three sentiment signals.

    Signals are converted to a unified [-1, 1] numeric space before
    averaging:

    * ``"positive"`` → ``+abs(score)``
    * ``"negative"`` → ``-abs(score)``
    * ``"neutral"``  → ``0.0``

    When a signal is ``None`` its weight is redistributed proportionally
    among the available signals.

    Parameters
    ----------
    finbert_weight:
        Weight assigned to the FinBERT signal.  Default ``0.45``.
    llm_weight:
        Weight assigned to the LLM signal.  Default ``0.35``.
    keyword_weight:
        Weight assigned to the keyword signal.  Default ``0.20``.
    positive_threshold:
        Ensemble score above which the label is ``"positive"``.
        Default ``0.1``.
    negative_threshold:
        Ensemble score below which the label is ``"negative"``.
        Default ``-0.1``.

    Examples
    --------
    >>> scorer = EnsembleScorer()
    >>> result = scorer.score(
    ...     finbert_result=("positive", 0.82),
    ...     llm_result=("positive", 0.75),
    ...     keyword_result=("neutral", 0.05),
    ... )
    >>> result.ensemble_label
    'positive'
    """

    def __init__(
        self,
        finbert_weight: float = 0.45,
        llm_weight: float = 0.35,
        keyword_weight: float = 0.20,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
    ) -> None:
        if abs(finbert_weight + llm_weight + keyword_weight - 1.0) > 1e-6:
            logger.warning(
                "EnsembleScorer weights sum to %.4f, not 1.0. "
                "They will be normalised automatically.",
                finbert_weight + llm_weight + keyword_weight,
            )
        self._base_weights: Dict[str, float] = {
            "finbert": finbert_weight,
            "llm": llm_weight,
            "keyword": keyword_weight,
        }
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(
        self,
        finbert_result: Signal,
        llm_result: Signal,
        keyword_result: Signal,
    ) -> EnsembleResult:
        """Combine three sentiment signals into a single ensemble result.

        Parameters
        ----------
        finbert_result:
            ``(label, score)`` tuple from FinBERT, or ``None``.
        llm_result:
            ``(label, score)`` tuple from an LLM provider, or ``None``.
        keyword_result:
            ``(label, score)`` tuple from the keyword scorer, or ``None``.

        Returns
        -------
        EnsembleResult
            Aggregated result with a final label and score plus all
            individual signal values for transparency.
        """
        signals: Dict[str, Signal] = {
            "finbert": finbert_result,
            "llm": llm_result,
            "keyword": keyword_result,
        }

        # Resolve effective weights (redistribute weight for missing signals)
        effective_weights = self._resolve_weights(signals)

        # Convert each signal to a signed numeric value in [-1, 1]
        numeric_signals: Dict[str, float] = {}
        for name, signal in signals.items():
            if signal is not None:
                numeric_signals[name] = self._to_numeric(signal)

        if not numeric_signals:
            # All signals missing — return neutral
            logger.warning("EnsembleScorer received no valid signals; returning neutral.")
            return EnsembleResult(
                finbert_label=None,
                finbert_score=None,
                llm_label=None,
                llm_score=None,
                keyword_label=None,
                keyword_score=None,
                ensemble_label="neutral",
                ensemble_score=0.0,
                effective_weights=effective_weights,
            )

        # Weighted average
        ensemble_score = sum(
            numeric_signals[name] * effective_weights.get(name, 0.0)
            for name in numeric_signals
        )
        # Clamp to [-1, 1]
        ensemble_score = max(-1.0, min(1.0, ensemble_score))
        ensemble_label = self._label(ensemble_score)

        # Unpack individual results safely
        fb_label, fb_score = _unpack(finbert_result)
        lm_label, lm_score = _unpack(llm_result)
        kw_label, kw_score = _unpack(keyword_result)

        return EnsembleResult(
            finbert_label=fb_label,
            finbert_score=fb_score,
            llm_label=lm_label,
            llm_score=lm_score,
            keyword_label=kw_label,
            keyword_score=kw_score,
            ensemble_label=ensemble_label,
            ensemble_score=ensemble_score,
            effective_weights=effective_weights,
        )

    def score_many(
        self,
        articles_with_signals: List[Dict[str, Signal]],
    ) -> List[EnsembleResult]:
        """Score multiple articles in a single call.

        Parameters
        ----------
        articles_with_signals:
            List of dictionaries, each containing up to three keys:
            ``"finbert"``, ``"llm"``, ``"keyword"``, each mapping to a
            ``(label, score)`` tuple or ``None``.

        Returns
        -------
        list of EnsembleResult
            One result per input dictionary, in the same order.

        Examples
        --------
        >>> scorer = EnsembleScorer()
        >>> results = scorer.score_many([
        ...     {"finbert": ("positive", 0.9), "llm": ("positive", 0.8), "keyword": None},
        ...     {"finbert": ("negative", 0.7), "llm": None, "keyword": ("negative", -0.6)},
        ... ])
        """
        results: List[EnsembleResult] = []
        for item in articles_with_signals:
            result = self.score(
                finbert_result=item.get("finbert"),
                llm_result=item.get("llm"),
                keyword_result=item.get("keyword"),
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, signals: Dict[str, Signal]) -> Dict[str, float]:
        """Compute effective weights after removing missing signals.

        The weights of absent signals are redistributed proportionally
        among the remaining signals so the total always equals 1.0.

        Parameters
        ----------
        signals:
            Dict mapping signal name → signal value (may be ``None``).

        Returns
        -------
        dict
            Mapping of signal name → effective weight (only includes
            present signals).
        """
        present = {name: w for name, w in self._base_weights.items() if signals.get(name) is not None}
        if not present:
            return {}

        total = sum(present.values())
        if total == 0:
            # Equal weights as last resort
            equal = 1.0 / len(present)
            return {name: equal for name in present}

        return {name: w / total for name, w in present.items()}

    @staticmethod
    def _to_numeric(signal: Tuple[str, float]) -> float:
        """Convert a ``(label, score)`` signal to a signed numeric value.

        Conversion rules:

        * ``"positive"`` → ``+abs(score)``
        * ``"negative"`` → ``-abs(score)``
        * ``"neutral"`` or any other label → ``0.0``

        Parameters
        ----------
        signal:
            ``(label, score)`` tuple where score is in [-1, 1].

        Returns
        -------
        float
            Signed numeric representation in [-1, 1].
        """
        label, score = signal
        label = label.lower().strip()
        if label == "positive":
            return abs(score)
        if label == "negative":
            return -abs(score)
        return 0.0

    def _label(self, score: float) -> str:
        """Map *score* to a sentiment label string.

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
        w = self._base_weights
        return (
            f"EnsembleScorer("
            f"finbert_weight={w['finbert']}, "
            f"llm_weight={w['llm']}, "
            f"keyword_weight={w['keyword']})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _unpack(signal: Signal) -> Tuple[Optional[str], Optional[float]]:
    """Unpack a Signal into ``(label, score)``, returning ``(None, None)``
    when the signal is absent."""
    if signal is None:
        return None, None
    return signal[0], signal[1]
