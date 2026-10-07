"""
Base abstractions for LLM providers used in financial news sentiment analysis.

Defines the shared response dataclass and the LLMProvider Protocol that all
concrete provider implementations must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response returned by any LLM provider.

    Attributes:
        sentiment_label: One of ``"positive"``, ``"negative"``, ``"neutral"``.
        sentiment_score: Numeric sentiment intensity in [-1.0, 1.0] where
            -1.0 is maximally negative and 1.0 is maximally positive.
        confidence_label: One of ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
        explanation: Prose explanation of the sentiment assessment.
        key_drivers: List of 3–5 factors that most influenced the sentiment.
        conflicting_signals: Any headlines or data points that contradict the
            dominant sentiment signal.
        supporting_headlines: Article titles or snippets used as primary
            evidence for the sentiment determination.
        raw_text: The raw text returned by the model before any parsing.
        latency_ms: Wall-clock time in milliseconds from request to response.
    """

    sentiment_label: str
    sentiment_score: float
    confidence_label: str
    explanation: str
    key_drivers: List[str]
    conflicting_signals: List[str]
    supporting_headlines: List[str]
    raw_text: str
    latency_ms: int


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface every LLM provider must implement.

    Concrete classes (e.g. :class:`AnthropicProvider`, :class:`OpenAIProvider`)
    must implement all five methods below.  Using :func:`typing.runtime_checkable`
    allows ``isinstance`` checks at runtime if needed.
    """

    def analyze_sentiment(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Analyse the sentiment of a collection of news articles.

        Args:
            context: A short description of the query context (e.g. topic,
                sector, time window) to guide the model's analysis.
            articles: List of article dicts.  Each dict should contain at
                minimum: ``title``, ``description``, ``source_name``,
                ``published_at``.

        Returns:
            A fully populated :class:`LLMResponse`.
        """
        ...

    def generate_market_commentary(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Generate a concise market commentary based on recent news articles.

        Produces both a structured JSON sentiment assessment and a two-paragraph
        narrative commentary suitable for display to end users.

        Args:
            context: Contextual framing for the commentary (e.g. asset class,
                relevant macro backdrop).
            articles: List of article dicts (same schema as
                :meth:`analyze_sentiment`).

        Returns:
            A :class:`LLMResponse` where ``explanation`` contains the narrative
            commentary and structured fields carry the sentiment assessment.
        """
        ...

    def summarize(self, texts: List[str]) -> str:
        """Summarise a list of text snippets into a single short paragraph.

        Args:
            texts: Individual text pieces to be consolidated (e.g. article
                descriptions, analyst quotes).

        Returns:
            A plain-text summary string.
        """
        ...

    def provider_name(self) -> str:
        """Return the canonical name of the provider (e.g. ``"anthropic"``).

        Returns:
            Lower-case provider identifier string.
        """
        ...

    def model_name(self) -> str:
        """Return the model identifier being used (e.g. ``"claude-haiku-4-5-20251001"``).

        Returns:
            Model identifier string as accepted by the provider's API.
        """
        ...
