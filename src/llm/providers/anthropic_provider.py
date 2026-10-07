"""
Anthropic (Claude) LLM provider for financial news sentiment analysis.

Implements the :class:`~src.llm.base.LLMProvider` protocol using the
Anthropic Python SDK.  The SDK is imported lazily to avoid a hard dependency
at import time; the package must be installed separately (``anthropic>=0.25``).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional

from ..base import LLMResponse
from ..prompts import (
    MARKET_COMMENTARY_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    format_articles_context,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 2048


class AnthropicProvider:
    """LLM provider backed by Anthropic's Claude API.

    Args:
        api_key: Anthropic API key.  Passed directly to the ``anthropic``
            client; can also be set via the ``ANTHROPIC_API_KEY`` environment
            variable (the SDK picks it up automatically when ``api_key`` is
            ``None``).
        model: Claude model identifier.  Defaults to
            ``"claude-haiku-4-5-20251001"`` for cost-efficient inference.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Optional[Any] = None  # lazy-initialised

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def provider_name(self) -> str:
        """Return the canonical provider name.

        Returns:
            ``"anthropic"``
        """
        return "anthropic"

    def model_name(self) -> str:
        """Return the Claude model identifier in use.

        Returns:
            Model string, e.g. ``"claude-haiku-4-5-20251001"``.
        """
        return self._model

    def analyze_sentiment(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Analyse sentiment of the provided articles using Claude.

        Calls the Claude API with :data:`~src.llm.prompts.SENTIMENT_ANALYSIS_PROMPT`
        as the system prompt, then parses the JSON response.

        Args:
            context: Short description of the query context (topic / asset).
            articles: List of article dicts with keys: title, description,
                source_name, published_at.

        Returns:
            A populated :class:`~src.llm.base.LLMResponse`.  On JSON parse
            failure a fallback response with ``explanation="Parse error"`` is
            returned rather than raising an exception.
        """
        articles_block = format_articles_context(articles)
        user_message = (
            f"Context: {context}\n\n"
            f"{articles_block}\n\n"
            "Analyse the sentiment of the above articles and return ONLY valid JSON."
        )

        raw_text, latency_ms = self._call_api(
            system_prompt=SENTIMENT_ANALYSIS_PROMPT,
            user_message=user_message,
        )

        return self._parse_sentiment_json(raw_text, latency_ms)

    def generate_market_commentary(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Generate a two-paragraph market commentary and structured sentiment JSON.

        Calls the Claude API with :data:`~src.llm.prompts.MARKET_COMMENTARY_PROMPT`
        as the system prompt.  The response is expected to contain narrative
        prose followed by a JSON block; this method extracts the JSON portion
        and places the narrative in ``explanation``.

        Args:
            context: Contextual framing for the commentary.
            articles: List of article dicts.

        Returns:
            A :class:`~src.llm.base.LLMResponse` where ``explanation`` holds
            the narrative commentary paragraphs.
        """
        articles_block = format_articles_context(articles)
        user_message = (
            f"Context: {context}\n\n"
            f"{articles_block}\n\n"
            "Write the two-paragraph market commentary followed by the JSON object."
        )

        raw_text, latency_ms = self._call_api(
            system_prompt=MARKET_COMMENTARY_PROMPT,
            user_message=user_message,
        )

        return self._parse_commentary_response(raw_text, latency_ms)

    def summarize(self, texts: List[str]) -> str:
        """Summarise a list of text snippets into a single short paragraph.

        Args:
            texts: Individual text pieces to consolidate.

        Returns:
            A plain-text summary string.  Returns an empty string on failure.
        """
        if not texts:
            return ""

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        system = (
            "You are a concise financial summariser. "
            "Combine the following snippets into a single coherent paragraph "
            "of 3–5 sentences. Return only the paragraph text with no preamble."
        )
        user_message = f"Summarise the following:\n\n{numbered}"

        raw_text, _ = self._call_api(system_prompt=system, user_message=user_message)
        return raw_text.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return (and lazily initialise) the Anthropic client.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
        """
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.debug("Anthropic client initialised (model=%s).", self._model)
        return self._client

    def _call_api(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, int]:
        """Make a single chat completion request to the Claude API.

        Args:
            system_prompt: System-role instructions for Claude.
            user_message: The user turn of the conversation.

        Returns:
            Tuple of (raw_response_text, latency_ms).

        Raises:
            Exception: Propagates any Anthropic SDK errors to the caller.
        """
        client = self._get_client()
        start = time.monotonic()

        logger.debug(
            "Calling Anthropic API | model=%s | user_msg_len=%d",
            self._model,
            len(user_message),
        )

        response = client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        latency_ms = int((time.monotonic() - start) * 1000)
        raw_text: str = response.content[0].text if response.content else ""

        logger.debug(
            "Anthropic API response received | latency_ms=%d | response_len=%d",
            latency_ms,
            len(raw_text),
        )
        return raw_text, latency_ms

    def _parse_sentiment_json(self, raw_text: str, latency_ms: int) -> LLMResponse:
        """Extract and validate the JSON payload from a sentiment response.

        Args:
            raw_text: Raw string returned by the model.
            latency_ms: Measured API latency.

        Returns:
            A populated :class:`~src.llm.base.LLMResponse`.  Returns a
            fallback response on any parse failure.
        """
        try:
            # Strip potential markdown fences
            clean = raw_text.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            data: dict = json.loads(clean)
            return LLMResponse(
                sentiment_label=str(data.get("sentiment_label", "neutral")),
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence_label=str(data.get("confidence_label", "LOW")),
                explanation=str(data.get("explanation", "")),
                key_drivers=list(data.get("key_drivers", [])),
                conflicting_signals=list(data.get("conflicting_signals", [])),
                supporting_headlines=list(data.get("supporting_headlines", [])),
                raw_text=raw_text,
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse Anthropic JSON response: %s", exc)
            return self._fallback_response(raw_text, latency_ms)

    def _parse_commentary_response(
        self, raw_text: str, latency_ms: int
    ) -> LLMResponse:
        """Extract narrative commentary and JSON from a market commentary response.

        The model is expected to return prose paragraphs followed by a JSON
        object.  This method splits on the first ``{`` character to separate
        the two sections.

        Args:
            raw_text: Raw string from the model containing prose + JSON.
            latency_ms: Measured API latency.

        Returns:
            A :class:`~src.llm.base.LLMResponse` with ``explanation`` set to
            the narrative prose and structured fields from the JSON block.
        """
        try:
            # Split narrative from JSON at the first opening brace
            json_start = raw_text.find("{")
            if json_start == -1:
                raise ValueError("No JSON object found in commentary response.")

            narrative = raw_text[:json_start].strip()
            json_str = raw_text[json_start:]

            # Strip trailing content after the closing brace
            json_end = json_str.rfind("}")
            if json_end != -1:
                json_str = json_str[: json_end + 1]

            data: dict = json.loads(json_str)

            return LLMResponse(
                sentiment_label=str(data.get("sentiment_label", "neutral")),
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                confidence_label=str(data.get("confidence_label", "LOW")),
                explanation=narrative or str(data.get("explanation", "")),
                key_drivers=list(data.get("key_drivers", [])),
                conflicting_signals=list(data.get("conflicting_signals", [])),
                supporting_headlines=list(data.get("supporting_headlines", [])),
                raw_text=raw_text,
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Failed to parse Anthropic commentary response: %s", exc
            )
            return self._fallback_response(raw_text, latency_ms)

    @staticmethod
    def _fallback_response(raw_text: str, latency_ms: int) -> LLMResponse:
        """Return a safe fallback :class:`~src.llm.base.LLMResponse` on parse error.

        Args:
            raw_text: The unparseable raw model output.
            latency_ms: API latency from the failed call.

        Returns:
            A neutral :class:`~src.llm.base.LLMResponse` with
            ``explanation="Parse error"``.
        """
        return LLMResponse(
            sentiment_label="neutral",
            sentiment_score=0.0,
            confidence_label="LOW",
            explanation="Parse error",
            key_drivers=[],
            conflicting_signals=[],
            supporting_headlines=[],
            raw_text=raw_text,
            latency_ms=latency_ms,
        )
