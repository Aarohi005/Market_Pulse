"""
Google Gemini LLM provider via LangChain.

Implements the :class:`~src.llm.base.LLMProvider` protocol using
``langchain-google-genai``.  The LangChain client is imported lazily so
the package is only required when this provider is actually used.

Install:
    pip install langchain-google-genai

Usage (via factory):
    from src.llm.factory import create_llm_provider
    provider = create_llm_provider("gemini", model="gemini-2.5-flash")

Usage (direct):
    from src.llm.providers.gemini_provider import GeminiProvider
    provider = GeminiProvider(api_key="...", model="gemini-2.5-flash")
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

_DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 2048


class GeminiProvider:
    """LLM provider backed by Google Gemini via LangChain.

    LangChain's ``ChatGoogleGenerativeAI`` is used so that the same
    prompt logic works across all providers.  The Gemini API does not
    support a separate ``system`` message natively; this provider
    prepends the system prompt to the human turn as a leading paragraph.

    Args:
        api_key: Google API key.  Falls back to the ``GOOGLE_API_KEY``
            environment variable when ``None``.
        model: Gemini model identifier.  Defaults to
            ``"gemini-2.5-flash"``.
        temperature: Sampling temperature.  Defaults to ``0.0`` for
            deterministic (greedy) output, which is preferable for
            structured JSON tasks.
        max_tokens: Maximum output tokens.  Defaults to 2 048.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm: Optional[Any] = None  # lazy-initialised

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def provider_name(self) -> str:
        return "gemini"

    def model_name(self) -> str:
        return self._model

    def analyze_sentiment(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Analyse sentiment of the provided articles using Gemini.

        Args:
            context: Short description of the query context (topic / asset).
            articles: List of article dicts.

        Returns:
            A populated :class:`~src.llm.base.LLMResponse`.
        """
        articles_block = format_articles_context(articles)
        user_message = (
            f"Context: {context}\n\n"
            f"{articles_block}\n\n"
            "Analyse the sentiment of the above articles and return ONLY valid JSON."
        )
        raw_text, latency_ms = self._call(SENTIMENT_ANALYSIS_PROMPT, user_message)
        return self._parse_sentiment_json(raw_text, latency_ms)

    def generate_market_commentary(
        self,
        context: str,
        articles: List[dict],
    ) -> LLMResponse:
        """Generate market commentary using Gemini.

        Args:
            context: Contextual framing for the commentary.
            articles: List of article dicts.

        Returns:
            A :class:`~src.llm.base.LLMResponse` where ``explanation``
            holds the narrative commentary paragraphs.
        """
        articles_block = format_articles_context(articles)
        user_message = (
            f"Context: {context}\n\n"
            f"{articles_block}\n\n"
            "Write the two-paragraph market commentary followed by the JSON object."
        )
        raw_text, latency_ms = self._call(MARKET_COMMENTARY_PROMPT, user_message)
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
        raw_text, _ = self._call(system, f"Summarise the following:\n\n{numbered}")
        return raw_text.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        """Lazily initialise and return the LangChain ChatGoogleGenerativeAI client.

        Raises:
            ImportError: If ``langchain-google-genai`` is not installed.
        """
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'langchain-google-genai' package is required for GeminiProvider. "
                    "Install it with: pip install langchain-google-genai"
                ) from exc

            kwargs: dict[str, Any] = dict(
                model=self._model,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            if self._api_key:
                kwargs["google_api_key"] = self._api_key

            self._llm = ChatGoogleGenerativeAI(**kwargs)
            logger.debug("GeminiProvider initialised (model=%s).", self._model)
        return self._llm

    def _call(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        """Invoke the Gemini model via LangChain.

        Gemini does not have a native system-message slot in the same way
        as OpenAI; LangChain forwards it as a system message using the
        ``HumanMessage`` / ``SystemMessage`` schema.  We compose a full
        ``messages`` list to keep the structure explicit.

        Args:
            system_prompt: System-role instructions.
            user_message: The user turn content.

        Returns:
            Tuple of (raw_response_text, latency_ms).
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        llm = self._get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        start = time.monotonic()
        logger.debug(
            "Calling Gemini API | model=%s | user_msg_len=%d",
            self._model, len(user_message),
        )

        response = llm.invoke(messages)
        latency_ms = int((time.monotonic() - start) * 1000)

        raw_text: str = response.content if isinstance(response.content, str) else ""
        logger.debug(
            "Gemini API response | latency_ms=%d | response_len=%d",
            latency_ms, len(raw_text),
        )
        return raw_text, latency_ms

    def _parse_sentiment_json(self, raw_text: str, latency_ms: int) -> LLMResponse:
        try:
            clean = raw_text.strip()
            # Strip markdown code fences if present
            if clean.startswith("```"):
                lines = clean.splitlines()
                clean = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
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
            logger.warning("Failed to parse Gemini JSON response: %s", exc)
            return self._fallback(raw_text, latency_ms)

    def _parse_commentary_response(self, raw_text: str, latency_ms: int) -> LLMResponse:
        try:
            json_start = raw_text.find("{")
            if json_start == -1:
                raise ValueError("No JSON object found in commentary response.")

            narrative = raw_text[:json_start].strip()
            json_str = raw_text[json_start:]
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
            logger.warning("Failed to parse Gemini commentary response: %s", exc)
            return self._fallback(raw_text, latency_ms)

    @staticmethod
    def _fallback(raw_text: str, latency_ms: int) -> LLMResponse:
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
