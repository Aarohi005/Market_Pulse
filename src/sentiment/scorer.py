"""
scorer.py — FinBERT sentiment scoring.

Model choice rationale
─────────────────────
We use ProsusAI/finbert, a BERT-base model fine-tuned on ~10,000 financial
news sentences from Financial PhraseBank (Malo et al., 2014).

Why FinBERT over alternatives:
  - VADER: rule-based, misreads finance idioms ("beat estimates" = positive,
    but VADER sees "beat" neutrally).
  - Twitter-roBERTa-sentiment: trained on social media, not financial prose.
  - FinBERT: trained specifically on analyst reports and financial news.
    Three-class output (positive/negative/neutral) maps cleanly to our schema.

Known limitations (called out honestly):
  - 512-token input limit: long articles get truncated. Mitigation: score
    headline + first 300 chars of description (see ArticleRecord.text_for_scoring).
  - English-only. Non-English articles get neutral scores.
  - Domain shift: fine-tuned on Bloomberg/Reuters-style prose; may underperform
    on social media reposts of financial news.
  - Batch size: defaults to 16 for CPU; GPU users can increase this.

Inference is CPU-safe by default.  Set device="cuda:0" for GPU acceleration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model identifier — can be overridden via environment variable
_DEFAULT_MODEL = "ProsusAI/finbert"
_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 32


@dataclass
class SentimentResult:
    """Sentiment output for a single text."""
    label: str          # "positive" | "negative" | "neutral"
    score: float        # softmax confidence of winning label (0..1)
    model_name: str


class SentimentScorer:
    """
    Wraps HuggingFace pipeline for FinBERT inference.

    The model is loaded lazily on first call to avoid import-time overhead
    and to allow the class to be instantiated in test environments without
    a GPU/model download.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  Defaults to ProsusAI/finbert.
    batch_size : int
        Number of texts to score in one forward pass.
    device : str | int
        "cpu", "cuda:0", etc.  Defaults to "cpu".
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._pipeline = None  # lazy initialisation

    def _load(self) -> None:
        """Download (on first run) and load the FinBERT pipeline."""
        if self._pipeline is not None:
            return

        # Respect HF_HOME env var for cache location
        hf_home = os.environ.get("HF_HOME", "./data/hf_cache")
        os.makedirs(hf_home, exist_ok=True)

        logger.info("Loading sentiment model '%s' (device=%s) …", self._model_name, self._device)
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="text-classification",
                model=self._model_name,
                tokenizer=self._model_name,
                device=self._device,
                top_k=1,              # return only the winning label
                truncation=True,
                max_length=_MAX_LENGTH,
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load sentiment model: %s", exc)
            raise

    def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Score a list of texts and return SentimentResult per text.

        Falls back to neutral/0.5 for any text that causes an error.
        """
        self._load()
        results = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_results = self._score_chunk(batch)
            results.extend(batch_results)

        return results

    def _score_chunk(self, texts: List[str]) -> List[SentimentResult]:
        """Run one batch through the pipeline."""
        try:
            raw = self._pipeline(texts)  # type: ignore[misc]
        except Exception as exc:
            logger.error("Sentiment pipeline error on batch: %s", exc)
            return [_fallback_result(self._model_name) for _ in texts]

        output = []
        for item in raw:
            # pipeline with top_k=1 returns a list-of-list [[{label, score}]]
            if isinstance(item, list):
                item = item[0] if item else {}
            label = (item.get("label") or "neutral").lower()
            score = float(item.get("score") or 0.5)
            # FinBERT labels are "positive", "negative", "neutral" — normalise
            label = _normalise_label(label)
            output.append(SentimentResult(label=label, score=score, model_name=self._model_name))

        return output

    @property
    def model_name(self) -> str:
        return self._model_name


def _normalise_label(label: str) -> str:
    """Map any label variant to one of: positive, negative, neutral."""
    label = label.strip().lower()
    if label in ("positive", "pos"):
        return "positive"
    if label in ("negative", "neg"):
        return "negative"
    return "neutral"


def _fallback_result(model_name: str) -> SentimentResult:
    return SentimentResult(label="neutral", score=0.5, model_name=model_name)
