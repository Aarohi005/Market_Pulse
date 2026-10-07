"""
embedder.py — text embedding wrapper using sentence-transformers.

This module provides TextEmbedder, a thin wrapper around a
sentence-transformers model with:
- Lazy model loading (the model is not loaded until the first call)
- Batch embedding with configurable batch size
- Single-string embedding convenience method
- Normalised embeddings (unit L2 norm) suitable for cosine similarity

Design notes
------------
sentence-transformers is used instead of the OpenAI embeddings API so
the system works fully offline with no API key required.  The default
model (all-MiniLM-L6-v2) is small (≈80 MB), fast on CPU, and produces
384-dimensional embeddings that balance quality and speed for a financial
news retrieval task.

You can swap the model by passing a different ``model_name`` to the
constructor; any model on the HuggingFace Hub compatible with
sentence-transformers will work.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model – small, fast, good quality for semantic similarity
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


class TextEmbedder:
    """Lazy-loading wrapper around a sentence-transformers embedding model.

    The underlying model is not loaded until the first call to
    :meth:`embed_batch` or :meth:`embed_single`.  This keeps import time
    fast and avoids loading the model if the embedder is never used.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path accepted by
        ``sentence_transformers.SentenceTransformer``.
        Defaults to ``"all-MiniLM-L6-v2"``.
    batch_size:
        Number of texts to encode in a single forward pass.
        Larger values use more memory but are faster on GPU.
    normalize_embeddings:
        If *True* (default) embeddings are L2-normalised so that dot
        product equals cosine similarity.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        If *None*, sentence-transformers auto-detects the best device.

    Examples
    --------
    >>> embedder = TextEmbedder()
    >>> vec = embedder.embed_single("Apple earnings beat expectations")
    >>> vec.shape
    (384,)

    >>> vecs = embedder.embed_batch(["Apple beat EPS", "Fed hikes rates"])
    >>> vecs.shape
    (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._device = device
        self._model = None  # lazily loaded

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (loads model if necessary)."""
        return self._get_model().get_sentence_embedding_dimension()

    @property
    def is_loaded(self) -> bool:
        """Return whether the underlying model has been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Parameters
        ----------
        texts:
            List of strings to embed.  Empty strings are replaced with a
            single space to avoid model errors.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(len(texts), embedding_dim)``.
            Rows are in the same order as the input list.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        RuntimeError
            If the model fails to load or encode.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        # Replace empty strings so the model does not error
        cleaned = [t if t.strip() else " " for t in texts]

        model = self._get_model()
        logger.debug("Embedding %d texts with model %s", len(cleaned), self._model_name)

        embeddings: np.ndarray = model.encode(
            cleaned,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string.

        Parameters
        ----------
        text:
            String to embed.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(embedding_dim,)``.
        """
        result = self.embed_batch([text])
        return result[0]

    def embed_articles(
        self,
        titles: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Embed articles by concatenating title and description.

        Parameters
        ----------
        titles:
            List of article headlines.
        descriptions:
            Optional list of article body excerpts, aligned with *titles*.
            If provided, each entry is concatenated to the corresponding
            title with a ``" | "`` separator before embedding.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(len(titles), embedding_dim)``.
        """
        if descriptions is not None:
            if len(descriptions) != len(titles):
                raise ValueError(
                    f"titles ({len(titles)}) and descriptions ({len(descriptions)}) "
                    "must have the same length"
                )
            texts = [
                f"{t} | {d}".strip() if d else t
                for t, d in zip(titles, descriptions)
            ]
        else:
            texts = titles

        return self.embed_batch(texts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the loaded model, loading it on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the sentence-transformers model.

        Raises
        ------
        ImportError
            If sentence-transformers is not installed.
        RuntimeError
            If the model cannot be loaded from the given name/path.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for TextEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", self._model_name)
        kwargs = {}
        if self._device is not None:
            kwargs["device"] = self._device

        try:
            model = SentenceTransformer(self._model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{self._model_name}': {exc}"
            ) from exc

        logger.info(
            "Model loaded | name=%s | dim=%d",
            self._model_name,
            model.get_sentence_embedding_dimension(),
        )
        return model
