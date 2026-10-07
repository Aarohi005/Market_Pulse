"""
vector_store.py — ChromaDB-backed vector store for financial news articles.

This module provides:
- SearchResult: a dataclass representing one search hit
- VectorStore: a ChromaDB wrapper with add, search, and management methods

Design notes
------------
ChromaDB is used as the vector store because it:
- Runs entirely in-process (no separate server required)
- Persists data to disk automatically
- Supports metadata filtering alongside vector similarity
- Has a Python-native API with no external dependencies beyond chromadb

The VectorStore wraps a single ChromaDB collection.  Each article is
stored as a document with its embedding and a metadata dict.  Metadata
fields are kept flat (no nested dicts) to stay within ChromaDB's
supported types (str, int, float, bool).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ChromaDB distance metric — cosine is appropriate for normalised embeddings
_DISTANCE_METRIC = "cosine"


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Attributes
    ----------
    article_id:
        The unique identifier of the article (as stored in the DB).
    score:
        Similarity score in [0, 1].  Higher is more similar.
        Derived from ChromaDB's distance: score = 1 - distance.
    text:
        The document text that was embedded (title + description).
    metadata:
        Flat dict of metadata stored alongside the embedding (e.g.
        source_type, asset_class, published_at).
    """

    article_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SearchResult(article_id={self.article_id!r}, "
            f"score={self.score:.4f}, "
            f"text={self.text[:60]!r})"
        )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """ChromaDB-backed vector store for financial news article embeddings.

    Each instance wraps a single ChromaDB collection.  The collection is
    created automatically if it does not already exist.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection.  Defaults to
        ``"financial_news"``.
    persist_directory:
        Path to the directory where ChromaDB stores its data on disk.
        If *None*, an in-memory (ephemeral) client is used — useful for
        testing.
    embedding_dim:
        Dimensionality of the embedding vectors.  Used only for
        validation; ChromaDB infers this from the first batch of vectors.

    Examples
    --------
    >>> store = VectorStore(persist_directory="/tmp/chroma")
    >>> store.add_articles(
    ...     ids=["art_001"],
    ...     embeddings=np.zeros((1, 384), dtype=np.float32),
    ...     texts=["Fed raises rates"],
    ...     metadatas=[{"source": "reuters", "published_at": "2024-01-01"}],
    ... )
    >>> results = store.search(query_embedding=np.zeros(384), top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "financial_news",
        persist_directory: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_dim = embedding_dim
        self._client = None
        self._collection = None
        self._init_client()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._collection_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_articles(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add articles (with their embeddings) to the collection.

        Parameters
        ----------
        ids:
            Unique string identifiers for each article.  Must be the same
            length as *embeddings* and *texts*.
        embeddings:
            Float32 array of shape ``(n, embedding_dim)``.
        texts:
            List of document strings (used for keyword search fallback
            and result display).  Same length as *ids*.
        metadatas:
            Optional list of flat metadata dicts, one per article.
            Values must be ``str``, ``int``, ``float``, or ``bool``.
            If *None*, empty dicts are used.

        Raises
        ------
        ValueError
            If lengths of ids, embeddings, and texts do not match.
        """
        n = len(ids)
        if embeddings.shape[0] != n or len(texts) != n:
            raise ValueError(
                f"ids ({n}), embeddings ({embeddings.shape[0]}), and "
                f"texts ({len(texts)}) must all have the same length"
            )

        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        elif len(metadatas) != n:
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must equal ids length ({n})"
            )

        # ChromaDB expects lists of lists, not numpy arrays
        embeddings_list = embeddings.tolist()

        # Sanitise metadata: ChromaDB rejects None values
        clean_metadatas = [
            {k: v for k, v in m.items() if v is not None}
            for m in metadatas
        ]
        logger.info("STEP C1: entering add_articles")
        logger.info("Embeddings shape: %s", embeddings.shape)
        logger.info("Number of ids: %d", len(ids))

        collection = self._get_collection()
        logger.info("STEP C2: collection ready")
        logger.info("STEP C3: calling collection.upsert()")
        collection.upsert(
            ids=ids[:1],
            embeddings=embeddings_list[:1],
            documents=texts[:1],
            metadatas=clean_metadatas[:1],
        )
        logger.debug("Upserted %d articles into collection '%s'", n, self._collection_name)
        logger.info("STEP C4: upsert completed")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for the most similar articles.

        Parameters
        ----------
        query_embedding:
            1-D float32 array of shape ``(embedding_dim,)``.
        top_k:
            Maximum number of results to return.
        where:
            ChromaDB metadata filter dict (e.g.
            ``{"source_type": {"$eq": "REUTERS"}}``).
            See ChromaDB docs for filter syntax.
        where_document:
            ChromaDB document content filter (e.g.
            ``{"$contains": "inflation"}``).

        Returns
        -------
        List[SearchResult]
            Up to *top_k* results ordered by descending similarity score.
        """
        collection = self._get_collection()

        # Guard: can't query an empty collection
        if collection.count() == 0:
            logger.debug("Search called on empty collection '%s'", self._collection_name)
            return []

        query_list = [query_embedding.tolist()]
        kwargs: Dict[str, Any] = {
            "query_embeddings": query_list,
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        raw = collection.query(**kwargs)

        results: List[SearchResult] = []
        ids_batch = raw.get("ids", [[]])[0]
        docs_batch = raw.get("documents", [[]])[0]
        metas_batch = raw.get("metadatas", [[]])[0]
        dists_batch = raw.get("distances", [[]])[0]

        for article_id, doc, meta, dist in zip(
            ids_batch, docs_batch, metas_batch, dists_batch
        ):
            # Convert cosine distance → similarity score
            score = float(max(0.0, 1.0 - dist))
            results.append(
                SearchResult(
                    article_id=article_id,
                    score=score,
                    text=doc or "",
                    metadata=meta or {},
                )
            )

        return results

    def get_collection_size(self) -> int:
        """Return the number of documents in the collection.

        Returns
        -------
        int
            Count of stored documents.
        """
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """Permanently delete the entire collection from ChromaDB.

        This is irreversible.  Use with caution.
        """
        client = self._get_client()
        client.delete_collection(self._collection_name)
        self._collection = None
        logger.warning("Deleted collection '%s'", self._collection_name)

    def delete_articles(self, ids: List[str]) -> None:
        """Delete specific articles from the collection by ID.

        Parameters
        ----------
        ids:
            List of article IDs to remove.
        """
        self._get_collection().delete(ids=ids)
        logger.debug("Deleted %d articles from collection '%s'", len(ids), self._collection_name)

    def get_article(self, article_id: str) -> Optional[SearchResult]:
        """Fetch a single article by its ID.

        Parameters
        ----------
        article_id:
            The unique identifier of the article.

        Returns
        -------
        SearchResult or None
            The article if found, otherwise *None*.
        """
        collection = self._get_collection()
        raw = collection.get(
            ids=[article_id],
            include=["documents", "metadatas"],
        )
        ids_out = raw.get("ids", [])
        if not ids_out:
            return None
        return SearchResult(
            article_id=ids_out[0],
            score=1.0,
            text=(raw.get("documents") or [""])[0],
            metadata=(raw.get("metadatas") or [{}])[0],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialise the ChromaDB client and collection."""
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for VectorStore. "
                "Install it with: pip install chromadb"
            ) from exc

        if self._persist_directory:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
            logger.info(
                "ChromaDB persistent client initialised at '%s'",
                self._persist_directory,
            )
        else:
            self._client = chromadb.EphemeralClient()
            logger.info("ChromaDB ephemeral (in-memory) client initialised")

    def _get_client(self):
        """Return the ChromaDB client, raising if not initialised."""
        if self._client is None:
            raise RuntimeError("ChromaDB client is not initialised")
        return self._client

    def _get_collection(self):
        """Return the ChromaDB collection, creating it if it does not exist."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": _DISTANCE_METRIC},
            )
            logger.debug(
                "Using collection '%s' (size=%d)",
                self._collection_name,
                self._collection.count(),
            )
        return self._collection
