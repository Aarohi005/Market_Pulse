"""
Topic clustering for financial news articles.

Uses K-Means (via scikit-learn) to group article embeddings into topic
clusters.  Each cluster is labelled using the most frequent non-stopword
tokens from its articles' titles, then mapped to a canonical Topic enum
category based on keyword overlap.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "has", "have", "had", "be", "been", "as", "it", "its",
        # Common news/article stopwords
        "says", "said", "that", "this", "will", "would", "could", "may",
        "new", "after", "over", "about", "up", "down", "out", "more",
    ]
)

# ---------------------------------------------------------------------------
# Topic keyword mapping — maps Topic enum values to keywords that help
# classify clusters into a topic category.
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "EARNINGS": ["earnings", "profit", "revenue", "eps", "quarterly", "results", "beat", "missed"],
    "MERGERS_ACQUISITIONS": ["acquisition", "merger", "takeover", "deal", "acquire", "buyout", "bid"],
    "CENTRAL_BANK": ["fed", "federal", "reserve", "rate", "rates", "interest", "central", "bank", "ecb", "boe"],
    "MACRO_ECONOMY": ["gdp", "inflation", "recession", "economy", "growth", "unemployment", "cpi"],
    "GEOPOLITICS": ["war", "sanctions", "geopolitical", "trade", "tariff", "china", "russia", "conflict"],
    "TECHNOLOGY": ["tech", "ai", "artificial", "intelligence", "software", "cloud", "chip", "semiconductor"],
    "ENERGY": ["oil", "gas", "energy", "opec", "crude", "renewable", "solar", "wind"],
    "CRYPTO": ["crypto", "bitcoin", "ethereum", "blockchain", "defi", "token", "nft"],
    "IPO": ["ipo", "listing", "public", "offering", "spac", "valuation"],
    "REGULATION": ["regulation", "sec", "ftc", "antitrust", "compliance", "law", "ruling"],
    "REAL_ESTATE": ["real", "estate", "housing", "mortgage", "property", "reit"],
    "BANKING": ["bank", "lending", "credit", "deposit", "capital", "loan", "fintech"],
    "DIVIDEND": ["dividend", "yield", "payout", "buyback", "shareholder", "return"],
    "ANALYST": ["analyst", "upgrade", "downgrade", "target", "rating", "recommendation"],
    "OTHER": [],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Represents a single topic cluster.

    Attributes
    ----------
    cluster_id:
        Zero-based integer cluster identifier assigned by K-Means.
    cluster_label:
        Human-readable label derived from the top title words.
    article_ids:
        List of article IDs that belong to this cluster.
    topic_enum:
        Canonical topic category string (one of the keys in
        :data:`_TOPIC_KEYWORDS`).
    dominant_sentiment:
        The most frequent sentiment label among articles in the cluster.
    avg_sentiment_score:
        Mean numeric sentiment score across articles in the cluster.
    size:
        Number of articles in the cluster (convenience alias for
        ``len(article_ids)``).
    """

    cluster_id: int
    cluster_label: str
    article_ids: List[int]
    topic_enum: str
    dominant_sentiment: str
    avg_sentiment_score: float
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.article_ids)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "article_ids": self.article_ids,
            "topic_enum": self.topic_enum,
            "dominant_sentiment": self.dominant_sentiment,
            "avg_sentiment_score": self.avg_sentiment_score,
            "size": self.size,
        }


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class TopicClusterer:
    """Groups articles into topic clusters using K-Means on embeddings.

    Parameters
    ----------
    n_clusters:
        Target number of clusters.  Automatically reduced when there are
        fewer articles than this value.
    min_cluster_size:
        Clusters with fewer articles than this threshold are excluded from
        the returned results.

    Examples
    --------
    >>> clusterer = TopicClusterer(n_clusters=5, min_cluster_size=2)
    >>> results = clusterer.fit(
    ...     article_ids=[0, 1, 2, 3, 4],
    ...     embeddings=[[0.1]*384] * 5,
    ...     titles=["Apple earnings beat", "Fed rate hike", "Apple profit record",
    ...             "Bitcoin crash", "Fed interest rates"],
    ...     sentiments=[0.8, -0.3, 0.7, -0.9, -0.2],
    ... )
    """

    def __init__(
        self,
        n_clusters: int = 15,
        min_cluster_size: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        article_ids: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        sentiments: List[float],
    ) -> List[ClusterResult]:
        """Cluster articles and return one :class:`ClusterResult` per cluster.

        Parameters
        ----------
        article_ids:
            Unique integer identifier for each article.
        embeddings:
            Dense embedding vector for each article.  Length must match
            ``article_ids``.
        titles:
            Article title strings.  Length must match ``article_ids``.
        sentiments:
            Numeric sentiment score in [-1, 1] for each article.  Length
            must match ``article_ids``.

        Returns
        -------
        list of ClusterResult
            Clusters with at least ``min_cluster_size`` members, sorted by
            ``cluster_id``.
        """
        n_articles = len(article_ids)
        if n_articles == 0:
            logger.warning("TopicClusterer.fit called with no articles.")
            return []

        # Guard: k must be ≤ n_articles
        k = min(self.n_clusters, n_articles)
        if k < self.n_clusters:
            logger.info(
                "Reducing n_clusters from %d to %d to match article count.",
                self.n_clusters, k,
            )

        # Run K-Means
        labels = self._run_kmeans(embeddings, k)

        # Build per-cluster data structures
        cluster_data: Dict[int, Dict] = {}
        for idx, (art_id, label) in enumerate(zip(article_ids, labels)):
            cid = int(label)
            if cid not in cluster_data:
                cluster_data[cid] = {
                    "article_ids": [],
                    "titles": [],
                    "sentiments": [],
                }
            cluster_data[cid]["article_ids"].append(art_id)
            cluster_data[cid]["titles"].append(titles[idx])
            cluster_data[cid]["sentiments"].append(sentiments[idx])

        # Convert to ClusterResult objects
        results: List[ClusterResult] = []
        for cid, data in cluster_data.items():
            if len(data["article_ids"]) < self.min_cluster_size:
                continue

            cluster_label = self._generate_label(data["titles"])
            topic_enum = self._map_to_topic(cluster_label, data["titles"])
            dominant_sentiment = self._dominant_sentiment(data["sentiments"])
            avg_sentiment_score = float(np.mean(data["sentiments"]))

            results.append(
                ClusterResult(
                    cluster_id=cid,
                    cluster_label=cluster_label,
                    article_ids=data["article_ids"],
                    topic_enum=topic_enum,
                    dominant_sentiment=dominant_sentiment,
                    avg_sentiment_score=round(avg_sentiment_score, 4),
                )
            )

        results.sort(key=lambda r: r.cluster_id)
        logger.info(
            "TopicClusterer produced %d clusters (min_size=%d) from %d articles.",
            len(results), self.min_cluster_size, n_articles,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_kmeans(
        self,
        embeddings: List[List[float]],
        k: int,
    ) -> np.ndarray:
        """Run K-Means and return an array of cluster label integers.

        Parameters
        ----------
        embeddings:
            List of embedding vectors.
        k:
            Number of clusters.

        Returns
        -------
        np.ndarray
            Integer cluster assignment for each embedding.
        """
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for TopicClusterer. "
                "Install it with: pip install scikit-learn"
            ) from exc

        X = np.array(embeddings, dtype=np.float32)
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(X)
        return kmeans.labels_

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise *text* into lowercase alphabetic tokens."""
        return [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]

    @classmethod
    def _generate_label(cls, titles: List[str]) -> str:
        """Generate a human-readable cluster label from article titles.

        Takes the top 3 most common non-stopword tokens across all titles
        and joins them with a space.

        Parameters
        ----------
        titles:
            Article title strings for articles in the cluster.

        Returns
        -------
        str
            A short label string such as ``"apple earnings record"``.
        """
        counter: Counter = Counter()
        for title in titles:
            tokens = cls._tokenize(title)
            for token in tokens:
                if token not in _STOPWORDS and len(token) > 2:
                    counter[token] += 1

        top_tokens = [word for word, _ in counter.most_common(3)]
        if not top_tokens:
            return "misc"
        return " ".join(top_tokens)

    @staticmethod
    def _map_to_topic(label: str, titles: List[str]) -> str:
        """Map a cluster label to the closest Topic enum string.

        Counts keyword matches for each topic category across the combined
        label and titles text, returning the category with the highest count.
        Falls back to ``"OTHER"`` when no matches are found.

        Parameters
        ----------
        label:
            Cluster label string generated by :meth:`_generate_label`.
        titles:
            List of title strings for context.

        Returns
        -------
        str
            One of the topic keys from :data:`_TOPIC_KEYWORDS`.
        """
        all_text = (label + " " + " ".join(titles)).lower()
        scores: Dict[str, int] = {}
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if topic == "OTHER":
                continue
            count = sum(1 for kw in keywords if kw in all_text)
            if count > 0:
                scores[topic] = count

        if not scores:
            return "OTHER"
        return max(scores, key=lambda t: scores[t])

    @staticmethod
    def _dominant_sentiment(sentiments: List[float]) -> str:
        """Determine the dominant sentiment label from numeric scores.

        Parameters
        ----------
        sentiments:
            List of numeric sentiment scores in [-1, 1].

        Returns
        -------
        str
            ``"positive"``, ``"negative"``, or ``"neutral"`` based on the
            mean score with thresholds ±0.1.
        """
        if not sentiments:
            return "neutral"
        mean = float(np.mean(sentiments))
        if mean > 0.1:
            return "positive"
        if mean < -0.1:
            return "negative"
        return "neutral"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TopicClusterer(n_clusters={self.n_clusters}, "
            f"min_cluster_size={self.min_cluster_size})"
        )
