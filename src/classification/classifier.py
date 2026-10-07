"""
classifier.py — rule-based article classification for financial news.

This module provides:
- ArticleClassification: a dataclass capturing all classification outputs
- RuleBasedClassifier: a deterministic keyword/pattern classifier
- classify_many: convenience function for batch classification

Design philosophy
-----------------
Rule-based classification is fast, transparent, and requires no model
inference budget.  It is used as a first-pass filter; the LLM ensemble
layer can override any field when confidence warrants it.

All rules are keyword sets defined as module-level constants so they can
be tuned without touching logic.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .enums import (
    AssetClass,
    Importance,
    Region,
    SourceType,
    Topic,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword rule tables
# ---------------------------------------------------------------------------

# --- Source URL / name → SourceType ---
_SOURCE_PATTERNS: List[tuple[str, SourceType]] = [
    (r"reuters\.com", SourceType.REUTERS),
    (r"cnbc\.com", SourceType.CNBC),
    (r"finance\.yahoo\.com|yahoo.*finance", SourceType.YAHOO),
    (r"marketwatch\.com", SourceType.MARKETWATCH),
    (r"bloomberg\.com", SourceType.BLOOMBERG),
]

# --- Keywords → AssetClass ---
_ASSET_CLASS_KEYWORDS: Dict[AssetClass, List[str]] = {
    AssetClass.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "defi", "nft", "blockchain", "coinbase", "binance", "solana",
        "ripple", "xrp", "dogecoin", "altcoin", "stablecoin",
    ],
    AssetClass.FOREX: [
        "forex", "currency", "exchange rate", "dollar index", "dxy",
        "eur/usd", "gbp/usd", "usd/jpy", "yuan", "renminbi", "yen",
        "euro", "sterling", "pound", "fx market", "central bank rate",
    ],
    AssetClass.COMMODITIES: [
        "oil", "crude", "brent", "wti", "natural gas", "gold", "silver",
        "copper", "wheat", "corn", "soybean", "commodity", "commodities",
        "futures", "opec", "energy price", "metals",
    ],
    AssetClass.FIXED_INCOME: [
        "treasury", "bond", "yield", "10-year", "2-year", "t-bill",
        "note yield", "corporate bond", "high yield", "junk bond",
        "investment grade", "duration", "coupon", "fixed income",
        "debt market", "credit market",
    ],
    AssetClass.MACRO: [
        "gdp", "inflation", "cpi", "ppi", "fed", "federal reserve",
        "interest rate", "rate hike", "rate cut", "recession", "unemployment",
        "jobs report", "nonfarm payroll", "fiscal policy", "monetary policy",
        "central bank", "ecb", "boj", "bank of england", "imf", "world bank",
    ],
    AssetClass.EQUITY: [
        "stock", "shares", "equity", "ipo", "earnings", "eps", "revenue",
        "nasdaq", "s&p", "s&p 500", "dow jones", "russell", "nyse",
        "market cap", "dividend", "buyback", "analyst", "upgrade", "downgrade",
    ],
}

# --- Keywords → Region ---
_REGION_KEYWORDS: Dict[Region, List[str]] = {
    Region.US: [
        "u.s.", "us ", "united states", "american", "wall street", "fed",
        "federal reserve", "treasury", "nasdaq", "nyse", "s&p", "dow",
        "washington", "congress", "biden", "trump",
    ],
    Region.EU: [
        "europe", "european", "eu ", "eurozone", "ecb", "euro",
        "germany", "france", "italy", "spain", "uk", "britain",
        "london", "frankfurt", "paris", "brexit",
    ],
    Region.APAC: [
        "china", "chinese", "japan", "japanese", "asia", "asian",
        "hong kong", "singapore", "south korea", "india", "australia",
        "boj", "pboc", "renminbi", "nikkei", "hang seng", "sensex",
    ],
}

# --- Keywords → Topic ---
_TOPIC_KEYWORDS: Dict[Topic, List[str]] = {
    Topic.EARNINGS: [
        "earnings", "eps", "quarterly results", "revenue beat", "revenue miss",
        "guidance", "profit", "net income", "operating income", "q1", "q2",
        "q3", "q4", "fiscal year", "annual results", "beats estimates",
        "misses estimates",
    ],
    Topic.FED_POLICY: [
        "federal reserve", "fed ", "fomc", "jerome powell", "rate hike",
        "rate cut", "interest rate", "basis points", "bps", "tapering",
        "quantitative easing", "qe", "balance sheet", "dot plot",
    ],
    Topic.INFLATION: [
        "inflation", "cpi", "ppi", "consumer price", "producer price",
        "core inflation", "price pressure", "cost of living", "disinflation",
        "deflation", "stagflation", "price index",
    ],
    Topic.RECESSION: [
        "recession", "contraction", "gdp decline", "economic slowdown",
        "bear market", "downturn", "hard landing", "soft landing",
        "negative growth",
    ],
    Topic.AI: [
        "artificial intelligence", "ai ", "machine learning", "large language",
        "llm", "gpt", "generative ai", "openai", "anthropic", "deepmind",
        "nvidia ai", "chatgpt", "copilot",
    ],
    Topic.TECH: [
        "technology", "tech sector", "semiconductor", "chip", "software",
        "cloud", "saas", "apple", "microsoft", "google", "alphabet",
        "meta", "amazon", "nvidia", "tsmc",
    ],
    Topic.BANKING: [
        "bank", "banking", "jpmorgan", "goldman sachs", "morgan stanley",
        "citigroup", "bank of america", "wells fargo", "credit suisse",
        "deutsche bank", "svb", "silicon valley bank", "deposit",
        "lending", "loan", "credit",
    ],
    Topic.ENERGY: [
        "energy", "oil", "gas", "opec", "renewable", "solar", "wind",
        "exxon", "chevron", "bp", "shell", "pipeline", "refinery",
        "petroleum", "lng",
    ],
    Topic.GEOPOLITICS: [
        "war", "conflict", "sanction", "tariff", "trade war", "geopolitic",
        "nato", "ukraine", "russia", "china taiwan", "middle east",
        "iran", "north korea", "diplomatic",
    ],
    Topic.CRYPTO: [
        "bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft",
        "coinbase", "binance", "sec crypto", "crypto regulation",
    ],
    Topic.HOUSING: [
        "housing", "real estate", "mortgage", "home price", "housing market",
        "reit", "commercial real estate", "cre", "home sales",
        "existing home", "new home",
    ],
    Topic.LABOR: [
        "jobs", "employment", "unemployment", "nonfarm payroll",
        "labor market", "wage", "hiring", "layoff", "workforce",
        "job openings", "jolts", "ism",
    ],
    Topic.TRADE: [
        "trade deficit", "trade surplus", "export", "import",
        "balance of trade", "tariff", "wto", "supply chain",
        "trade deal", "trade agreement",
    ],
}

# --- HIGH importance signals ---
_HIGH_IMPORTANCE_PATTERNS: List[str] = [
    r"\bfed\b.*decision",
    r"rate (hike|cut|decision)",
    r"earnings (beat|miss|report)",
    r"breaking[:\s]",
    r"market (crash|rally|surge|plunge)",
    r"record (high|low)",
    r"\bipo\b",
    r"(bankruptcy|default|collapse)",
    r"(gdp|cpi|ppi)\s+(report|data|print)",
    r"(merger|acquisition|takeover)",
    r"(sec|doj|cftc)\s+(charges|investigation|settlement)",
]

# --- LOW importance signals (editorial / opinion / listicles) ---
_LOW_IMPORTANCE_PATTERNS: List[str] = [
    r"(top|best|worst)\s+\d+\s+stocks",
    r"stocks to (watch|buy|sell)",
    r"opinion[:\s]",
    r"commentary[:\s]",
    r"(explainer|guide|how-to|tutorial)",
    r"weekly (roundup|digest|recap)",
]

# --- Common S&P 500 / major tickers for mention extraction ---
_TICKER_PATTERN = re.compile(
    r"\b(AAPL|MSFT|GOOGL|GOOG|AMZN|NVDA|META|TSLA|BRK\.?[AB]|JPM|"
    r"V|UNH|XOM|JNJ|WMT|MA|PG|CVX|HD|MRK|ABBV|PEP|COST|AVGO|"
    r"BAC|MCD|KO|ADBE|CRM|TMO|ACN|LLY|DHR|TXN|NEE|PM|QCOM|"
    r"AMGN|HON|IBM|SBUX|GS|MS|C|WFC|BLK|SPGI|AXP|RTX|CAT|"
    r"DE|MMM|BA|LMT|GE|F|GM|INTC|AMD|ORCL|NFLX|DIS|PYPL|"
    r"BTC|ETH|SOL|XRP|DOGE)\b"
)


# ---------------------------------------------------------------------------
# ArticleClassification dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArticleClassification:
    """All classification outputs for a single article.

    Attributes
    ----------
    source_type:
        Enumerated source of the article (e.g. REUTERS, CNBC).
    asset_class:
        Primary asset class the article pertains to.
    region:
        Geographic focus of the article.
    importance:
        Estimated editorial importance (HIGH / MEDIUM / LOW).
    topic:
        Primary financial topic detected.
    ticker_mentions:
        List of recognised equity/crypto ticker symbols found in the text.
    classified_by:
        Identifier of the classification strategy used (e.g. "rule_based_v1").
    raw_scores:
        Optional dict of intermediate scoring data for debugging / audit.
    """

    source_type: SourceType = SourceType.UNKNOWN
    asset_class: AssetClass = AssetClass.UNKNOWN
    region: Region = Region.UNKNOWN
    importance: Importance = Importance.MEDIUM
    topic: Topic = Topic.OTHER
    ticker_mentions: List[str] = field(default_factory=list)
    classified_by: str = "rule_based_v1"
    raw_scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all classification fields."""
        return {
            "source_type": self.source_type.value,
            "asset_class": self.asset_class.value,
            "region": self.region.value,
            "importance": self.importance.value,
            "topic": self.topic.value,
            "ticker_mentions": self.ticker_mentions,
            "classified_by": self.classified_by,
        }


# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """Deterministic rule-based classifier for financial news articles.

    All classification decisions are made via keyword matching and regular
    expression patterns defined as module-level constants.  No external
    model calls are required; classification is O(n * k) where n is the
    number of articles and k is the total number of keyword rules.

    Usage
    -----
    >>> clf = RuleBasedClassifier()
    >>> result = clf.classify(
    ...     title="Fed raises rates by 25bps, Powell signals pause",
    ...     description="The Federal Reserve raised interest rates...",
    ...     source_name="Reuters",
    ...     url="https://reuters.com/markets/fed-raises-rates",
    ... )
    >>> result.topic
    <Topic.FED_POLICY: 'FED_POLICY'>
    """

    def __init__(self) -> None:
        # Pre-compile importance patterns for speed
        self._high_patterns = [
            re.compile(p, re.IGNORECASE) for p in _HIGH_IMPORTANCE_PATTERNS
        ]
        self._low_patterns = [
            re.compile(p, re.IGNORECASE) for p in _LOW_IMPORTANCE_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        title: str,
        description: str,
        source_name: str = "",
        url: str = "",
    ) -> ArticleClassification:
        """Classify a single article.

        Parameters
        ----------
        title:
            Article headline.
        description:
            Article body excerpt or summary (may be empty string).
        source_name:
            Display name of the news source (e.g. "Reuters").
        url:
            Full URL of the article.

        Returns
        -------
        ArticleClassification
            Populated classification dataclass.
        """
        # Combine title + description into one lowercase search corpus
        corpus = f"{title} {description}".lower()
        corpus_raw = f"{title} {description}"  # keep case for ticker regex

        result = ArticleClassification()

        result.source_type = self._classify_source(source_name, url)
        result.asset_class, asset_scores = self._classify_asset_class(corpus)
        result.region, region_scores = self._classify_region(corpus)
        result.importance = self._classify_importance(corpus)
        result.topic, topic_scores = self._classify_topic(corpus)
        result.ticker_mentions = self._extract_tickers(corpus_raw)

        result.raw_scores = {
            "asset_scores": asset_scores,
            "region_scores": region_scores,
            "topic_scores": topic_scores,
        }

        logger.debug(
            "Classified article | title=%r | topic=%s | asset=%s | region=%s",
            title[:80],
            result.topic.value,
            result.asset_class.value,
            result.region.value,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_source(self, source_name: str, url: str) -> SourceType:
        """Infer SourceType from the source name and/or URL."""
        combined = f"{source_name} {url}".lower()
        for pattern, source_type in _SOURCE_PATTERNS:
            if re.search(pattern, combined):
                return source_type
        # Fallback: check if url contains "rss" or "feed"
        if re.search(r"(rss|feed)", combined):
            return SourceType.RSS
        return SourceType.UNKNOWN

    def _classify_asset_class(
        self, corpus: str
    ) -> tuple[AssetClass, Dict[str, int]]:
        """Count keyword hits per asset class and return the winner."""
        scores: Dict[str, int] = {}
        for asset_class, keywords in _ASSET_CLASS_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in corpus)
            if count > 0:
                scores[asset_class.value] = count

        if not scores:
            return AssetClass.UNKNOWN, scores

        best = max(scores, key=lambda k: scores[k])
        return AssetClass(best), scores

    def _classify_region(
        self, corpus: str
    ) -> tuple[Region, Dict[str, int]]:
        """Count keyword hits per region and return the winner."""
        scores: Dict[str, int] = {}
        for region, keywords in _REGION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in corpus)
            if count > 0:
                scores[region.value] = count

        if not scores:
            return Region.UNKNOWN, scores

        # If multiple regions tie or are close, classify as GLOBAL
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] <= 1:
            return Region.GLOBAL, scores

        best = max(scores, key=lambda k: scores[k])
        return Region(best), scores

    def _classify_importance(self, corpus: str) -> Importance:
        """Apply high / low importance heuristics; default to MEDIUM."""
        for pattern in self._high_patterns:
            if pattern.search(corpus):
                return Importance.HIGH
        for pattern in self._low_patterns:
            if pattern.search(corpus):
                return Importance.LOW
        return Importance.MEDIUM

    def _classify_topic(
        self, corpus: str
    ) -> tuple[Topic, Dict[str, int]]:
        """Count keyword hits per topic and return the highest-scoring one."""
        scores: Dict[str, int] = {}
        for topic, keywords in _TOPIC_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in corpus)
            if count > 0:
                scores[topic.value] = count

        if not scores:
            return Topic.OTHER, scores

        best = max(scores, key=lambda k: scores[k])
        return Topic(best), scores

    def _extract_tickers(self, corpus_raw: str) -> List[str]:
        """Extract recognised ticker symbols from the raw (case-sensitive) corpus."""
        found = _TICKER_PATTERN.findall(corpus_raw)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for ticker in found:
            t = ticker.replace(".", "")  # normalise BRK.A → BRKA
            if t not in seen:
                seen.add(t)
                unique.append(ticker)
        return unique


# ---------------------------------------------------------------------------
# Convenience batch function
# ---------------------------------------------------------------------------

def classify_many(
    articles: Sequence[Dict[str, str]],
    classifier: Optional[RuleBasedClassifier] = None,
) -> List[ArticleClassification]:
    """Classify a batch of article dicts.

    Parameters
    ----------
    articles:
        Sequence of dicts, each with keys:
        ``title``, ``description``, ``source_name`` (optional),
        ``url`` (optional).
    classifier:
        An existing RuleBasedClassifier instance to reuse.  If *None*,
        a new instance is created.

    Returns
    -------
    List[ArticleClassification]
        One classification per input article, in the same order.

    Example
    -------
    >>> results = classify_many([
    ...     {"title": "Fed hikes rates", "description": "The FOMC voted..."},
    ...     {"title": "Bitcoin hits $100k", "description": "Crypto markets..."},
    ... ])
    >>> [r.topic.value for r in results]
    ['FED_POLICY', 'CRYPTO']
    """
    clf = classifier or RuleBasedClassifier()
    classifications: List[ArticleClassification] = []

    for article in articles:
        try:
            result = clf.classify(
                title=article.get("title", ""),
                description=article.get("description", ""),
                source_name=article.get("source_name", ""),
                url=article.get("url", ""),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Classification failed for article %r: %s", article, exc)
            result = ArticleClassification()  # safe default
        classifications.append(result)

    return classifications
