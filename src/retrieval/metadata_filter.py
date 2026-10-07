"""
metadata_filter.py — structured metadata filtering for article queries.

This module provides:
- MetadataFilter: a dataclass capturing all supported filter dimensions
- apply_metadata_filter: builds a ChromaDB ``where`` dict from a filter
- build_sql_where_clause: builds a parameterised SQL WHERE clause for
  SQLite queries (used by the storage layer)

Design notes
------------
The same MetadataFilter dataclass is used for both vector store queries
(ChromaDB) and relational queries (SQLite).  Two separate builder
functions translate the filter into each backend's native format so
calling code never needs to know which backend is in use.

ChromaDB filter syntax uses ``$eq``, ``$in``, ``$gte``, ``$lte`` etc.
inside a ``where`` dict.  When multiple conditions are present they are
combined with ``$and``.

SQLite uses parameterised ``?`` placeholders to prevent SQL injection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetadataFilter dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetadataFilter:
    """Declarative filter specification for article metadata queries.

    All fields are optional.  Only non-None fields are included in the
    generated filter expressions.

    Attributes
    ----------
    source_types:
        Restrict to articles from these source types (e.g. ``["REUTERS", "CNBC"]``).
    asset_classes:
        Restrict to articles covering these asset classes
        (e.g. ``["EQUITY", "MACRO"]``).
    regions:
        Restrict to articles focused on these regions
        (e.g. ``["US", "EU"]``).
    importance_levels:
        Restrict to articles of these importance levels
        (e.g. ``["HIGH", "MEDIUM"]``).
    topics:
        Restrict to articles tagged with these topics
        (e.g. ``["EARNINGS", "FED_POLICY"]``).
    sentiment_categories:
        Restrict to articles with these sentiment labels
        (e.g. ``["POSITIVE", "NEGATIVE"]``).
    ticker_mentions:
        Restrict to articles mentioning at least one of these tickers.
        Note: ChromaDB metadata stores tickers as a comma-separated string;
        this filter uses a document content search as a fallback.
    published_after:
        Only include articles published at or after this datetime.
    published_before:
        Only include articles published before or at this datetime.
    min_sentiment_score:
        Only include articles with a sentiment score >= this value.
    max_sentiment_score:
        Only include articles with a sentiment score <= this value.
    min_confidence:
        Only include articles with a confidence score >= this value.
    keyword:
        Full-text keyword that must appear in the article text
        (used for ChromaDB ``where_document`` filter).
    """

    source_types: Optional[List[str]] = None
    asset_classes: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    importance_levels: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sentiment_categories: Optional[List[str]] = None
    ticker_mentions: Optional[List[str]] = None
    published_after: Optional[datetime] = None
    published_before: Optional[datetime] = None
    min_sentiment_score: Optional[float] = None
    max_sentiment_score: Optional[float] = None
    min_confidence: Optional[float] = None
    keyword: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        return all(
            v is None
            for v in [
                self.source_types, self.asset_classes, self.regions,
                self.importance_levels, self.topics, self.sentiment_categories,
                self.ticker_mentions, self.published_after, self.published_before,
                self.min_sentiment_score, self.max_sentiment_score,
                self.min_confidence, self.keyword,
            ]
        )


# ---------------------------------------------------------------------------
# ChromaDB filter builder
# ---------------------------------------------------------------------------

def apply_metadata_filter(
    metadata_filter: MetadataFilter,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convert a MetadataFilter into ChromaDB query dicts.

    Parameters
    ----------
    metadata_filter:
        The filter specification to convert.

    Returns
    -------
    (where, where_document)
        A tuple of:
        - ``where``: ChromaDB metadata filter dict, or *None* if no
          metadata conditions are active.
        - ``where_document``: ChromaDB document content filter dict, or
          *None* if no keyword is set.

    Notes
    -----
    ChromaDB requires that a ``$and`` clause contain at least two
    conditions.  This function collapses single-condition lists into a
    bare condition dict automatically.

    Examples
    --------
    >>> f = MetadataFilter(source_types=["REUTERS"], regions=["US"])
    >>> where, where_doc = apply_metadata_filter(f)
    >>> where
    {'$and': [{'source_type': {'$eq': 'REUTERS'}}, {'region': {'$eq': 'US'}}]}
    >>> where_doc  # None — no keyword set
    """
    if metadata_filter.is_empty():
        return None, None

    conditions: List[Dict[str, Any]] = []

    # --- Enum list fields → $eq (single value) or $in (multiple values) ---
    _add_list_condition(conditions, "source_type", metadata_filter.source_types)
    _add_list_condition(conditions, "asset_class", metadata_filter.asset_classes)
    _add_list_condition(conditions, "region", metadata_filter.regions)
    _add_list_condition(conditions, "importance", metadata_filter.importance_levels)
    _add_list_condition(conditions, "topic", metadata_filter.topics)
    _add_list_condition(conditions, "sentiment_category", metadata_filter.sentiment_categories)

    # --- Date range (stored as ISO 8601 string in metadata) ---
    if metadata_filter.published_after is not None:
        ts = metadata_filter.published_after.isoformat()
        conditions.append({"published_at": {"$gte": ts}})

    if metadata_filter.published_before is not None:
        ts = metadata_filter.published_before.isoformat()
        conditions.append({"published_at": {"$lte": ts}})

    # --- Numeric range: sentiment score ---
    if metadata_filter.min_sentiment_score is not None:
        conditions.append(
            {"sentiment_score": {"$gte": float(metadata_filter.min_sentiment_score)}}
        )
    if metadata_filter.max_sentiment_score is not None:
        conditions.append(
            {"sentiment_score": {"$lte": float(metadata_filter.max_sentiment_score)}}
        )

    # --- Numeric range: confidence ---
    if metadata_filter.min_confidence is not None:
        conditions.append(
            {"confidence": {"$gte": float(metadata_filter.min_confidence)}}
        )

    # --- Build the final where dict ---
    where: Optional[Dict[str, Any]] = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # --- Document-level keyword filter ---
    where_document: Optional[Dict[str, Any]] = None
    if metadata_filter.keyword:
        where_document = {"$contains": metadata_filter.keyword}

    logger.debug(
        "Built ChromaDB filter | where=%s | where_document=%s",
        where,
        where_document,
    )
    return where, where_document


def _add_list_condition(
    conditions: List[Dict[str, Any]],
    field_name: str,
    values: Optional[List[str]],
) -> None:
    """Append an $eq or $in condition to *conditions* if *values* is set."""
    if not values:
        return
    if len(values) == 1:
        conditions.append({field_name: {"$eq": values[0]}})
    else:
        conditions.append({field_name: {"$in": values}})


# ---------------------------------------------------------------------------
# SQLite WHERE clause builder
# ---------------------------------------------------------------------------

def build_sql_where_clause(
    metadata_filter: MetadataFilter,
    table_alias: str = "",
) -> Tuple[str, List[Any]]:
    """Build a parameterised SQL WHERE clause from a MetadataFilter.

    Designed for use with the SQLite storage layer.  Returns a clause
    string (without the ``WHERE`` keyword) and the corresponding list of
    parameter values for safe parameterised execution.

    Parameters
    ----------
    metadata_filter:
        The filter specification to convert.
    table_alias:
        Optional table alias prefix (e.g. ``"a"`` to produce
        ``a.source_type = ?``).  Include the dot in the alias
        (e.g. ``"a."``), or omit for no alias.

    Returns
    -------
    (clause, params)
        - ``clause``: SQL fragment like ``"source_type = ? AND region IN (?, ?)"``
          or an empty string if no conditions are active.
        - ``params``: list of values to bind to the ``?`` placeholders.

    Examples
    --------
    >>> f = MetadataFilter(source_types=["REUTERS", "CNBC"], regions=["US"])
    >>> clause, params = build_sql_where_clause(f, table_alias="a.")
    >>> clause
    'a.source_type IN (?, ?) AND a.region = ?'
    >>> params
    ['REUTERS', 'CNBC', 'US']
    """
    if metadata_filter.is_empty():
        return "", []

    p = table_alias  # prefix for column names
    clauses: List[str] = []
    params: List[Any] = []

    # --- Enum list fields ---
    _add_sql_list(clauses, params, f"{p}source_type", metadata_filter.source_types)
    _add_sql_list(clauses, params, f"{p}asset_class", metadata_filter.asset_classes)
    _add_sql_list(clauses, params, f"{p}region", metadata_filter.regions)
    _add_sql_list(clauses, params, f"{p}importance", metadata_filter.importance_levels)
    _add_sql_list(clauses, params, f"{p}topic", metadata_filter.topics)
    _add_sql_list(clauses, params, f"{p}sentiment_category", metadata_filter.sentiment_categories)

    # --- Ticker mentions (stored as JSON array string; use LIKE search) ---
    if metadata_filter.ticker_mentions:
        ticker_clauses = [f"{p}ticker_mentions LIKE ?" for _ in metadata_filter.ticker_mentions]
        clauses.append(f"({' OR '.join(ticker_clauses)})")
        for ticker in metadata_filter.ticker_mentions:
            params.append(f"%{ticker}%")

    # --- Date range ---
    if metadata_filter.published_after is not None:
        clauses.append(f"{p}published_at >= ?")
        params.append(metadata_filter.published_after.isoformat())

    if metadata_filter.published_before is not None:
        clauses.append(f"{p}published_at <= ?")
        params.append(metadata_filter.published_before.isoformat())

    # --- Numeric: sentiment score ---
    if metadata_filter.min_sentiment_score is not None:
        clauses.append(f"{p}sentiment_score >= ?")
        params.append(float(metadata_filter.min_sentiment_score))

    if metadata_filter.max_sentiment_score is not None:
        clauses.append(f"{p}sentiment_score <= ?")
        params.append(float(metadata_filter.max_sentiment_score))

    # --- Numeric: confidence ---
    if metadata_filter.min_confidence is not None:
        clauses.append(f"{p}confidence >= ?")
        params.append(float(metadata_filter.min_confidence))

    # --- Full-text keyword ---
    if metadata_filter.keyword:
        clauses.append(f"({p}title LIKE ? OR {p}description LIKE ?)")
        params.append(f"%{metadata_filter.keyword}%")
        params.append(f"%{metadata_filter.keyword}%")

    sql = " AND ".join(clauses)
    logger.debug("Built SQL WHERE clause: %s | params=%s", sql, params)
    return sql, params


def _add_sql_list(
    clauses: List[str],
    params: List[Any],
    column: str,
    values: Optional[List[str]],
) -> None:
    """Append an ``= ?`` or ``IN (?, ...)`` clause to *clauses*."""
    if not values:
        return
    if len(values) == 1:
        clauses.append(f"{column} = ?")
        params.append(values[0])
    else:
        placeholders = ", ".join("?" * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
