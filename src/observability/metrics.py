"""
Pipeline metrics collection and reporting.

Provides an in-memory :class:`MetricsCollector` that records float
metric values keyed by pipeline stage.  Includes a context-manager
helper for automatic latency timing and a module-level singleton that
can be reset between pipeline runs.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetric:
    """A single recorded metric observation.

    Attributes
    ----------
    stage:
        Pipeline stage name (e.g. ``"ingestion"``, ``"embedding"``).
    metric_name:
        Name of the metric (e.g. ``"articles_fetched"``,
        ``"embedding_latency_ms"``).
    metric_value:
        Numeric value of the metric.
    recorded_at:
        ISO-8601 UTC timestamp string when the metric was recorded.
    """

    stage: str
    metric_name: str
    metric_value: float
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "stage": self.stage,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "recorded_at": self.recorded_at,
        }


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """In-memory store for pipeline stage metrics.

    Metrics are keyed by ``(stage, metric_name)`` and stored as a list of
    :class:`PipelineMetric` instances (allowing multiple observations of
    the same metric during a run).

    Parameters
    ----------
    None

    Examples
    --------
    >>> collector = MetricsCollector()
    >>> collector.record("ingestion", "articles_fetched", 42)
    >>> with collector.timed_stage("embedding"):
    ...     time.sleep(0.1)  # simulate work
    >>> summary = collector.get_summary()
    >>> summary["ingestion"]["articles_fetched"]
    42.0
    """

    def __init__(self) -> None:
        # stage -> list of PipelineMetric
        self._store: Dict[str, List[PipelineMetric]] = {}
        # For context-manager timing (stack to allow nesting)
        self._timing_stack: List[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------

    def record(self, stage: str, metric_name: str, value: float) -> None:
        """Record a single metric observation.

        Appends the metric to the in-memory store and emits a DEBUG log
        message.

        Parameters
        ----------
        stage:
            Pipeline stage identifier string.
        metric_name:
            Name of the metric being recorded.
        value:
            Numeric value to record.
        """
        metric = PipelineMetric(
            stage=stage,
            metric_name=metric_name,
            metric_value=float(value),
        )
        self._store.setdefault(stage, []).append(metric)
        logger.debug(
            "METRIC | stage=%-20s | %-35s | %s",
            stage,
            metric_name,
            value,
        )

    def reset(self) -> None:
        """Clear all recorded metrics.

        Call this at the start of each pipeline run to ensure the
        collector starts fresh.
        """
        self._store.clear()
        self._timing_stack.clear()
        logger.debug("MetricsCollector reset.")

    # ------------------------------------------------------------------
    # Summary / export
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return the latest value of every metric keyed by stage.

        When the same metric has been recorded multiple times for a
        stage, the **last** recorded value is returned.

        Returns
        -------
        dict
            ``{stage: {metric_name: latest_value}}``
        """
        summary: Dict[str, Dict[str, float]] = {}
        for stage, metrics in self._store.items():
            stage_summary: Dict[str, float] = {}
            for m in metrics:
                stage_summary[m.metric_name] = m.metric_value
            summary[stage] = stage_summary
        return summary

    def get_all(self) -> List[PipelineMetric]:
        """Return every recorded :class:`PipelineMetric` in insertion order.

        Returns
        -------
        list of PipelineMetric
        """
        all_metrics: List[PipelineMetric] = []
        for metrics in self._store.values():
            all_metrics.extend(metrics)
        return all_metrics

    def to_db_rows(self, run_id: int) -> List[Dict[str, Any]]:
        """Convert all recorded metrics to database-ready row dictionaries.

        Parameters
        ----------
        run_id:
            Pipeline run identifier to attach to each row.

        Returns
        -------
        list of dict
            One dict per :class:`PipelineMetric` with keys:
            ``run_id``, ``stage``, ``metric_name``, ``metric_value``,
            ``recorded_at``.
        """
        rows: List[Dict[str, Any]] = []
        for metric in self.get_all():
            row = metric.to_dict()
            row["run_id"] = run_id
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Context manager: overall timing
    # ------------------------------------------------------------------

    def __enter__(self) -> "MetricsCollector":
        """Begin timing the whole collector block (stage name = 'total')."""
        self._timing_stack.append(("total", time.perf_counter()))
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Record the elapsed time when the block exits."""
        if self._timing_stack:
            stage, start = self._timing_stack.pop()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(stage, f"{stage}_latency_ms", elapsed_ms)
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # timed_stage context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def timed_stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager that records ``{stage_name}_latency_ms``.

        Usage::

            with collector.timed_stage("embedding"):
                embeddings = embedder.embed(texts)

        The latency in milliseconds is recorded under the given stage
        regardless of whether the block raises an exception.

        Parameters
        ----------
        stage_name:
            Name of the pipeline stage being timed.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(stage_name, f"{stage_name}_latency_ms", elapsed_ms)
            logger.info(
                "Stage '%s' completed in %.1f ms.", stage_name, elapsed_ms
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def increment(self, stage: str, metric_name: str, by: float = 1.0) -> None:
        """Increment a counter metric by *by* (default 1).

        Reads the current latest value (0 if not yet set), adds *by*,
        then records the new value.

        Parameters
        ----------
        stage:
            Pipeline stage identifier.
        metric_name:
            Name of the counter metric.
        by:
            Amount to increment by.  Defaults to ``1.0``.
        """
        current = self.get_summary().get(stage, {}).get(metric_name, 0.0)
        self.record(stage, metric_name, current + by)

    def stage_metrics(self, stage: str) -> Dict[str, float]:
        """Return all latest metric values for a single *stage*.

        Parameters
        ----------
        stage:
            Stage name to query.

        Returns
        -------
        dict
            ``{metric_name: value}`` for the given stage.
            Returns an empty dict if the stage has no recorded metrics.
        """
        return self.get_summary().get(stage, {})

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        total = sum(len(v) for v in self._store.values())
        return (
            f"MetricsCollector(stages={len(self._store)}, total_metrics={total})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global metrics collector singleton.  Reset at the start of each pipeline
#: run by calling ``CURRENT_RUN_METRICS.reset()``.
CURRENT_RUN_METRICS: MetricsCollector = MetricsCollector()
