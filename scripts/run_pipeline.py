#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the ingestion + scoring pipeline.

Usage:
    # Normal run
    python scripts/run_pipeline.py

    # Skip market data refresh (faster, offline-safe)
    python scripts/run_pipeline.py --skip-market

    # Dry run (no DB writes — useful for CI smoke tests)
    python scripts/run_pipeline.py --dry-run

Exit codes:
    0 — pipeline completed successfully
    1 — pipeline failed (check logs)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial news sentiment pipeline.")
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Skip yfinance market data refresh",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM analysis stage (faster, no API cost)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip topic clustering stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and normalise without writing to the database",
    )
    args = parser.parse_args()

    result = run_pipeline(
        skip_market_refresh=args.skip_market,
        skip_llm=args.skip_llm,
        skip_clustering=args.skip_clustering,
        dry_run=args.dry_run,
    )

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
