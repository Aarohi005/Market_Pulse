#!/usr/bin/env python3
"""
init_db.py — initialise the SQLite database.

Usage:
    python scripts/init_db.py

Safe to run multiple times (CREATE IF NOT EXISTS).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings
from src.storage.database import init_db

if __name__ == "__main__":
    print(f"Initialising database at: {settings.db_path_resolved}")
    init_db(settings.db_path_resolved)
    print("Done.")
