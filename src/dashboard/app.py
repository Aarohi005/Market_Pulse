"""
app.py — Streamlit multi-page app entry point.

Streamlit reads pages/ sub-directory automatically when using st.Page.
This file sets global configuration and renders the navigation shell.

Run:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path fix ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ────────────────────────────────────────────────────────────────
pages_dir = Path(__file__).parent / "pages"

pg = st.navigation(
    [
        st.Page(str(pages_dir / "1_Market_Overview.py"),     title="Market Overview",     icon="🌐"),
        st.Page(str(pages_dir / "2_Sentiment_Trends.py"),    title="Sentiment Trends",    icon="📈"),
        st.Page(str(pages_dir / "3_Source_Analysis.py"),     title="Source Analysis",     icon="📰"),
        st.Page(str(pages_dir / "4_Topic_Analysis.py"),      title="Topic Analysis",      icon="🏷️"),
        st.Page(str(pages_dir / "5_Confidence_Analysis.py"), title="Confidence Analysis", icon="🎯"),
        st.Page(str(pages_dir / "6_Market_Correlation.py"),  title="Market Correlation",  icon="📉"),
    ]
)
pg.run()

if __name__ == "__main__":
    print("=" * 10)

    print("Current file:")
    print(Path(__file__).resolve())

    print("\nProject root:")
    print(_ROOT)

    print("\nPages directory:")
    print(pages_dir.resolve())

    print("\nDirectory exists:")
    print(pages_dir.exists())

    page_files = [
        "1_Market_Overview.py",
        "2_Sentiment_Trends.py",
        "3_Source_Analysis.py",
        "4_Topic_Analysis.py",
        "5_Confidence_Analysis.py",
        "6_Market_Correlation.py",
    ]

    print("\nChecking pages...\n")

    for page in page_files:
        path = pages_dir / page
        print(f"{page:35} -> {path.exists()}")

    print("=" * 60)
