"""
Prompt templates for LLM-based financial news sentiment analysis.

Module-level constants hold the system prompts for each task.
The :func:`format_articles_context` helper formats a list of article dicts
into the numbered context block that both prompts expect.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Sentiment analysis prompt
# ---------------------------------------------------------------------------

SENTIMENT_ANALYSIS_PROMPT: str = """\
You are a financial market analyst with deep expertise in interpreting news flow, \
macroeconomic data, and corporate events. Your task is to read the provided set of \
financial news articles and produce a rigorous, evidence-based sentiment assessment.

Follow these steps carefully:

1. **Read all articles** in the context block below before forming any conclusion.

2. **Determine overall market sentiment** for the topic described in the user message. \
Choose exactly one label: "positive", "negative", or "neutral".

3. **Explain your reasoning** in 2–4 sentences, grounding every claim in specific \
articles from the context. Do not speculate beyond the provided evidence.

4. **Identify 3–5 key drivers** — the most important factors (events, data releases, \
corporate actions, macro signals) that shaped the overall sentiment. Each driver \
should be a concise phrase of 5–15 words.

5. **Highlight conflicting signals** — if any articles present information that \
contradicts the dominant sentiment, list them explicitly. Return an empty array \
if there are none.

6. **Rate your confidence** as HIGH, MEDIUM, or LOW based on:
   - HIGH: strong consensus across multiple high-quality sources, clear directional signal
   - MEDIUM: moderate evidence, some ambiguity or limited source diversity
   - LOW: sparse evidence, high uncertainty, or contradictory signals

7. **List supporting headlines** — include the titles of the 2–5 articles that most \
directly support the overall sentiment label.

8. **Return ONLY valid JSON** — no markdown fences, no prose before or after the JSON \
object. The response must be parseable by `json.loads()` without any pre-processing.

Required JSON schema:
{
  "sentiment_label": "<positive|negative|neutral>",
  "sentiment_score": <float between -1.0 and 1.0>,
  "confidence_label": "<HIGH|MEDIUM|LOW>",
  "explanation": "<2-4 sentence prose explanation>",
  "key_drivers": ["<driver 1>", "<driver 2>", "..."],
  "conflicting_signals": ["<signal 1>", "..."],
  "supporting_headlines": ["<headline 1>", "..."]
}

Field definitions:
- sentiment_score: -1.0 = maximally negative, 0.0 = neutral, 1.0 = maximally positive. \
  Must be consistent with sentiment_label (positive → score > 0, negative → score < 0).
- key_drivers: between 3 and 5 items; never fewer than 3.
- conflicting_signals: empty array [] is valid when there are none.
- supporting_headlines: exact or near-exact titles from the provided articles.
"""


# ---------------------------------------------------------------------------
# Market commentary prompt
# ---------------------------------------------------------------------------

MARKET_COMMENTARY_PROMPT: str = """\
You are a senior financial market analyst producing a daily market commentary for \
institutional investors. Your writing is precise, confident, and free of jargon. \
You cite evidence from the news articles provided and avoid speculative language.

Your task has two parts:

**Part A — Market Commentary (narrative)**
Write exactly two paragraphs:
  - Paragraph 1: Summarise the key market developments from the articles and describe \
    the overall sentiment direction with supporting evidence.
  - Paragraph 2: Discuss the most significant risks or opportunities implied by the \
    news flow, and note any conflicting signals that could alter the outlook.

Keep each paragraph to 4–6 sentences. Use precise financial language. \
Do not introduce information not present in the articles.

**Part B — Structured Sentiment Assessment (JSON)**
Immediately after the two paragraphs, output a JSON object (no markdown fences) \
with the following schema. The JSON must be on a single logical block that can be \
extracted by splitting on the first `{` character after the narrative text.

Required JSON schema:
{
  "sentiment_label": "<positive|negative|neutral>",
  "sentiment_score": <float between -1.0 and 1.0>,
  "confidence_label": "<HIGH|MEDIUM|LOW>",
  "explanation": "<1-2 sentence summary matching the narrative>",
  "key_drivers": ["<driver 1>", "<driver 2>", "..."],
  "conflicting_signals": ["<signal 1>", "..."],
  "supporting_headlines": ["<headline 1>", "..."]
}

Field definitions:
- sentiment_score: -1.0 = maximally negative, 0.0 = neutral, 1.0 = maximally positive.
- key_drivers: 3–5 concise factor phrases drawn from the articles.
- conflicting_signals: empty array [] if none present.
- supporting_headlines: titles taken directly from the provided articles.

Confidence rating guidance:
  HIGH   → strong consensus, multiple corroborating tier-1 sources, clear direction
  MEDIUM → moderate evidence, some source diversity, mild ambiguity
  LOW    → sparse evidence, single source, high uncertainty or contradictory signals

Important: the narrative paragraphs come FIRST, then the JSON object. \
Do not wrap either section in markdown.
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def format_articles_context(articles: List[dict]) -> str:
    """Format a list of article dicts into a numbered context block for LLM prompts.

    Each article is rendered as a numbered entry containing its title, source,
    publication timestamp, and description.  Fields that are missing or empty
    are omitted from the rendered entry.

    Args:
        articles: List of dicts.  Recognised keys: ``title``, ``description``,
            ``source_name``, ``published_at``.  Additional keys are ignored.

    Returns:
        A multi-line string formatted as a numbered article context block,
        ready to be inserted into a prompt.  Returns an empty string if
        ``articles`` is empty.

    Example output::

        === NEWS ARTICLES CONTEXT ===

        [1] Title: Fed holds rates steady amid inflation uncertainty
            Source: Reuters | Published: 2024-01-15T14:30:00Z
            Description: The Federal Reserve kept its benchmark interest rate ...

        [2] Title: S&P 500 falls 0.8% on rate-hold disappointment
            ...
        =============================
    """
    if not articles:
        return ""

    lines: List[str] = ["=== NEWS ARTICLES CONTEXT ===\n"]

    for idx, article in enumerate(articles, start=1):
        title = (article.get("title") or "").strip()
        source_name = (article.get("source_name") or "").strip()
        published_at = (article.get("published_at") or "").strip()
        description = (article.get("description") or "").strip()

        entry_lines: List[str] = []

        header = f"[{idx}]"
        if title:
            header += f" Title: {title}"
        entry_lines.append(header)

        meta_parts: List[str] = []
        if source_name:
            meta_parts.append(f"Source: {source_name}")
        if published_at:
            meta_parts.append(f"Published: {published_at}")
        if meta_parts:
            entry_lines.append("    " + " | ".join(meta_parts))

        if description:
            # Wrap long descriptions at ~120 chars per line for readability
            words = description.split()
            current_line: List[str] = ["    Description:"]
            char_count = len(current_line[0])
            for word in words:
                if char_count + len(word) + 1 > 120:
                    entry_lines.append(" ".join(current_line))
                    current_line = ["      " + word]
                    char_count = len(current_line[0])
                else:
                    current_line.append(word)
                    char_count += len(word) + 1
            if current_line:
                entry_lines.append(" ".join(current_line))

        lines.append("\n".join(entry_lines))
        lines.append("")  # blank line between entries

    lines.append("=============================")
    return "\n".join(lines)
