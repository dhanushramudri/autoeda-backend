"""Generate a concise AI narrative for a dataset's overview page."""
import logging
from typing import Optional

from .llm import generate

logger = logging.getLogger("autoeda.ai.narrator")

_PROMPT = """You are a data analyst writing a concise executive summary for a dataset overview page.
Given the dataset metadata below, write 2-4 sentences that:
1. Describe what the dataset appears to contain
2. Highlight the most important data quality facts (missing data, duplicates)
3. Point out the most interesting insight (skewed columns, strong correlations, outliers)

Plain paragraph only. No bullet points. No markdown.

Dataset name: {name}
Rows: {rows} | Columns: {cols}
Memory: {memory_mb} MB | Missing: {missing_pct}% | Duplicates: {duplicate_pct}%
Column types: {type_summary}
Top issues: {issues}
Top suggestions: {suggestions}
Sample columns: {sample_cols}"""


def build_narrative(
    name: str,
    rows: int,
    cols: int,
    memory_mb: float,
    missing_pct: float,
    duplicate_pct: float,
    column_types: dict,
    issues: list[str],
    suggestions: list[str],
    sample_cols: list[str],
) -> Optional[str]:
    type_summary = ", ".join(f"{v} {k}" for k, v in column_types.items() if v > 0) or "mixed"
    prompt = _PROMPT.format(
        name=name,
        rows=rows,
        cols=cols,
        memory_mb=round(memory_mb, 2),
        missing_pct=round(missing_pct, 1),
        duplicate_pct=round(duplicate_pct, 1),
        type_summary=type_summary,
        issues="; ".join(issues[:3]) or "none",
        suggestions="; ".join(suggestions[:3]) or "none",
        sample_cols=", ".join(sample_cols[:10]),
    )
    return generate(prompt, temperature=0.4, max_tokens=300)
