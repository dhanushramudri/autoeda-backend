"""Generate a concise AI narrative for a dataset's overview page."""
import logging
from typing import Optional

from .gemini_client import generate

logger = logging.getLogger("autoeda.ai.narrator")

_PROMPT = """You are a data analyst writing a concise executive summary for a dataset overview page.
Given the dataset metadata below, write 2-4 sentences that:
1. Describe what the dataset appears to contain
2. Highlight the most important data quality facts (e.g. missing data, duplicates)
3. Point out the most interesting insight (e.g. skewed columns, strong correlations, outliers)

Be direct, informative, and professional. No bullet points. Plain paragraph only.

Dataset name: {name}
Rows: {rows} | Columns: {cols}
Memory: {memory_mb} MB
Missing data: {missing_pct}% of all cells
Duplicate rows: {duplicate_pct}%
Column types: {type_summary}
Top quality issues: {issues}
Top suggestions: {suggestions}
Sample column names: {sample_cols}"""


def build_narrative(
    name: str,
    rows: int,
    cols: int,
    memory_mb: float,
    missing_pct: float,
    duplicate_pct: float,
    column_types: dict,          # {"numeric": 5, "categorical": 3, ...}
    issues: list[str],
    suggestions: list[str],
    sample_cols: list[str],
) -> Optional[str]:
    type_summary = ", ".join(f"{v} {k}" for k, v in column_types.items() if v > 0)
    issues_str = "; ".join(issues[:3]) if issues else "none"
    suggestions_str = "; ".join(suggestions[:3]) if suggestions else "none"
    sample_str = ", ".join(sample_cols[:10])

    prompt = _PROMPT.format(
        name=name,
        rows=rows,
        cols=cols,
        memory_mb=round(memory_mb, 2),
        missing_pct=round(missing_pct, 1),
        duplicate_pct=round(duplicate_pct, 1),
        type_summary=type_summary or "mixed",
        issues=issues_str,
        suggestions=suggestions_str,
        sample_cols=sample_str,
    )

    return generate(prompt, temperature=0.4)
