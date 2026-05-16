"""AI chat handler for dataset-level conversational Q&A."""
import logging
from typing import Optional

from .gemini_client import generate

logger = logging.getLogger("autoeda.ai.chat")

_SYSTEM = """You are an AI data analyst assistant embedded in AutoEDA, a data exploration platform.
The user is analysing a dataset with the following details:

Dataset: {name}
Rows: {rows} | Columns: {cols}
Column names: {col_names}
Column types: {type_summary}
Missing data: {missing_pct}% overall
Quality score: {quality_score}/100
Top issues: {issues}

Answer the user's questions concisely and helpfully. Focus on data analysis insights.
If you suggest a chart or analysis, mention the relevant page name in AutoEDA (e.g. "open the Distributions page").
Keep responses under 200 words unless a longer answer is clearly needed.

Conversation so far:
{history}

User: {message}
Assistant:"""


def chat_reply(
    message: str,
    history: list[dict],   # [{"role": "user"|"assistant", "content": str}]
    dataset_context: dict,
) -> str:
    history_text = ""
    for turn in history[-8:]:   # keep last 8 turns
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_text += f"{role}: {turn.get('content', '')}\n"

    ctx = dataset_context
    col_names = ", ".join(ctx.get("columns", [])[:30])
    type_summary = ", ".join(f"{v} {k}" for k, v in ctx.get("column_types", {}).items() if v > 0)
    issues = "; ".join(ctx.get("issues", [])[:3]) or "none"

    prompt = _SYSTEM.format(
        name=ctx.get("name", "Unknown"),
        rows=ctx.get("rows", "?"),
        cols=ctx.get("cols", "?"),
        col_names=col_names or "(unknown)",
        type_summary=type_summary or "mixed",
        missing_pct=ctx.get("missing_pct", 0),
        quality_score=ctx.get("quality_score", "?"),
        issues=issues,
        history=history_text.strip() or "(no prior messages)",
        message=message,
    )

    reply = generate(prompt, temperature=0.5)
    if not reply:
        return (
            "I'm currently unavailable — the AI service is not configured or unreachable. "
            "Please ensure GEMINI_API_KEY is set in your environment."
        )
    return reply
