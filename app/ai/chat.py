"""AI chat handler for dataset-level conversational Q&A."""
import logging
from typing import Optional

from .llm import generate

logger = logging.getLogger("autoeda.ai.chat")

_SYSTEM = """You are an AI data analyst assistant inside AutoEDA.

Dataset: {name} | Rows: {rows} | Columns: {cols}
Column names (sample): {col_names}
Column types: {type_summary}
Missing: {missing_pct}% | Quality score: {quality_score}/100
Top issues: {issues}

{page_context_block}

Answer concisely. If you suggest an analysis, mention the AutoEDA page by name.
Keep responses under 150 words unless a longer answer is clearly needed.

Conversation history:
{history}

User: {message}
Assistant:"""

_PAGE_CONTEXT_BLOCK = """Current page context:
{details}"""


def chat_reply(
    message: str,
    history: list[dict],
    dataset_context: dict,
    page_context: Optional[dict] = None,
) -> str:
    history_text = ""
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_text += f"{role}: {turn.get('content', '')}\n"

    ctx = dataset_context
    col_names = ", ".join(ctx.get("columns", [])[:20])
    type_summary = ", ".join(f"{v} {k}" for k, v in ctx.get("column_types", {}).items() if v > 0)
    issues = "; ".join(ctx.get("issues", [])[:3]) or "none"

    page_block = ""
    if page_context:
        details = "\n".join(f"  {k}: {v}" for k, v in page_context.items())
        page_block = _PAGE_CONTEXT_BLOCK.format(details=details)

    prompt = _SYSTEM.format(
        name=ctx.get("name", "Unknown"),
        rows=ctx.get("rows", "?"),
        cols=ctx.get("cols", "?"),
        col_names=col_names or "(unknown)",
        type_summary=type_summary or "mixed",
        missing_pct=ctx.get("missing_pct", 0),
        quality_score=ctx.get("quality_score", "?"),
        issues=issues,
        page_context_block=page_block,
        history=history_text.strip() or "(no prior messages)",
        message=message,
    )

    reply = generate(prompt, temperature=0.5, max_tokens=512)
    if not reply:
        return (
            "AI service is unavailable. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY in your environment."
        )
    return reply
