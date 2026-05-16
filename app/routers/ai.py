"""AI endpoints: dataset narrative + Ask AI chat."""
import json
import logging
from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..cache import get_cached_result
from ..database import get_db
from ..models.dataset import Dataset
from ..models.user import User

logger = logging.getLogger("autoeda.routers.ai")
router = APIRouter(tags=["ai"])


def _get_ds(dataset_id: int, user: User, db: Session) -> Dataset:
    from ..models.workspace import WorkspaceMember
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == ds.workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()
    if not member:
        raise HTTPException(403, "Not a workspace member")
    return ds


def _build_dataset_context(ds: Dataset, db: Session) -> dict:
    """Assemble lightweight context dict from cached EDA results."""
    content_hash = ds.content_hash or ""
    profile_result = get_cached_result(db, ds.id, "profile", {}, content_hash)
    quality_result = get_cached_result(db, ds.id, "quality_score", {}, content_hash)

    profile = profile_result or {}
    quality = quality_result or {}

    columns_raw = profile.get("columns", [])
    col_names = [c.get("name", "") for c in columns_raw]

    type_counts: Counter = Counter()
    for c in columns_raw:
        t = c.get("semantic_type", "unknown")
        type_counts[t] += 1

    missing_pct = 0.0
    total_rows = profile.get("total_rows", ds.row_count or 0)
    total_cols = profile.get("total_columns", ds.column_count or 0)
    if total_rows and total_cols:
        total_missing = sum(c.get("missing_count", 0) for c in columns_raw)
        total_cells = total_rows * total_cols
        missing_pct = round((total_missing / total_cells) * 100, 1) if total_cells else 0.0

    issues = [i.get("description", "") for i in quality.get("issues", [])]
    suggestions = quality.get("suggestions", [])

    return {
        "name": ds.name,
        "rows": total_rows,
        "cols": total_cols,
        "columns": col_names,
        "column_types": dict(type_counts),
        "missing_pct": missing_pct,
        "duplicate_pct": round(profile.get("duplicate_pct", 0), 1),
        "memory_mb": round(profile.get("memory_mb", 0), 2),
        "quality_score": quality.get("overall_score", "?"),
        "issues": issues,
        "suggestions": suggestions,
    }


# ── Narrative ────────────────────────────────────────────────────────────────

@router.get("/datasets/{dataset_id}/ai/narrative")
def get_narrative(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    if ds.status != "ready":
        return {"narrative": None, "message": "Dataset is not ready yet."}

    ctx = _build_dataset_context(ds, db)

    from ..ai.narrator import build_narrative
    narrative = build_narrative(
        name=ctx["name"],
        rows=ctx["rows"],
        cols=ctx["cols"],
        memory_mb=ctx["memory_mb"],
        missing_pct=ctx["missing_pct"],
        duplicate_pct=ctx["duplicate_pct"],
        column_types=ctx["column_types"],
        issues=ctx["issues"],
        suggestions=ctx["suggestions"],
        sample_cols=ctx["columns"],
    )

    return {"narrative": narrative}


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []


@router.post("/datasets/{dataset_id}/ai/chat")
def ai_chat(
    dataset_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    ctx = _build_dataset_context(ds, db)

    from ..ai.chat import chat_reply
    reply = chat_reply(
        message=body.message,
        history=body.history,
        dataset_context=ctx,
    )
    return {"reply": reply}
