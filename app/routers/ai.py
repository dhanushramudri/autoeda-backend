"""AI endpoints: dataset narrative + Ask AI chat."""
import json
import logging
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..cache import get_cached_result
from ..dataset_access import assert_dataset_access
from ..database import get_db
from ..models.dataset import Dataset
from ..models.user import User

logger = logging.getLogger("autoeda.routers.ai")
router = APIRouter(tags=["ai"])


def _get_ds(dataset_id: int, user: User, db: Session) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    assert_dataset_access(ds, user, db)
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
    page_context: Optional[dict[str, Any]] = None


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
        page_context=body.page_context,
    )
    return {"reply": reply}


@router.get("/datasets/{dataset_id}/ai/transform-suggestions")
def get_transform_suggestions(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    if ds.status != "ready":
        return {"suggestions": []}

    content_hash = ds.content_hash or ""
    profile = get_cached_result(db, ds.id, "profile", {}, content_hash) or {}
    quality = get_cached_result(db, ds.id, "quality_score", {}, content_hash) or {}

    from ..ai.transform_advisor import get_transform_suggestions as _get_suggestions
    suggestions = _get_suggestions(profile, quality)
    return {"suggestions": suggestions}


@router.get("/ai/provider")
def get_provider_info():
    """Return the active AI provider name (for UI display)."""
    from ..ai.llm import provider_name
    return {"provider": provider_name()}


# ── NL → Transform Step ──────────────────────────────────────────────────────

class NLTransformRequest(BaseModel):
    prompt: str


@router.post("/datasets/{dataset_id}/ai/nl-transform")
def nl_transform(
    dataset_id: int,
    body: NLTransformRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    if ds.status != "ready":
        raise HTTPException(400, "Dataset not ready for AI operations")
    if not body.prompt.strip():
        raise HTTPException(422, "Prompt cannot be empty")

    content_hash = ds.content_hash or ""
    profile = get_cached_result(db, ds.id, "profile", {}, content_hash) or {}
    columns = [c.get("name", "") for c in profile.get("columns", [])]
    column_types = {
        c.get("name", ""): c.get("semantic_type", "unknown")
        for c in profile.get("columns", [])
    }

    from ..ai.nl_transform import generate_transform_step
    try:
        result = generate_transform_step(body.prompt, columns, column_types)
    except ValueError as e:
        raise HTTPException(422, str(e))

    return result


# ── Hypothesis Generation ─────────────────────────────────────────────────────

@router.get("/datasets/{dataset_id}/ai/hypotheses")
def get_hypotheses(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    ds = _get_ds(dataset_id, current_user, db)
    if ds.status != "ready":
        return {"hypotheses": [], "source": "none", "message": "Dataset is not ready yet."}

    content_hash = ds.content_hash or ""

    profile = get_cached_result(db, ds.id, "profile", {}, content_hash) or {}
    correlations = (
        get_cached_result(db, ds.id, "correlations", {"type": "correlations", "method": "pearson"}, content_hash)
        or {}
    )
    outliers = (
        get_cached_result(db, ds.id, "outliers", {"type": "outliers", "method": "iqr", "column": None}, content_hash)
        or {}
    )

    # Feature importance: fetch whatever target was last computed (target varies per user)
    from ..models.dataset import EDAResult
    fi_row = (
        db.query(EDAResult)
        .filter(
            EDAResult.dataset_id == ds.id,
            EDAResult.analysis_type == "feature_importance",
            EDAResult.dataset_version == content_hash,
        )
        .order_by(EDAResult.computed_at.desc())
        .first()
    )
    feature_importance: dict = {}
    if fi_row:
        import json as _json
        try:
            feature_importance = _json.loads(fi_row.result_data)
        except Exception:
            pass

    from ..ai.hypothesis_generator import generate_hypotheses
    cards = generate_hypotheses(
        name=ds.name,
        profile=profile,
        correlations=correlations,
        outliers=outliers,
        feature_importance=feature_importance,
    )

    from ..ai.llm import provider_name
    source = "ai" if provider_name() != "none" else "rules"

    return {"hypotheses": cards, "source": source, "count": len(cards)}
