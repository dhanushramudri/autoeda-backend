"""AI endpoints: NL → transform step + provider info."""
import logging

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
