import json
from typing import Any

from sqlalchemy.orm import Session

from .models.dataset import EDAResult


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def get_cached_result(
    db: Session,
    dataset_id: int,
    analysis_type: str,
    params: dict,
    current_hash: str,
) -> dict | None:
    key = _params_key(params)
    row = (
        db.query(EDAResult)
        .filter(
            EDAResult.dataset_id == dataset_id,
            EDAResult.analysis_type == analysis_type,
            EDAResult.parameters == key,
            EDAResult.dataset_version == current_hash,
        )
        .order_by(EDAResult.computed_at.desc())
        .first()
    )
    if row:
        try:
            return json.loads(row.result_data)
        except Exception:
            return None
    return None


def store_result(
    db: Session,
    dataset_id: int,
    analysis_type: str,
    params: dict,
    result: dict,
    content_hash: str,
) -> None:
    key = _params_key(params)
    row = EDAResult(
        dataset_id=dataset_id,
        analysis_type=analysis_type,
        parameters=key,
        result_data=json.dumps(result, default=str),
        dataset_version=content_hash,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
