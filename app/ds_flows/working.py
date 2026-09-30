"""The flow's cleaned modelling table, saved as a hidden dataset so the existing AutoEDA pages (profile, analysis,
distributions, correlations, missing, outliers, feature importance...) can show THIS flow's data.

Kept deliberately small (sampled rows, strongest features) so the existing EDA endpoints stay light on a small
server. The dataset is hidden from every dataset list (see the source_config marker) and deleted with the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import pandas as pd

from .churn import META, _feature_frame

logger = logging.getLogger("autoeda.ds_flows.working")

MARKER = '"ds_flow_run"'
MAX_ROWS = 40_000
MAX_NUMERIC = 60
MAX_CATEGORICAL = 8


def build_working_frame(work: pd.DataFrame, roles: dict, excluded: dict, uni: pd.DataFrame, drop: set[str] | None = None) -> pd.DataFrame:
    drop = drop or set()
    ent, dcol = roles.get("entity"), roles.get("date")
    skip = set(excluded) | drop | {c for c in (ent, dcol) if c}
    strength = dict(zip(uni["feature"], uni["strength"])) if len(uni) else {}
    feats = _feature_frame(work, skip)
    ranked = sorted(feats.columns, key=lambda c: -strength.get(c, 0))[:MAX_NUMERIC]

    out = pd.DataFrame(index=work.index)
    if ent and ent in work.columns:
        out[ent] = work[ent]
    if dcol and work["__date"].notna().any():
        out[dcol] = work["__date"]
    out["churned"] = work["__y"]
    for c in ranked:
        out[c] = feats[c]
    added = 0
    for c in work.columns:
        if added >= MAX_CATEGORICAL:
            break
        if c in META or c in skip or c in out.columns:
            continue
        s = work[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and 2 <= s.nunique() <= 15:
            out[c] = s
            added += 1
    if len(out) > MAX_ROWS:
        out = out.sample(MAX_ROWS, random_state=42).sort_index()
    return out.reset_index(drop=True)


def upsert_working_dataset(db, run, frame: pd.DataFrame, existing_id: int | None = None) -> int:
    from ..models.dataset import Dataset

    data = frame.to_csv(index=False).encode("utf-8")
    ds = db.query(Dataset).filter(Dataset.id == existing_id).first() if existing_id else None
    if ds is None:
        ds = Dataset(workspace_id=run.workspace_id, created_by=run.created_by, source_type="file")
        db.add(ds)
    ds.name = re.sub(r"\s+", " ", f"DS Flow · {run.title or 'churn'} (working table)")[:250]
    ds.description = "Cleaned modelling table produced by a Data Science Flow run. Hidden from dataset lists."
    ds.source_config = json.dumps({"ds_flow_run": run.id})
    ds.file_path = f"ds_flow_{run.id}_working.csv"
    ds.file_data = data
    ds.file_size_bytes = len(data)
    ds.content_hash = hashlib.md5(data).hexdigest()
    ds.row_count = int(len(frame))
    ds.column_count = int(frame.shape[1])
    ds.schema_info = json.dumps({c: str(t) for c, t in frame.dtypes.items()})
    ds.status = "ready"
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds.id
