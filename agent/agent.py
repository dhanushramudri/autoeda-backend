"""AutoEDA local training agent.

Runs on the engineer's own laptop (as a Docker container, so nothing needs
to be pip-installed by hand) instead of the cloud server. It logs in with
the engineer's normal AutoEDA credentials, polls for experiments they
queued from the AutoML page, downloads the dataset through the existing
export endpoint, trains the DS playbook's algorithm shortlist, and reports
every run back so the leaderboard fills in live.

Env vars:
  AUTOEDA_API_URL     e.g. http://<your-ec2-host>:8000/api/v1   (required)
  AUTOEDA_EMAIL       your AutoEDA login email                  (required)
  POLL_INTERVAL_SECONDS   default 10
  MAX_TRAINING_ROWS       default 1000 — datasets larger than this get
                          sampled down (stratified, for classification)
                          before training. 1000 is a "does the pipeline
                          work" speed, not a "trust this accuracy" one —
                          raise it (e.g. 20000+) once you're past testing
                          and want a model that's actually worth keeping.
"""
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import joblib
import pandas as pd
import requests

from algorithms import (
    ALGORITHM_LABELS, determine_problem_type, detect_categorical_columns,
    shortlist_algorithms, train_one,
)
from feature_tools import apply_engineered_features

API_URL = os.environ.get("AUTOEDA_API_URL", "").rstrip("/")
EMAIL = os.environ.get("AUTOEDA_EMAIL", "")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
MAX_TRAINING_ROWS = int(os.environ.get("MAX_TRAINING_ROWS", "1000"))


def log(msg: str):
    print(f"[autoeda-agent] {msg}", flush=True)


def login() -> str:
    # Same passwordless mock-auth the web app itself uses (see
    # POST /auth/microsoft-mock) — an email is all AutoEDA accounts have.
    resp = requests.post(f"{API_URL}/auth/microsoft-mock", json={"email": EMAIL}, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def load_dataset(session: requests.Session, dataset_id: int) -> pd.DataFrame:
    resp = session.get(f"{API_URL}/datasets/{dataset_id}/export", timeout=120)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    disposition = resp.headers.get("content-disposition", "")
    filename = "dataset.csv"
    if "filename=" in disposition:
        filename = disposition.split("filename=")[-1].strip('"; ')

    buf = io.BytesIO(resp.content)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
    if ext in ("xlsx", "xls") or "spreadsheet" in content_type:
        return pd.read_excel(buf)
    if ext == "parquet":
        return pd.read_parquet(buf)
    if ext == "json":
        return pd.read_json(buf)
    return pd.read_csv(buf)


def run_experiment(session: requests.Session, exp: dict):
    exp_id = exp["id"]
    log(f"Claiming experiment #{exp_id} ({exp['name']})")
    claim = session.post(f"{API_URL}/agent/experiments/{exp_id}/claim", timeout=30)
    if claim.status_code == 409:
        log(f"Experiment #{exp_id} was already claimed elsewhere — skipping")
        return
    claim.raise_for_status()

    try:
        log(f"Downloading dataset #{exp['dataset_id']}…")
        df = load_dataset(session, exp["dataset_id"])
        target = exp["target_column"]
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        recipes = exp.get("engineered_features", [])
        if recipes:
            log(f"Applying {len(recipes)} engineered feature(s)…")
            df = apply_engineered_features(df, recipes, log=log)

        excluded = [c for c in exp.get("excluded_columns", []) if c in df.columns and c != target]
        if excluded:
            log(f"Dropping {len(excluded)} excluded column(s): {excluded}")
            df = df.drop(columns=excluded)

        problem_type = exp.get("problem_type") or determine_problem_type(df[target])
        feature_cols = [c for c in df.columns if c != target]
        categorical_cols = detect_categorical_columns(df, feature_cols)
        algorithms = shortlist_algorithms(len(df), len(categorical_cols))

        sample_note = f" (sampled down to {MAX_TRAINING_ROWS:,} for training)" if len(df) > MAX_TRAINING_ROWS else ""
        log(
            f"{problem_type} · {len(df):,} rows · {len(feature_cols)} features "
            f"({len(categorical_cols)} categorical) → trying {[ALGORITHM_LABELS[a] for a in algorithms]}{sample_note}"
        )

        any_success = False
        for algo in algorithms:
            label = ALGORITHM_LABELS[algo]
            log(f"  Training {label}…")
            try:
                estimator, params, metrics, feature_importance, seconds = train_one(
                    df, target, algo, problem_type, categorical_cols, max_rows=MAX_TRAINING_ROWS,
                )
                log(f"  {label} done in {seconds}s — {metrics}")
                run_resp = session.post(
                    f"{API_URL}/agent/experiments/{exp_id}/runs",
                    json={
                        "algorithm": label, "params": params, "metrics": metrics,
                        "feature_importance": feature_importance,
                        "training_seconds": seconds, "status": "completed",
                    },
                    timeout=30,
                )
                run_resp.raise_for_status()
                run_id = run_resp.json()["id"]

                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                    joblib.dump(estimator, tmp.name)
                    tmp_path = tmp.name
                try:
                    with open(tmp_path, "rb") as f:
                        session.post(
                            f"{API_URL}/agent/experiments/{exp_id}/runs/{run_id}/artifact",
                            files={"file": (f"{algo}.joblib", f, "application/octet-stream")},
                            timeout=60,
                        ).raise_for_status()
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                any_success = True
            except Exception as e:
                log(f"  {label} FAILED: {e}")
                session.post(
                    f"{API_URL}/agent/experiments/{exp_id}/runs",
                    json={"algorithm": label, "status": "failed", "error": str(e)},
                    timeout=30,
                )

        status = "completed" if any_success else "failed"
        error = None if any_success else "Every shortlisted algorithm failed — see individual run errors."
        session.post(f"{API_URL}/agent/experiments/{exp_id}/complete", json={"status": status, "error": error}, timeout=30)
        log(f"Experiment #{exp_id} {status}")

    except Exception as e:
        log(f"Experiment #{exp_id} FAILED: {e}\n{traceback.format_exc()}")
        session.post(
            f"{API_URL}/agent/experiments/{exp_id}/complete",
            json={"status": "failed", "error": str(e)},
            timeout=30,
        )


def main():
    if not API_URL or not EMAIL:
        log("Missing required env vars: AUTOEDA_API_URL, AUTOEDA_EMAIL")
        sys.exit(1)

    log(f"Starting — API: {API_URL}, user: {EMAIL}")
    token = login()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    log("Logged in. Polling for queued experiments…")

    while True:
        try:
            resp = session.get(f"{API_URL}/agent/experiments/queued", timeout=30)
            if resp.status_code == 401:
                log("Token expired — logging in again")
                token = login()
                session.headers["Authorization"] = f"Bearer {token}"
                continue
            resp.raise_for_status()
            queued = resp.json()
            for exp in queued:
                run_experiment(session, exp)
        except requests.RequestException as e:
            log(f"Poll failed (will retry): {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
