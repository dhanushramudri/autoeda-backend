"""Executes a Data Science Flow run in the background and persists progress after every stage.

Zero configuration: the run loads every dataset it was given (by default, the whole workspace), a first
"discover" stage works out the outcome column, account key, date, revenue and how the tables link, and the
remaining stages run on the combined table. Each stage runs in the shared process pool (app/process_pool.py),
so a heavy model fit can't starve other users and a crash/OOM only fails that stage. The UI polls the run row,
which always holds the stage timeline and every finished stage's results.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from ..process_pool import AnalysisCrashed, AnalysisTimeout, run_isolated

logger = logging.getLogger("autoeda.ds_flows.runner")

FLOW_LABELS: dict[str, str] = {
    "churn":           "Churn",
    "revenue_growth":  "Revenue Growth",
    "pricing":         "Pricing",
    "efficiency_cost": "Efficiency & Cost",
}

def _flow_label(flow_key: str) -> str:
    return FLOW_LABELS.get(flow_key, flow_key.replace("_", " ").title())

_FLOW_EDA_CONTEXT: dict[str, str] = {
    "churn": (
        "Churn analysis. The outcome column is 1 when the account churned and 0 when it renewed. "
        "Focus on what drives churn, which segments churn most, data quality problems and revenue at risk."
    ),
    "revenue_growth": (
        "Revenue growth analysis. The outcome column identifies high-growth vs. lower-growth accounts or products. "
        "Focus on what drives revenue growth, which segments grow fastest, and the key leading indicators."
    ),
    "pricing": (
        "Pricing analysis. The outcome column identifies pricing tiers, premium vs. standard, or above-average margin. "
        "Focus on price sensitivity, segment willingness-to-pay, and the features that predict price tier."
    ),
    "efficiency_cost": (
        "Efficiency and cost analysis. The outcome column identifies high-cost or low-efficiency records. "
        "Focus on cost drivers, inefficient segments, and the operational features that predict high cost."
    ),
    "forecasting": (
        "Time-series forecasting analysis. The table is a single series: one row per period, with a Period column "
        "and the forecasted measure. Focus on trend, seasonality, turning points and anything unusual in the history."
    ),
}

def _flow_eda_context(flow_key: str) -> str:
    return _FLOW_EDA_CONTEXT.get(flow_key, f"{_flow_label(flow_key)} analysis. Focus on the key drivers of the outcome column.")

# A stage failing here aborts the run (nothing downstream can work without it); the others degrade gracefully.
CRITICAL = {"discover", "understand", "leakage", "features", "select", "models", "value", "build"}
HEAVY_ART = {"model_bytes", "enriched_csv", "accounts_csv", "dictionary_csv"}
# Artifacts each stage actually reads. Only these are sent to the worker process, and anything no later stage
# needs is dropped from memory (the feature matrix and trained models are hundreds of MB and were the cause of
# out-of-memory worker crashes on a small server when every stage received everything).
NEEDS = {
    "discover": [],
    "understand": [],
    "leakage": ["work"],
    "eda": ["work", "excluded", "uni"],
    "hypotheses": ["work", "excluded", "uni"],
    "features": ["work", "excluded", "uni"],
    "select": ["work", "X", "uni"],
    "models": ["work", "X", "selected", "train_idx", "test_idx", "uni"],
    "explain": ["work", "X", "selected", "test_idx", "model_train", "model_final", "prob"],
    "value": ["work", "prob", "test_idx", "p_hold_cal", "drivers"],
    "validate": ["work", "selected", "excluded", "split_kind", "test_idx", "p_hold_cal"],
    "build": ["work", "prob", "tiers", "acct", "drivers", "score_type", "threshold", "best", "excluded"],
}
STAGE_TIMEOUT = {"discover": 600, "models": 1800, "explain": 600, "hypotheses": 600}
DEFAULT_TIMEOUT = 600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _all_stages():
    from .churn import STAGES
    from .discover import stage_discover

    return [("discover", "Discover & link your data", stage_discover, False)] + list(STAGES)


def initial_stages(flow_key: str) -> list[dict]:
    def _s(key, title):
        return {"key": key, "title": title, "status": "pending", "summary": None, "logs": [],
                "started_at": None, "finished_at": None, "seconds": None}

    if flow_key == "forecasting":
        from .forecast import STAGES as FS
        return [_s(k, t) for k, t, _fn, _needs in FS]
    else:
        # All other flows (churn, revenue_growth, pricing, efficiency_cost) run the same full pipeline.
        stages = [_s(k, t) for k, t, _fn, _needs in _all_stages()]
        stages.append(_s("report", "Write summary"))
        return stages


def unique_table_names(datasets) -> dict[int, str]:
    """dataset id -> table name, disambiguating duplicate names."""
    seen: dict[str, int] = {}
    out: dict[int, str] = {}
    for d in datasets:
        n = d.name
        if n in seen:
            n = f"{n} ({d.id})"
        seen[n] = d.id
        out[d.id] = n
    return out


def _refresh_working_dataset(db, run, params: dict, roles: dict, art: dict, drop: set) -> None:
    """Save (or update) the cleaned modelling table as a hidden dataset so the AutoEDA pages can show it.
    Never fatal: the flow itself doesn't depend on it."""
    try:
        from .working import build_working_frame, upsert_working_dataset

        frame = build_working_frame(art["work"], roles, art["excluded"], art["uni"], drop)
        wid = upsert_working_dataset(db, run, frame, params.get("working_dataset_id"))
        params["working_dataset_id"] = wid
        run.params_json = json.dumps(params)
        db.add(run)
        db.commit()
    except Exception:
        logger.exception("DS flow %s: could not save the working dataset", run.id)
        db.rollback()


def _generate_ai_hypotheses(run, wid: int) -> str:
    """Run the real Hypotheses generator (same as the Hypotheses page's "Generate with AI") on the working table, so the
    embedded Hypotheses page is already filled in. Synchronous and never fatal."""
    try:
        from ..routers.hypotheses import _run_generate_bg

        _run_generate_bg(run.workspace_id, wid, 6, run.created_by)
        return "AI hypotheses generated"
    except Exception as e:
        logger.exception("DS flow %s: AI hypotheses failed", run.id)
        return f"AI hypotheses unavailable ({str(e)[:80]})"


def _warm_pages(run, wid: int) -> str:
    """Compute, once and cached, every analysis the embedded AutoEDA pages show (profile, quality, missing, correlations,
    outliers, analysis, feature importance, time series) by calling the same endpoints the pages call. After this the
    pages open instantly. Sequential on purpose (memory). Never fatal."""
    import os

    import httpx

    from ..auth import create_access_token
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        from ..models.user import User

        user = db.query(User).filter(User.id == run.created_by).first()
        if user is None:
            return "skipped"
        token = create_access_token({"sub": str(user.id)})
    finally:
        db.close()
    headers = {"Authorization": f"Bearer {token}"}
    bases = [b for b in (os.environ.get("DS_FLOW_SELF_URL"), "http://127.0.0.1:8000", "http://127.0.0.1:8001") if b]
    with httpx.Client(timeout=300) as c:
        base = None
        for b in bases:
            try:
                if c.get(f"{b}/api/v1/health", timeout=5).status_code == 200:
                    base = b
                    break
            except Exception:
                continue
        if base is None:
            return "skipped (server address not found)"
        root = f"{base}/api/v1/datasets/{wid}"
        calls = [
            ("profile", {}), ("quality-score", {}), ("missing", {}), ("correlations", {"method": "pearson", "methods": "numeric"}),
            ("outliers", {"method": "iqr"}), ("analysis", {}), ("feature-importance", {"target": "churned", "methods": "rf"}),
        ]
        ok = 0
        for path, q in calls:
            try:
                ok += c.get(f"{root}/{path}", params=q, headers=headers).status_code == 200
            except Exception:
                logger.exception("DS flow %s: warming %s failed", run.id, path)
        try:
            cols = c.get(f"{root}/timeseries-columns", headers=headers).json()
            rec = cols.get("recommended") or {}
            if rec.get("time_col") and rec.get("value_col"):
                ok += c.get(f"{root}/timeseries", params={"time_col": rec["time_col"], "value_col": rec["value_col"], "methods": "overview"}, headers=headers).status_code == 200
                calls.append(("timeseries", {}))
        except Exception:
            pass
    return f"{ok}/{len(calls)} page analyses ready"


def _run_auto_eda(db, run, params: dict) -> str:
    """Run the real Auto EDA agent (plan -> auto-approve -> execute -> report) on the flow's working table, so the
    flow's EDA step is exactly what the Auto EDA page does. Never fatal for the flow."""
    try:
        from ..models.auto_eda import AutoEdaRun
        from ..routers.auto_eda import _run_in_background

        wid = params.get("working_dataset_id")
        if not wid:
            return "skipped (no working table)"
        er = AutoEdaRun(
            workspace_id=run.workspace_id, dataset_ids_json=json.dumps([wid]), created_by=run.created_by, status="pending", max_items=12,
            business_context=_flow_eda_context(run.flow_key),
        )
        db.add(er)
        db.commit()
        db.refresh(er)
        params["auto_eda_run_id"] = er.id
        run.params_json = json.dumps(params)
        db.add(run)
        db.commit()
        title = f"{_flow_label(run.flow_key)} EDA — {run.dataset_name or 'data'}"
        _run_in_background(run.workspace_id, [wid], er.id, run.created_by, False, title)  # plans, then stops at "planned"
        db.refresh(er)
        if er.status == "planned":  # the approval gate is automatic inside a flow
            er.status = "running"
            db.add(er)
            db.commit()
            _run_in_background(run.workspace_id, [wid], er.id, run.created_by, True)
            db.refresh(er)
        status = er.status
        warm = _warm_pages(run, wid)
        return f"{status} · {warm}"
    except Exception as e:
        logger.exception("DS flow %s: Auto EDA failed", run.id)
        db.rollback()
        return f"failed ({str(e)[:120]})"


def execute_run(run_id: int) -> None:
    from ..database import SessionLocal
    from ..models.dataset import Dataset
    from ..models.ds_flow import DsFlowRun
    from ..routers.eda import _load_df
    from .report import apply_quarantine, build_headline, build_markdown, build_narrative, stage_summary

    db = SessionLocal()
    try:
        run = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
        if run is None:
            return
        stages = json.loads(run.stages_json)
        params = json.loads(run.params_json or "{}")
        results: dict = {}

        def persist(**fields):
            run.stages_json = json.dumps(stages)
            run.results_json = json.dumps(results, default=str)
            for k, v in fields.items():
                setattr(run, k, v)
            db.add(run)
            db.commit()

        # ---- load every dataset we were given --------------------------------------------------
        try:
            datasets = db.query(Dataset).filter(Dataset.id.in_(params.get("dataset_ids") or [run.dataset_id])).all()
            if not datasets:
                raise RuntimeError("No datasets to analyse")
            names = unique_table_names(datasets)
            by_name = {names[d.id]: d for d in datasets}
            run.status = "running"
            persist()
            tables = {}
            for d in datasets:
                tables[names[d.id]] = _load_df(d)
            logger.info("DS flow %s: loaded %s", run_id, {n: t.shape for n, t in tables.items()})
        except Exception as e:
            logger.exception("DS flow %s failed to load data", run_id)
            persist(status="error", error=f"Could not load the datasets: {e}")
            return

        art: dict = {}
        outputs: dict = {}
        by_key = {s["key"]: s for s in stages}
        merged = base_df = None
        roles: dict = {}

        for key, title, fn, needs_df in _all_stages():
            st = by_key[key]
            auto_status = None
            st["status"], st["started_at"] = "running", _now_iso()
            persist()
            t0 = time.time()
            if key == "discover":
                ctx = {"tables": tables, "roles": {}, "params": params, "results": {}, "art": {}}
            else:
                ctx = {
                    "df": (base_df if key == "build" else merged) if needs_df else None,
                    "roles": roles, "params": params, "results": results,
                    "art": {k: art[k] for k in NEEDS.get(key, []) if k in art},
                }
            try:
                res, new_art = run_isolated(fn, ctx, timeout=STAGE_TIMEOUT.get(key, DEFAULT_TIMEOUT))
                results[key] = res
                if key == "discover":
                    merged, base_df = new_art["merged"], new_art["base"]
                    roles = res["roles"]
                    params = {**params, "exclude_columns": res["exclude_columns"], "label_column": res["label"]["column"]}
                    base_ds = by_name.get(res.get("source_table") or res["base_table"])
                    if base_ds is not None:
                        run.dataset_id = base_ds.id
                        run.dataset_name = base_ds.name
                        run.source_filename = os.path.basename(base_ds.file_path or "") or base_ds.name
                        run.title = f"{_flow_label(run.flow_key)} — {base_ds.name}"
                    run.roles_json = json.dumps(roles)
                    tables.clear()  # free memory: the combined table now carries everything
                else:
                    for k, v in new_art.items():
                        (outputs if k in HEAVY_ART else art)[k] = v
                    if key == "leakage":
                        _refresh_working_dataset(db, run, params, roles, art, drop=set())
                    if key == "eda":
                        results[key] = res
                        st["summary"], st["logs"] = stage_summary(key, res)
                        persist()  # the flow's own EDA visuals are visible while Auto EDA runs
                        auto_status = _run_auto_eda(db, run, params)
                    if key == "hypotheses" and params.get("working_dataset_id"):
                        auto_status = _generate_ai_hypotheses(run, params["working_dataset_id"])
                    if key == "understand":
                        merged = None  # the working table now lives in `art`; don't hold a second copy in the server process
                    if key == "models":
                        apply_quarantine(results)
                        q = {x["feature"] for x in (res.get("quarantined") or [])}
                        if q:
                            _refresh_working_dataset(db, run, params, roles, art, drop=q)
                    # free everything no later stage reads
                    order = [k for k, _t, _f, _n in _all_stages()]
                    later = set()
                    for k in order[order.index(key) + 1:]:
                        later.update(NEEDS.get(k, []))
                    for k in [k for k in art if k not in later]:
                        del art[k]
                st["summary"], st["logs"] = stage_summary(key, res)
                if auto_status and key == "eda":
                    st["logs"] = [f"Auto EDA report: {auto_status}"] + st["logs"]
                    st["summary"] = f"{st['summary']} · Auto EDA {auto_status}"
                elif auto_status:
                    st["logs"] = [auto_status] + st["logs"]
                st["status"] = "done"
            except (AnalysisTimeout, AnalysisCrashed) as e:
                st["status"], st["summary"], st["logs"] = "error", str(e), [str(e)]
            except Exception as e:
                logger.exception("DS flow %s stage %s failed", run_id, key)
                msg = str(e) or e.__class__.__name__
                st["status"], st["summary"], st["logs"] = "error", msg[:400], [msg[:800]]
            st["finished_at"], st["seconds"] = _now_iso(), round(time.time() - t0, 1)
            persist()
            if st["status"] == "error" and key in CRITICAL:
                for later in stages:
                    if later["status"] == "pending":
                        later["status"], later["summary"] = "skipped", "Skipped because an earlier critical stage failed"
                persist(status="error", error=st["summary"] if key == "discover" else f"Stage '{title}' failed: {st['summary']}")
                return

        rep = by_key["report"]
        rep["status"], rep["started_at"] = "running", _now_iso()
        persist()
        t0 = time.time()
        try:
            headline = build_headline(results)
            narrative = build_narrative(results, headline)
            markdown = build_markdown(run.title or "Churn analysis", run.dataset_name or "", results, headline, narrative)
            rep["summary"] = f"Summary written ({'AI-worded' if narrative['source'] == 'llm' else 'template'})"
            rep["status"] = "done"
            rep["logs"] = [f"Wording source: {narrative['source']}. Every number comes from the computed results."]
            persist(headline_json=json.dumps(headline, default=str), narrative_json=json.dumps(narrative, default=str), markdown=markdown)
        except Exception as e:
            logger.exception("DS flow %s report failed", run_id)
            rep["status"], rep["summary"] = "error", str(e)[:300]
        rep["finished_at"], rep["seconds"] = _now_iso(), round(time.time() - t0, 1)

        persist(
            status="completed",
            enriched_csv=outputs.get("enriched_csv"), accounts_csv=outputs.get("accounts_csv"),
            dictionary_csv=outputs.get("dictionary_csv"), model_blob=outputs.get("model_bytes"),
        )
    except Exception as e:  # last-resort guard so a run can never sit in "running" forever
        logger.exception("DS flow %s crashed", run_id)
        try:
            db.rollback()
            r = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
            if r and r.status not in ("completed", "error"):
                r.status, r.error = "error", f"Run crashed: {e}"
                db.add(r)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def execute_forecast_run(run_id: int) -> None:
    """Run the fully-implemented forecast pipeline for a ds-flow run."""
    from .forecast import STAGES as FORECAST_STAGES

    from ..database import SessionLocal
    from ..models.dataset import Dataset
    from ..models.ds_flow import DsFlowRun
    from ..routers.eda import _load_df

    FORECAST_CRITICAL = {"detect", "models", "forecast"}

    db = SessionLocal()
    try:
        run = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
        if run is None:
            return
        stages = json.loads(run.stages_json)
        params = json.loads(run.params_json or "{}")
        results: dict = {}

        def persist(**fields):
            run.stages_json = json.dumps(stages)
            run.results_json = json.dumps(results, default=str)
            for k, v in fields.items():
                setattr(run, k, v)
            db.add(run)
            db.commit()

        try:
            datasets = db.query(Dataset).filter(Dataset.id.in_(params.get("dataset_ids") or [run.dataset_id])).all()
            if not datasets:
                raise RuntimeError("No datasets to analyse")
            names = unique_table_names(datasets)
            run.status = "running"
            persist()
            tables = {names[d.id]: _load_df(d) for d in datasets}
            logger.info("DS flow %s (forecast): loaded %s", run_id, {n: t.shape for n, t in tables.items()})
        except Exception as e:
            logger.exception("DS flow %s (forecast) failed to load data", run_id)
            persist(status="error", error=f"Could not load the datasets: {e}")
            return

        art: dict = {}
        by_key = {s["key"]: s for s in stages}

        for key, title, fn, _ in FORECAST_STAGES:
            st = by_key.get(key)
            if st is None:
                continue
            st["status"], st["started_at"] = "running", _now_iso()
            persist()
            t0 = time.time()
            ctx = {"tables": tables, "params": params, "results": results, "art": art} if key == "detect" \
                else {"params": params, "results": results, "art": art}
            try:
                res, new_art = run_isolated(fn, ctx, timeout=STAGE_TIMEOUT.get(key, DEFAULT_TIMEOUT))
                results[key] = res
                art.update(new_art)
                if key == "detect":
                    tables.clear()
                    first_ds = datasets[0]
                    run.dataset_id = first_ds.id
                    run.dataset_name = first_ds.name
                    run.source_filename = os.path.basename(first_ds.file_path or "") or first_ds.name
                    run.title = f"Forecasting — {first_ds.name}"
                    # Same real working-table + Auto EDA + Hypotheses agents churn gets (see working.py / _run_auto_eda /
                    # _generate_ai_hypotheses above) — the prepared series is already a clean table, so no churn-specific
                    # feature-frame building is needed, just save it and point the two generic agents at it.
                    agent_logs = []
                    try:
                        working_frame = new_art.get("working_frame")
                        if working_frame is not None:
                            from .working import upsert_working_dataset

                            wid = upsert_working_dataset(db, run, working_frame, params.get("working_dataset_id"))
                            params["working_dataset_id"] = wid
                            run.params_json = json.dumps(params)
                            db.add(run)
                            db.commit()
                            agent_logs.append(_run_auto_eda(db, run, params))
                            agent_logs.append(_generate_ai_hypotheses(run, wid))
                    except Exception:
                        logger.exception("DS flow %s (forecast): could not prepare the working dataset", run_id)
                        db.rollback()
                    if agent_logs:
                        st["logs"] = agent_logs
                if isinstance(res, dict):
                    if "periods" in res and "frequency" in res:
                        st["summary"] = f"{res.get('periods')} {res.get('frequency')} periods, table '{res.get('table', '')}'"
                    elif "models" in res:
                        best = min(res.get("models", []), key=lambda m: m.get("aic", 0), default={})
                        st["summary"] = f"Best model: {best.get('model', 'n/a')}" if best else f"{key}: done"
                    else:
                        st["summary"] = f"{key}: done"
                else:
                    st["summary"] = f"{key}: done"
                st["status"] = "done"
                if key == "report" and isinstance(res, dict):
                    # Mirrors execute_run: the headline/narrative/markdown live on the run row, not just
                    # inside results_json, so the frontend's Dashboard/Analysis toggle (gated on run.headline) shows up.
                    persist(
                        headline_json=json.dumps(res.get("headline"), default=str),
                        narrative_json=json.dumps(res.get("narrative"), default=str),
                        markdown=res.get("markdown"),
                    )
                if key == "deliver":
                    persist(
                        enriched_csv=art.get("enriched_csv"),
                        accounts_csv=art.get("accounts_csv"),
                        dictionary_csv=art.get("dictionary_csv"),
                    )
            except (AnalysisTimeout, AnalysisCrashed) as e:
                st["status"], st["summary"], st["logs"] = "error", str(e), [str(e)]
            except Exception as e:
                logger.exception("DS flow %s (forecast) stage %s failed", run_id, key)
                msg = str(e) or e.__class__.__name__
                st["status"], st["summary"], st["logs"] = "error", msg[:400], [msg[:800]]
            st["finished_at"], st["seconds"] = _now_iso(), round(time.time() - t0, 1)
            persist()
            if st["status"] == "error" and key in FORECAST_CRITICAL:
                for later in stages:
                    if later["status"] == "pending":
                        later["status"], later["summary"] = "skipped", "Skipped because an earlier critical stage failed"
                persist(status="error", error=f"Stage '{title}' failed: {st['summary']}")
                return

        persist(status="completed")
    except Exception as e:
        logger.exception("DS flow %s (forecast) crashed", run_id)
        try:
            db.rollback()
            r = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
            if r and r.status not in ("completed", "error"):
                r.status, r.error = "error", f"Run crashed: {e}"
                db.add(r)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def execute_scope_run(run_id: int) -> None:
    """For flows without a full pipeline: run the feasibility scan and record a scope report."""
    from ..database import SessionLocal
    from ..models.dataset import Dataset
    from ..models.ds_flow import DsFlowRun
    from ..routers.eda import _load_df
    from .registry import get_flow, scan_dataset

    db = SessionLocal()
    try:
        run = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
        if run is None:
            return
        stages = json.loads(run.stages_json)
        params = json.loads(run.params_json or "{}")

        def persist(**fields):
            run.stages_json = json.dumps(stages)
            for k, v in fields.items():
                setattr(run, k, v)
            db.add(run)
            db.commit()

        by_key = {s["key"]: s for s in stages}
        st = by_key.get("scope")
        if st:
            st["status"], st["started_at"] = "running", _now_iso()
        run.status = "running"
        persist()
        t0 = time.time()

        try:
            datasets = db.query(Dataset).filter(Dataset.id.in_(params.get("dataset_ids") or [run.dataset_id])).all()
            if not datasets:
                raise RuntimeError("No datasets to analyse")
            names = unique_table_names(datasets)
            tables = {names[d.id]: _load_df(d) for d in datasets}
            flow = get_flow(run.flow_key)
            best_scan: dict | None = None
            best_score = -1
            for df in tables.values():
                for f in scan_dataset(df).get("flows", []):
                    if f["key"] == run.flow_key and f["feasibility"]["score"] > best_score:
                        best_score = f["feasibility"]["score"]
                        best_scan = f
            first_ds = datasets[0]
            run.dataset_id = first_ds.id
            run.dataset_name = first_ds.name
            run.source_filename = os.path.basename(first_ds.file_path or "") or first_ds.name
            run.title = f"{flow['category'] if flow else run.flow_key} — {first_ds.name}"
            if st:
                verdict = best_scan["feasibility"]["verdict"] if best_scan else "not_detected"
                signals = best_scan["feasibility"].get("signals", []) if best_scan else []
                missing = best_scan["feasibility"].get("missing", []) if best_scan else []
                summary_parts = [f"Verdict: {verdict} (score {best_score})"]
                if signals:
                    summary_parts.append(f"Signals: {'; '.join(signals[:3])}")
                if missing:
                    summary_parts.append(f"Missing: {'; '.join(missing[:2])}")
                st["summary"] = " · ".join(summary_parts)
                st["logs"] = signals + [f"Missing: {m}" for m in missing]
                st["status"] = "done"
                st["finished_at"], st["seconds"] = _now_iso(), round(time.time() - t0, 1)
        except Exception as e:
            logger.exception("DS flow %s (scope) failed", run_id)
            if st:
                st["status"], st["summary"] = "error", str(e)[:300]
                st["finished_at"], st["seconds"] = _now_iso(), round(time.time() - t0, 1)
            persist(status="error", error=f"Scope analysis failed: {e}")
            return

        persist(status="completed")
    except Exception as e:
        logger.exception("DS flow %s (scope) crashed", run_id)
        try:
            db.rollback()
            r = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
            if r and r.status not in ("completed", "error"):
                r.status, r.error = "error", f"Run crashed: {e}"
                db.add(r)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def execute_flow_run(run_id: int) -> None:
    """Dispatcher: routes to the correct executor based on the run's flow_key."""
    from ..database import SessionLocal
    from ..models.ds_flow import DsFlowRun

    db = SessionLocal()
    try:
        run = db.query(DsFlowRun).filter(DsFlowRun.id == run_id).first()
        flow_key = run.flow_key if run else "churn"
    finally:
        db.close()

    if flow_key == "forecasting":
        execute_forecast_run(run_id)
    else:
        # churn, revenue_growth, pricing, efficiency_cost all run the same full pipeline.
        # The discover stage auto-detects the outcome column for each dataset.
        execute_run(run_id)
