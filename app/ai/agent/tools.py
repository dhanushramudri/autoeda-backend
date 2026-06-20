"""Scout's tool layer.

Every tool here is a thin wrapper around EDA/SQL logic that already exists
in the routers — Scout doesn't reimplement analysis, it orchestrates it.
Each tool returns the *full* result (so the frontend can render it with the
existing chart components); context-window trimming for the LLM happens in
orchestrator.py, not here.
"""
import json
from typing import Any

from sqlalchemy.orm import Session

from ...dataset_access import dataset_visibility_filter
from ...models.data_quality_rule import DataQualityRule
from ...models.dataset import Dataset
from ...models.scout import ScoutJoinFact
from ...models.user import User
from ...routers.eda import _get_authorized_dataset, _load_df, _run_isolated
from ...cache import get_cached_result, store_result


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_datasets",
        "description": "List every dataset in the current workspace, with row/column counts and status. Use this first when the user asks about \"the data\" generically, or about what tables/datasets exist.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_dataset_schema",
        "description": "Get the column names, data types, and semantic types (numeric/categorical/datetime/text) for a dataset.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "integer", "description": "The dataset's id"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_profile",
        "description": "Get summary statistics for a dataset: row/column counts, duplicate rows, and per-column stats (missing %, mean/median/std for numeric columns, top values for categorical columns).",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "integer"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_missing",
        "description": "Get a missing-values breakdown for a dataset: which columns have missing data, how much, and imputation suggestions.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "integer"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_correlations",
        "description": "Get correlations between columns in a dataset (numeric-numeric via Pearson/Spearman, categorical-categorical via Cramer's V). Use this to answer questions about relationships between variables.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "method": {"type": "string", "enum": ["pearson", "spearman"], "description": "Correlation method for numeric columns. Default pearson."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_outliers",
        "description": "Detect outliers in a dataset's numeric columns. Use isolation_forest for a holistic multivariate view across all columns at once, or iqr/zscore for a single column.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "method": {"type": "string", "enum": ["iqr", "zscore", "isolation_forest"]},
                "column": {"type": "string", "description": "Required for iqr/zscore; omit for isolation_forest."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_feature_importance",
        "description": "Find which columns are most predictive of a target column (e.g. \"what drives revenue?\"). Requires a target column name.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "target": {"type": "string", "description": "The column to predict/explain"},
            },
            "required": ["dataset_id", "target"],
        },
    },
    {
        "name": "get_quality_score",
        "description": "Get an overall data-quality score (completeness, consistency, uniqueness) for a dataset plus flagged issues.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "integer"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "run_sql",
        "description": "Run a read-only SQL SELECT query against a single dataset (registered as table `df`) for precise aggregations, filtering, or grouping the other tools don't cover. Only SELECT statements are allowed.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "sql": {"type": "string", "description": "A SELECT query, e.g. SELECT category, COUNT(*) FROM df GROUP BY category"},
            },
            "required": ["dataset_id", "sql"],
        },
    },
    {
        "name": "get_distribution",
        "description": "Get the distribution (histogram, KDE, box-plot stats, normality test) for one column. Use this to answer questions about a column's spread, shape, or skew.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "column": {"type": "string"},
            },
            "required": ["dataset_id", "column"],
        },
    },
    {
        "name": "get_text_analysis",
        "description": "Analyze a text column: word frequency, sentiment, length stats. Use for free-text columns (reviews, comments, descriptions).",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "column": {"type": "string"},
            },
            "required": ["dataset_id", "column"],
        },
    },
    {
        "name": "get_timeseries",
        "description": "Analyze a value column over a datetime column: trend, seasonality, change points. Use for \"how has X changed over time\" questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "time_col": {"type": "string", "description": "The datetime column"},
                "value_col": {"type": "string", "description": "The numeric column to analyze over time"},
            },
            "required": ["dataset_id", "time_col", "value_col"],
        },
    },
    {
        "name": "search_columns",
        "description": "Search column names across every dataset in the workspace. Use this when the user mentions a column but you don't know which dataset it's in.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Column name or partial name to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_workspace_sql",
        "description": "Run a read-only SQL SELECT across MULTIPLE datasets in the workspace at once — use this to join/compare/combine datasets (e.g. \"how are these tables related\"). Reference datasets by their slug from list_datasets (e.g. `electricity_data`), not by id. Only SELECT statements are allowed.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A SELECT query referencing one or more dataset slugs, e.g. SELECT a.*, b.* FROM dataset_a a JOIN dataset_b b ON a.id = b.id"},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "run_workspace_python",
        "description": "Run Python (pandas as pd, numpy as np, scipy.stats as stats) over the result of a cross-dataset SQL query — use this for a correlation, statistical test, or custom calculation that spans MULTIPLE datasets (run_python only sees one dataset at a time; this tool is the cross-dataset equivalent). The SQL result is preloaded as `df`. Assign your answer to `result`.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A SELECT query (using dataset slugs from list_datasets) whose result becomes `df`."},
                "code": {"type": "string", "description": "Python source run against `df`. Must set `result = ...`."},
            },
            "required": ["sql", "code"],
        },
    },
    {
        "name": "get_known_relationships",
        "description": "Recall dataset relationships discovered in past conversations in this workspace (e.g. join keys between two datasets). Call this early — it can save you from re-discovering a relationship via trial-and-error SQL.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remember_relationship",
        "description": "Save a confirmed relationship between two datasets for future conversations to reuse (e.g. after successfully joining them with run_workspace_sql). Only call this once you've actually verified the relationship works.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_a_id": {"type": "integer"},
                "dataset_b_id": {"type": "integer"},
                "description": {"type": "string", "description": "Plain description of the relationship, e.g. \"orders.customer_id joins to customers.id\""},
            },
            "required": ["dataset_a_id", "dataset_b_id", "description"],
        },
    },
    {
        "name": "get_shap_explanations",
        "description": "Compute SHAP values for a target column — explains *how* each feature pushes individual predictions up or down, not just overall importance ranking. Slower than get_feature_importance; use it when the user specifically wants per-feature impact direction/magnitude, not just a ranking.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "target": {"type": "string"},
            },
            "required": ["dataset_id", "target"],
        },
    },
    {
        "name": "evaluate_quality_rules",
        "description": "Run this dataset's saved data-quality rules (not_null, range, regex, unique, allowed_values) and return pass/fail rates with sample violations. Returns an empty list if no rules are configured yet.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "integer"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "add_quality_rule",
        "description": "Add a new data-quality rule for this dataset based on something you found (e.g. a column that should never be null). Adds to existing rules without removing any.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "rule_type": {"type": "string", "enum": ["not_null", "range", "regex", "unique", "allowed_values"]},
                "column_name": {"type": "string", "description": "Column the rule applies to (omit only for dataset-wide rules, if any)."},
                "params": {
                    "type": "object",
                    "description": "Rule-specific params: range -> {min, max}; regex -> {pattern}; allowed_values -> {values: [...]}; not_null/unique -> {}",
                    "properties": {},
                },
            },
            "required": ["dataset_id", "rule_type"],
        },
    },
    {
        "name": "save_chart",
        "description": "Save a chart (visible later in this dataset's Chart Builder) so the user can revisit a finding. Use after identifying a relationship/distribution worth keeping.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "name": {"type": "string"},
                "chart_type": {"type": "string", "enum": ["bar", "line", "area", "scatter", "pie"]},
                "x_col": {"type": "string"},
                "y_col": {"type": "string"},
                "color_col": {"type": "string", "description": "Optional grouping/color column"},
            },
            "required": ["dataset_id", "name", "chart_type", "x_col", "y_col"],
        },
    },
    {
        "name": "create_segment",
        "description": "Save a named, reusable row filter (segment) for this dataset based on a pattern you've found (e.g. \"high-value customers\" = revenue > 1000).",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "name": {"type": "string"},
                "filters": {
                    "type": "array",
                    "description": "List of {column, operator, value} filter conditions, ANDed together. operator is one of: ==, !=, >, >=, <, <=, contains, is_null, is_not_null",
                    "items": {"type": "object", "properties": {}},
                },
            },
            "required": ["dataset_id", "name", "filters"],
        },
    },
    {
        "name": "preview_transform",
        "description": "Preview the effect of cleaning/transform operations WITHOUT saving anything — use to show the user what a fix would look like before they apply it in Transform Studio. Supports: drop, rename, fill_missing, cast_type, drop_high_missing, normalize, standardize, log_transform, one_hot_encode, label_encode, filter_rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "operations": {
                    "type": "array",
                    "description": "List of {operation, column, params} steps applied in order.",
                    "items": {"type": "object", "properties": {}},
                },
            },
            "required": ["dataset_id", "operations"],
        },
    },
    {
        "name": "run_statistical_test",
        "description": "Run a rigorous statistical hypothesis test instead of eyeballing numbers. ttest_ind/mannwhitney/ks_2samp compare `column` between two specific values of `group_column` (set group_a/group_b). anova compares `column` across ALL groups of `group_column`. chi2 tests independence between `column` and `group_column` (both categorical). shapiro tests `column` alone for normality.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "test": {"type": "string", "enum": ["ttest_ind", "anova", "mannwhitney", "ks_2samp", "chi2", "shapiro"]},
                "column": {"type": "string"},
                "group_column": {"type": "string", "description": "Required for all tests except shapiro"},
                "group_a": {"type": "string", "description": "Required for ttest_ind/mannwhitney/ks_2samp"},
                "group_b": {"type": "string", "description": "Required for ttest_ind/mannwhitney/ks_2samp"},
            },
            "required": ["dataset_id", "test", "column"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute Python (pandas as pd, numpy as np, scipy.stats as stats; dataset preloaded as `df`) for analysis the other tools don't cover — custom aggregations, multi-step calculations, ad-hoc logic. Assign your final answer to a variable named `result` (a number, string, list, dict, or small DataFrame/Series). Runs in a restricted, isolated sandbox: no file/network/import access, ~20s time limit. Prefer the dedicated tools when one already covers the question — use this for genuine gaps.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "integer"},
                "code": {"type": "string", "description": "Python source. Must set `result = ...`."},
            },
            "required": ["dataset_id", "code"],
        },
    },
]


def _dataset(dataset_id: int, db: Session, user: User) -> Dataset:
    return _get_authorized_dataset(int(dataset_id), user, db)


def _list_datasets(workspace_id: int, db: Session) -> dict:
    from ...routers.warehouse import _slugify

    rows = (
        db.query(Dataset)
        .filter(dataset_visibility_filter(db, workspace_id))
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return {
        "datasets": [
            {
                "id": d.id, "name": d.name, "status": d.status,
                "row_count": d.row_count, "column_count": d.column_count,
                "source_type": d.source_type,
                "slug": _slugify(d.name),  # use this name when calling run_workspace_sql
            }
            for d in rows
        ]
    }


def _get_dataset_schema(dataset_id: int, db: Session, user: User) -> dict:
    profile = _get_profile(dataset_id, db, user)
    return {
        "columns": [
            {"name": c["name"], "dtype": c["dtype"], "semantic_type": c["semantic_type"]}
            for c in profile["columns"]
        ]
    }


def _get_profile(dataset_id: int, db: Session, user: User) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "profile"}
    cached = get_cached_result(db, dataset_id, "profile", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.profiler import run_profile
    result = _run_isolated(run_profile, df)
    store_result(db, dataset_id, "profile", cache_key, result, ds.content_hash or "")
    return result


def _get_missing(dataset_id: int, db: Session, user: User) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "missing"}
    cached = get_cached_result(db, dataset_id, "missing", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.missing import run_missing_analysis
    result = _run_isolated(run_missing_analysis, df)
    store_result(db, dataset_id, "missing", cache_key, result, ds.content_hash or "")
    return result


def _get_correlations(dataset_id: int, db: Session, user: User, method: str = "pearson") -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "correlations", "method": method, "v": 2}
    cached = get_cached_result(db, dataset_id, "correlations", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.correlations import run_correlations
    result = _run_isolated(run_correlations, df, method)
    store_result(db, dataset_id, "correlations", cache_key, result, ds.content_hash or "")
    return result


def _get_outliers(dataset_id: int, db: Session, user: User, method: str = "iqr", column: str | None = None) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "outliers", "method": method, "column": column}
    cached = get_cached_result(db, dataset_id, "outliers", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.outliers import run_outlier_detection
    result = _run_isolated(run_outlier_detection, df, method, column)
    store_result(db, dataset_id, "outliers", cache_key, result, ds.content_hash or "")
    return result


def _get_feature_importance(dataset_id: int, db: Session, user: User, target: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    methods_list = ["rf", "metadata"]
    cache_key = {"type": "feature_importance", "target": target, "methods": sorted(methods_list)}
    cached = get_cached_result(db, dataset_id, "feature_importance", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.feature_importance import run_feature_importance
    result = _run_isolated(run_feature_importance, df, target, methods=methods_list, timeout=240)
    store_result(db, dataset_id, "feature_importance", cache_key, result, ds.content_hash or "")
    return result


def _get_quality_score(dataset_id: int, db: Session, user: User) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "quality_score"}
    cached = get_cached_result(db, dataset_id, "quality_score", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.quality_score import run_quality_score
    result = _run_isolated(run_quality_score, df)
    store_result(db, dataset_id, "quality_score", cache_key, result, ds.content_hash or "")
    return result


def _run_sql(dataset_id: int, db: Session, user: User, sql: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return {"error": "Only SELECT statements are allowed."}

    from ...routers.sql_editor import _load_duckdb, _load_dataset_df
    duckdb = _load_duckdb()
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", _load_dataset_df(str(ds.id), db, user))
        limited = stripped
        if "LIMIT" not in limited.upper():
            limited = f"SELECT * FROM ({limited}) __q LIMIT 200"
        result = con.execute(limited)
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
        return {"columns": columns, "rows": [list(r) for r in rows], "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        con.close()


def _get_distribution(dataset_id: int, db: Session, user: User, column: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "distributions", "column": column}
    cached = get_cached_result(db, dataset_id, "distributions", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.distributions import run_distribution
    result = _run_isolated(run_distribution, df, column)
    store_result(db, dataset_id, "distributions", cache_key, result, ds.content_hash or "")
    return result


def _get_text_analysis(dataset_id: int, db: Session, user: User, column: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "text", "column": column}
    cached = get_cached_result(db, dataset_id, "text", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.text_analysis import run_text_analysis
    result = _run_isolated(run_text_analysis, df, column)
    store_result(db, dataset_id, "text", cache_key, result, ds.content_hash or "")
    return result


def _get_timeseries(dataset_id: int, db: Session, user: User, time_col: str, value_col: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    cache_key = {"type": "timeseries", "time_col": time_col, "value_col": value_col}
    cached = get_cached_result(db, dataset_id, "timeseries", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.timeseries import run_timeseries
    result = _run_isolated(run_timeseries, df, time_col, value_col)
    store_result(db, dataset_id, "timeseries", cache_key, result, ds.content_hash or "")
    return result


def _search_columns(workspace_id: int, db: Session, query: str) -> dict:
    from ...routers.warehouse import _get_ready_datasets, _columns_from_schema_info

    q = query.strip().lower()
    matches = []
    for ds in _get_ready_datasets(workspace_id, db, load_data=False):
        for col in _columns_from_schema_info(ds):
            if q in col["name"].lower():
                matches.append({"dataset_id": ds.id, "dataset_name": ds.name, "column": col["name"], "type": col["type"]})
    return {"matches": matches[:30]}


def _run_workspace_sql(workspace_id: int, db: Session, user: User, sql: str) -> dict:
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return {"error": "Only SELECT statements are allowed."}

    from ...routers.warehouse import execute_warehouse_sql, WarehouseExecuteRequest
    result = execute_warehouse_sql(str(workspace_id), WarehouseExecuteRequest(sql=stripped, limit=200), db, user)
    return result


def _run_workspace_python(workspace_id: int, db: Session, user: User, sql: str, code: str) -> dict:
    sql_result = _run_workspace_sql(workspace_id, db, user, sql)
    if "error" in sql_result:
        return sql_result
    columns = sql_result.get("columns") or []
    rows = sql_result.get("rows") or []
    if not columns:
        return {"error": "Query returned no columns to build a dataframe from."}

    import pandas as pd
    df = pd.DataFrame(rows, columns=columns)

    from .sandbox import exec_sandboxed
    return _run_isolated(exec_sandboxed, df, code, timeout=20)


def _get_known_relationships(workspace_id: int, db: Session) -> dict:
    facts = db.query(ScoutJoinFact).filter(ScoutJoinFact.workspace_id == workspace_id).order_by(ScoutJoinFact.created_at.desc()).all()
    dataset_ids = {f.dataset_a_id for f in facts} | {f.dataset_b_id for f in facts}
    names = {d.id: d.name for d in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()} if dataset_ids else {}
    return {
        "relationships": [
            {
                "dataset_a": names.get(f.dataset_a_id, f.dataset_a_id),
                "dataset_b": names.get(f.dataset_b_id, f.dataset_b_id),
                "description": f.description,
            }
            for f in facts
        ]
    }


def _remember_relationship(workspace_id: int, db: Session, user: User, dataset_a_id: int, dataset_b_id: int, description: str) -> dict:
    # Validate both datasets actually belong to this workspace before storing.
    _dataset(dataset_a_id, db, user)
    _dataset(dataset_b_id, db, user)
    fact = ScoutJoinFact(
        workspace_id=workspace_id, dataset_a_id=dataset_a_id, dataset_b_id=dataset_b_id,
        description=description, created_by=user.id,
    )
    db.add(fact)
    db.commit()
    return {"remembered": True}


def _get_shap_explanations(dataset_id: int, db: Session, user: User, target: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    methods_list = ["shap", "metadata"]
    cache_key = {"type": "feature_importance", "target": target, "methods": sorted(methods_list)}
    cached = get_cached_result(db, dataset_id, "feature_importance", cache_key, ds.content_hash or "")
    if cached:
        return cached
    df = _load_df(ds)
    from ...eda.feature_importance import run_feature_importance
    # Shorter timeout than the default tool — SHAP is opt-in precisely because
    # it's slow; fail fast rather than tying up a worker for the full 240s.
    result = _run_isolated(run_feature_importance, df, target, methods=methods_list, timeout=90)
    store_result(db, dataset_id, "feature_importance", cache_key, result, ds.content_hash or "")
    return result


def _evaluate_quality_rules(dataset_id: int, db: Session, user: User) -> dict:
    from ...routers.extra import get_rule_results
    return get_rule_results(dataset_id, db, user)


def _add_quality_rule(dataset_id: int, db: Session, user: User, rule_type: str, column_name: str | None, params: dict) -> dict:
    _dataset(dataset_id, db, user)  # access check
    rule = DataQualityRule(
        dataset_id=dataset_id, column_name=column_name,
        rule_type=rule_type, params_json=json.dumps(params or {}),
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return {"added": True, "rule_id": rule.id}


def _save_chart(dataset_id: int, db: Session, user: User, name: str, chart_type: str, x_col: str, y_col: str, color_col: str | None = None) -> dict:
    from ...routers.extra import save_chart as _persist_chart, SavedChartRequest
    body = SavedChartRequest(
        name=name, chart_type=chart_type,
        config={"xCol": x_col, "yCol": y_col, "colorCol": color_col, "chartType": chart_type},
    )
    return _persist_chart(dataset_id, body, db, user)


def _create_segment(dataset_id: int, db: Session, user: User, name: str, filters: list[dict]) -> dict:
    from ...routers.extra import create_segment as _persist_segment, SegmentRequest
    return _persist_segment(dataset_id, SegmentRequest(name=name, filters=filters), db, user)


def _preview_transform(dataset_id: int, db: Session, user: User, operations: list[dict]) -> dict:
    ds = _dataset(dataset_id, db, user)
    df = _load_df(ds)
    from ...routers.extra import _apply_pipeline
    result_df = _apply_pipeline(df, operations)
    return {
        "row_count": len(result_df),
        "column_count": len(result_df.columns),
        "preview": result_df.head(20).fillna("").to_dict(orient="records"),
    }


def _run_statistical_test(
    dataset_id: int, db: Session, user: User, test: str, column: str,
    group_column: str | None = None, group_a: str | None = None, group_b: str | None = None,
) -> dict:
    ds = _dataset(dataset_id, db, user)
    df = _load_df(ds)
    from ...eda.stats_tests import run_statistical_test
    return _run_isolated(run_statistical_test, df, test, column, group_column, group_a, group_b, timeout=60)


def _run_python(dataset_id: int, db: Session, user: User, code: str) -> dict:
    ds = _dataset(dataset_id, db, user)
    df = _load_df(ds)
    from .sandbox import exec_sandboxed
    return _run_isolated(exec_sandboxed, df, code, timeout=20)


def execute_tool(name: str, arguments: dict, *, workspace_id: int, db: Session, user: User) -> dict:
    """Dispatch a tool call by name. Always returns a dict — errors are
    returned as {"error": ...} rather than raised, so the agent loop can
    keep going and let the LLM react to the failure."""
    try:
        if name == "list_datasets":
            return _list_datasets(workspace_id, db)
        if name == "get_dataset_schema":
            return _get_dataset_schema(int(arguments["dataset_id"]), db, user)
        if name == "get_profile":
            return _get_profile(int(arguments["dataset_id"]), db, user)
        if name == "get_missing":
            return _get_missing(int(arguments["dataset_id"]), db, user)
        if name == "get_correlations":
            return _get_correlations(int(arguments["dataset_id"]), db, user, arguments.get("method", "pearson"))
        if name == "get_outliers":
            return _get_outliers(int(arguments["dataset_id"]), db, user, arguments.get("method", "iqr"), arguments.get("column"))
        if name == "get_feature_importance":
            return _get_feature_importance(int(arguments["dataset_id"]), db, user, arguments["target"])
        if name == "get_quality_score":
            return _get_quality_score(int(arguments["dataset_id"]), db, user)
        if name == "run_sql":
            return _run_sql(int(arguments["dataset_id"]), db, user, arguments["sql"])
        if name == "get_distribution":
            return _get_distribution(int(arguments["dataset_id"]), db, user, arguments["column"])
        if name == "get_text_analysis":
            return _get_text_analysis(int(arguments["dataset_id"]), db, user, arguments["column"])
        if name == "get_timeseries":
            return _get_timeseries(int(arguments["dataset_id"]), db, user, arguments["time_col"], arguments["value_col"])
        if name == "search_columns":
            return _search_columns(workspace_id, db, arguments["query"])
        if name == "run_workspace_sql":
            return _run_workspace_sql(workspace_id, db, user, arguments["sql"])
        if name == "run_workspace_python":
            return _run_workspace_python(workspace_id, db, user, arguments["sql"], arguments["code"])
        if name == "get_known_relationships":
            return _get_known_relationships(workspace_id, db)
        if name == "remember_relationship":
            return _remember_relationship(
                workspace_id, db, user,
                int(arguments["dataset_a_id"]), int(arguments["dataset_b_id"]), arguments["description"],
            )
        if name == "get_shap_explanations":
            return _get_shap_explanations(int(arguments["dataset_id"]), db, user, arguments["target"])
        if name == "evaluate_quality_rules":
            return _evaluate_quality_rules(int(arguments["dataset_id"]), db, user)
        if name == "add_quality_rule":
            return _add_quality_rule(
                int(arguments["dataset_id"]), db, user,
                arguments["rule_type"], arguments.get("column_name"), arguments.get("params", {}),
            )
        if name == "save_chart":
            return _save_chart(
                int(arguments["dataset_id"]), db, user,
                arguments["name"], arguments["chart_type"], arguments["x_col"], arguments["y_col"],
                arguments.get("color_col"),
            )
        if name == "create_segment":
            return _create_segment(int(arguments["dataset_id"]), db, user, arguments["name"], arguments["filters"])
        if name == "preview_transform":
            return _preview_transform(int(arguments["dataset_id"]), db, user, arguments["operations"])
        if name == "run_statistical_test":
            return _run_statistical_test(
                int(arguments["dataset_id"]), db, user, arguments["test"], arguments["column"],
                arguments.get("group_column"), arguments.get("group_a"), arguments.get("group_b"),
            )
        if name == "run_python":
            return _run_python(int(arguments["dataset_id"]), db, user, arguments["code"])
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}
