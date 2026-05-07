"""Rule-based natural language query parser. No LLM required."""
import re


def parse_nl_query(query: str, columns: list[str]) -> dict:
    """
    Parse a plain-English query into a structured action.

    Returns: { action: str, params: dict, message: str }
    """
    q = query.lower().strip()
    cols_lower = {c.lower(): c for c in columns}

    # Helper: find first column mentioned in query
    def find_column(text: str) -> str | None:
        for col_lower, col_orig in cols_lower.items():
            if col_lower in text:
                return col_orig
        return None

    def find_two_columns(text: str) -> tuple[str | None, str | None]:
        found = [orig for low, orig in cols_lower.items() if low in text]
        if len(found) >= 2:
            return found[0], found[1]
        return found[0] if found else None, None

    # ── Distribution / histogram ────────────────────────────────────────────
    if any(kw in q for kw in ["distribution", "histogram", "show", "plot", "visualize"]):
        col = find_column(q)
        if col:
            return {
                "action": "navigate",
                "params": {"page": "distributions", "column": col},
                "message": f"Opening distribution chart for '{col}'.",
            }

    # ── Correlation ─────────────────────────────────────────────────────────
    if any(kw in q for kw in ["correlation", "correlate", "relate", "relationship", "between"]):
        col1, col2 = find_two_columns(q)
        params: dict = {"page": "correlations"}
        if col1:
            params["col1"] = col1
        if col2:
            params["col2"] = col2
        return {
            "action": "navigate",
            "params": params,
            "message": "Opening correlation analysis.",
        }

    # ── Missing values ───────────────────────────────────────────────────────
    if any(kw in q for kw in ["missing", "null", "empty", "nan", "na "]):
        return {
            "action": "navigate",
            "params": {"page": "missing"},
            "message": "Opening missing value analysis.",
        }

    # ── Outliers ─────────────────────────────────────────────────────────────
    if any(kw in q for kw in ["outlier", "anomaly", "anomalies", "extreme"]):
        return {
            "action": "navigate",
            "params": {"page": "outliers"},
            "message": "Opening outlier detection.",
        }

    # ── Feature importance ───────────────────────────────────────────────────
    if any(kw in q for kw in ["feature importance", "important features", "importance"]):
        col = find_column(q)
        params = {"page": "feature-importance"}
        if col:
            params["target"] = col
        return {
            "action": "navigate",
            "params": params,
            "message": "Opening feature importance analysis.",
        }

    # ── Time series ──────────────────────────────────────────────────────────
    if any(kw in q for kw in ["time series", "timeseries", "trend", "seasonal", "forecast"]):
        return {
            "action": "navigate",
            "params": {"page": "timeseries"},
            "message": "Opening time series analysis.",
        }

    # ── Text analysis ────────────────────────────────────────────────────────
    if any(kw in q for kw in ["text", "sentiment", "word", "nlp", "language"]):
        col = find_column(q)
        params = {"page": "text"}
        if col:
            params["column"] = col
        return {
            "action": "navigate",
            "params": params,
            "message": "Opening text analysis.",
        }

    # ── Profile ──────────────────────────────────────────────────────────────
    if any(kw in q for kw in ["profile", "statistics", "stats", "column info", "dtypes"]):
        return {
            "action": "navigate",
            "params": {"page": "profile"},
            "message": "Opening column profile.",
        }

    # ── Overview / summarize ─────────────────────────────────────────────────
    if any(kw in q for kw in ["summarize", "overview", "describe", "summary", "quality"]):
        return {
            "action": "navigate",
            "params": {"page": "overview"},
            "message": "Opening dataset overview.",
        }

    # ── DROP column ──────────────────────────────────────────────────────────
    if any(kw in q for kw in ["drop", "remove", "delete column", "exclude"]):
        col = find_column(q)
        if col:
            return {
                "action": "transform",
                "params": {"op": "drop", "column": col},
                "message": f"Queuing drop operation for column '{col}'.",
            }

    # ── FILL / IMPUTE ────────────────────────────────────────────────────────
    if any(kw in q for kw in ["fill", "impute", "replace missing"]):
        col = find_column(q)
        method = "mean"
        if "median" in q:
            method = "median"
        elif "mode" in q:
            method = "mode"
        elif "zero" in q or "0" in q:
            method = "custom"
        if col:
            return {
                "action": "transform",
                "params": {"op": "fill_missing", "column": col, "method": method},
                "message": f"Queuing fill-missing ({method}) for column '{col}'.",
            }
        return {
            "action": "transform",
            "params": {"op": "fill_missing", "method": method},
            "message": f"Fill missing with {method} — please specify a column.",
        }

    # ── FILTER ───────────────────────────────────────────────────────────────
    filter_kws = ["filter", "where", "rows where", "only rows"]
    if any(kw in q for kw in filter_kws):
        col = find_column(q)
        # Try to detect operator + value
        operator = "equals"
        value: str | None = None
        op_map = [
            (r">=\s*([\d.]+)", "gte"), (r"<=\s*([\d.]+)", "lte"),
            (r">\s*([\d.]+)", "gt"),   (r"<\s*([\d.]+)", "lt"),
            (r"!=\s*(.+)", "not_equals"),
        ]
        for pattern, op in op_map:
            m = re.search(pattern, q)
            if m:
                operator = op
                value = m.group(1).strip()
                break
        if not value:
            m = re.search(r"(?:equals?|is|=)\s+(['\"]?)(\S+)\1", q)
            if m:
                value = m.group(2)
        params = {"page": "filter"}
        if col:
            params["column"] = col
        if operator:
            params["operator"] = operator
        if value:
            params["value"] = value
        return {
            "action": "filter",
            "params": params,
            "message": f"Adding filter: {col or '?'} {operator} {value or '?'}",
        }

    # ── DROP columns with high missing ───────────────────────────────────────
    if "more than" in q and "missing" in q and ("drop" in q or "remove" in q):
        m = re.search(r"(\d+)\s*%", q)
        threshold = int(m.group(1)) if m else 50
        return {
            "action": "transform",
            "params": {"op": "drop_high_missing", "threshold": threshold},
            "message": f"Queuing drop of columns with >{threshold}% missing values.",
        }

    # ── Graph / relationship ─────────────────────────────────────────────────
    if any(kw in q for kw in ["graph", "network", "relationship graph"]):
        return {
            "action": "navigate",
            "params": {"page": "graph"},
            "message": "Opening column relationship graph.",
        }

    # ── Pivot ────────────────────────────────────────────────────────────────
    if any(kw in q for kw in ["pivot", "crosstab", "cross tab"]):
        return {
            "action": "navigate",
            "params": {"page": "pivot"},
            "message": "Opening pivot table.",
        }

    # ── Charts ───────────────────────────────────────────────────────────────
    if any(kw in q for kw in ["chart", "scatter", "bar chart", "line chart"]):
        col = find_column(q)
        params = {"page": "charts"}
        if col:
            params["column"] = col
        return {
            "action": "navigate",
            "params": params,
            "message": "Opening chart builder.",
        }

    # ── Fallback ─────────────────────────────────────────────────────────────
    return {
        "action": "unknown",
        "params": {},
        "message": (
            "I couldn't understand that query. Try: "
            "'show distribution of <column>', "
            "'drop column <name>', "
            "'filter where <column> > <value>', or "
            "'summarize this dataset'."
        ),
    }
