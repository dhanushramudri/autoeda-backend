import pandas as pd

from .profiler import run_profile


def run_quality_score(df: pd.DataFrame) -> dict:
    profile = run_profile(df)
    issues: list[dict] = []
    suggestions: list[str] = []

    total_cells = profile["total_rows"] * profile["total_columns"]
    total_missing = sum(c["missing_count"] for c in profile["columns"])
    overall_missing_pct = total_missing / max(total_cells, 1) * 100

    # Completeness
    completeness = max(0, int(100 - overall_missing_pct))
    for col in profile["columns"]:
        if col["missing_pct"] > 50:
            issues.append({
                "type": "missing_critical", "column": col["name"],
                "description": f"'{col['name']}' has {col['missing_pct']:.1f}% missing values",
                "severity": "danger",
            })
            suggestions.append(f"Consider dropping column '{col['name']}' (>{col['missing_pct']:.0f}% missing)")
        elif col["missing_pct"] > 20:
            issues.append({
                "type": "missing_moderate", "column": col["name"],
                "description": f"'{col['name']}' has {col['missing_pct']:.1f}% missing values",
                "severity": "warning",
            })
            suggestions.append(f"Impute missing values in '{col['name']}' before modeling")

    # Consistency
    constant_cols = [c for c in profile["columns"] if c["semantic_type"] == "constant"]
    consistency = max(0, int(100 - len(constant_cols) / max(len(profile["columns"]), 1) * 100))
    for col in constant_cols:
        issues.append({
            "type": "constant_column", "column": col["name"],
            "description": f"'{col['name']}' has zero variance (constant value)",
            "severity": "warning",
        })
        suggestions.append(f"Remove constant column '{col['name']}' before modeling")

    # Skewness warnings
    for col in profile["columns"]:
        if col["semantic_type"] == "numeric" and col["skewness"] is not None:
            if abs(col["skewness"]) > 2:
                issues.append({
                    "type": "high_skew", "column": col["name"],
                    "description": f"'{col['name']}' is highly skewed (skewness={col['skewness']:.2f})",
                    "severity": "warning",
                })
                direction = "right" if col["skewness"] > 0 else "left"
                suggestions.append(f"Apply log/sqrt transformation to '{col['name']}' ({direction}-skewed)")

    # Uniqueness
    uniqueness = max(0, int(100 - profile["duplicate_pct"]))
    if profile["duplicate_pct"] > 5:
        issues.append({
            "type": "duplicates", "column": "all",
            "description": f"{profile['duplicate_count']} duplicate rows ({profile['duplicate_pct']:.1f}%)",
            "severity": "warning",
        })
        suggestions.append(f"Remove {profile['duplicate_count']} duplicate rows before analysis")

    # High cardinality
    for col in profile["columns"]:
        if col["semantic_type"] == "id_like":
            issues.append({
                "type": "high_cardinality", "column": col["name"],
                "description": f"'{col['name']}' appears to be an ID column (very high cardinality)",
                "severity": "info",
            })
            suggestions.append(f"Drop ID column '{col['name']}' before ML modeling")

    # Validity
    validity = 85
    for col in profile["columns"]:
        if col["semantic_type"] == "numeric" and col.get("skewness") and abs(col["skewness"]) > 3:
            validity -= 3
    validity = max(0, min(100, validity))

    overall = int(completeness * 0.4 + consistency * 0.2 + uniqueness * 0.2 + validity * 0.2)

    return {
        "overall": overall,
        "completeness": completeness,
        "consistency": consistency,
        "uniqueness": uniqueness,
        "validity": validity,
        "issues": issues[:20],
        "suggestions": list(dict.fromkeys(suggestions))[:10],
    }
