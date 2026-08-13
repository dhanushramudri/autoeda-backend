import difflib
import re
from collections import defaultdict

import pandas as pd

MAX_EXAMPLES = 5
CASING_MAX_UNIQUE = 500  # skip casing-group detection on huge free-text columns
DATE_MIN_PARSE_RATE = 0.7
DATE_MIN_SHAPE_SHARE = 0.02
FUZZY_MAX_UNIQUE = 200  # pairwise comparison is O(u^2) — keep u bounded
FUZZY_MIN_RATIO = 0.84
FUZZY_MIN_LEN = 3  # very short strings produce too many false-positive matches

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _visible_whitespace(s: str) -> str:
    """Wrap in guillemets so leading/trailing whitespace is visible in a diff."""
    return f"«{s}»"


def _date_shape_signature(s: str) -> str:
    """Collapse digit runs to 'D' so '2024-01-15' and '2024-03-09' share a signature
    but '2024-01-15' and '01/15/2024' don't."""
    return re.sub(r"\d+", "D", s.strip())


def _detect_whitespace(col: str, series: pd.Series, n: int) -> dict | None:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    stripped = non_null.str.strip()
    mismatch = stripped != non_null
    affected = int(mismatch.sum())
    if affected == 0:
        return None

    examples = []
    for orig, fixed in zip(non_null[mismatch].head(MAX_EXAMPLES), stripped[mismatch].head(MAX_EXAMPLES)):
        examples.append({"before": _visible_whitespace(orig), "after": fixed})

    return {
        "column": col,
        "issue_type": "whitespace",
        "severity": "low",
        "description": f"{affected} value(s) have leading/trailing whitespace",
        "affected_count": affected,
        "affected_pct": round(affected / max(n, 1) * 100, 2),
        "operation": {"type": "text_clean", "column": col, "strip": True},
        "examples": examples,
    }


def _detect_inconsistent_casing(col: str, series: pd.Series, n: int) -> dict | None:
    non_null = series.dropna().astype(str)
    unique_vals = non_null.unique()
    if not (1 < len(unique_vals) <= CASING_MAX_UNIQUE):
        return None

    groups: dict[str, list[str]] = defaultdict(list)
    for v in unique_vals:
        groups[v.strip().lower()].append(v)

    value_counts = non_null.value_counts()
    mapping: dict[str, str] = {}
    examples = []
    affected = 0

    for variants in groups.values():
        if len(variants) <= 1:
            continue
        canonical = max(variants, key=lambda v: value_counts.get(v, 0))
        for v in variants:
            if v == canonical:
                continue
            mapping[v] = canonical
            affected += int(value_counts.get(v, 0))
            if len(examples) < MAX_EXAMPLES:
                examples.append({"before": v, "after": canonical})

    if not mapping:
        return None

    return {
        "column": col,
        "issue_type": "inconsistent_casing",
        "severity": "medium",
        "description": (
            f"{len(mapping)} value variant(s) collapse to {len(set(mapping.values()))} "
            f"canonical value(s) once case/whitespace differences are ignored — affects {affected} row(s)"
        ),
        "affected_count": affected,
        "affected_pct": round(affected / max(n, 1) * 100, 2),
        "operation": {"type": "map_values", "column": col, "mapping": mapping},
        "examples": examples,
    }


def _normalize_key(v: str) -> str:
    return re.sub(r"\s+", " ", v.strip().lower())


def _detect_fuzzy_duplicates(col: str, series: pd.Series, n: int) -> dict | None:
    """Catches spelling-variant near-duplicates ('Jonh' vs 'John', 'St. Louis' vs
    'St Louis') via edit-distance similarity — not true abbreviations ('NY' vs
    'New York'), which need semantic matching, not string similarity."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    value_counts = non_null.value_counts()

    # Collapse case/whitespace-identical values to one representative first —
    # those are already handled by _detect_inconsistent_casing.
    key_to_values: dict[str, list[str]] = defaultdict(list)
    for v in value_counts.index:
        key_to_values[_normalize_key(v)].append(v)

    representatives: list[str] = []
    rep_count: dict[str, int] = {}
    for variants in key_to_values.values():
        best = max(variants, key=lambda v: value_counts.get(v, 0))
        representatives.append(best)
        rep_count[best] = sum(int(value_counts.get(v, 0)) for v in variants)

    candidates = [r for r in representatives if len(r) >= FUZZY_MIN_LEN]
    if not (1 < len(candidates) <= FUZZY_MAX_UNIQUE):
        return None

    parent = {r: r for r in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    ordered = sorted(candidates, key=len)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if len(b) > len(a) * 1.6 + 2:
                break  # too different in length to plausibly be a typo — prune rest
            if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= FUZZY_MIN_RATIO:
                union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for r in candidates:
        groups[find(r)].append(r)

    mapping: dict[str, str] = {}
    examples = []
    affected = 0
    for members in groups.values():
        if len(members) <= 1:
            continue
        canonical = max(members, key=lambda v: rep_count.get(v, 0))
        for m in members:
            if m == canonical:
                continue
            mapping[m] = canonical
            affected += rep_count.get(m, 0)
            if len(examples) < MAX_EXAMPLES:
                examples.append({"before": m, "after": canonical})

    if not mapping:
        return None

    return {
        "column": col,
        "issue_type": "fuzzy_duplicates",
        "severity": "medium",
        "description": (
            f"{len(mapping)} value(s) look like near-duplicate spelling variants of "
            f"{len(set(mapping.values()))} other value(s) — review before applying, affects {affected} row(s)"
        ),
        "affected_count": affected,
        "affected_pct": round(affected / max(n, 1) * 100, 2),
        "operation": {"type": "map_values", "column": col, "mapping": mapping},
        "examples": examples,
    }


def _detect_mixed_date_format(col: str, series: pd.Series, n: int) -> dict | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return None
    non_null = series.dropna().astype(str)
    if len(non_null) < 5:
        return None

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    parse_rate = float(parsed.notna().mean())
    if parse_rate < DATE_MIN_PARSE_RATE:
        return None

    parseable_mask = parsed.notna()
    shapes = non_null[parseable_mask].map(_date_shape_signature)
    shape_counts = shapes.value_counts()
    significant = shape_counts[shape_counts / max(shape_counts.sum(), 1) >= DATE_MIN_SHAPE_SHARE]
    if len(significant) <= 1:
        return None

    examples = []
    seen_shapes = set()
    for orig, shp, dt in zip(non_null[parseable_mask], shapes, parsed[parseable_mask]):
        if shp in seen_shapes:
            continue
        seen_shapes.add(shp)
        examples.append({"before": orig, "after": str(dt.date())})
        if len(examples) >= MAX_EXAMPLES:
            break

    return {
        "column": col,
        "issue_type": "mixed_date_format",
        "severity": "high",
        "description": (
            f"{len(significant)} different date formats detected in this column "
            f"({round(parse_rate * 100)}% of values parse as dates)"
        ),
        "affected_count": int(len(non_null)),
        "affected_pct": round(parse_rate * 100, 2),
        "operation": {"type": "cast_type", "column": col, "to_type": "datetime"},
        "examples": examples,
    }


def detect_smart_clean_issues(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    if n == 0:
        return []

    suggestions: list[dict] = []
    for col in df.columns:
        series = df[col]
        is_textlike = series.dtype == object or pd.api.types.is_string_dtype(series)
        if not is_textlike:
            continue

        for detector in (
            _detect_whitespace,
            _detect_inconsistent_casing,
            _detect_fuzzy_duplicates,
            _detect_mixed_date_format,
        ):
            try:
                found = detector(col, series, n)
            except Exception:
                found = None
            if found:
                suggestions.append(found)

    suggestions.sort(key=lambda s: (_SEVERITY_RANK.get(s["severity"], 3), -s["affected_count"]))
    return suggestions
