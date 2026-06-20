"""Restricted Python execution for Scout's run_python tool.

Security model — be clear-eyed about what this is and isn't:
  - This is defense-in-depth (restricted builtins + a pattern blocklist +
    process isolation + a short timeout), NOT a hard security boundary.
    Pure-Python restricted-exec sandboxes are well known to be incompletely
    escape-proof against a sufficiently determined attacker (e.g. reaching
    dangerous objects via __class__/__bases__/__subclasses__ chains).
  - The realistic threat model here is an LLM writing code in good faith
    that occasionally needs reining in, not a hostile user directly typing
    exploit code — Scout's caller is already an authenticated workspace
    member who can already run arbitrary SQL via run_sql/run_workspace_sql,
    a comparable-risk capability.
  - Runs inside the existing process-pool isolation (_run_isolated), so a
    crash or runaway loop only takes down one isolated worker, not the API
    process, and is bounded by a short timeout.
"""
import builtins
import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats

_MAX_RESULT_ROWS = 100
_MAX_STDOUT_CHARS = 2000

_FORBIDDEN_SUBSTRINGS = [
    "__import__", "import ", "open(", "exec(", "eval(", "compile(",
    "globals(", "locals(", "vars(", "getattr(", "setattr(", "delattr(",
    "__class__", "__bases__", "__subclasses__", "__globals__", "__builtins__",
    "__dict__", "__getattribute__", "__mro__", "input(", "breakpoint(",
    # pandas/numpy I/O and (de)serialization — file and code-execution vectors.
    "read_csv(", "read_excel(", "read_sql(", "read_parquet(", "read_json(",
    "read_pickle(", "to_pickle(", "read_html(", "HDFStore(", "to_sql(",
    "np.load(", "np.save(",
]

_ALLOWED_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "print", "isinstance", "type",
    "ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError", "Exception",
]
_ALLOWED_BUILTINS: dict = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}
_ALLOWED_BUILTINS.update({"True": True, "False": False, "None": None})


def _check_code_safety(code: str) -> str | None:
    if len(code) > 8000:
        return "Code is too long (max 8000 characters)."
    lowered = code.lower()
    for pattern in _FORBIDDEN_SUBSTRINGS:
        if pattern.lower() in lowered:
            return f"Code contains a disallowed pattern: {pattern!r}"
    return None


def _to_jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.head(_MAX_RESULT_ROWS).fillna("").to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(_MAX_RESULT_ROWS).fillna("").to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()[:_MAX_RESULT_ROWS]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in list(value.items())[:_MAX_RESULT_ROWS]}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value[:_MAX_RESULT_ROWS]]
    return value


def exec_sandboxed(df: pd.DataFrame, code: str) -> dict:
    """Top-level (picklable) entry point — must stay a plain module function
    since it's submitted to a ProcessPoolExecutor by _run_isolated."""
    safety_error = _check_code_safety(code)
    if safety_error:
        return {"error": safety_error}

    sandbox_globals = {
        "__builtins__": _ALLOWED_BUILTINS,
        "pd": pd, "np": np, "stats": stats, "df": df,
    }
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, sandbox_globals)  # noqa: S102 — restricted globals above
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "stdout": stdout_buf.getvalue()[-_MAX_STDOUT_CHARS:]}

    if "result" not in sandbox_globals:
        return {
            "error": "Code ran but didn't set a `result` variable.",
            "stdout": stdout_buf.getvalue()[-_MAX_STDOUT_CHARS:],
        }

    try:
        result = _to_jsonable(sandbox_globals["result"])
    except Exception as e:
        return {"error": f"Result isn't serializable: {e}", "stdout": stdout_buf.getvalue()[-_MAX_STDOUT_CHARS:]}

    return {"result": result, "stdout": stdout_buf.getvalue()[-_MAX_STDOUT_CHARS:]}
