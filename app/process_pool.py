"""Bounded process pool for CPU/memory-heavy EDA computations.

Running heavy pandas/numpy/statsmodels work in separate OS processes (instead of
directly in a request handler) means:
  - it can't starve the event loop or other requests for the GIL/CPU
  - a worker that OOMs or hangs only fails that one request — the API process
    and every other in-flight request for other users are unaffected
  - the pool size caps how many heavy analyses run at once, so concurrent users
    queue behind the pool instead of all piling onto the box's CPU/RAM together
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock

logger = logging.getLogger("autoeda.process_pool")

MAX_WORKERS = int(os.environ.get("EDA_POOL_WORKERS", max(1, os.cpu_count() or 1)))
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("EDA_POOL_TIMEOUT_SECONDS", 600))

_pool: ProcessPoolExecutor | None = None
_lock = Lock()


class AnalysisTimeout(Exception):
    pass


class AnalysisCrashed(Exception):
    pass


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            logger.info("Started EDA process pool with %d worker(s)", MAX_WORKERS)
        return _pool


def shutdown_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            for proc in list(getattr(_pool, "_processes", {}).values()):
                if proc.is_alive():
                    proc.kill()
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def run_isolated(fn, *args, timeout: float = DEFAULT_TIMEOUT_SECONDS, **kwargs):
    """Run fn(*args, **kwargs) in the shared process pool and block for the result.

    fn and all args/kwargs must be picklable (top-level module functions, DataFrames,
    primitives) — no SQLAlchemy sessions/ORM objects or closures.
    """
    future = _get_pool().submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.error("Analysis exceeded %.0fs — killing stuck worker by restarting the pool", timeout)
        shutdown_pool()
        raise AnalysisTimeout(f"Analysis timed out after {timeout:.0f}s")
    except BrokenExecutor:
        logger.error("EDA process pool crashed (likely OOM) — restarting pool")
        shutdown_pool()
        raise AnalysisCrashed("Analysis worker crashed, likely out of memory")
