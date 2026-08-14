"""Dedicated single-worker process pool for TabFM inference.

TabFM's pretrained weights are ~6.5GB and loading + inference is genuinely slow
and memory-heavy on CPU (observed: ~8 min load, ~10 min inference on an 8-row
toy example on a 15.5GB RAM dev box). Two things follow from that:

  1. This must run in its own process, isolated from the shared EDA pool
     (process_pool.py) — a crash here must never take down fast analyses like
     Profile/Correlations, and vice versa.
  2. It must be a *single*-worker pool. The shared EDA pool sizes to CPU count;
     if TabFM jobs could land on multiple of those workers, each one would load
     its own ~6.5GB copy of the model, which would OOM this class of machine
     almost immediately. Pinning to one worker guarantees the model is loaded
     into memory exactly once and reused across requests.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock

logger = logging.getLogger("autoeda.tabfm_pool")

# Observed end-to-end (load + inference) on a tiny toy example was ~18 minutes on
# CPU with tight RAM — default timeout leaves real headroom above that.
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("TABFM_TIMEOUT_SECONDS", 1800))

_pool: ProcessPoolExecutor | None = None
_lock = Lock()


class TabFMTimeout(Exception):
    pass


class TabFMCrashed(Exception):
    pass


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1)
            logger.info("Started dedicated single-worker TabFM process pool")
        return _pool


def shutdown_tabfm_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            for proc in list(getattr(_pool, "_processes", {}).values()):
                if proc.is_alive():
                    proc.kill()
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def run_tabfm_isolated(fn, *args, timeout: float = DEFAULT_TIMEOUT_SECONDS, **kwargs):
    """Run fn(*args, **kwargs) in the dedicated TabFM worker and block for the result.

    fn and all args/kwargs must be picklable — no SQLAlchemy sessions/ORM objects.
    """
    future = _get_pool().submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.error("TabFM job exceeded %.0fs — restarting worker", timeout)
        shutdown_tabfm_pool()
        raise TabFMTimeout(f"TabFM job timed out after {timeout:.0f}s")
    except BrokenExecutor:
        logger.error("TabFM worker crashed (likely OOM) — restarting worker")
        shutdown_tabfm_pool()
        raise TabFMCrashed("TabFM worker crashed, likely out of memory")
