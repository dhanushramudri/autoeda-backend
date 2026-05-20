"""
In-memory presence manager.

Tracks which users are currently viewing which dataset, per workspace.
Designed to be swapped for a Redis-backed implementation when horizontal
scaling is needed — same interface, different storage layer.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger("autoeda.presence")

# Users inactive longer than this are excluded from presence snapshots.
STALE_AFTER = 35  # seconds — slightly above client heartbeat interval (25s)


@dataclass
class _Entry:
    user_id: int
    email: str
    name: str
    dataset_id: Optional[str] = None
    last_seen: float = field(default_factory=time.monotonic)


class PresenceManager:
    def __init__(self) -> None:
        # workspace_id → {user_id: _Entry}
        self._entries: dict[str, dict[int, _Entry]] = defaultdict(dict)
        # workspace_id → set of live WebSocket connections
        self._sockets: dict[str, set] = defaultdict(set)

    # ── connection lifecycle ──────────────────────────────────────────────────

    def connect(self, workspace_id: str, ws: "WebSocket") -> None:
        self._sockets[workspace_id].add(ws)

    def disconnect(self, workspace_id: str, ws: "WebSocket", user_id: int) -> None:
        self._sockets[workspace_id].discard(ws)
        self._entries[workspace_id].pop(user_id, None)

    # ── state mutations ───────────────────────────────────────────────────────

    def update(
        self,
        workspace_id: str,
        user_id: int,
        email: str,
        name: str,
        dataset_id: Optional[str],
    ) -> None:
        self._entries[workspace_id][user_id] = _Entry(
            user_id=user_id,
            email=email,
            name=name,
            dataset_id=dataset_id,
            last_seen=time.monotonic(),
        )

    def heartbeat(self, workspace_id: str, user_id: int) -> None:
        if entry := self._entries[workspace_id].get(user_id):
            entry.last_seen = time.monotonic()

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self, workspace_id: str) -> dict[str, list[dict]]:
        """Return {dataset_id: [{email, name}, ...]} for all live users."""
        cutoff = time.monotonic() - STALE_AFTER
        result: dict[str, list[dict]] = defaultdict(list)
        for e in self._entries[workspace_id].values():
            if e.dataset_id and e.last_seen >= cutoff:
                result[e.dataset_id].append({"email": e.email, "name": e.name})
        return dict(result)

    # ── broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, workspace_id: str, msg: dict) -> None:
        dead: set = set()
        for ws in set(self._sockets[workspace_id]):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._sockets[workspace_id] -= dead


presence_manager = PresenceManager()
