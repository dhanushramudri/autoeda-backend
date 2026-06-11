"""
Event bus for real-time notifications (in-memory only).

Usage from async code:
    await event_bus.publish("workspace:42", {"type": "dataset_created", ...})

Usage from sync FastAPI endpoints:
    emit_nowait("workspace:42", {"type": "dataset_deleted", ...})
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("autoeda.eventbus")

_main_loop: Optional[asyncio.AbstractEventLoop] = None

def init(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop

class _InMemoryBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subs[channel].append(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        try:
            self._subs[channel].remove(q)
        except ValueError:
            pass
        if not self._subs.get(channel):
            self._subs.pop(channel, None)

    async def publish(self, channel: str, event: dict) -> None:
        targets = (
            [ch for ch in list(self._subs) if ch.startswith(channel[:-1])]
            if channel.endswith(":*") else [channel]
        )
        for target in targets:
            dead: list[asyncio.Queue] = []
            for q in list(self._subs.get(target, [])):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "EventBus: queue full on channel=%s, dropping slow consumer",
                        target,
                    )
                    dead.append(q)
            for q in dead:
                self.unsubscribe(target, q)

    def drain_all(self) -> None:
        sentinel = {"_sentinel": True}
        for channel, queues in list(self._subs.items()):
            for q in list(queues):
                try:
                    q.put_nowait(sentinel)
                except asyncio.QueueFull:
                    pass
        logger.info("EventBus: drained %d channel(s) for shutdown", len(self._subs))

def _build():
    logger.info("EventBus → in-memory")
    return _InMemoryBus()

_bus_instance = None

def _get_bus():
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = _build()
    return _bus_instance

class _BusProxy:
    def subscribe(self, channel: str) -> asyncio.Queue:
        return _get_bus().subscribe(channel)

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        _get_bus().unsubscribe(channel, q)

    async def publish(self, channel: str, event: dict) -> None:
        await _get_bus().publish(channel, event)

    def drain_all(self) -> None:
        _get_bus().drain_all()

event_bus = _BusProxy()

def emit_nowait(channel: str, event: dict) -> None:
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(event_bus.publish(channel, event), _main_loop)