"""
Event bus for real-time notifications.

Default backend: in-memory asyncio queues (single-process/single-instance).
Set REDIS_URL in .env to enable Redis Pub/Sub (multi-instance / horizontally scaled).

Usage from async code:
    await event_bus.publish("workspace:42", {"type": "dataset_created", ...})

Usage from sync FastAPI endpoints (thread-pool context):
    emit_nowait("workspace:42", {"type": "dataset_deleted", ...})
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("autoeda.eventbus")

# Stored by init() at ASGI startup so sync code can schedule coroutines.
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def init(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop


# ── In-memory backend ─────────────────────────────────────────────────────────

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
        # "workspace:*" broadcasts to every workspace channel
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
        """
        FIX: push a sentinel into every live queue so all SSE generators wake
        up and exit immediately during server shutdown, rather than waiting up
        to KEEPALIVE_TIMEOUT seconds for the next asyncio.wait_for to expire.

        Call this from the ASGI lifespan shutdown hook:
            @asynccontextmanager
            async def lifespan(app):
                yield
                event_bus.drain_all()   
        """
        sentinel = {"_sentinel": True}
        for channel, queues in list(self._subs.items()):
            for q in list(queues):
                try:
                    q.put_nowait(sentinel)
                except asyncio.QueueFull:
                    pass
        logger.info("EventBus: drained %d channel(s) for shutdown", len(self._subs))


# ── Redis backend ─────────────────────────────────────────────────────────────

class _RedisBus:
    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._r = aioredis.from_url(redis_url, decode_responses=True)
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._tasks: dict[str, asyncio.Task] = {}

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subs[channel].append(q)
        if channel not in self._tasks or self._tasks[channel].done():
            self._tasks[channel] = asyncio.create_task(self._listen(channel))
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        try:
            self._subs[channel].remove(q)
        except ValueError:
            pass
        if not self._subs.get(channel):
            self._subs.pop(channel, None)

    async def _listen(self, channel: str) -> None:
        try:
            async with self._r.pubsub() as ps:
                await ps.subscribe(channel)
                async for msg in ps.listen():
                    if msg["type"] != "message":
                        continue
                    event = json.loads(msg["data"])
                    dead: list[asyncio.Queue] = []
                    for q in list(self._subs.get(channel, [])):
                        try:
                            q.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning(
                                "EventBus(Redis): queue full on channel=%s, dropping slow consumer",
                                channel,
                            )
                            dead.append(q)
                    for q in dead:
                        self.unsubscribe(channel, q)
        except Exception as exc:
            logger.warning("Redis listener error on %s: %s", channel, exc)
        finally:
            self._tasks.pop(channel, None)

    async def publish(self, channel: str, event: dict) -> None:
        await self._r.publish(channel, json.dumps(event))

    def drain_all(self) -> None:
        """No-op for Redis backend — listeners are cancelled by the event loop."""
        pass


# ── Singleton ─────────────────────────────────────────────────────────────────

def _build(redis_url: Optional[str]):
    if redis_url:
        logger.info("EventBus → Redis (%s)", redis_url)
        return _RedisBus(redis_url)
    logger.info("EventBus → in-memory")
    return _InMemoryBus()


_bus_instance = None


def _get_bus():
    global _bus_instance
    if _bus_instance is None:
        from ..config import settings
        _bus_instance = _build(getattr(settings, "REDIS_URL", None) or None)
    return _bus_instance


class _BusProxy:
    """Thin proxy so callers can `from .event_bus import event_bus` and get live singleton."""
    def subscribe(self, channel: str) -> asyncio.Queue:
        return _get_bus().subscribe(channel)

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        _get_bus().unsubscribe(channel, q)

    async def publish(self, channel: str, event: dict) -> None:
        await _get_bus().publish(channel, event)

    def drain_all(self) -> None:
        """Wake all live SSE queues with a sentinel so they exit immediately."""
        _get_bus().drain_all()


event_bus = _BusProxy()


def emit_nowait(channel: str, event: dict) -> None:
    """Fire-and-forget publish from a sync (thread-pool) FastAPI endpoint."""
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(event_bus.publish(channel, event), _main_loop)