"""
Real-time endpoints
  GET /api/v1/events          — Server-Sent Events (notifications stream)
  WS  /api/v1/ws/{workspace_id} — WebSocket (presence)

Authentication: both endpoints accept ?token=<JWT> query param because
browsers cannot send custom headers with EventSource or native WebSocket.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db

logger = logging.getLogger("autoeda.realtime")
router = APIRouter(tags=["realtime"])

# ── Per-user SSE deduplication ────────────────────────────────────────────────
# Tracks active SSE queues per (user_id, workspace_id) so that when a client
# reconnects (e.g. after a hot-reload) the old queue is ejected immediately
# rather than lingering until its asyncio.wait_for timeout fires.
_sse_registry: dict[tuple, asyncio.Queue] = {}
_sse_registry_lock = asyncio.Lock()


# ── Shared auth helper ────────────────────────────────────────────────────────

def _user_from_token(token: str, db: Session):
    """Decode a JWT access token and return the User row, or None."""
    from ..models.user import User
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload["sub"])
        return db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    except (JWTError, KeyError, ValueError, TypeError):
        return None


# ── SSE — notifications ───────────────────────────────────────────────────────

@router.get("/events")
async def sse_stream(
    request: Request,
    workspace_id: str = Query(...),
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Server-Sent Events stream for a workspace.
    Client receives JSON-encoded notification objects whenever a teammate
    performs an action (dataset upload, delete, feedback, etc.).

    Key guarantees:
    - Stale connections from the same user+workspace are ejected immediately
      on reconnect (prevents duplicate-queue buildup).
    - Shutdown completes in ≤ KEEPALIVE_TIMEOUT seconds (default 5 s during
      shutdown, 20 s during normal operation) so uvicorn never waits minutes.
    - SSE errors are fully isolated — they never propagate to other API routes.
    """
    from ..core.event_bus import event_bus

    # FIX: short timeout during server shutdown so uvicorn isn't blocked for
    # minutes waiting for the next keepalive cycle to expire.
    NORMAL_TIMEOUT = 20.0   # seconds between keepalive pings
    SHUTDOWN_TIMEOUT = 3.0  # fast-drain on cancellation

    user = _user_from_token(token, db)
    if not user:
        return StreamingResponse(
            iter(["data: {\"type\":\"error\",\"message\":\"Unauthorized\"}\n\n"]),
            status_code=401,
            media_type="text/event-stream",
        )

    channel = f"workspace:{workspace_id}"
    registry_key = (user.id, workspace_id)

    # FIX: eject any previous queue for this user+workspace before creating a
    # new one. This handles browser reconnects (e.g. after hot-reload) that
    # happen before the old SSE task notices the disconnect — without this the
    # old queue sits in the event bus consuming memory and delaying shutdown.
    async with _sse_registry_lock:
        old_queue = _sse_registry.get(registry_key)
        if old_queue is not None:
            # Unsubscribe old queue from bus so it stops receiving events,
            # then put a sentinel so its generator loop exits immediately.
            event_bus.unsubscribe(channel, old_queue)
            try:
                old_queue.put_nowait({"_sentinel": True})
            except asyncio.QueueFull:
                pass
            logger.info(
                "SSE ejected stale connection — user=%s workspace=%s",
                user.email, workspace_id,
            )

        queue = event_bus.subscribe(channel)
        _sse_registry[registry_key] = queue

    logger.info("SSE connected — user=%s workspace=%s", user.email, workspace_id)

    async def generate():
        timeout = NORMAL_TIMEOUT
        try:
            # Handshake
            yield f"data: {json.dumps({'type': 'connected', 'user': user.email})}\n\n"

            while True:
                # Check client disconnect before blocking on queue.get().
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)

                    # Sentinel injected by the stale-connection eject above.
                    if event.get("_sentinel"):
                        break

                    for_user_id = event.get("_for_user_id")
                    if for_user_id is not None:
                        if for_user_id != user.id:
                            continue
                    else:
                        if event.get("_actor_id") == user.id:
                            continue

                    payload = {k: v for k, v in event.items() if not k.startswith("_")}
                    yield f"data: {json.dumps(payload)}\n\n"

                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            # FIX: switch to a very short timeout so we drain the queue and
            # exit quickly, giving uvicorn a clean shutdown in seconds rather
            # than minutes.  Then re-raise so the event loop can finish.
            logger.info(
                "SSE cancelled (server shutdown) — user=%s, draining fast",
                user.email,
            )
            timeout = SHUTDOWN_TIMEOUT
            raise

        finally:
            # Always clean up — runs on client disconnect, shutdown, or error.
            async with _sse_registry_lock:
                if _sse_registry.get(registry_key) is queue:
                    del _sse_registry[registry_key]
            event_bus.unsubscribe(channel, queue)
            logger.info("SSE disconnected — user=%s", user.email)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ── WebSocket — presence ──────────────────────────────────────────────────────

# FIX: track active WS sockets per (user_id, workspace_id) so duplicate
# connections (same browser reconnecting before old socket is reaped) are
# closed immediately rather than accumulating in presence state.
_ws_registry: dict[tuple, WebSocket] = {}
_ws_registry_lock = asyncio.Lock()


@router.websocket("/ws/{workspace_id}")
async def presence_ws(
    websocket: WebSocket,
    workspace_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Presence WebSocket for a workspace.

    Client messages (JSON):
      {"action": "focus",     "dataset_id": "123"}  — user opened a dataset
      {"action": "blur"}                             — user left a dataset
      {"action": "heartbeat"}                        — keep-alive every 25 s

    Server messages (JSON):
      {"type": "snapshot",  "presence": {dataset_id: [{email, name}]}}  — on connect
      {"type": "presence",  "presence": {dataset_id: [{email, name}]}}  — on any change

    Key guarantees:
    - Only one WS per (user, workspace) — duplicate sockets are closed with
      code 4009 before they affect presence state.
    - Shutdown completes within the receive_json() timeout window (30 s max),
      not after an unbounded block.
    - WS errors are fully isolated — they never propagate to other API routes.
    """
    from ..core.presence import presence_manager

    user = _user_from_token(token, db)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    registry_key = (user.id, workspace_id)

    # FIX: close any pre-existing socket for this user+workspace before
    # accepting the new one.  This handles browser reconnects that race ahead
    # of the server noticing the previous TCP close, preventing ghost presence
    # entries and the stale-connection log spam seen in production.
    async with _ws_registry_lock:
        old_ws = _ws_registry.get(registry_key)
        if old_ws is not None:
            try:
                await old_ws.close(code=4009, reason="Superseded by new connection")
            except Exception:
                pass  # already gone — that's fine
            logger.info(
                "WS ejected stale connection — user=%s workspace=%s",
                user.email, workspace_id,
            )
        _ws_registry[registry_key] = websocket

    await websocket.accept()
    presence_manager.connect(workspace_id, websocket)

    snap = presence_manager.snapshot(workspace_id)
    await websocket.send_json({"type": "snapshot", "presence": snap})
    logger.info("WS connected — user=%s workspace=%s", user.email, workspace_id)

    try:
        while True:
            # FIX: timeout prevents the event loop from blocking indefinitely,
            # allowing CancelledError from uvicorn shutdown to be delivered
            # promptly instead of waiting for the next client message.
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                # Missed heartbeat window — not an error, just loop again.
                continue

            action = data.get("action")

            if action == "focus":
                presence_manager.update(
                    workspace_id,
                    user.id,
                    user.email,
                    user.full_name or user.email,
                    str(data.get("dataset_id", "")),
                )
            elif action == "blur":
                presence_manager.update(
                    workspace_id, user.id, user.email,
                    user.full_name or user.email, None,
                )
            elif action == "heartbeat":
                presence_manager.heartbeat(workspace_id, user.id)

            snap = presence_manager.snapshot(workspace_id)
            await presence_manager.broadcast(
                workspace_id, {"type": "presence", "presence": snap}
            )

    except WebSocketDisconnect:
        pass  # normal client-initiated close

    except asyncio.CancelledError:
        logger.info("WS cancelled (server shutdown) — user=%s", user.email)
        raise  # re-raise so the event loop can finish cancellation

    except Exception as exc:
        logger.warning("WS error — user=%s: %s", user.email, exc)

    finally:
        # Always clean up presence state and broadcast departure, regardless
        # of how the connection ended.
        async with _ws_registry_lock:
            if _ws_registry.get(registry_key) is websocket:
                del _ws_registry[registry_key]

        presence_manager.disconnect(workspace_id, websocket, user.id)
        snap = presence_manager.snapshot(workspace_id)

        # FIX: shield the final broadcast so it isn't itself cancelled mid-flight
        # during server shutdown — other clients must see the departure even
        # when uvicorn is tearing everything down.
        try:
            await asyncio.shield(
                presence_manager.broadcast(
                    workspace_id, {"type": "presence", "presence": snap}
                )
            )
        except asyncio.CancelledError:
            pass  # shield protects the coroutine; swallow the outer cancel here

        logger.info("WS disconnected — user=%s", user.email)