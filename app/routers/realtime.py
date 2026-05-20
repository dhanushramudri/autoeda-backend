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
    """
    from ..core.event_bus import event_bus

    user = _user_from_token(token, db)
    if not user:
        return StreamingResponse(
            iter(["data: {\"type\":\"error\",\"message\":\"Unauthorized\"}\n\n"]),
            status_code=401,
            media_type="text/event-stream",
        )

    channel = f"workspace:{workspace_id}"
    queue = event_bus.subscribe(channel)
    logger.info("SSE connected — user=%s workspace=%s", user.email, workspace_id)

    async def generate():
        try:
            # Handshake
            yield f"data: {json.dumps({'type': 'connected', 'user': user.email})}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    # Never echo an event back to the actor who triggered it
                    if event.get("_actor_id") == user.id:
                        continue
                    # Strip internal field before sending to client
                    payload = {k: v for k, v in event.items() if not k.startswith("_")}
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive comment — prevents proxy / load-balancer timeouts
                    yield ": keepalive\n\n"
        finally:
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
    """
    from ..core.presence import presence_manager

    user = _user_from_token(token, db)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    presence_manager.connect(workspace_id, websocket)

    # Send current workspace snapshot immediately on connect
    snap = presence_manager.snapshot(workspace_id)
    await websocket.send_json({"type": "snapshot", "presence": snap})
    logger.info("WS connected — user=%s workspace=%s", user.email, workspace_id)

    try:
        while True:
            data = await websocket.receive_json()
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
        pass
    except Exception as exc:
        logger.warning("WS error — user=%s: %s", user.email, exc)
    finally:
        presence_manager.disconnect(workspace_id, websocket, user.id)
        snap = presence_manager.snapshot(workspace_id)
        await presence_manager.broadcast(
            workspace_id, {"type": "presence", "presence": snap}
        )
        logger.info("WS disconnected — user=%s", user.email)
