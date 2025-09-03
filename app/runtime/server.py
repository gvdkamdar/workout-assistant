# app/runtime/server.py
from __future__ import annotations

import asyncio
import json
import pathlib
import time
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.counter.session import RepSessionManager

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app + static / web
app = FastAPI()

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"

# Serve the 'web' folder (index.html lives here)
app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

@app.get("/", response_class=FileResponse)
async def home():
    return FileResponse(str(WEB_DIR / "index.html"))

@app.get("/favicon.ico")
async def favicon():
    # avoid noisy 404s in logs
    return Response(status_code=204)

# ──────────────────────────────────────────────────────────────────────────────
# Session manager singleton + event wiring
MANAGER = RepSessionManager(trainer_mode=True)

def ACTIVE_MANAGER() -> RepSessionManager:
    return MANAGER

# WS clients to broadcast server → browser events (trace/rep/partial)
WS_CLIENTS: Set[WebSocket] = set()

async def broadcast(obj: dict):
    """Send a JSON event to all connected WS clients."""
    dead: list[WebSocket] = []
    payload = json.dumps(obj)
    for ws in list(WS_CLIENTS):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for d in dead:
        try:
            WS_CLIENTS.remove(d)
        except Exception:
            pass

# Hook the manager so on_rep / on_partial / errors show up in the page Trace panel
def _sink(obj: dict):
    # manager is synchronous; hop to loop to broadcast
    asyncio.create_task(broadcast(obj))

MANAGER.set_event_sink(_sink)

# ──────────────────────────────────────────────────────────────────────────────
# REST API

@app.get("/sessions/current")
async def current():
    m = ACTIVE_MANAGER()
    state = "idle"
    if m.active_id and m.active_pipeline:
        state = "running"
    return JSONResponse({
        "state": state,
        "count": getattr(m, "count", 0),
        "session_id": m.active_id,
        "exercise": getattr(m, "active_cfg", None).exercise if m.active_cfg else None,
        "side": getattr(m, "active_cfg", None).side if m.active_cfg else None,
        "web_mode": getattr(m, "web_mode", False),
    })

@app.post("/counter/start")
async def start(
    exercise: str = Query(..., regex="^(bicep_curl|bench_press|lateral_raise|shoulder_press)$"),
    side: str = Query("both", regex="^(left|right|both)$"),
    target_reps: Optional[int] = Query(None, ge=1, le=500),
):
    m = ACTIVE_MANAGER()
    sid, status = m.start(exercise=exercise, side=side, target_reps=target_reps)
    # Let the UI know what tool/action was routed
    await broadcast({"type": "trace", "msg": f"router: selected tool → start_rep_counter({exercise}, side={side})"})
    return JSONResponse({"session_id": sid, "status": status})

@app.post("/counter/stop")
async def stop():
    m = ACTIVE_MANAGER()
    sid = m.active_id
    try:
        m.stop(sid)
    except Exception:
        pass
    await broadcast({"type": "trace", "msg": "router: stop"})
    return JSONResponse({"stopped": True, "session_id": sid})

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket: browser → server (angles), plus server → browser (events)

@app.websocket("/ws/angles")
async def ws_angles(ws: WebSocket):
    await ws.accept()
    WS_CLIENTS.add(ws)
    # flip web mode on
    ACTIVE_MANAGER().set_web_mode(True)
    await broadcast({"type": "trace", "msg": "ws: client connected"})

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            if data.get("type") == "angle":
                # push the sample into the web pipeline (if active)
                ACTIVE_MANAGER().push_angle(
                    float(data["angle"]),
                    float(data.get("ts", time.time())),
                    str(data.get("side", "both")),
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        # ignore malformed frames; keep loop alive if desired
        pass
    finally:
        # unregister
        try:
            WS_CLIENTS.remove(ws)
        except Exception:
            pass

        # turn web mode off when last client disconnects
        if not WS_CLIENTS:
            ACTIVE_MANAGER().set_web_mode(False)
        await broadcast({"type": "trace", "msg": "ws: client disconnected"})
