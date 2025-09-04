from __future__ import annotations
import asyncio
import json
import pathlib
import time
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.counter.session import RepSessionManager, Exercise, Side

app = FastAPI()

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

@app.get("/", response_class=FileResponse)
async def home():
    return FileResponse(str(WEB_DIR / "index.html"))

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

MANAGER = RepSessionManager(trainer_mode=True)

def ACTIVE_MANAGER() -> RepSessionManager:
    return MANAGER

# let the manager emit traces to all WS clients
def _sink(ev: dict):
    try:
        asyncio.create_task(broadcast(ev))
    except Exception:
        pass

MANAGER.set_event_sink(_sink)

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
        "exercise": getattr(m.active_cfg, "exercise", None),
        "side": getattr(m.active_cfg, "side", None),
        "web_mode": getattr(m, "web_mode", False),
    })

@app.post("/counter/start")
async def start(exercise: Exercise, side: Side = "both", target_reps: int | None = None):
    m = ACTIVE_MANAGER()
    sid, status = m.start(exercise=exercise, side=side, target_reps=target_reps)
    await broadcast({"type": "trace", "msg": f"router: selected tool → start_rep_counter({exercise}, side={side})"})
    return {"session_id": sid, "status": status}

@app.post("/counter/stop")
async def stop():
    m = ACTIVE_MANAGER()
    sid = m.active_id
    try:
        m.stop(sid)
    except Exception:
        pass
    return JSONResponse({"stopped": True, "session_id": sid})

@app.websocket("/ws/angles")
async def ws_angles(ws: WebSocket):
    await ws.accept()
    WS_CLIENTS.add(ws)
    ACTIVE_MANAGER().set_web_mode(True)
    await broadcast({"type": "trace", "msg": "ws: client connected"})
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            if data.get("type") != "angle":
                continue
            # ignore low-confidence frames from client
            if data.get("vis") is False:
                continue
            angle = data.get("angle")
            ts    = data.get("ts", time.time())
            side  = data.get("side", "both")
            if angle is None:
                continue
            ACTIVE_MANAGER().push_angle(float(angle), float(ts), str(side))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try: WS_CLIENTS.remove(ws)
        except: pass
        if not WS_CLIENTS:
            ACTIVE_MANAGER().set_web_mode(False)
        await broadcast({"type": "trace", "msg": "ws closed"})

WS_CLIENTS: Set[WebSocket] = set()

async def broadcast(obj: dict):
    dead = []
    for ws in list(WS_CLIENTS):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            dead.append(ws)
    for d in dead:
        try: WS_CLIENTS.remove(d)
        except: pass
