from __future__ import annotations
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.counter.session import ACTIVE_MANAGER

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SayBody(BaseModel):
    text: str

@app.post("/agent/say")
async def say_cmd(body: SayBody):
    # Optional endpoint to send text commands from a simple web UI
    from app.agent.router import build_chain
    chain = build_chain()
    out = chain.invoke(body.text)
    return {"ok": True, "result": out}

@app.post("/counter/start")
async def start(exercise: str, side: str = "both", target_reps: int | None = None):
    sid, status = ACTIVE_MANAGER().start(exercise=exercise, side=side, target_reps=target_reps)
    return {"session_id": sid, "status": status}

@app.post("/counter/pause")
async def pause():
    sid = ACTIVE_MANAGER().pause()
    return {"session_id": sid}

@app.post("/counter/resume")
async def resume():
    sid = ACTIVE_MANAGER().resume()
    return {"session_id": sid}

@app.post("/counter/stop")
async def stop():
    final = ACTIVE_MANAGER().stop()
    return {"session_id": final.session_id, "total_reps": final.total_reps}

@app.get("/sessions/current")
async def current():
    st = ACTIVE_MANAGER().status()
    return st.__dict__

# Minimal live WebSocket (placeholder: push count every 250ms)
@app.websocket("/ws/live")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            st = ACTIVE_MANAGER().status()
            await ws.send_json({"session_id": st.session_id, "state": st.state, "count": st.count})
            await ws.receive_text()  # simple ping/pong; client should send any text periodically
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
