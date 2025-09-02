from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.counter.session import RepSessionManager, ACTIVE_MANAGER

# Shared enums
Exercise = Literal["bicep_curl", "bench_press", "lateral_raise", "shoulder_press"]
Side = Literal["left", "right", "both"]
Mode = Literal["count", "timed"]


class StartArgs(BaseModel):
    exercise: Exercise = Field(..., description="Exercise to count")
    side: Optional[Side] = Field("both", description="Body side focus")
    target_reps: Optional[int] = Field(None, description="Optional target reps")
    mode: Mode = Field("count", description="Mode: count or timed")


class SessionArg(BaseModel):
    session_id: Optional[str] = Field(None, description="Explicit session ID; default to active")


@tool("start_rep_counter", args_schema=StartArgs)
def start_rep_counter(exercise: Exercise, side: Optional[Side] = "both", target_reps: Optional[int] = None, mode: Mode = "count") -> str:
    """Start a rep counting session for the given exercise. Returns a session_id and a short status line."""
    mgr: RepSessionManager = ACTIVE_MANAGER()
    sid, status = mgr.start(exercise=exercise, side=side or "both", target_reps=target_reps, mode=mode)
    return f"session_id={sid}; status={status}"


@tool("pause_rep_counter", args_schema=SessionArg)
def pause_rep_counter(session_id: Optional[str] = None) -> str:
    """Pause the current or specified rep session."""
    mgr: RepSessionManager = ACTIVE_MANAGER()
    sid = mgr.pause(session_id)
    return f"paused session_id={sid}"


@tool("resume_rep_counter", args_schema=SessionArg)
def resume_rep_counter(session_id: Optional[str] = None) -> str:
    """Resume the current or specified rep session."""
    mgr: RepSessionManager = ACTIVE_MANAGER()
    sid = mgr.resume(session_id)
    return f"resumed session_id={sid}"


@tool("stop_rep_counter", args_schema=SessionArg)
def stop_rep_counter(session_id: Optional[str] = None) -> str:
    """Stop the current or specified rep session and finalize stats."""
    mgr: RepSessionManager = ACTIVE_MANAGER()
    final = mgr.stop(session_id)
    return f"stopped session_id={final.session_id}; total_reps={final.total_reps}"


@tool("status_rep_counter", args_schema=SessionArg)
def status_rep_counter(session_id: Optional[str] = None) -> str:
    """Return the current count and state for the active or given session."""
    mgr: RepSessionManager = ACTIVE_MANAGER()
    st = mgr.status(session_id)
    return f"session_id={st.session_id}; state={st.state}; reps={st.count}"


TOOLS = [start_rep_counter, pause_rep_counter, resume_rep_counter, stop_rep_counter, status_rep_counter]
