from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_STOPPED = "session_stopped"
    REP = "rep"
    PARTIAL = "partial"

@dataclass
class SessionEvent:
    type: EventType
    session_id: str
    exercise: str
    side: str
    ts: float
    count: int = 0

@dataclass
class RepEvent:
    type: EventType
    session_id: str
    ts: float
    rep_index: int
    rep_count: int
    rom_deg: float
    tut_ms_total: int
    tut_ms_concentric: int
    tut_ms_eccentric: int
    peak_ang_vel_deg_s: float
    avg_ang_vel_deg_s: float
    side: str = "both"

@dataclass
class PartialEvent:
    type: EventType
    session_id: str
    ts: float
    reason: str  # e.g., "insufficient_rom", "unstable"
    side: str = "both"
