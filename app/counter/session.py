from __future__ import annotations
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Literal
from app.audio.tts import TTSEngine
from app.common.events import EventType, SessionEvent, RepEvent, PartialEvent
from app.data import db
from app.counter.pipeline import PosePipeline, RepConfig

Exercise = Literal["bicep_curl", "bench_press", "lateral_raise", "shoulder_press"]
Side = Literal["left", "right", "both"]


@dataclass
class SessionStatus:
    session_id: str
    state: str
    count: int


@dataclass
class FinalSummary:
    session_id: str
    total_reps: int


class RepSessionManager:
    def __init__(self, trainer_mode: bool = True):
        self.trainer_mode = trainer_mode
        self.tts = TTSEngine()
        self.active_id: Optional[str] = None
        self.active_pipeline: Optional[PosePipeline] = None
        self.active_cfg: Optional[RepConfig] = None
        self.count = 0

    def _on_rep(self, metrics: dict):
        self.count += 1
        ts = time.time()
        if self.active_id:
            db.insert_event(self.active_id, ts, self.count, 1, metrics.get("rom_deg", 0.0), self.active_cfg.side if self.active_cfg else "both", "")
            db.insert_rep(
                session_id=self.active_id,
                rep_index=self.count,
                start_ts=ts - (metrics.get("tut_ms_total", 0) / 1000.0),
                end_ts=ts,
                tut_ms_total=metrics.get("tut_ms_total", 0),
                tut_ms_concentric=metrics.get("tut_ms_concentric", 0),
                tut_ms_eccentric=metrics.get("tut_ms_eccentric", 0),
                rom_deg=metrics.get("rom_deg", 0.0),
                peak_ang_vel=metrics.get("peak_ang_vel_deg_s", 0.0),
                avg_ang_vel=metrics.get("avg_ang_vel_deg_s", 0.0),
                side=self.active_cfg.side if self.active_cfg else "both",
            )
        if self.trainer_mode:
            self.tts.say(str(self.count))

    def _on_partial(self, reason: str):
        ts = time.time()
        if self.active_id:
            db.insert_event(self.active_id, ts, self.count, 0, 0.0, self.active_cfg.side if self.active_cfg else "both", f"partial:{reason}")
        if self.trainer_mode:
            self.tts.say("partial rep")

    def start(self, exercise: Exercise, side: Side = "both", target_reps: Optional[int] = None, mode: str = "count"):
        # stop existing
        if self.active_pipeline is not None:
            self.tts.say("stopping current session")
            self.stop(self.active_id)
        sid = str(uuid.uuid4())
        self.active_id = sid
        self.count = 0
        # Per-exercise config (thresholds can be tuned here)
        cfg = RepConfig(exercise=exercise, side=side, min_angle=45, max_angle=165, min_rom=40, velocity_eps=15, dwell_ms=150, concentric_angle_down=True)
        if exercise in ("lateral_raise", "shoulder_press"):
            # For these, concentric typically decreases shoulder angle relative to torso
            cfg.concentric_angle_down = False  # example tweak; refine as needed
        self.active_cfg = cfg

        # persist session
        db.insert_session(sid, exercise, side, time.time(), target_reps)

        # pipeline
        pipe = PosePipeline(cfg, on_rep=self._on_rep, on_partial=self._on_partial)
        self.active_pipeline = pipe
        pipe.start()

        self.tts.say(f"starting counter for {exercise.replace('_', ' ')}")
        return sid, f"started {exercise}"

    def pause(self, session_id: Optional[str] = None) -> str:
        if self.active_pipeline is None:
            return self.active_id or ""
        self.active_pipeline.pause()
        self.tts.say("paused")
        return self.active_id or ""

    def resume(self, session_id: Optional[str] = None) -> str:
        if self.active_pipeline is None:
            return self.active_id or ""
        self.active_pipeline.resume()
        self.tts.say("resuming")
        return self.active_id or ""

    def stop(self, session_id: Optional[str] = None) -> FinalSummary:
        if self.active_pipeline is not None:
            self.active_pipeline.stop()
            self.active_pipeline.join(timeout=1.0)
        sid = self.active_id or ""
        db.stop_session(sid, time.time())
        total = self.count
        self.active_pipeline = None
        self.active_cfg = None
        self.active_id = None
        self.tts.say("stopping counter")
        return FinalSummary(session_id=sid, total_reps=total)

    def status(self, session_id: Optional[str] = None) -> SessionStatus:
        return SessionStatus(session_id=self.active_id or "", state=("running" if self.active_pipeline else "stopped"), count=self.count)


# Global manager factory (so tools can access a singleton cleanly)
_ACTIVE: Optional[RepSessionManager] = None

def ACTIVE_MANAGER() -> RepSessionManager:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = RepSessionManager(trainer_mode=True)
    return _ACTIVE
