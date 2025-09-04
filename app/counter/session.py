from __future__ import annotations
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Literal, Callable
from app.audio.tts import TTSEngine
from app.common.events import EventType, SessionEvent, RepEvent, PartialEvent
from app.data import db
from app.counter.pipeline import PosePipeline, RepConfig
from app.counter.web_pipeline import WebAnglePipeline



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
        self.web_mode: bool = False           # ← browser is feeding angles?
        self.web_session_id: Optional[str] = None        
        self._event_sink: Optional[Callable[[dict], None]] = None  # ← type hint

    def set_event_sink(self, sink: Callable[[dict], None]):
        self._event_sink = sink

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
        # if self.trainer_mode:
        #     self.tts.say(str(self.count))

        # Speak numbers only if not in web mode (browser will TTS if web)
        if self.trainer_mode and not self.web_mode:
            self.tts.say(str(self.count))

        if self._event_sink:
            try:
                self._event_sink({"type":"rep", "count": self.count})
            except Exception:
                pass

    def _emit_debug(self, ev):
        """
        Accepts either a dict like {"type":"trace","msg": "..."} or any object;
        normalizes and forwards to the sink so it appears in the Trace panel.
        """
        if self._event_sink is None:
            return
        try:
            if isinstance(ev, dict):
                payload = ev
            else:
                payload = {"type": "trace", "msg": str(ev)}
            self._event_sink(payload)
        except Exception:
            pass

    def _on_partial(self, reason: str):
        ts = time.time()
        if self.active_id:
            db.insert_event(self.active_id, ts, self.count, 0, 0.0,
                            self.active_cfg.side if self.active_cfg else "both",
                            f"partial:{reason}")
        if self.trainer_mode and not self.web_mode:
            self.tts.say("partial rep")

        if self._event_sink:
            try:
                self._event_sink({"type":"partial", "reason": reason})
            except Exception:
                pass

        # if self.active_id:
        #     db.insert_event(self.active_id, ts, self.count, 0, 0.0, self.active_cfg.side if self.active_cfg else "both", f"partial:{reason}")
        # if self.trainer_mode:
        #     self.tts.say("partial rep")

    # def start(self, exercise: Exercise, side: Side = "both", target_reps: Optional[int] = None, mode: str = "count"):
    #     # stop existing
    #     if self.active_pipeline is not None:
    #         self.tts.say("stopping current session")
    #         self.stop(self.active_id)
        
    #     sid = str(uuid.uuid4())
    #     self.active_id = sid
    #     self.count = 0
    #     # Per-exercise config (thresholds can be tuned here)
    #     cfg = RepConfig(exercise=exercise, side=side, min_angle=45, max_angle=165, min_rom=35, velocity_eps=8, dwell_ms=150, concentric_angle_down=True)
    #     # Friendlier per-exercise thresholds for front-camera 2D landmarks
    #     if exercise == "bicep_curl":
    #         cfg.min_rom = 22
    #         cfg.velocity_eps = 10
    #         cfg.concentric_angle_down = True
    #     elif exercise == "lateral_raise":
    #         cfg.min_rom = 28
    #         cfg.velocity_eps = 12
    #         cfg.concentric_angle_down = False
    #     elif exercise == "shoulder_press":
    #         cfg.min_rom = 24
    #         cfg.velocity_eps = 10
    #         cfg.concentric_angle_down = False
    #     elif exercise == "bench_press":
    #         cfg.min_rom = 18
    #         cfg.velocity_eps = 10
    #         cfg.concentric_angle_down = False

    #     self.active_cfg = cfg

    #     # persist session
    #     db.insert_session(sid, exercise, side, time.time(), target_reps)

    #     # # pipeline
    #     # pipe = PosePipeline(cfg, on_rep=self._on_rep, on_partial=self._on_partial, show_window=False, on_error=self._on_error, )
    #     # self.active_pipeline = pipe
    #     # pipe.start()

    #     # >>> choose pipeline based on web mode <<<
    #     if self.web_mode:
    #         pipe = WebAnglePipeline(cfg, on_rep=self._on_rep, on_partial=self._on_partial)
    #     else:
    #         # Native Pose pipeline; tolerate older signatures
    #         try:
    #             pipe = PosePipeline(
    #                 cfg,
    #                 on_rep=self._on_rep,
    #                 on_partial=self._on_partial,
    #                 show_window=False,
    #                 on_error=self._on_error,
    #             )
    #         except TypeError:
    #             # fallback for older PosePipeline without show_window/on_error
    #             pipe = PosePipeline(
    #                 cfg,
    #                 on_rep=self._on_rep,
    #                 on_partial=self._on_partial,
    #             )

    #         # pipe = PosePipeline(cfg, on_rep=self._on_rep, on_partial=self._on_partial, show_window=False, on_error=self._on_error)

    #     self.active_pipeline = pipe
    #     # Only PosePipeline is threaded; WebAnglePipeline is passive
    #     if hasattr(pipe, "start"):
    #         pipe.start()

    #     self.tts.say(f"starting counter for {exercise.replace('_', ' ')}")
    #     return sid, f"started {exercise}"

    def start(self, exercise: Exercise, side: Side = "both", target_reps: Optional[int] = None, mode: str = "count"):
        # stop existing session if any
        if self.active_pipeline is not None:
            self.tts.say("stopping current session")
            self.stop(self.active_id)

        sid = str(uuid.uuid4())
        self.active_id = sid
        self.count = 0

        # Build config (your per-exercise tuning preserved)
        cfg = RepConfig(exercise=exercise, side=side, min_angle=45, max_angle=165, min_rom=35, velocity_eps=8, dwell_ms=150, concentric_angle_down=True)
        if exercise == "bicep_curl":
            cfg.min_rom = 22; cfg.velocity_eps = 10; cfg.concentric_angle_down = True
        elif exercise == "lateral_raise":
            cfg.min_rom = 28; cfg.velocity_eps = 12; cfg.concentric_angle_down = False
        elif exercise == "shoulder_press":
            cfg.min_rom = 24; cfg.velocity_eps = 10; cfg.concentric_angle_down = False
        elif exercise == "bench_press":
            cfg.min_rom = 18; cfg.velocity_eps = 10; cfg.concentric_angle_down = False

        self.active_cfg = cfg

        # Persist session
        db.insert_session(sid, exercise, side, time.time(), target_reps)

        # Choose pipeline
        if self.web_mode:
            pipe = WebAnglePipeline(
                cfg,
                on_rep=self._on_rep,
                on_partial=self._on_partial,
                debug_cb=self._emit_debug,     # ← pass debug down to detector
            )
        else:
            # Try PosePipeline with debug if it supports it; fall back gracefully
            try:
                pipe = PosePipeline(
                    cfg,
                    on_rep=self._on_rep,
                    on_partial=self._on_partial,
                    show_window=False,
                    on_error=self._on_error,
                    debug_cb=self._emit_debug,  # ← some versions may not accept this
                )
            except TypeError:
                try:
                    pipe = PosePipeline(
                        cfg,
                        on_rep=self._on_rep,
                        on_partial=self._on_partial,
                        show_window=False,
                        on_error=self._on_error,
                    )
                except TypeError:
                    pipe = PosePipeline(
                        cfg,
                        on_rep=self._on_rep,
                        on_partial=self._on_partial,
                    )

        self.active_pipeline = pipe
        if hasattr(pipe, "start"):
            pipe.start()

        self.tts.say(f"starting counter for {exercise.replace('_', ' ')}")
        # emit an explicit trace line too
        self._emit_debug({"type":"trace","msg":f"session started: {exercise} ({side})"})
        return sid, f"started {exercise}"

    # push_angle stays the same, but rename to match server if needed
    def push_angle(self, angle: float, ts: Optional[float] = None, side: str = "both"):
        if isinstance(self.active_pipeline, WebAnglePipeline):
            self.active_pipeline.push_angle(angle, ts, side)

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

    # def stop(self, session_id: Optional[str] = None) -> FinalSummary:
    #     if self.active_pipeline is not None:
    #         self.active_pipeline.stop()
    #         self.active_pipeline.join(timeout=1.0)
    #     sid = self.active_id or ""
    #     db.stop_session(sid, time.time())
    #     total = self.count
    #     self.active_pipeline = None
    #     self.active_cfg = None
    #     self.active_id = None
    #     self.tts.say("stopping counter")
    #     return FinalSummary(session_id=sid, total_reps=total)

    def stop(self, session_id: Optional[str]):
        # stop active pipeline if any
        if self.active_pipeline is not None:
            # ask pipeline to stop if it supports it
            if hasattr(self.active_pipeline, "stop"):
                try:
                    self.active_pipeline.stop()
                except Exception:
                    pass
            # join only if it's a thread-like object
            if hasattr(self.active_pipeline, "join"):
                try:
                    self.active_pipeline.join(timeout=1.0)
                except Exception:
                    pass

        # mark DB + clear state
        end = time.time()
        try:
            if session_id:
                db.stop_session(session_id, end)
        except Exception:
            pass
        self.active_pipeline = None
        self.active_cfg = None
        self.active_id = None
        if self.trainer_mode:
            try:
                self.tts.say("stopping counter")
            except Exception:
                pass

    def status(self, session_id: Optional[str] = None) -> SessionStatus:
        return SessionStatus(session_id=self.active_id or "", state=("running" if self.active_pipeline else "stopped"), count=self.count)

    def _on_error(self, msg: str):
    # Called from pipeline thread on error
        try:
            self.tts.say("camera error")
        except Exception:
            pass
        # mark session stopped
        sid = self.active_id or ""
        try:
            db.stop_session(sid, time.time())
        except Exception:
            pass
        self.active_pipeline = None
        self.active_cfg = None
        self.active_id = None

        if self._event_sink:
            try:
                self._event_sink({"type":"trace", "msg": f"pipeline error: {msg}"})
            except Exception:
                pass


    # Enable/disable web mode (called by WS on connect/disconnect)
    def set_web_mode(self, active: bool):
        self.web_mode = bool(active)

    # Push one angle sample from the web client
    def push_angle(self, angle: float, ts: Optional[float] = None, side: str = "both"):
        if isinstance(self.active_pipeline, WebAnglePipeline):
            self.active_pipeline.push_angle(angle, ts)



# Global manager factory (so tools can access a singleton cleanly)
_ACTIVE: Optional[RepSessionManager] = None

def ACTIVE_MANAGER() -> RepSessionManager:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = RepSessionManager(trainer_mode=True)
    return _ACTIVE
