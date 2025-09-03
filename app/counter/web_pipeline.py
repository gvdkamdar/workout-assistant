# app/counter/web_pipeline.py
from __future__ import annotations
import time
from typing import Optional, Literal, Callable
from dataclasses import dataclass
from app.counter.pipeline import RepConfig, SingleJointRepDetector

Side = Literal["left", "right", "both"]

class WebAnglePipeline:
    """
    A minimal 'pipeline' that consumes pre-computed joint angles from the browser.
    No camera, no threads. Just call push_angle(angle, ts).
    """
    def __init__(
        self,
        cfg: RepConfig,
        on_rep: Callable[[dict], None],
        on_partial: Callable[[str], None],
    ):
        self.cfg = cfg
        self.detector = SingleJointRepDetector(cfg)
        self.on_rep = on_rep
        self.on_partial = on_partial
        self._running = True

    def push_angle(self, angle: float, ts: Optional[float] = None):
        if not self._running:
            return
        t = ts if ts is not None else time.time()
        out = self.detector.step(angle=angle, t=t)
        if not out:
            return
        if "partial" in out:
            self.on_partial(out.get("reason", "partial"))
        else:
            self.on_rep(out)

    # these exist to match the PosePipeline interface used in session manager
    def stop(self):
        self._running = False

    def pause(self):
        # web client can simply stop sending angles; nothing to do here
        pass

    def resume(self):
        pass
