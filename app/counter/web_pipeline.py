# app/counter/web_pipeline.py
from __future__ import annotations
import time
from typing import Callable, Optional

from app.counter.pipeline import RepConfig, SingleJointRepDetector

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
        debug_cb: Optional[Callable[[dict], None]] = None,   # ← NEW
    ):
        self.cfg = cfg
        self.on_rep = on_rep
        self.on_partial = on_partial
        self.debug_cb = debug_cb                              # ← NEW
        # pass debug_cb into the detector so it can emit "state→..." etc.
        self.detector = SingleJointRepDetector(cfg, debug_cb=debug_cb)
        self._running = True

    # keep for API parity with PosePipeline
    def start(self): 
        self._running = True

    def stop(self):
        self._running = False

    def pause(self):
        self._running = False

    def resume(self):
        self._running = True

    def push_angle(self, angle: float, ts: Optional[float] = None, side: str = "both"):
        """Feed one angle sample (deg) at timestamp ts (sec)."""
        if not self._running:
            return
        t = float(ts) if ts is not None else time.time()
        res = self.detector.step(float(angle), t)
        if res is None:
            return
        # Detector returns either a full rep metrics dict, or {"partial": True, ...}
        if "partial" in res:
            if self.debug_cb:
                self.debug_cb({"type": "trace", "msg": f"partial: {res.get('reason','unknown')}"})
            self.on_partial(res.get("reason", "partial"))
        else:
            if self.debug_cb:
                self.debug_cb({"type": "trace", "msg": f"rep++ ({res.get('rom_deg',0):.1f}°)"} )
            self.on_rep(res)




# # app/counter/web_pipeline.py
# from __future__ import annotations
# import time
# from typing import Optional, Literal, Callable
# from dataclasses import dataclass
# from app.counter.pipeline import RepConfig, SingleJointRepDetector

# Side = Literal["left", "right", "both"]

# class WebAnglePipeline:
#     """
#     A minimal 'pipeline' that consumes pre-computed joint angles from the browser.
#     No camera, no threads. Just call push_angle(angle, ts).
#     """
#     def __init__(
#         self,
#         cfg: RepConfig,
#         on_rep: Callable[[dict], None],
#         on_partial: Callable[[str], None],
#         debug_cb: Optional[Callable[[dict], None]] = None,   # ← NEW

#     ):
#         self.cfg = cfg
#         self.debug_cb = debug_cb                              # ← NEW

#         self.detector = SingleJointRepDetector(cfg, debug_cb=debug_cb)
#         self.on_rep = on_rep
        
#         self.on_partial = on_partial
#         self._running = True

#     def push_angle(self, angle: float, ts: Optional[float] = None):
#         if not self._running:
#             return
#         t = ts if ts is not None else time.time()
#         out = self.detector.step(angle=angle, t=t)
#         if not out:
#             return
#         if "partial" in out:
#             self.on_partial(out.get("reason", "partial"))
#         else:
#             self.on_rep(out)

#     # these exist to match the PosePipeline interface used in session manager
#     def stop(self):
#         self._running = False

#     def pause(self):
#         # web client can simply stop sending angles; nothing to do here
#         pass

#     def resume(self):
#         pass

#     def join(self, timeout: float | None = None):
#     # Provided only so Session.stop() can call join() uniformly.
#         return

