from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Literal

import cv2
import numpy as np
import mediapipe as mp
from app.counter.pose_core import angle_3pt

Exercise = Literal["bicep_curl", "bench_press", "lateral_raise", "shoulder_press"]
Side = Literal["left", "right", "both"]

@dataclass
class RepConfig:
    exercise: Exercise
    side: Side = "both"
    # Thresholds
    min_angle: float = 45.0   # contracted
    max_angle: float = 165.0  # extended
    min_rom: float = 40.0     # must travel at least this many degrees
    velocity_eps: float = 8.0  # deg/s considered moving
    dwell_ms: int = 150       # minimum dwell to confirm top/bottom
    # Direction: True=angle decreases for concentric (e.g., curls), False=increases
    concentric_angle_down: bool = True


@dataclass
class RepPhase:
    active: bool = False
    start_ts: float = 0.0
    end_ts: float = 0.0

#     def __init__(self, cfg: RepConfig):
#         self.cfg = cfg
#         self.count = 0
#         self.partial = False
#         self.phase_conc = RepPhase(False, 0.0, 0.0)
#         self.phase_ecc = RepPhase(False, 0.0, 0.0)
#         self.last_angle = None
#         self.peak_vel = 0.0
#         self.vel_sum = 0.0
#         self.vel_n = 0
#         self.rep_start_ts = 0.0
#         self.rep_rom = 0.0
#         self.last_top_dwell = 0.0
#         self.last_bottom_dwell = 0.0
#         self.rep_min = 1e9
#         self.rep_max = -1e9
#         self._ema = None            # simple angle smoother (deg)
#         self._ema_alpha = 0.35   

#     def _phase_start(self, conc: bool, t: float):
#         if conc:
#             self.phase_conc = RepPhase(True, t, 0.0)
#         else:
#             self.phase_ecc = RepPhase(True, t, 0.0)

#     def _phase_end(self, conc: bool, t: float) -> float:
#         ph = self.phase_conc if conc else self.phase_ecc
#         if ph.active:
#             ph.active = False
#             ph.end_ts = t
#             return max(0.0, (ph.end_ts - ph.start_ts) * 1000.0)
#         return 0.0

#     def _update_phases(self, angle: float, vel: float, t: float):
#         moving = abs(vel) >= self.cfg.velocity_eps
#         if not moving:
#             return
#         # Determine direction of concentric given config
#         conc_dir = -1 if self.cfg.concentric_angle_down else 1
#         if vel * conc_dir < 0:  # moving concentric
#             if not self.phase_conc.active:
#                 self._phase_start(True, t)
#             if self.phase_ecc.active:
#                 self._phase_end(False, t)
#         else:  # moving eccentric
#             if not self.phase_ecc.active:
#                 self._phase_start(False, t)
#             if self.phase_conc.active:
#                 self._phase_end(True, t)

#     # def step(self, angle: float, t: float) -> Optional[dict]:
#     #     # velocity estimate
#     #     if self.last_angle is None:
#     #         self.last_angle = angle
#     #         self.rep_start_ts = t

#     #         return None
#     #     dt = max(1e-3, t - (self._last_t if hasattr(self, "_last_t") else t))
#     #     vel = (angle - self.last_angle) / dt
#     #     self._last_t = t
#     #     self.last_angle = angle

#     #     self.peak_vel = max(self.peak_vel, abs(vel))
#     #     self.vel_sum += abs(vel)
#     #     self.vel_n += 1
#     #     self._update_phases(angle, vel, t)

#     #     # ROM tracking
#     #     if self.cfg.concentric_angle_down:
#     #         # reps go from high angle -> low angle -> high angle
#     #         top = self.cfg.max_angle
#     #         bottom = self.cfg.min_angle
#     #     else:
#     #         top = self.cfg.min_angle
#     #         bottom = self.cfg.max_angle

#     #     # Detect top/bottom dwell
#     #     if abs(angle - top) < 5:
#     #         if self.last_top_dwell == 0.0:
#     #             self.last_top_dwell = t
#     #     else:
#     #         self.last_top_dwell = 0.0

#     #     if abs(angle - bottom) < 5:
#     #         if self.last_bottom_dwell == 0.0:
#     #             self.last_bottom_dwell = t
#     #     else:
#     #         self.last_bottom_dwell = 0.0

#     #     # Determine if a full rep completed (lenient on long holds)
#     #     # A full rep requires crossing from top to bottom and back (or vice versa),
#     #     # with ROM >= min_rom. Long holds are okay as long as camera session continues.
#     #     # We’ll approximate by checking whether both phases saw non-trivial durations
#     #     # and ROM change exceeded threshold.
#     #     self.rep_rom = max(self.rep_rom, abs(angle - top), abs(angle - bottom))

#     #     conc_ms = (self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0 if (self.phase_conc.end_ts and not self.phase_conc.active) else 0.0
#     #     ecc_ms = (self.phase_ecc.end_ts - self.phase_ecc.start_ts) * 1000.0 if (self.phase_ecc.end_ts and not self.phase_ecc.active) else 0.0

#     #     full_rom = (self.cfg.max_angle - self.cfg.min_angle)
#     #     rom_now = full_rom - abs(angle - (top if self.cfg.concentric_angle_down else bottom))

#     #     # Heuristic: when both phases completed at least once and ROM exceeded min_rom, we count one rep
#     #     if conc_ms > 0 and ecc_ms > 0 and (full_rom >= self.cfg.min_rom + 10):
#     #         # finalize rep
#     #         rep_end = t
#     #         rep_start = self.rep_start_ts
#     #         total_ms = int((rep_end - rep_start) * 1000.0)
#     #         conc_ms_i = int(conc_ms)
#     #         ecc_ms_i = int(ecc_ms)
#     #         avg_vel = (self.vel_sum / max(1, self.vel_n))
#     #         out = {
#     #             "tut_ms_total": total_ms,
#     #             "tut_ms_concentric": conc_ms_i,
#     #             "tut_ms_eccentric": ecc_ms_i,
#     #             "rom_deg": full_rom,
#     #             "peak_ang_vel_deg_s": float(self.peak_vel),
#     #             "avg_ang_vel_deg_s": float(avg_vel),
#     #         }
#     #         # reset for next rep
#     #         self.count += 1
#     #         self.phase_conc = RepPhase(False, 0.0, 0.0)
#     #         self.phase_ecc = RepPhase(False, 0.0, 0.0)
#     #         self.vel_sum = 0.0
#     #         self.vel_n = 0
#     #         self.peak_vel = 0.0
#     #         self.rep_start_ts = t
#     #         self.rep_rom = 0.0
#     #         return out

#     #     # Partial detection: exceeded movement but failed criteria for full rep
#     #     # E.g., ROM moved but never completed both phases or ROM < min_rom after long time
#     #     if (abs(vel) < self.cfg.velocity_eps) and (t - self.rep_start_ts) > 3.5:  # long attempt
#     #         # reset attempt
#     #         self.phase_conc = RepPhase(False, 0.0, 0.0)
#     #         self.phase_ecc = RepPhase(False, 0.0, 0.0)
#     #         self.vel_sum = 0.0
#     #         self.vel_n = 0
#     #         self.peak_vel = 0.0
#     #         self.rep_start_ts = t
#     #         self.rep_rom = 0.0
#     #         return {"partial": True, "reason": "insufficient_rom"}

#     #     return None

#     def step(self, angle: float, t: float) -> Optional[dict]:
#         # EMA smoothing to reduce jitter before velocity
#         if self._ema is None:
#             self._ema = angle
#         else:
#             self._ema = self._ema_alpha * angle + (1 - self._ema_alpha) * self._ema
#         angle_s = self._ema
        
#         # velocity estimate
#         if self.last_angle is None:
#             self.last_angle = angle_s
#             self.rep_start_ts = t
#             # init last timestamp for next dt
#             self._last_t = t
#             # init the per-rep observed range
#             self.rep_min = min(self.rep_min, angle_s)
#             self.rep_max = max(self.rep_max, angle_s)
#             return None

#         dt = max(1e-3, t - (self._last_t if hasattr(self, "_last_t") else t))
#         vel = (angle_s - self.last_angle) / dt
#         self._last_t = t
#         self.last_angle = angle_s

#         # Update per-rep observed range (REAL ROM tracking)
#         self.rep_min = min(self.rep_min, angle_s)
#         self.rep_max = max(self.rep_max, angle_s)
#         current_rom = self.rep_max - self.rep_min

#         self.peak_vel = max(self.peak_vel, abs(vel))
#         self.vel_sum += abs(vel)
#         self.vel_n += 1
#         self._update_phases(angle_s, vel, t)

#         # Determine canonical top/bottom for dwell checks (unchanged)
#         if self.cfg.concentric_angle_down:
#             top = self.cfg.max_angle
#             bottom = self.cfg.min_angle
#         else:
#             top = self.cfg.min_angle
#             bottom = self.cfg.max_angle

#         # Detect top/bottom dwell (unchanged)
#         if abs(angle_s - top) < 5:
#             if self.last_top_dwell == 0.0:
#                 self.last_top_dwell = t
#         else:
#             self.last_top_dwell = 0.0

#         if abs(angle_s - bottom) < 5:
#             if self.last_bottom_dwell == 0.0:
#                 self.last_bottom_dwell = t
#         else:
#             self.last_bottom_dwell = 0.0

#         # Phase durations (ms) once a phase has ended (unchanged logic)
#         conc_ms = (
#             (self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0
#             if (self.phase_conc.end_ts and not self.phase_conc.active) else 0.0
#         )
#         ecc_ms = (
#             (self.phase_ecc.end_ts - self.phase_ecc.start_ts) * 1000.0
#             if (self.phase_ecc.end_ts and not self.phase_ecc.active) else 0.0
#         )

#         # NEW: Use true per-rep ROM to decide a full rep
#         if conc_ms > 0 and ecc_ms > 0 and (current_rom >= self.cfg.min_rom):
#             # finalize rep
#             rep_end = t
#             rep_start = self.rep_start_ts
#             total_ms = int((rep_end - rep_start) * 1000.0)
#             conc_ms_i = int(conc_ms)
#             ecc_ms_i = int(ecc_ms)
#             avg_vel = (self.vel_sum / max(1, self.vel_n))
#             out = {
#                 "tut_ms_total": total_ms,
#                 "tut_ms_concentric": conc_ms_i,
#                 "tut_ms_eccentric": ecc_ms_i,
#                 "rom_deg": float(current_rom),                # report real ROM
#                 "peak_ang_vel_deg_s": float(self.peak_vel),
#                 "avg_ang_vel_deg_s": float(avg_vel),
#             }
#             # reset for next rep
#             self.count += 1
#             self.phase_conc = RepPhase(False, 0.0, 0.0)
#             self.phase_ecc = RepPhase(False, 0.0, 0.0)
#             self.vel_sum = 0.0
#             self.vel_n = 0
#             self.peak_vel = 0.0
#             self.rep_start_ts = t
#             self.rep_rom = 0.0
#             # reset the per-rep ROM trackers
#             self.rep_min = 1e9
#             self.rep_max = -1e9
#             return out

#         # Partial detection: exceeded time without sufficient ROM/phase completion
#         if (abs(vel) < self.cfg.velocity_eps) and (t - self.rep_start_ts) > 3.5:
#             # reset attempt
#             self.phase_conc = RepPhase(False, 0.0, 0.0)
#             self.phase_ecc = RepPhase(False, 0.0, 0.0)
#             self.vel_sum = 0.0
#             self.vel_n = 0
#             self.peak_vel = 0.0
#             self.rep_start_ts = t
#             self.rep_rom = 0.0
#             # also reset per-rep ROM trackers on partial reset
#             self.rep_min = 1e9
#             self.rep_max = -1e9
#             return {"partial": True, "reason": "insufficient_rom"}

#         return None

# class SingleJointRepDetector:
#     """
#     Robust elbow rep detector using a 4-state hysteresis machine:
#       UNKNOWN → AT_TOP ↔ MOVING_DOWN ↔ AT_BOTTOM ↔ MOVING_UP → AT_TOP (rep++)
#     “Top” and “Bottom” are defined from cfg + hysteresis margins.
#     """
#     def __init__(self, cfg: RepConfig, debug_cb=None):
#         self.cfg = cfg
#         self.count = 0
#         self.phase_conc = RepPhase(False, 0.0, 0.0)
#         self.phase_ecc  = RepPhase(False, 0.0, 0.0)

#         self._ema = None
#         self._ema_alpha = 0.35
#         self._last_t = None
#         self._last_angle = None

#         # Per-rep trackers
#         self.rep_start_ts = 0.0
#         self.rep_min = 1e9
#         self.rep_max = -1e9
#         self.vel_sum = 0.0
#         self.vel_n = 0
#         self.peak_vel = 0.0

#         # Dwell trackers
#         self._dwell_top_start = 0.0
#         self._dwell_bot_start = 0.0

#         # State
#         self.state = "UNKNOWN"

#         # Hysteresis margins
#         self._hi_enter_margin = 8.0    # deg
#         self._hi_exit_margin  = 20.0
#         self._lo_enter_margin = 8.0
#         self._lo_exit_margin  = 20.0

#         # Timing gates
#         self._min_phase_ms = 180
#         self._min_rep_ms   = 600
#         self._max_attempt_s = 5.0

#         self.debug_cb = debug_cb

#     def _emit(self, msg):
#         if self.debug_cb:
#             try: self.debug_cb({"type": "trace", "msg": msg})
#             except: pass

#     def _extremes(self):
#         # Define canonical extremes given concentric direction
#         if self.cfg.concentric_angle_down:
#             top_ref = self.cfg.max_angle   # extended
#             bot_ref = self.cfg.min_angle   # contracted
#         else:
#             top_ref = self.cfg.min_angle   # contracted
#             bot_ref = self.cfg.max_angle   # extended
#         hi_enter = top_ref - self._hi_enter_margin if self.cfg.concentric_angle_down else top_ref + self._hi_enter_margin
#         hi_exit  = top_ref - self._hi_exit_margin  if self.cfg.concentric_angle_down else top_ref + self._hi_exit_margin
#         lo_enter = bot_ref + self._lo_enter_margin if self.cfg.concentric_angle_down else bot_ref - self._lo_enter_margin
#         lo_exit  = bot_ref + self._lo_exit_margin  if self.cfg.concentric_angle_down else bot_ref - self._lo_exit_margin
#         return top_ref, bot_ref, hi_enter, hi_exit, lo_enter, lo_exit

#     def step(self, angle: float, t: float) -> Optional[dict]:
#         # Smooth
#         self._ema = angle if self._ema is None else (self._ema_alpha * angle + (1 - self._ema_alpha) * self._ema)
#         a = self._ema

#         # Time/vel
#         if self._last_t is None:
#             self._last_t = t
#             self._last_angle = a
#             return None
#         dt = max(1e-3, t - self._last_t)
#         vel = (a - self._last_angle) / dt
#         self._last_t = t
#         self._last_angle = a

#         # Accumulate simple kinematics
#         self.peak_vel = max(self.peak_vel, abs(vel))
#         self.vel_sum += abs(vel)
#         self.vel_n += 1

#         # Per-rep ROM
#         self.rep_min = min(self.rep_min, a)
#         self.rep_max = max(self.rep_max, a)
#         current_rom = self.rep_max - self.rep_min

#         # Extremes & hysteresis bands
#         top_ref, bot_ref, hi_enter, hi_exit, lo_enter, lo_exit = self._extremes()
#         at_top = (a >= hi_enter) if self.cfg.concentric_angle_down else (a <= hi_enter)
#         at_bot = (a <= lo_enter) if self.cfg.concentric_angle_down else (a >= lo_enter)

#         # Dwell update
#         if at_top:
#             if self._dwell_top_start == 0.0: self._dwell_top_start = t
#         else:
#             self._dwell_top_start = 0.0
#         if at_bot:
#             if self._dwell_bot_start == 0.0: self._dwell_bot_start = t
#         else:
#             self._dwell_bot_start = 0.0

#         dwell_top_ms = (t - self._dwell_top_start) * 1000.0 if self._dwell_top_start else 0.0
#         dwell_bot_ms = (t - self._dwell_bot_start) * 1000.0 if self._dwell_bot_start else 0.0

#         # Velocity gate
#         moving_conc = (vel < -self.cfg.velocity_eps) if self.cfg.concentric_angle_down else (vel > self.cfg.velocity_eps)
#         moving_ecc  = (vel >  self.cfg.velocity_eps) if self.cfg.concentric_angle_down else (vel < -self.cfg.velocity_eps)

#         # STATE MACHINE
#         s = self.state
#         if s == "UNKNOWN":
#             if dwell_top_ms >= self.cfg.dwell_ms:
#                 self.state = "AT_TOP"
#                 self.rep_start_ts = t
#                 self.rep_min = a; self.rep_max = a
#                 self._emit("state→AT_TOP")
#             elif dwell_bot_ms >= self.cfg.dwell_ms:
#                 self.state = "AT_BOTTOM"
#                 self.rep_start_ts = t
#                 self.rep_min = a; self.rep_max = a
#                 self._emit("state→AT_BOTTOM")

#         elif s == "AT_TOP":
#             # leave top band through exit and move conc
#             leave_top = (a <= hi_exit) if self.cfg.concentric_angle_down else (a >= hi_exit)
#             if leave_top and moving_conc:
#                 self.state = "MOVING_DOWN" if self.cfg.concentric_angle_down else "MOVING_UP"
#                 self.phase_conc = RepPhase(True, t, 0.0)
#                 self._emit("state→CONC_START")

#         elif s in ("MOVING_DOWN", "MOVING_UP"):
#             # reached bottom dwell?
#             if dwell_bot_ms >= self.cfg.dwell_ms:
#                 self.phase_conc.active = False
#                 self.phase_conc.end_ts = t
#                 self.state = "AT_BOTTOM"
#                 self._emit("state→AT_BOTTOM")
#             # timeout: if not really moving ROM-wise, don't spam partials—wait up to max_attempt
#             elif (t - self.rep_start_ts) > self._max_attempt_s and current_rom < (self.cfg.min_rom * 0.5):
#                 # soft reset attempt
#                 self.state = "UNKNOWN"
#                 self.phase_conc = RepPhase(False, 0.0, 0.0)
#                 self.vel_sum = 0.0; self.vel_n = 0; self.peak_vel = 0.0
#                 self.rep_start_ts = t
#                 self.rep_min = a; self.rep_max = a
#                 self._emit("reset: idle/noise")

#         elif s == "AT_BOTTOM":
#             leave_bot = (a >= lo_exit) if self.cfg.concentric_angle_down else (a <= lo_exit)
#             if leave_bot and moving_ecc:
#                 self.state = "MOVING_UP" if self.cfg.concentric_angle_down else "MOVING_DOWN"
#                 self.phase_ecc = RepPhase(True, t, 0.0)
#                 self._emit("state→ECC_START")

#         elif s in ("MOVING_UP", "MOVING_DOWN"):
#             if dwell_top_ms >= self.cfg.dwell_ms:
#                 self.phase_ecc.active = False
#                 self.phase_ecc.end_ts = t
#                 # Candidate rep finish
#                 total_ms = int((t - self.rep_start_ts) * 1000.0)
#                 conc_ms = int((self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0) if self.phase_conc.end_ts else 0
#                 ecc_ms  = int((self.phase_ecc.end_ts  - self.phase_ecc.start_ts)  * 1000.0) if self.phase_ecc.end_ts  else 0

#                 # Gates: ROM + phase durations + total duration
#                 if current_rom >= self.cfg.min_rom and conc_ms >= self._min_phase_ms and ecc_ms >= self._min_phase_ms and total_ms >= self._min_rep_ms:
#                     avg_vel = self.vel_sum / max(1, self.vel_n)
#                     out = {
#                         "tut_ms_total": total_ms,
#                         "tut_ms_concentric": conc_ms,
#                         "tut_ms_eccentric": ecc_ms,
#                         "rom_deg": float(current_rom),
#                         "peak_ang_vel_deg_s": float(self.peak_vel),
#                         "avg_ang_vel_deg_s": float(avg_vel),
#                     }
#                     # reset
#                     self.count += 1
#                     self._emit(f"rep++ → {self.count} (ROM={current_rom:.1f}, ms={total_ms})")
#                     self.state = "AT_TOP"
#                     self.phase_conc = RepPhase(False, 0.0, 0.0)
#                     self.phase_ecc  = RepPhase(False, 0.0, 0.0)
#                     self.vel_sum = 0.0; self.vel_n = 0; self.peak_vel = 0.0
#                     self.rep_start_ts = t
#                     self.rep_min = 1e9; self.rep_max = -1e9
#                     return out
#                 else:
#                     # not enough evidence → go back to AT_TOP but do NOT count
#                     self._emit(f"reject rep (ROM={current_rom:.1f}, conc={conc_ms}ms, ecc={ecc_ms}ms, total={total_ms}ms)")
#                     self.state = "AT_TOP"
#                     self.phase_conc = RepPhase(False, 0.0, 0.0)
#                     self.phase_ecc  = RepPhase(False, 0.0, 0.0)
#                     self.vel_sum = 0.0; self.vel_n = 0; self.peak_vel = 0.0
#                     self.rep_start_ts = t
#                     self.rep_min = a; self.rep_max = a

#         return None

# ... keep your imports and RepConfig/RepPhase dataclasses unchanged ...

class SingleJointRepDetector:
    """
    Robust single-joint rep detector with hysteresis and friendlier top/bottom bands.
    Works with browser angles that can exceed cfg.max_angle (e.g., 175–180° elbow).
    """
    def __init__(self, cfg: RepConfig, debug_cb: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self._dbg = debug_cb or (lambda *_: None)

        # Public-ish counters/flags
        self.count = 0

        # Per-rep/stream trackers
        self._ema = None
        self._ema_alpha = 0.35  # smoothing on angles

        self.last_angle = None
        self._last_t = None
        self.peak_vel = 0.0
        self.vel_sum = 0.0
        self.vel_n = 0

        # Phases (durations are derived; we don't need separate RepPhase objects)
        self.phase_conc = RepPhase(False, 0.0, 0.0)
        self.phase_ecc  = RepPhase(False, 0.0, 0.0)

        # Dwell helpers
        self.last_top_dwell = 0.0
        self.last_bottom_dwell = 0.0

        # Per-rep ROM (reset when we *leave top* into concentric)
        self.rep_min = 1e9
        self.rep_max = -1e9
        self.rep_start_ts = 0.0

        # Simple FSM
        self.state = "AT_TOP"   # assume we start extended
        self._dbg("state→AT_TOP")

        # Bands / hysteresis (degrees)
        # Give generous bands for front camera noise & lite pose model.
        self._TOP_BAND    = 14.0   # within this of top counts as "at top"
        self._BOTTOM_BAND = 12.0   # within this of bottom counts as "at bottom"
        self._SETTLE_BAND = 10.0   # when returning to top, allow count without long dwell

    def _enter_state(self, new_state: str):
        if new_state != self.state:
            self.state = new_state
            self._dbg(f"state→{new_state}")

    def _start_phase(self, conc: bool, t: float):
        ph = self.phase_conc if conc else self.phase_ecc
        if not ph.active:
            ph.active = True
            ph.start_ts = t
            ph.end_ts = 0.0

    def _end_phase(self, conc: bool, t: float):
        ph = self.phase_conc if conc else self.phase_ecc
        if ph.active:
            ph.active = False
            ph.end_ts = t
            return max(0.0, (ph.end_ts - ph.start_ts) * 1000.0)  # ms
        return 0.0

    def step(self, angle: float, t: float) -> Optional[dict]:
        # ----- smoothing & clamping -----
        a = max(0.0, min(180.0, float(angle)))
        if self._ema is None:
            self._ema = a
        else:
            self._ema = self._ema_alpha * a + (1 - self._ema_alpha) * self._ema
        ang = self._ema

        # ----- velocity -----
        if self.last_angle is None:
            self.last_angle = ang
            self._last_t = t
            self.rep_start_ts = t
            return None
        dt = max(1e-3, t - (self._last_t or t))
        vel = (ang - self.last_angle) / dt  # deg/s
        self._last_t = t
        self.last_angle = ang

        # Track velocity aggregates
        self.peak_vel = max(self.peak_vel, abs(vel))
        self.vel_sum += abs(vel)
        self.vel_n += 1

        # Determine movement direction expected for concentric
        conc_dir = -1.0 if self.cfg.concentric_angle_down else 1.0
        moving_conc = vel * conc_dir < -self.cfg.velocity_eps   # "definitely into concentric"
        moving_ecc  = vel * conc_dir >  self.cfg.velocity_eps   # "definitely into eccentric"
        slow_near_top = abs(vel) < (self.cfg.velocity_eps * 0.6)

        # ----- bands for top/bottom detection -----
        # NOTE: we don't require exact equality with cfg.max_angle/min_angle.
        top_hit    = ang >= (self.cfg.max_angle - self._TOP_BAND)
        bottom_hit = ang <= (self.cfg.min_angle + self._BOTTOM_BAND)

        # ----- per-rep ROM tracking -----
        # Reset ROM when we LEAVE top into concentric (start of a rep)
        if self.state in ("AT_TOP",) and moving_conc:
            self.rep_min = ang
            self.rep_max = ang
            self.rep_start_ts = t

        # Accumulate ROM while moving
        self.rep_min = min(self.rep_min, ang)
        self.rep_max = max(self.rep_max, ang)
        current_rom = max(0.0, self.rep_max - self.rep_min)

        # ----- phase bookkeeping -----
        if moving_conc:
            if not self.phase_conc.active:
                self._start_phase(True, t)
            if self.phase_ecc.active:
                self._end_phase(False, t)
        elif moving_ecc:
            if not self.phase_ecc.active:
                self._start_phase(False, t)
            if self.phase_conc.active:
                self._end_phase(True, t)

        # Durations (when a phase has ended)
        conc_ms = ((self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0
                   if (self.phase_conc.end_ts and not self.phase_conc.active) else 0.0)
        ecc_ms  = ((self.phase_ecc.end_ts - self.phase_ecc.start_ts) * 1000.0
                   if (self.phase_ecc.end_ts and not self.phase_ecc.active) else 0.0)

        # ----- FSM with hysteresis -----
        if top_hit and slow_near_top:
            self.last_top_dwell = self.last_top_dwell or t
        else:
            self.last_top_dwell = 0.0

        if bottom_hit and abs(vel) < (self.cfg.velocity_eps * 0.8):
            self.last_bottom_dwell = self.last_bottom_dwell or t
        else:
            self.last_bottom_dwell = 0.0

        # State transitions
        if self.state == "AT_TOP":
            if moving_conc:
                self._enter_state("CONC_START")
        elif self.state == "CONC_START":
            if bottom_hit:
                self._enter_state("AT_BOTTOM")
        elif self.state == "AT_BOTTOM":
            if moving_ecc:
                self._enter_state("ECC_START")
        elif self.state == "ECC_START":
            # Two ways to finish the rep:
            # 1) reach top band and slow down (no long dwell required), or
            # 2) short dwell at top (lenient) if available on some reps.
            if top_hit and (slow_near_top or (self.last_top_dwell and (t - self.last_top_dwell) * 1000.0 >= max(80, self.cfg.dwell_ms * 0.5))):
                # Check ROM threshold (friendlier in web mode)
                if current_rom >= self.cfg.min_rom:
                    # finalize rep
                    rep_end = t
                    total_ms = int((rep_end - self.rep_start_ts) * 1000.0)
                    conc_ms_i = int(conc_ms)
                    ecc_ms_i = int(ecc_ms)
                    avg_vel = (self.vel_sum / max(1, self.vel_n))

                    out = {
                        "tut_ms_total": total_ms,
                        "tut_ms_concentric": conc_ms_i,
                        "tut_ms_eccentric": ecc_ms_i,
                        "rom_deg": float(current_rom),
                        "peak_ang_vel_deg_s": float(self.peak_vel),
                        "avg_ang_vel_deg_s": float(avg_vel),
                    }

                    # Reset for next rep
                    self.count += 1
                    self.phase_conc = RepPhase(False, 0.0, 0.0)
                    self.phase_ecc  = RepPhase(False, 0.0, 0.0)
                    self.vel_sum = 0.0
                    self.vel_n = 0
                    self.peak_vel = 0.0
                    self.rep_start_ts = t
                    self.rep_min = 1e9
                    self.rep_max = -1e9

                    self._enter_state("AT_TOP")
                    return out
                else:
                    # Not enough ROM → partial, reset cycle at top
                    self._enter_state("AT_TOP")
                    self.phase_conc = RepPhase(False, 0.0, 0.0)
                    self.phase_ecc  = RepPhase(False, 0.0, 0.0)
                    self.vel_sum = 0.0
                    self.vel_n = 0
                    self.peak_vel = 0.0
                    self.rep_start_ts = t
                    self.rep_min = 1e9
                    self.rep_max = -1e9
                    return {"partial": True, "reason": "insufficient_rom"}

        # Safety: if we’ve been “stuck” too long with tiny movement, consider partial
        if abs(vel) < self.cfg.velocity_eps and (t - self.rep_start_ts) > 4.0:
            self.phase_conc = RepPhase(False, 0.0, 0.0)
            self.phase_ecc  = RepPhase(False, 0.0, 0.0)
            self.vel_sum = 0.0
            self.vel_n = 0
            self.peak_vel = 0.0
            self.rep_start_ts = t
            self.rep_min = 1e9
            self.rep_max = -1e9
            self._enter_state("AT_TOP")  # reset
            return {"partial": True, "reason": "timeout_low_motion"}

        return None


class PosePipeline(threading.Thread):
    def __init__(
            self, 
            cfg: RepConfig, 
            on_rep: Callable[[dict], None], 
            on_partial: Callable[[str], None], 
            show_window: bool = False, 
            on_error: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_rep = on_rep
        self.on_partial = on_partial
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.detector = SingleJointRepDetector(cfg)
        self.cap = None
        self.pose = None

    def run(self):
        mp_pose = mp.solutions.pose

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Webcam not available")

            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            if self.show_window:
                try:
                    cv2.namedWindow("Workout", cv2.WINDOW_NORMAL)
                except Exception:
                    # If window creation fails, fallback to headless
                    self.show_window = False
                
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)

                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.pose.process(image)
                t = time.time()
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    # Use right side by default; if both, average angles (simple heuristic)
                    # Bicep curls -> elbow angle: shoulder (11/12), elbow (13/14), wrist (15/16)
                    if self.cfg.side in ("right", "both"):
                        shoulder = (lm[12].x, lm[12].y)
                        elbow = (lm[14].x, lm[14].y)
                        wrist = (lm[16].x, lm[16].y)
                        angle_r = angle_3pt(shoulder, elbow, wrist)  # normalized later
                    else:
                        angle_r = None

                    if self.cfg.side in ("left", "both"):
                        shoulder = (lm[11].x, lm[11].y)
                        elbow = (lm[13].x, lm[13].y)
                        wrist = (lm[15].x, lm[15].y)
                        angle_l = angle_3pt(shoulder, elbow, wrist)
                    else:
                        angle_l = None

                    angle = 0.0
                    if angle_r is not None and angle_l is not None:
                        angle = (angle_r + angle_l) / 2.0
                    elif angle_r is not None:
                        angle = angle_r
                    else:
                        angle = angle_l or 0.0

                    # Normalize mediapipe angle output if needed (already degrees 0..180 from our function)
                    res_step = self.detector.step(angle=angle, t=t)
                    if res_step is not None:
                        if "partial" in res_step:
                            self.on_partial(res_step.get("reason", "partial"))
                        else:
                            self.on_rep(res_step)

                if self.show_window:
                    try:
                        cv2.putText(frame, f"Count: {self.detector.count}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.imshow("Workout", frame)
                        # macOS: imshow requires waitKey even if we ignore keys
                        _ = cv2.waitKey(1)
                    except Exception as e:
                        # Stop showing rather than crashing
                        self.show_window = False

                # # Draw overlay minimal
                # cv2.putText(frame, f"Count: {self.detector.count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # cv2.imshow("Workout", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        except Exception as e:
            # surface error to manager
            if self.on_error:
                try:
                    self.on_error(str(e))
                except Exception:
                    pass
            else:
                # at least print it once for dev
                print("PosePipeline error:", e)

        
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def stop(self):
        self._stop.set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()
