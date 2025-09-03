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


# class SingleJointRepDetector:
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

class SingleJointRepDetector:
    """
    Robust single-joint rep detector with:
      • EMA smoothing for elbow angle
      • Adaptive ROM gate (bootstraps from the motion you actually do)
      • Concentric/Eccentric phase timing
      • Lenient partial rule (only when ROM is really tiny for long)
    """
    def __init__(self, cfg: RepConfig):
        self.cfg = cfg
        self.count = 0

        # phases
        self.phase_conc = RepPhase(False, 0.0, 0.0)
        self.phase_ecc  = RepPhase(False, 0.0, 0.0)

        # kinematics
        self._ema = None
        self._ema_alpha = 0.35
        self.last_angle = None
        self._last_t = None
        self.peak_vel = 0.0
        self.vel_sum  = 0.0
        self.vel_n    = 0

        # per-rep tracking
        self.rep_start_ts = 0.0
        self.rep_min = 1e9
        self.rep_max = -1e9

        # adaptive ROM bootstrap (per rep)
        self._boot_start = None
        self._boot_min   = 1e9
        self._boot_max   = -1e9
        self._boot_ms    = 1200.0  # first ~1.2s of motion gives a good ROM estimate

        # misc
        self.last_top_dwell = 0.0
        self.last_bottom_dwell = 0.0

    def _phase_start(self, conc: bool, t: float):
        if conc:
            self.phase_conc = RepPhase(True, t, 0.0)
        else:
            self.phase_ecc  = RepPhase(True, t, 0.0)

    def _phase_end(self, conc: bool, t: float) -> float:
        ph = self.phase_conc if conc else self.phase_ecc
        if ph.active:
            ph.active = False
            ph.end_ts = t
            return max(0.0, (ph.end_ts - ph.start_ts) * 1000.0)
        return 0.0

    def _update_phases(self, angle: float, vel: float, t: float):
        # Hysteresis around velocity_eps to avoid chatter
        eps_on  = self.cfg.velocity_eps * 1.1
        eps_off = self.cfg.velocity_eps * 0.7
        moving = abs(vel) >= eps_on

        conc_dir = -1 if self.cfg.concentric_angle_down else 1  # for curls, conc vel < 0
        if moving:
            if vel * conc_dir < 0:      # concentric
                if not self.phase_conc.active:
                    self._phase_start(True, t)
                if self.phase_ecc.active:
                    self._phase_end(False, t)
            else:                        # eccentric
                if not self.phase_ecc.active:
                    self._phase_start(False, t)
                if self.phase_conc.active:
                    self._phase_end(True, t)
        else:
            # allow phases to end when we slow down sufficiently
            if self.phase_conc.active and abs(vel) < eps_off:
                self._phase_end(True, t)
            if self.phase_ecc.active and abs(vel) < eps_off:
                self._phase_end(False, t)

    def _rom_gate(self, current_rom: float, now: float) -> float:
        """Adaptive ROM threshold per rep."""
        boot_rom = max(0.0, self._boot_max - self._boot_min)
        # 65% of observed boot ROM, but never below the static min_rom
        return max(self.cfg.min_rom, 0.65 * boot_rom)

    def _reset_rep(self, t: float):
        self.phase_conc = RepPhase(False, 0.0, 0.0)
        self.phase_ecc  = RepPhase(False, 0.0, 0.0)
        self.vel_sum = 0.0
        self.vel_n   = 0
        self.peak_vel = 0.0
        self.rep_start_ts = t
        self.rep_min = 1e9
        self.rep_max = -1e9
        self._boot_start = None
        self._boot_min   = 1e9
        self._boot_max   = -1e9

    def step(self, angle: float, t: float) -> Optional[dict]:
        # EMA smoothing
        if self._ema is None:
            self._ema = angle
        else:
            self._ema = self._ema_alpha * angle + (1.0 - self._ema_alpha) * self._ema
        angle_s = self._ema

        # init
        if self.last_angle is None:
            self.last_angle = angle_s
            self._last_t = t
            self.rep_start_ts = t
            self.rep_min = min(self.rep_min, angle_s)
            self.rep_max = max(self.rep_max, angle_s)
            return None

        dt  = max(1e-3, t - self._last_t)
        vel = (angle_s - self.last_angle) / dt
        self._last_t   = t
        self.last_angle = angle_s

        # per-rep extrema
        self.rep_min = min(self.rep_min, angle_s)
        self.rep_max = max(self.rep_max, angle_s)
        current_rom = self.rep_max - self.rep_min

        # bootstrap ROM during early motion
        if self._boot_start is None:
            self._boot_start = t
        if abs(vel) >= (self.cfg.velocity_eps * 0.5) and (t - self._boot_start) <= (self._boot_ms / 1000.0):
            self._boot_min = min(self._boot_min, angle_s)
            self._boot_max = max(self._boot_max, angle_s)

        # kinematics aggregates
        self.peak_vel = max(self.peak_vel, abs(vel))
        self.vel_sum += abs(vel)
        self.vel_n   += 1
        self._update_phases(angle_s, vel, t)

        # dwell helpers (not strictly required with adaptive ROM; keep for HUD/options)
        if self.cfg.concentric_angle_down:
            top, bottom = self.cfg.max_angle, self.cfg.min_angle
        else:
            top, bottom = self.cfg.min_angle, self.cfg.max_angle
        self.last_top_dwell    = t if abs(angle_s - top) < 5 else 0.0
        self.last_bottom_dwell = t if abs(angle_s - bottom) < 5 else 0.0

        # phase durations (ms) if ended
        conc_ms = ((self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0
                   if (self.phase_conc.end_ts and not self.phase_conc.active) else 0.0)
        ecc_ms  = ((self.phase_ecc.end_ts  - self.phase_ecc.start_ts)  * 1000.0
                   if (self.phase_ecc.end_ts  and not self.phase_ecc.active) else 0.0)

        rom_gate = self._rom_gate(current_rom, t)

        # Count a rep when we've seen *both* phases and ROM crosses the adaptive gate
        if conc_ms > 0 and ecc_ms > 0 and current_rom >= rom_gate:
            rep_end   = t
            total_ms  = int((rep_end - self.rep_start_ts) * 1000.0)
            conc_ms_i = int(conc_ms)
            ecc_ms_i  = int(ecc_ms)
            avg_vel   = (self.vel_sum / max(1, self.vel_n))
            out = {
                "tut_ms_total": total_ms,
                "tut_ms_concentric": conc_ms_i,
                "tut_ms_eccentric": ecc_ms_i,
                "rom_deg": float(current_rom),
                "peak_ang_vel_deg_s": float(self.peak_vel),
                "avg_ang_vel_deg_s": float(avg_vel),
            }
            self.count += 1
            self._reset_rep(t)
            return out

        # Partial (much gentler): only if we've been at it for a while AND ROM is tiny
        time_in_attempt = t - self.rep_start_ts
        if time_in_attempt > 6.0 and current_rom < (0.7 * rom_gate) and abs(vel) < (self.cfg.velocity_eps * 0.5):
            self._reset_rep(t)
            return {"partial": True, "reason": "insufficient_rom"}

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
