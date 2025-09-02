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
    velocity_eps: float = 15.0  # deg/s considered moving
    dwell_ms: int = 150       # minimum dwell to confirm top/bottom
    # Direction: True=angle decreases for concentric (e.g., curls), False=increases
    concentric_angle_down: bool = True


@dataclass
class RepPhase:
    active: bool = False
    start_ts: float = 0.0
    end_ts: float = 0.0


class SingleJointRepDetector:
    def __init__(self, cfg: RepConfig):
        self.cfg = cfg
        self.count = 0
        self.partial = False
        self.phase_conc = RepPhase(False, 0.0, 0.0)
        self.phase_ecc = RepPhase(False, 0.0, 0.0)
        self.last_angle = None
        self.peak_vel = 0.0
        self.vel_sum = 0.0
        self.vel_n = 0
        self.rep_start_ts = 0.0
        self.rep_rom = 0.0
        self.last_top_dwell = 0.0
        self.last_bottom_dwell = 0.0

    def _phase_start(self, conc: bool, t: float):
        if conc:
            self.phase_conc = RepPhase(True, t, 0.0)
        else:
            self.phase_ecc = RepPhase(True, t, 0.0)

    def _phase_end(self, conc: bool, t: float) -> float:
        ph = self.phase_conc if conc else self.phase_ecc
        if ph.active:
            ph.active = False
            ph.end_ts = t
            return max(0.0, (ph.end_ts - ph.start_ts) * 1000.0)
        return 0.0

    def _update_phases(self, angle: float, vel: float, t: float):
        moving = abs(vel) >= self.cfg.velocity_eps
        if not moving:
            return
        # Determine direction of concentric given config
        conc_dir = -1 if self.cfg.concentric_angle_down else 1
        if vel * conc_dir < 0:  # moving concentric
            if not self.phase_conc.active:
                self._phase_start(True, t)
            if self.phase_ecc.active:
                self._phase_end(False, t)
        else:  # moving eccentric
            if not self.phase_ecc.active:
                self._phase_start(False, t)
            if self.phase_conc.active:
                self._phase_end(True, t)

    def step(self, angle: float, t: float) -> Optional[dict]:
        # velocity estimate
        if self.last_angle is None:
            self.last_angle = angle
            self.rep_start_ts = t
            return None
        dt = max(1e-3, t - (self._last_t if hasattr(self, "_last_t") else t))
        vel = (angle - self.last_angle) / dt
        self._last_t = t
        self.last_angle = angle

        self.peak_vel = max(self.peak_vel, abs(vel))
        self.vel_sum += abs(vel)
        self.vel_n += 1
        self._update_phases(angle, vel, t)

        # ROM tracking
        if self.cfg.concentric_angle_down:
            # reps go from high angle -> low angle -> high angle
            top = self.cfg.max_angle
            bottom = self.cfg.min_angle
        else:
            top = self.cfg.min_angle
            bottom = self.cfg.max_angle

        # Detect top/bottom dwell
        if abs(angle - top) < 5:
            if self.last_top_dwell == 0.0:
                self.last_top_dwell = t
        else:
            self.last_top_dwell = 0.0

        if abs(angle - bottom) < 5:
            if self.last_bottom_dwell == 0.0:
                self.last_bottom_dwell = t
        else:
            self.last_bottom_dwell = 0.0

        # Determine if a full rep completed (lenient on long holds)
        # A full rep requires crossing from top to bottom and back (or vice versa),
        # with ROM >= min_rom. Long holds are okay as long as camera session continues.
        # We’ll approximate by checking whether both phases saw non-trivial durations
        # and ROM change exceeded threshold.
        self.rep_rom = max(self.rep_rom, abs(angle - top), abs(angle - bottom))

        conc_ms = (self.phase_conc.end_ts - self.phase_conc.start_ts) * 1000.0 if (self.phase_conc.end_ts and not self.phase_conc.active) else 0.0
        ecc_ms = (self.phase_ecc.end_ts - self.phase_ecc.start_ts) * 1000.0 if (self.phase_ecc.end_ts and not self.phase_ecc.active) else 0.0

        full_rom = (self.cfg.max_angle - self.cfg.min_angle)
        rom_now = full_rom - abs(angle - (top if self.cfg.concentric_angle_down else bottom))

        # Heuristic: when both phases completed at least once and ROM exceeded min_rom, we count one rep
        if conc_ms > 0 and ecc_ms > 0 and (full_rom >= self.cfg.min_rom + 10):
            # finalize rep
            rep_end = t
            rep_start = self.rep_start_ts
            total_ms = int((rep_end - rep_start) * 1000.0)
            conc_ms_i = int(conc_ms)
            ecc_ms_i = int(ecc_ms)
            avg_vel = (self.vel_sum / max(1, self.vel_n))
            out = {
                "tut_ms_total": total_ms,
                "tut_ms_concentric": conc_ms_i,
                "tut_ms_eccentric": ecc_ms_i,
                "rom_deg": full_rom,
                "peak_ang_vel_deg_s": float(self.peak_vel),
                "avg_ang_vel_deg_s": float(avg_vel),
            }
            # reset for next rep
            self.count += 1
            self.phase_conc = RepPhase(False, 0.0, 0.0)
            self.phase_ecc = RepPhase(False, 0.0, 0.0)
            self.vel_sum = 0.0
            self.vel_n = 0
            self.peak_vel = 0.0
            self.rep_start_ts = t
            self.rep_rom = 0.0
            return out

        # Partial detection: exceeded movement but failed criteria for full rep
        # E.g., ROM moved but never completed both phases or ROM < min_rom after long time
        if (abs(vel) < self.cfg.velocity_eps) and (t - self.rep_start_ts) > 3.5:  # long attempt
            # reset attempt
            self.phase_conc = RepPhase(False, 0.0, 0.0)
            self.phase_ecc = RepPhase(False, 0.0, 0.0)
            self.vel_sum = 0.0
            self.vel_n = 0
            self.peak_vel = 0.0
            self.rep_start_ts = t
            self.rep_rom = 0.0
            return {"partial": True, "reason": "insufficient_rom"}

        return None


class PosePipeline(threading.Thread):
    def __init__(self, cfg: RepConfig, on_rep: Callable[[dict], None], on_partial: Callable[[str], None]):
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
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not available")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                ok, frame = self.cap.read()
                if not ok:
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

                # Draw overlay minimal
                cv2.putText(frame, f"Count: {self.detector.count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Workout", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self._stop.set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()
