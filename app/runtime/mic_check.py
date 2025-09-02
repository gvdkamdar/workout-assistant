from __future__ import annotations
import numpy as np
import sounddevice as sd
import time

def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def main():
    dev = 0  # your mic
    sr = 48000
    secs = 3
    print("Recording 3s from device", dev, "at", sr, "Hz ... speak into the mic.")
    audio = sd.rec(int(sr * secs), samplerate=sr, channels=1, dtype="int16", device=dev)
    sd.wait()
    a = audio.reshape(-1)
    print("Samples:", a.shape[0], "RMS:", rms(a))
    if rms(a) < 0.005:
        print("⚠️ Very low level detected — check macOS input volume / permissions.")
    else:
        print("✅ Mic levels look OK.")

if __name__ == "__main__":
    main()
