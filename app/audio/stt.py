from __future__ import annotations
import io
import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad
from typing import Optional

# Local STT (faster-whisper)
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

# Cloud fallback (OpenAI Whisper)
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# class VADRecorder:
#     """Records audio chunks until silence using WebRTC VAD.
#     Produces 16kHz mono 16-bit PCM frames suitable for Whisper.
#     """
#     def __init__(self, sample_rate=16000, frame_ms=30, aggressiveness=2, max_len_s=8):
#         self.sample_rate = sample_rate
#         self.frame_len = int(sample_rate * frame_ms / 1000)
#         self.vad = webrtcvad.Vad(aggressiveness)
#         self.max_len = int(sample_rate * max_len_s)

#     def record_once(self) -> np.ndarray:
#         sd.default.samplerate = self.sample_rate
#         sd.default.channels = 1
#         buffer = []
#         voiced_timeout = 0.8  # seconds after last voice to stop
#         last_voice_time = None

#         def callback(indata, frames, time_info, status):
#             nonlocal last_voice_time
#             pcm16 = (indata * 32768).astype(np.int16)
#             # chunk into VAD frames
#             for i in range(0, len(pcm16), self.frame_len):
#                 frame = pcm16[i:i + self.frame_len]
#                 if len(frame) < self.frame_len:
#                     continue
#                 is_speech = False
#                 try:
#                     is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
#                 except Exception:
#                     pass
#                 buffer.append(frame)
#                 if is_speech:
#                     last_voice_time = time.time()

#         with sd.InputStream(callback=callback):
#             start = time.time()
#             while True:
#                 sd.sleep(50)
#                 if last_voice_time is None and (time.time() - start) > 1.5:
#                     # no voice detected, keep listening until max_len
#                     pass
#                 elif last_voice_time is not None and (time.time() - last_voice_time) > voiced_timeout:
#                     break
#                 if len(buffer) * 1.0 >= self.max_len:
#                     break
#         if not buffer:
#             return np.zeros((0,), dtype=np.int16)
#         audio = np.concatenate(buffer, axis=0)
#         return audio

# # app/audio/stt.py  — replace the VADRecorder class definition 2

# class VADRecorder:


#     """Records audio chunks until silence using WebRTC VAD.
#     Produces 16kHz mono 16-bit PCM frames suitable for Whisper.
#     """
#     def __init__(
#         self,
#         sample_rate: int = 16000,
#         frame_ms: int = 30,
#         aggressiveness: int = 2,
#         max_len_s: float = 8.0,
#         listen_timeout_s: float = 6.0,   # hard cap if no speech detected
#         device_index: Optional[int] = None,
#     ):
#         self.sample_rate = sample_rate
#         self.frame_ms = frame_ms
#         self.frame_len = int(sample_rate * frame_ms / 1000)      # samples per frame
#         self.vad = webrtcvad.Vad(aggressiveness)
#         self.max_len_samples = int(sample_rate * max_len_s)       # <-- samples, not frames
#         self.listen_timeout_s = listen_timeout_s
#         self.device_index = device_index

#     def record_once(self) -> np.ndarray:
#         sd.default.samplerate = self.sample_rate
#         sd.default.channels = 1

#         buffer: list[np.ndarray] = []
#         total_samples = 0
#         voiced_timeout = 0.8  # seconds after last voice to stop
#         last_voice_time: Optional[float] = None

#         def callback(indata, frames, time_info, status):
#             nonlocal last_voice_time, total_samples
#             # indata is float32 [-1,1]; convert to int16
#             pcm16 = np.clip(indata, -1.0, 1.0)
#             pcm16 = (pcm16 * 32768.0).astype(np.int16)

#             # chunk into VAD frames
#             for i in range(0, len(pcm16), self.frame_len):
#                 frame = pcm16[i:i + self.frame_len]
#                 if len(frame) < self.frame_len:
#                     continue
#                 try:
#                     is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
#                 except Exception:
#                     is_speech = False
#                 buffer.append(frame)
#                 total_samples += len(frame)
#                 if is_speech:
#                     last_voice_time = time.time()

#         try:
#             with sd.InputStream(
#                 samplerate=self.sample_rate,
#                 channels=1,
#                 dtype="float32",
#                 callback=callback,
#                 device=self.device_index,  # None = system default
#             ):
#                 start = time.time()
#                 while True:
#                     sd.sleep(50)
#                     now = time.time()

#                     # If no speech at all, bail after listen_timeout_s
#                     if last_voice_time is None and (now - start) > self.listen_timeout_s:
#                         break

#                     # If we had speech and then enough silence → stop
#                     if last_voice_time is not None and (now - last_voice_time) > voiced_timeout:
#                         break

#                     # Hard maximum length (samples)
#                     if total_samples >= self.max_len_samples:
#                         break
#         except KeyboardInterrupt:
#             # surface to caller so CLI can exit cleanly
#             raise

#         if not buffer:
#             return np.zeros((0,), dtype=np.int16)

#         audio = np.concatenate(buffer, axis=0)
#         return audio

# app/audio/stt.py  — replace ONLY the VADRecorder class 3 48k capture

class VADRecorder:
    """
    Records audio until silence using WebRTC VAD.
    Captures at capture_rate (e.g., 48000) and resamples to 16000 for VAD/Whisper.
    Produces 16kHz mono int16 PCM suitable for Whisper/VAD.
    """
    def __init__(
        self,
        sample_rate: int = 16000,          # VAD/Whisper rate (keep 16000)
        frame_ms: int = 30,
        aggressiveness: int = 2,
        max_len_s: float = 8.0,
        listen_timeout_s: float = 6.0,     # bail if no speech in this time
        device_index: Optional[int] = None,
        capture_rate: int = 48000,         # <-- native mac mic is 48k
    ):
        self.vad_rate = sample_rate
        self.capture_rate = capture_rate
        self.frame_ms = frame_ms
        self.vad_frame_len = int(self.vad_rate * frame_ms / 1000)        # samples per 30ms @16k
        self.cap_frame_len = int(self.capture_rate * frame_ms / 1000)    # samples per 30ms @48k
        self.vad = webrtcvad.Vad(aggressiveness)
        self.max_len_samples = int(self.vad_rate * max_len_s)            # cap in 16k samples
        self.listen_timeout_s = listen_timeout_s
        self.device_index = device_index

    # simple, fast resampler 48k -> 16k (decimate by 3). Generic fallback uses linear interp.
    def _to_16k(self, pcm16_cap: np.ndarray) -> np.ndarray:
        if self.capture_rate == self.vad_rate:
            return pcm16_cap
        if self.capture_rate == 48000 and self.vad_rate == 16000:
            return pcm16_cap[::3]
        # generic linear resample
        src = pcm16_cap.astype(np.float32)
        new_len = int(round(len(src) * (self.vad_rate / self.capture_rate)))
        if new_len <= 1:
            return np.zeros((0,), dtype=np.int16)
        x = np.linspace(0, len(src) - 1, num=len(src), dtype=np.float32)
        xi = np.linspace(0, len(src) - 1, num=new_len, dtype=np.float32)
        yi = np.interp(xi, x, src)
        yi = np.clip(yi, -32768, 32767).astype(np.int16)
        return yi

    def record_once(self) -> np.ndarray:
        sd.default.channels = 1  # mono

        buffer_16k: list[np.ndarray] = []
        resample_fifo = np.zeros((0,), dtype=np.int16)
        total_16k = 0
        voiced_timeout = 0.8
        last_voice_time: Optional[float] = None

        def callback(indata, frames, time_info, status):
            nonlocal last_voice_time, resample_fifo, total_16k
            # indata dtype=int16 because we open stream as int16 below
            pcm16_cap = np.array(indata, copy=False).reshape(-1).astype(np.int16)
            # resample chunk to 16k and append to fifo
            chunk_16k = self._to_16k(pcm16_cap)
            if chunk_16k.size == 0:
                return
            resample_fifo = np.concatenate([resample_fifo, chunk_16k], axis=0)

            # slice out exact 30ms VAD frames from fifo
            while len(resample_fifo) >= self.vad_frame_len:
                frame = resample_fifo[:self.vad_frame_len]
                resample_fifo = resample_fifo[self.vad_frame_len:]
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.vad_rate)
                except Exception:
                    is_speech = False
                buffer_16k.append(frame)
                total_16k += len(frame)
                if is_speech:
                    last_voice_time = time.time()

        try:
            with sd.InputStream(
                samplerate=self.capture_rate,   # capture at native 48k
                channels=1,
                dtype="int16",
                callback=callback,
                device=self.device_index,       # 0 for your MacBook Air Microphone
                blocksize=self.cap_frame_len,   # ~30ms blocks at capture rate
            ):
                start = time.time()
                while True:
                    sd.sleep(50)
                    now = time.time()

                    # If no speech at all, bail after listen_timeout_s
                    if last_voice_time is None and (now - start) > self.listen_timeout_s:
                        break

                    # If we had speech and then enough silence → stop
                    if last_voice_time is not None and (now - last_voice_time) > voiced_timeout:
                        break

                    # Hard maximum length (16k samples)
                    if total_16k >= self.max_len_samples:
                        break
        except KeyboardInterrupt:
            raise

        if not buffer_16k:
            return np.zeros((0,), dtype=np.int16)

        audio_16k = np.concatenate(buffer_16k, axis=0)
        return audio_16k


class STTEngine:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._whisper_local = None
        if WhisperModel is not None:
            try:
                self._whisper_local = WhisperModel(model_size, device=device, compute_type="int8")
            except Exception:
                self._whisper_local = None

    def transcribe(self, wav_pcm16: np.ndarray, sample_rate: int = 16000) -> str:
        """Try local faster-whisper; fall back to OpenAI Whisper if key set."""
        if wav_pcm16 is None or len(wav_pcm16) == 0:
            return ""
        # Local first
        if self._whisper_local is not None:
            try:
                segments, _ = self._whisper_local.transcribe(wav_pcm16.astype(np.float32) / 32768.0, language="en")
                text = " ".join(s.text.strip() for s in segments)
                return text.strip()
            except Exception:
                pass
        # Cloud fallback
        if _OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI()
                import tempfile, wave
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    # write WAV
                    with wave.open(f, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(wav_pcm16.tobytes())
                    tmpname = f.name
                with open(tmpname, "rb") as fh:
                    res = client.audio.transcriptions.create(model="whisper-1", file=fh)
                return (res.text or "").strip()
            except Exception:
                return ""
        return ""
