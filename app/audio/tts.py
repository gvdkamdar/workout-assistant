from __future__ import annotations
import os
import queue
import threading
import subprocess
from typing import Optional

class TTSEngine:
    def __init__(self, prefer_mac_say: bool = True):
        self.prefer_mac_say = prefer_mac_say and (os.uname().sysname == "Darwin")
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self._pyttsx3 = None
        self._speaking = False  # <-- add


    def _speak_mac(self, text: str):
        try:
            subprocess.run(["say", text], check=False)
        except Exception:
            raise

    # app/audio/tts.py — inside class TTSEngine

    def say_sync(self, text: str):
        """Speak synchronously (blocks). Use for the 'beep' so ASR doesn't capture it."""
        if not text:
            return
        try:
            self._speaking = True
            if self.prefer_mac_say:
                subprocess.run(["say", text], check=False)
            else:
                self._ensure_pyttsx3()
                self._pyttsx3.say(text)
                self._pyttsx3.runAndWait()
        finally:
            self._speaking = False  # <-- add

    def is_speaking(self) -> bool:
        return bool(self._speaking or not self.q.empty())  # <-- add

    def wait_until_idle(self, timeout: float | None = None):
        """Block until all queued speech is done (best-effort)."""
        # quick drain of queue tasks
        try:
            self.q.join()
        except Exception:
            pass
        # small extra wait if synth is still in-progress
        if timeout:
            import time
            t0 = time.time()
            while self.is_speaking() and (time.time() - t0) < timeout:
                time.sleep(0.05)

    def _ensure_pyttsx3(self):
        if self._pyttsx3 is None:
            import pyttsx3  # lazy import
            self._pyttsx3 = pyttsx3.init()

    def _speak_fallback(self, text: str):
        try:
            self._ensure_pyttsx3()
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()
        except Exception:
            # last resort: swallow
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._speaking = True  # <-- add

                if self.prefer_mac_say:
                    self._speak_mac(text)
                else:
                    self._speak_fallback(text)
            finally:
                self.q.task_done()

    def say(self, text: str):
        if not text:
            return
        self.q.put(text)

    def shutdown(self):
        self._stop.set()
        try:
            self.q.put_nowait("")
        except Exception:
            pass
