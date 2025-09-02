# app/runtime/cli.py
from __future__ import annotations
import sys
import time
from app.audio.stt import VADRecorder, STTEngine
from app.agent.router import route_and_execute
from app.audio.tts import TTSEngine
from app.data import db  # <-- for ASR logging added below
import json

def main():
    print("Voice assistant ready. Press Ctrl+C to exit.", flush=True)
    tts = TTSEngine()
    asr = STTEngine(model_size="base")

    # Use your input device index (0 = MacBook Air Microphone)
    rec = VADRecorder(device_index=0, capture_rate=48000)

    while True:
        try:
            tts.wait_until_idle(timeout=2.0)

            tts.say_sync("beep")
            time.sleep(0.4)

            # tts.wait_until_idle(timeout=2.0)
            print("\n---", flush=True)
            print("listening… (speak now)", flush=True)

            audio = rec.record_once()
            if audio is None or len(audio) == 0:
                print("STT: (no audio captured)", flush=True)
                continue

            text = asr.transcribe(audio)
            if not text:
                print("STT: (no speech detected / empty transcript)", flush=True)
                continue

            print(f"STT: {text!r}", flush=True)

            # Route + execute (runs tool if selected)
            trace = route_and_execute(text, verbose=True)

            # Show the step-by-step trace
            for s in trace["steps"]:
                print(s, flush=True)

            try:
                db.insert_asr_log(
                    ts=time.time(),
                    transcript=text,
                    action=trace.get("action","noop"),
                    tool_name=trace.get("action") if trace.get("action") != "noop" else "",
                    args_json=json.dumps(trace.get("args", {})),
                )
            except Exception:
                pass

            # If no tool was called, JUST PRINT (do NOT TTS) to avoid feedback into mic
            if trace["action"] == "noop":
                msg = trace["llm_text"] or "Not a workout command. Do not call tools."
                print(f"assistant: {msg}", flush=True)
                # intentionally no tts.say(msg)

            # Ensure any tool confirmations finish before next listen cycle (half-duplex)
            tts.wait_until_idle(timeout=2.0)

            # # If no tool was called, give a short verbal nudge
            # if trace["action"] == "noop":
            #     msg = trace["llm_text"] or "that's not a workout command i recognize"
            #     print(f"assistant: {msg}", flush=True)
            #     tts.say(msg)

            # Tool side-effects (start/pause/resume/stop) will speak via session manager TTS
        except KeyboardInterrupt:
            print("\nExiting…", flush=True)
            break
        except Exception as e:
            print("Error:", e, file=sys.stderr)
            time.sleep(0.5)

if __name__ == "__main__":
    main()
    rec = VADRecorder(device_index=0, capture_rate=48000, aggressiveness=1)

