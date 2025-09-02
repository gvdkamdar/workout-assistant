from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

_DB_PATH = Path("./workout.db")

SCHEMA = r"""
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  exercise TEXT NOT NULL,
  side TEXT NOT NULL,
  started_at REAL NOT NULL,
  stopped_at REAL,
  target_reps INTEGER,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  t REAL NOT NULL,
  rep_count INTEGER NOT NULL,
  rep_delta INTEGER NOT NULL,
  rom_deg REAL,
  side TEXT,
  flag_form TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS reps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  rep_index INTEGER NOT NULL,
  start_ts REAL NOT NULL,
  end_ts REAL NOT NULL,
  tut_ms_total INTEGER NOT NULL,
  tut_ms_concentric INTEGER NOT NULL,
  tut_ms_eccentric INTEGER NOT NULL,
  rom_deg REAL,
  peak_ang_vel_deg_s REAL,
  avg_ang_vel_deg_s REAL,
  side TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS asr_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  transcript TEXT NOT NULL,
  action TEXT,
  tool_name TEXT,
  args_json TEXT
);
"""

_conn: Optional[sqlite3.Connection] = None

def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH.as_posix(), check_same_thread=False)
        _conn.execute("PRAGMA foreign_keys=ON;")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn

# Session-level writes

def insert_session(session_id: str, exercise: str, side: str, started_at: float, target_reps: Optional[int]):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (id, exercise, side, started_at, target_reps) VALUES (?,?,?,?,?)",
        (session_id, exercise, side, started_at, target_reps),
    )
    conn.commit()


def stop_session(session_id: str, stopped_at: float):
    conn = get_conn()
    conn.execute("UPDATE sessions SET stopped_at=? WHERE id=?", (stopped_at, session_id))
    conn.commit()

# Event & rep writes

def insert_event(session_id: str, t: float, rep_count: int, rep_delta: int, rom_deg: float, side: str, flag_form: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO events (session_id, t, rep_count, rep_delta, rom_deg, side, flag_form) VALUES (?,?,?,?,?,?,?)",
        (session_id, t, rep_count, rep_delta, rom_deg, side, flag_form),
    )
    conn.commit()


def insert_rep(
    session_id: str,
    rep_index: int,
    start_ts: float,
    end_ts: float,
    tut_ms_total: int,
    tut_ms_concentric: int,
    tut_ms_eccentric: int,
    rom_deg: float,
    peak_ang_vel: float,
    avg_ang_vel: float,
    side: str,
):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO reps (
          session_id, rep_index, start_ts, end_ts, tut_ms_total, tut_ms_concentric, tut_ms_eccentric,
          rom_deg, peak_ang_vel_deg_s, avg_ang_vel_deg_s, side
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            session_id,
            rep_index,
            start_ts,
            end_ts,
            tut_ms_total,
            tut_ms_concentric,
            tut_ms_eccentric,
            rom_deg,
            peak_ang_vel,
            avg_ang_vel,
            side,
        ),
    )
    conn.commit()
