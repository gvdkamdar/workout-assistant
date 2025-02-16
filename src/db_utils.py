import sqlite3
from datetime import datetime

DB_NAME = "workout_logs.db"

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            exercise TEXT,
            count INTEGER,
            time_per_count FLOAT,
            total_time FLOAT,
            date DATE,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

# Add a workout log
def add_log(user_id, exercise, count, time_per_count, total_time):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        INSERT INTO workout_logs (user_id, exercise, count, time_per_count, total_time, date, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, exercise, count, time_per_count, total_time, date, timestamp))
    conn.commit()
    conn.close()

# Fetch all logs
def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM workout_logs')
    rows = cursor.fetchall()
    conn.close()
    return rows


def add_dummy_data():
    add_log("user_001", "bicep_curl", 10, 3.0, 30.0)
    add_log("user_001", "bench_press", 8, 2.5, 20.0)
    add_log("user_001", "lateral_raise", 12, 1.8, 21.6)
    print("Dummy data added.")
