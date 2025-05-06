import sqlite3
from pathlib import Path
from contextlib import contextmanager

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY,
    strategy TEXT,
    attempt INTEGER,
    prompt TEXT,
    response TEXT,
    score REAL,
    topic TEXT,
    model TEXT,
    ts DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_run_topic ON run(topic);
"""

def init_db(path: Path):
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)

@contextmanager
def open_db(path: Path):
    init_db(path)
    con = sqlite3.connect(path)
    try:
        yield con
    finally:
        con.commit()
        con.close()

def log_run(con, **row):
    con.execute(
        "INSERT INTO run (strategy, attempt, prompt, response, score, topic, model) "
        "VALUES (:strategy, :attempt, :prompt, :response, :score, :topic, :model)",
        row
    )