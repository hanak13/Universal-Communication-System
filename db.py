"""
UCS Database — Phase 1 Schema & CRUD
"""
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, BACKUP_DIR, SCHEMA_VERSION

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

_TABLES_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL UNIQUE,
    label       TEXT    NOT NULL,
    speaker     TEXT    NOT NULL DEFAULT 'unknown',
    mic_type    TEXT    NOT NULL DEFAULT 'built-in',
    noise_level REAL,
    created_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS words (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical     TEXT    NOT NULL,
    meaning       TEXT,
    word_type     TEXT,
    person_number TEXT,
    status        TEXT    NOT NULL DEFAULT 'unverified',
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS word_variants (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id  INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    spelling TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS phrases (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    meaning    TEXT,
    status     TEXT    NOT NULL DEFAULT 'unverified',
    created_at TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS rules (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type  TEXT NOT NULL,
    definition TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'unverified',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_sounds (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    row_id     INTEGER NOT NULL,
    old_value  TEXT,
    new_value  TEXT,
    ts         TEXT NOT NULL
);
"""

def _current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
        return row["version"] if row else 0
    except sqlite3.OperationalError:
        return 0

def _set_version(conn: sqlite3.Connection, ver: int) -> None:
    conn.execute("DELETE FROM schema_version;")
    conn.execute("INSERT INTO schema_version (version) VALUES (?);", (ver,))

def init_db(db_path: str | None = None) -> sqlite3.Connection:
    conn = get_connection(db_path)
    cur_ver = _current_version(conn)
    if cur_ver < 1:
        conn.executescript(_TABLES_V1)
        _set_version(conn, 1)
        conn.commit()
    return conn

def save_clip(conn: sqlite3.Connection, filename: str, label: str,
              speaker: str = "unknown", mic_type: str = "built-in",
              noise_level: float | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO clips (filename, label, speaker, mic_type, noise_level, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (filename, label, speaker, mic_type, noise_level, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid

def get_clips(conn: sqlite3.Connection, label: str | None = None,
              speaker: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM clips WHERE 1=1"
    params = []
    if label:
        sql += " AND label = ?"
        params.append(label)
    if speaker:
        sql += " AND speaker = ?"
        params.append(speaker)
    sql += " ORDER BY created_at DESC"
    return conn.execute(sql, params).fetchall()

def get_clip_by_id(conn: sqlite3.Connection, clip_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()

def distinct_labels(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT label FROM clips ORDER BY label").fetchall()
    return [r["label"] for r in rows]

def distinct_speakers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT speaker FROM clips ORDER BY speaker").fetchall()
    return [r["speaker"] for r in rows]
