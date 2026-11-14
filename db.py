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

_MIGRATION_V2 = """
CREATE TABLE IF NOT EXISTS features (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id    INTEGER NOT NULL UNIQUE REFERENCES clips(id) ON DELETE CASCADE,
    vector     TEXT    NOT NULL,
    duration   REAL    NOT NULL,
    created_at TEXT    NOT NULL
);
"""

def save_features(conn: sqlite3.Connection, clip_id: int, vector: list[float], duration: float) -> int:
    vec_json = json.dumps(vector)
    cur = conn.execute(
        "INSERT OR REPLACE INTO features (clip_id, vector, duration, created_at) VALUES (?, ?, ?, ?)",
        (clip_id, vec_json, duration, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid

def get_features(conn: sqlite3.Connection, clip_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM features WHERE clip_id = ?", (clip_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "clip_id": row["clip_id"],
        "vector": json.loads(row["vector"]),
        "duration": row["duration"],
        "created_at": row["created_at"],
    }

def clips_without_features(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT c.* FROM clips c LEFT JOIN features f ON c.id = f.clip_id WHERE f.id IS NULL ORDER BY c.id"
    ).fetchall()

def init_db(db_path: str | None = None) -> sqlite3.Connection:
    conn = get_connection(db_path)
    cur_ver = _current_version(conn)
    if cur_ver < 1:
        conn.executescript(_TABLES_V1)
        conn.executescript(_MIGRATION_V2)
        _safe_alter(conn, "ALTER TABLE pending_sounds ADD COLUMN clip_id INTEGER REFERENCES clips(id);\nALTER TABLE pending_sounds ADD COLUMN vector TEXT;")
        _set_version(conn, 3)
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

def _safe_alter(conn: sqlite3.Connection, sql: str) -> None:
    for line in sql.strip().splitlines():
        line = line.strip()
        if not line: continue
        try:
            conn.execute(line)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower(): raise

def save_pending_sound(conn: sqlite3.Connection, name: str, clip_id: int | None = None, vector: list[float] | None = None) -> int:
    vec_json = json.dumps(vector) if vector is not None else None
    cur = conn.execute(
        "INSERT INTO pending_sounds (name, clip_id, vector, created_at) VALUES (?, ?, ?, ?)",
        (name, clip_id, vec_json, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid

def get_all_pending_sounds(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM pending_sounds ORDER BY id").fetchall()
    res = []
    for r in rows:
        d = dict(r)
        if d.get("vector"): d["vector"] = json.loads(d["vector"])
        res.append(d)
    return res

def get_pending_sounds_with_vectors(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM pending_sounds WHERE vector IS NOT NULL ORDER BY id").fetchall()
    res = []
    for r in rows:
        d = dict(r)
        d["vector"] = json.loads(d["vector"])
        res.append(d)
    return res

def delete_pending_by_name(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM pending_sounds WHERE name = ?", (name,))
    conn.commit()
