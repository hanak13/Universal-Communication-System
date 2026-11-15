"""
UCS Database — schema, CRUD helpers, and migrations.
Auto-backs up ucs.db before any migration.
"""
from pathlib import Path
import json
import shutil
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, BACKUP_DIR, SCHEMA_VERSION


# ── helpers ────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a connection with row-factory and WAL mode."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── schema bootstrap ──────────────────────────────────────────────

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
    status        TEXT    NOT NULL DEFAULT 'unverified'
                          CHECK (status IN ('verified', 'unverified')),
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
    status     TEXT    NOT NULL DEFAULT 'unverified'
                       CHECK (status IN ('verified', 'unverified')),
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS rules (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type  TEXT NOT NULL,
    definition TEXT NOT NULL,          -- JSON blob
    status     TEXT NOT NULL DEFAULT 'unverified'
                    CHECK (status IN ('verified', 'unverified')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_sounds (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,       -- e.g. new_word_1
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


def _backup_db() -> str | None:
    """Copy ucs.db → backups/ucs_<timestamp>.db. Returns backup path."""
    db_file = Path(DB_PATH)
    if not db_file.is_file():
        return None
    backup_dir = Path(BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"ucs_{stamp}.db"
    shutil.copy2(str(db_file), str(dst))
    return str(dst)


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the stored schema version, or 0 if table missing."""
    try:
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
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
    vector     TEXT    NOT NULL,          -- JSON array of floats
    duration   REAL    NOT NULL,
    created_at TEXT    NOT NULL
);
"""

_MIGRATION_V3 = """
ALTER TABLE pending_sounds ADD COLUMN clip_id INTEGER REFERENCES clips(id);
ALTER TABLE pending_sounds ADD COLUMN vector TEXT;
"""

_MIGRATION_V4 = """
ALTER TABLE words ADD COLUMN confirmations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE words ADD COLUMN rejections INTEGER NOT NULL DEFAULT 0;
"""


_MIGRATION_V5 = """
CREATE TABLE IF NOT EXISTS sequences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    recognized  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
"""

_MIGRATION_V6 = """
CREATE TABLE IF NOT EXISTS transcription_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_index INTEGER NOT NULL,
    heard_guess   TEXT,
    user_edit     TEXT,
    final_spelling TEXT,
    clip_hash     TEXT,
    created_at    TEXT NOT NULL
);
"""


def _safe_alter(conn: sqlite3.Connection, sql: str) -> None:
    """Run ALTER TABLE statements one by one, ignoring 'duplicate column'."""
    for line in sql.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            conn.execute(line)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Create tables if needed, run migrations, return connection."""
    conn = get_connection(db_path)
    cur_ver = _current_version(conn)

    if cur_ver < SCHEMA_VERSION:
        if cur_ver > 0:
            _backup_db()          # back up before migration

        # V1: base tables
        if cur_ver < 1:
            conn.executescript(_TABLES_V1)

        # V2: features table
        if cur_ver < 2:
            conn.executescript(_MIGRATION_V2)

        # V3: extend pending_sounds with clip_id + vector
        if cur_ver < 3:
            _safe_alter(conn, _MIGRATION_V3)

        # V4: add confirmations + rejections to words
        if cur_ver < 4:
            _safe_alter(conn, _MIGRATION_V4)

        # V5: sequences table
        if cur_ver < 5:
            conn.executescript(_MIGRATION_V5)

        # V6: transcription_log table
        if cur_ver < 6:
            conn.executescript(_MIGRATION_V6)

        _set_version(conn, SCHEMA_VERSION)
        conn.commit()

    return conn


# ── CRUD: clips ───────────────────────────────────────────────────

def save_clip(
    conn: sqlite3.Connection,
    filename: str,
    label: str,
    speaker: str = "unknown",
    mic_type: str = "built-in",
    noise_level: float | None = None,
) -> int:
    """Insert a clip row and return its id."""
    cur = conn.execute(
        """
        INSERT INTO clips (filename, label, speaker, mic_type, noise_level, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (filename, label, speaker, mic_type, noise_level, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_clips(
    conn: sqlite3.Connection,
    label: str | None = None,
    speaker: str | None = None,
) -> list[sqlite3.Row]:
    """Return clips, optionally filtered."""
    sql = "SELECT * FROM clips WHERE 1=1"
    params: list = []
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


# ── CRUD: words ───────────────────────────────────────────────────

def save_word(
    conn: sqlite3.Connection,
    canonical: str,
    meaning: str | None = None,
    word_type: str | None = None,
    person_number: str | None = None,
    status: str = "unverified",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO words (canonical, meaning, word_type, person_number, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (canonical, meaning, word_type, person_number, status, _utcnow()),
    )
    conn.commit()
    wid = cur.lastrowid
    log_history(
        conn, "words", wid,
        json.dumps({"_action": "create"}),
        json.dumps({"canonical": canonical, "meaning": meaning, "status": status}),
    )
    return wid


def get_word_by_id(conn: sqlite3.Connection, word_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM words WHERE id = ?", (word_id,)).fetchone()
    return dict(row) if row else None


def get_all_words(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM words ORDER BY canonical").fetchall()
    return [dict(r) for r in rows]


def get_word_by_canonical(conn: sqlite3.Connection, canonical: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM words WHERE canonical = ?", (canonical,)
    ).fetchone()
    return dict(row) if row else None


def update_word(
    conn: sqlite3.Connection,
    word_id: int,
    **fields,
) -> None:
    """Update arbitrary fields on a word row. Log each change to history."""
    old = get_word_by_id(conn, word_id)
    if old is None:
        return
    for col, new_val in fields.items():
        if col in ("id", "created_at"):
            continue
        old_val = old.get(col)
        if str(old_val) != str(new_val):
            log_history(conn, "words", word_id,
                        json.dumps({col: old_val}),
                        json.dumps({col: new_val}))
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [word_id]
    conn.execute(f"UPDATE words SET {sets} WHERE id = ?", vals)
    conn.commit()


def confirm_word(conn: sqlite3.Connection, word_id: int) -> None:
    """Increment confirmations by 1."""
    conn.execute(
        "UPDATE words SET confirmations = confirmations + 1 WHERE id = ?",
        (word_id,),
    )
    conn.commit()
    log_history(conn, "words", word_id, None, '{"action":"confirm"}')


def reject_word(conn: sqlite3.Connection, word_id: int,
                correction: str | None = None) -> None:
    """Increment rejections by 1, optionally log the correction."""
    conn.execute(
        "UPDATE words SET rejections = rejections + 1 WHERE id = ?",
        (word_id,),
    )
    conn.commit()
    log_history(conn, "words", word_id, None,
                json.dumps({"action": "reject", "correction": correction}))


def word_confidence(word: dict) -> tuple[float, int]:
    """Return (confidence_ratio, total_tests)."""
    c = word.get("confirmations", 0)
    r = word.get("rejections", 0)
    total = c + r
    if total == 0:
        return 0.0, 0
    return c / total, total


# ── CRUD: word_variants ───────────────────────────────────────────

def add_variant(conn: sqlite3.Connection, word_id: int, spelling: str) -> int:
    cur = conn.execute(
        "INSERT INTO word_variants (word_id, spelling) VALUES (?, ?)",
        (word_id, spelling),
    )
    conn.commit()
    log_history(conn, "word_variants", cur.lastrowid, None,
                json.dumps({"word_id": word_id, "spelling": spelling}))
    return cur.lastrowid


def get_variants(conn: sqlite3.Connection, word_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT spelling FROM word_variants WHERE word_id = ?", (word_id,)
    ).fetchall()
    return [r["spelling"] for r in rows]


def lookup_by_variant(conn: sqlite3.Connection, spelling: str) -> dict | None:
    """Find a word by any of its variant spellings (or canonical)."""
    # Check canonical first
    row = conn.execute(
        "SELECT * FROM words WHERE canonical = ?", (spelling,)
    ).fetchone()
    if row:
        return dict(row)
    # Check variants
    vrow = conn.execute(
        "SELECT word_id FROM word_variants WHERE spelling = ?", (spelling,)
    ).fetchone()
    if vrow:
        return get_word_by_id(conn, vrow["word_id"])
    return None


# ── CRUD: pending_sounds (convert to word) ────────────────────────

def delete_pending_by_name(conn: sqlite3.Connection, name: str) -> None:
    """Remove all pending_sounds rows with this name."""
    conn.execute("DELETE FROM pending_sounds WHERE name = ?", (name,))
    conn.commit()


def get_pending_clip_ids(conn: sqlite3.Connection, name: str) -> list[int]:
    """Return clip IDs associated with a pending sound name."""
    rows = conn.execute(
        "SELECT clip_id FROM pending_sounds WHERE name = ? AND clip_id IS NOT NULL",
        (name,),
    ).fetchall()
    return [r["clip_id"] for r in rows]


# ── CRUD: pending_sounds ──────────────────────────────────────────

def next_pending_name(conn: sqlite3.Connection) -> str:
    """Return the next auto-name like new_word_3."""
    row = conn.execute("SELECT COUNT(*) AS n FROM pending_sounds").fetchone()
    return f"new_word_{row['n'] + 1}"


def save_pending_sound(
    conn: sqlite3.Connection,
    name: str,
    clip_id: int | None = None,
    vector: list[float] | None = None,
) -> int:
    vec_json = json.dumps(vector) if vector is not None else None
    cur = conn.execute(
        "INSERT INTO pending_sounds (name, clip_id, vector, created_at) VALUES (?, ?, ?, ?)",
        (name, clip_id, vec_json, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_all_pending_sounds(conn: sqlite3.Connection) -> list[dict]:
    """Return all pending sounds with parsed vectors."""
    rows = conn.execute(
        "SELECT * FROM pending_sounds ORDER BY id"
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("vector"):
            d["vector"] = json.loads(d["vector"])
        result.append(d)
    return result


def get_pending_sounds_with_vectors(conn: sqlite3.Connection) -> list[dict]:
    """Return only pending sounds that have a stored vector."""
    rows = conn.execute(
        "SELECT * FROM pending_sounds WHERE vector IS NOT NULL ORDER BY id"
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["vector"] = json.loads(d["vector"])
        result.append(d)
    return result


# ── CRUD: history ─────────────────────────────────────────────────

def log_history(
    conn: sqlite3.Connection,
    table_name: str,
    row_id: int,
    old_value: str | None,
    new_value: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO history (table_name, row_id, old_value, new_value, ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        (table_name, row_id, old_value, new_value, _utcnow()),
    )
    conn.commit()


def get_last_history(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent history entry."""
    row = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def undo_last_change(conn: sqlite3.Connection) -> str | None:
    """
    Revert the last history entry where old_value is not None.
    Returns a description of what was undone, or None if nothing to undo.
    """
    row = conn.execute(
        "SELECT * FROM history WHERE old_value IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None

    entry = dict(row)
    table = entry["table_name"]
    row_id = entry["row_id"]
    old = json.loads(entry["old_value"]) if entry["old_value"] else {}

    # Check if this was a creation action
    if isinstance(old, dict) and old.get("_action") == "create":
        if table in ("words", "clips", "phrases", "rules", "pending_sounds", "word_variants"):
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            if table == "words":
                conn.execute("DELETE FROM word_variants WHERE word_id = ?", (row_id,))
            conn.commit()
        desc = f"Undid creation of {table} #{row_id}"
    elif old and table in ("words", "clips", "phrases", "rules"):
        # Apply the old values back
        sets = ", ".join(f"{k} = ?" for k in old)
        vals = list(old.values()) + [row_id]
        conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", vals)
        conn.commit()
        desc = f"Reverted {table} #{row_id}: {old}"
    else:
        desc = f"Reverted {table} #{row_id}"

    # Remove this history entry so it can't be undone twice
    conn.execute("DELETE FROM history WHERE id = ?", (entry["id"],))
    conn.commit()

    return desc


def get_recent_history(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Utility: list distinct values (for filters) ──────────────────

def distinct_labels(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT label FROM clips ORDER BY label").fetchall()
    return [r["label"] for r in rows]


def distinct_speakers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT speaker FROM clips ORDER BY speaker"
    ).fetchall()
    return [r["speaker"] for r in rows]


# ── CRUD: features ────────────────────────────────────────────────

def save_features(
    conn: sqlite3.Connection,
    clip_id: int,
    vector: list[float],
    duration: float,
) -> int:
    """Insert or replace a feature row for a clip."""
    vec_json = json.dumps(vector)
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO features (clip_id, vector, duration, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (clip_id, vec_json, duration, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_features(conn: sqlite3.Connection, clip_id: int) -> dict | None:
    """Return features dict for a clip, or None."""
    row = conn.execute(
        "SELECT * FROM features WHERE clip_id = ?", (clip_id,)
    ).fetchone()
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
    """Return clips that have no entry in the features table."""
    return conn.execute(
        """
        SELECT c.* FROM clips c
        LEFT JOIN features f ON c.id = f.clip_id
        WHERE f.id IS NULL
        ORDER BY c.id
        """
    ).fetchall()


# ── CRUD: phrases ─────────────────────────────────────────────────

def save_phrase(
    conn: sqlite3.Connection,
    text: str,
    meaning: str | None = None,
    status: str = "unverified",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO phrases (text, meaning, status, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (text, meaning, status, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_phrase_by_text(conn: sqlite3.Connection, text: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM phrases WHERE text = ?", (text,)
    ).fetchone()
    return dict(row) if row else None


def get_all_phrases(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM phrases ORDER BY text").fetchall()
    return [dict(r) for r in rows]


def update_phrase(
    conn: sqlite3.Connection,
    phrase_id: int,
    **fields,
) -> None:
    old = conn.execute("SELECT * FROM phrases WHERE id = ?", (phrase_id,)).fetchone()
    if old is None:
        return
    old = dict(old)
    for col, new_val in fields.items():
        if col in ("id", "created_at"):
            continue
        old_val = old.get(col)
        if str(old_val) != str(new_val):
            log_history(conn, "phrases", phrase_id,
                        json.dumps({col: old_val}),
                        json.dumps({col: new_val}))
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [phrase_id]
    conn.execute(f"UPDATE phrases SET {sets} WHERE id = ?", vals)
    conn.commit()


# ── CRUD: rules ───────────────────────────────────────────────────

def save_rule(
    conn: sqlite3.Connection,
    rule_type: str,
    definition: dict,
    status: str = "unverified",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO rules (rule_type, definition, status, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (rule_type, json.dumps(definition), status, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_all_rules(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM rules ORDER BY rule_type, id").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["definition"] = json.loads(d["definition"])
        result.append(d)
    return result


def get_rules_by_type(conn: sqlite3.Connection, rule_type: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM rules WHERE rule_type = ? ORDER BY id",
        (rule_type,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["definition"] = json.loads(d["definition"])
        result.append(d)
    return result


def get_rule_by_id(conn: sqlite3.Connection, rule_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["definition"] = json.loads(d["definition"])
    return d


def update_rule(
    conn: sqlite3.Connection,
    rule_id: int,
    **fields,
) -> None:
    old = get_rule_by_id(conn, rule_id)
    if old is None:
        return
    for col, new_val in fields.items():
        if col in ("id", "created_at"):
            continue
        old_val = old.get(col)
        if col == "definition":
            old_ser = json.dumps(old_val)
            new_ser = json.dumps(new_val)
            if old_ser != new_ser:
                log_history(conn, "rules", rule_id, json.dumps({col: old_val}),
                            json.dumps({col: new_val}))
        elif str(old_val) != str(new_val):
            log_history(conn, "rules", rule_id,
                        json.dumps({col: old_val}),
                        json.dumps({col: new_val}))
    update_fields = {}
    for k, v in fields.items():
        if k == "definition":
            update_fields[k] = json.dumps(v)
        else:
            update_fields[k] = v
    sets = ", ".join(f"{k} = ?" for k in update_fields)
    vals = list(update_fields.values()) + [rule_id]
    conn.execute(f"UPDATE rules SET {sets} WHERE id = ?", vals)
    conn.commit()


def delete_rule(conn: sqlite3.Connection, rule_id: int) -> None:
    conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
    conn.commit()


# ── CRUD: sequences ──────────────────────────────────────────────

def save_sequence(
    conn: sqlite3.Connection,
    filename: str,
    recognized: list[dict],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO sequences (filename, recognized, created_at)
        VALUES (?, ?, ?)
        """,
        (filename, json.dumps(recognized), _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_all_sequences(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM sequences ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["recognized"] = json.loads(d["recognized"])
        result.append(d)
    return result


# ── CRUD: transcription_log ──────────────────────────────────────

def save_transcription_log(
    conn: sqlite3.Connection,
    segment_index: int,
    heard_guess: str | None,
    user_edit: str | None,
    final_spelling: str | None,
    clip_hash: str | None = None,
) -> int:
    """Log a transcription guess and the user's edit for accuracy tracking."""
    cur = conn.execute(
        """
        INSERT INTO transcription_log
            (segment_index, heard_guess, user_edit, final_spelling, clip_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (segment_index, heard_guess, user_edit, final_spelling, clip_hash, _utcnow()),
    )
    conn.commit()
    return cur.lastrowid


def get_transcription_log(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Return recent transcription log entries."""
    rows = conn.execute(
        "SELECT * FROM transcription_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def transcription_accuracy(conn: sqlite3.Connection) -> dict:
    """Compute how often the heard guess matched the final spelling."""
    rows = conn.execute(
        "SELECT heard_guess, final_spelling FROM transcription_log "
        "WHERE heard_guess IS NOT NULL AND final_spelling IS NOT NULL"
    ).fetchall()
    total = len(rows)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(
        1 for r in rows
        if r["heard_guess"].strip().lower() == r["final_spelling"].strip().lower()
    )
    return {"total": total, "correct": correct, "accuracy": correct / total}

