"""
create_phase_zips.py — Generates 4 cumulative project zip archives:
  - ucs_phase1.zip: Phase 1 only (Recording, Library, Audio I/O, DB V1)
  - ucs_phase2.zip: Phase 1 + 2 (Feature extraction, Spectrogram/MFCC Analysis page, DB V2)
  - ucs_phase3.zip: Phase 1 + 2 + 3 (Clustering, Pure-numpy KNN recognizer, Pending sounds, DB V3)
  - ucs_phase4.zip: Phase 1 + 2 + 3 + 4 (Words dictionary, Variants, Confidence Right/Wrong, Undo, DB V4)
  - ucs_phase5.zip: Phase 1 + 2 + 3 + 4 + 5 (Sequences, Rules, Sentence arrangement, DB V5)

Run:
  .venv\\Scripts\\python.exe create_phase_zips.py
"""
import os
import io
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_zip(zip_name: str, files_dict: dict[str, str]):
    out_path = os.path.join(BASE_DIR, zip_name)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, content in files_dict.items():
            zf.writestr(file_path, content.strip() + "\n")
    print(f"Created {zip_name} ({len(files_dict)} files, {os.path.getsize(out_path):,} bytes)")

# ─────────────────────────────────────────────────────────────────
# COMMON & PHASE 1 CODE
# ─────────────────────────────────────────────────────────────────

TESTS_INIT = '"""UCS Test Suite"""\n'

AUDIO_IO_PY = '''"""
UCS Audio I/O — recording, saving, loading WAV files.
Filename convention:  label_speaker_take.wav
"""
import io
import os
import re
import wave

from config import AUDIO_DIR, SAMPLE_RATE, CHANNELS, FILENAME_TEMPLATE


_FNAME_RE = re.compile(
    r"^(?P<label>[a-zA-Z0-9_]+)_(?P<speaker>[a-zA-Z0-9_]+)_(?P<take>\\d{3})\\.wav$"
)


def parse_filename(fname: str) -> dict | None:
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return {
        "label": m.group("label"),
        "speaker": m.group("speaker"),
        "take": int(m.group("take")),
    }


def next_take(label: str, speaker: str) -> int:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    max_take = 0
    prefix = f"{label}_{speaker}_"
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(prefix) and f.endswith(".wav"):
            parts = parse_filename(f)
            if parts and parts["take"] > max_take:
                max_take = parts["take"]
    return max_take + 1


def build_filename(label: str, speaker: str, take: int | None = None) -> str:
    if take is None:
        take = next_take(label, speaker)
    return FILENAME_TEMPLATE.format(label=label, speaker=speaker, take=take)


def save_wav(
    audio_bytes: bytes,
    filename: str,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sampwidth: int = 2,
) -> str:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return path


def save_wav_from_file(uploaded_bytes: bytes, filename: str) -> str:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_bytes)
    return path


def load_wav(filename: str) -> bytes | None:
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def wav_duration(filename: str) -> float:
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate if rate else 0.0


def list_recordings() -> list[str]:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
'''

TEST_CLIP_PY = '''"""
tests/test_clip.py — save a WAV, read it back, verify header values.
"""
import os
import sys
import tempfile
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_io import save_wav, parse_filename, build_filename
from db import init_db, save_clip, get_clips


def test_save_and_read_clip():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_wav = os.path.join(tmpdir, "test_alice_001.wav")
        sample_rate = 44100
        n_frames = sample_rate // 2
        raw_frames = b"\\x00\\x00" * n_frames

        with wave.open(test_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_frames)

        with wave.open(test_wav, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() == n_frames

    meta = parse_filename("hello_alice_001.wav")
    assert meta == {"label": "hello", "speaker": "alice", "take": 1}
    assert parse_filename("badname.wav") is None


def test_tables_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = init_db(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        expected = {
            "schema_version", "clips", "words", "word_variants",
            "phrases", "rules", "pending_sounds", "history",
        }
        assert expected.issubset(table_names)
        cid = save_clip(conn, filename="hi_bob_001.wav", label="hi", speaker="bob")
        clips = get_clips(conn)
        assert len(clips) == 1
        assert clips[0]["label"] == "hi"
        conn.close()
'''

# ─────────────────────────────────────────────────────────────────
# 1. PHASE 1 SNAPSHOT
# ─────────────────────────────────────────────────────────────────
REQ_P1 = """streamlit>=1.45.0
pytest>=8.0
"""

CONFIG_P1 = '''"""
UCS Configuration — Phase 1
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ucs.db")
AUDIO_DIR = os.path.join(BASE_DIR, "recordings")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_DURATION_S = 0.5
MAX_RECORD_S = 30
FILENAME_TEMPLATE = "{label}_{speaker}_{take:03d}.wav"
SCHEMA_VERSION = 1
'''

DB_P1 = '''"""
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
'''

APP_P1 = '''"""
UCS — Universal Communication System (Phase 1)
"""
import os
import streamlit as st

from db import (
    init_db, save_clip, get_clips,
    distinct_labels, distinct_speakers,
)
from audio_io import build_filename, save_wav_from_file, load_wav

if "conn" not in st.session_state:
    st.session_state.conn = init_db()

conn = st.session_state.conn

st.set_page_config(page_title="UCS — Universal Communication System", layout="wide")
page = st.sidebar.radio("Navigate", ["🎙️ Record", "📚 Library"])

if page == "🎙️ Record":
    st.title("🎙️ Record a Clip")
    st.caption("Record a word or phrase. Every meaning is a *hypothesis* until verified.")

    col1, col2 = st.columns(2)
    with col1:
        label = st.text_input("Label (what you think this sound means)").strip().lower().replace(" ", "_")
    with col2:
        speaker = st.text_input("Speaker name", value="speaker1").strip().lower().replace(" ", "_")

    meaning = st.text_input("Meaning / translation (optional hypothesis)").strip()
    audio_data = st.audio_input("🎤 Press to record")

    if audio_data is not None:
        if not label:
            st.warning("Please enter a label before saving.")
        else:
            filename = build_filename(label, speaker)
            wav_bytes = audio_data.getvalue()
            save_wav_from_file(wav_bytes, filename)
            clip_id = save_clip(conn, filename=filename, label=label, speaker=speaker)
            st.success(f"✅ Saved **{filename}** (clip #{clip_id})")
            st.audio(wav_bytes, format="audio/wav")

elif page == "📚 Library":
    st.title("📚 Clip Library")
    labels = ["(all)"] + distinct_labels(conn)
    speakers = ["(all)"] + distinct_speakers(conn)

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_label = st.selectbox("Filter by label", labels)
    with fc2:
        sel_speaker = st.selectbox("Filter by speaker", speakers)

    lbl = sel_label if sel_label != "(all)" else None
    spk = sel_speaker if sel_speaker != "(all)" else None
    clips = get_clips(conn, label=lbl, speaker=spk)

    if not clips:
        st.info("No clips yet. Go to **Record** to add some!")
    else:
        st.write(f"**{len(clips)}** clip(s)")
        for clip in clips:
            with st.expander(f"🔊 {clip['filename']}", expanded=False):
                wav = load_wav(clip["filename"])
                if wav:
                    st.audio(wav, format="audio/wav")
                st.markdown(
                    f"**Label:** {clip['label']}  \\n"
                    f"**Speaker:** {clip['speaker']}  \\n"
                    f"**Recorded:** {clip['created_at']}"
                )
'''

# ─────────────────────────────────────────────────────────────────
# 2. PHASE 2 SNAPSHOT (Phase 1 + 2)
# ─────────────────────────────────────────────────────────────────
REQ_P2 = """streamlit>=1.45.0
pytest>=8.0
librosa>=0.10.2
matplotlib>=3.8
numpy>=1.26
"""

def get_p2_files():
    # Read current features.py and test_features.py from disk
    with open(os.path.join(BASE_DIR, "features.py"), "r") as f:
        features_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_features.py"), "r") as f:
        test_features_py = f.read()

    config_p2 = CONFIG_P1.replace("SCHEMA_VERSION = 1", """FEATURE_SR = 16000
N_MFCC = 20
SILENCE_TOP_DB = 30
PITCH_FMIN = 50.0
PITCH_FMAX = 500.0
N_FFT = 2048
HOP_LENGTH = 512
FEATURE_VECTOR_LEN = 45
SCHEMA_VERSION = 2""")

    db_p2 = DB_P1.replace("def init_db(db_path", """_MIGRATION_V2 = \"\"\"
CREATE TABLE IF NOT EXISTS features (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id    INTEGER NOT NULL UNIQUE REFERENCES clips(id) ON DELETE CASCADE,
    vector     TEXT    NOT NULL,
    duration   REAL    NOT NULL,
    created_at TEXT    NOT NULL
);
\"\"\"

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

def init_db(db_path""").replace("_set_version(conn, 1)", """conn.executescript(_MIGRATION_V2)
        _set_version(conn, 2)""")

    app_p2 = APP_P1.replace('["🎙️ Record", "📚 Library"]', '["🎙️ Record", "📚 Library", "📊 Analysis"]').replace(
        "from db import (", "import numpy as np\nfrom db import (\n    save_features, get_features, clips_without_features,"
    ) + '''
elif page == "📊 Analysis":
    st.title("📊 Clip Analysis")
    from config import AUDIO_DIR, HOP_LENGTH
    pending = clips_without_features(conn)
    if pending:
        st.info(f"**{len(pending)}** clip(s) have no features yet.")
        if st.button("⚡ Analyze all clips"):
            from features import extract_features
            progress = st.progress(0, text="Analyzing…")
            for i, clip in enumerate(pending):
                clip_path = os.path.join(AUDIO_DIR, clip["filename"])
                if os.path.isfile(clip_path):
                    vector, raw = extract_features(clip_path)
                    save_features(conn, clip["id"], vector, raw["duration"])
                progress.progress((i + 1) / len(pending))
            st.success("✅ All clips analyzed!")
            st.rerun()
    
    all_clips = get_clips(conn)
    if all_clips:
        clip_names = {c["filename"]: c["id"] for c in all_clips}
        selected_file = st.selectbox("Select clip", list(clip_names.keys()))
        if selected_file:
            clip_id = clip_names[selected_file]
            clip_path = os.path.join(AUDIO_DIR, selected_file)
            wav = load_wav(selected_file)
            if wav:
                st.audio(wav, format="audio/wav")
            feat = get_features(conn, clip_id)
            if feat:
                from features import extract_features
                _, raw = extract_features(clip_path)
                import matplotlib.pyplot as plt
                import librosa.display
                st.subheader("Waveform")
                fig1, ax1 = plt.subplots(figsize=(8, 2))
                librosa.display.waveshow(raw["y"], sr=raw["sr"], ax=ax1)
                st.pyplot(fig1)
                st.subheader("Spectrogram")
                fig2, ax2 = plt.subplots(figsize=(8, 2.5))
                S_db = librosa.amplitude_to_db(raw["stft"], ref=np.max)
                librosa.display.specshow(S_db, sr=raw["sr"], hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=ax2)
                st.pyplot(fig2)
                st.subheader("Feature Values")
                st.json({"duration": raw["duration"], "pitch_mean": raw["pitch_mean"], "centroid_mean": raw["centroid_mean"]})
'''

    return {
        "requirements.txt": REQ_P2,
        "config.py": config_p2,
        "audio_io.py": AUDIO_IO_PY,
        "db.py": db_p2,
        "features.py": features_py,
        "app.py": app_p2,
        "tests/__init__.py": TESTS_INIT,
        "tests/test_clip.py": TEST_CLIP_PY,
        "tests/test_features.py": test_features_py,
    }

# ─────────────────────────────────────────────────────────────────
# 3. PHASE 3 SNAPSHOT (Phase 1 + 2 + 3)
# ─────────────────────────────────────────────────────────────────
def get_p3_files():
    p2 = get_p2_files()
    with open(os.path.join(BASE_DIR, "clustering.py"), "r") as f:
        clustering_py = f.read()
    with open(os.path.join(BASE_DIR, "recognizer.py"), "r") as f:
        recognizer_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_clustering.py"), "r") as f:
        test_clustering_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_recognizer.py"), "r") as f:
        test_recognizer_py = f.read()

    config_p3 = p2["config.py"].replace("SCHEMA_VERSION = 2", """RECOGNIZE_THRESHOLD = 15.0
PENDING_MERGE_THRESHOLD = 10.0
KNN_NEIGHBORS = 3
TSNE_PERPLEXITY = 5
DBSCAN_EPS_DEFAULT = 5.0
DBSCAN_MIN_SAMPLES_DEFAULT = 2
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.json")
MIN_CLIPS_FOR_CLUSTERING = 4
SCHEMA_VERSION = 3""")

    db_p3 = p2["db.py"].replace("_set_version(conn, 2)", """_safe_alter(conn, "ALTER TABLE pending_sounds ADD COLUMN clip_id INTEGER REFERENCES clips(id);\\nALTER TABLE pending_sounds ADD COLUMN vector TEXT;")
        _set_version(conn, 3)""") + '''
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
'''

    # For app_p3, extract lines up to the end of Pending page from current app.py
    with open(os.path.join(BASE_DIR, "app.py"), "r") as f:
        curr_app = f.read()

    # App p3 has Record, Library, Analysis, Clustering, Pending
    app_p3_cut = curr_app.split('#  PAGE: Words')[0]
    app_p3 = app_p3_cut.replace(
        'PAGES = [\n    "🎙️ Record",\n    "📚 Library",\n    "📊 Analysis",\n    "🔬 Clustering",\n    "❓ Pending",\n    "📝 Words",\n    "🔗 Sequences",\n    "📏 Rules",\n    "✅ Test",\n    "📋 Summary",\n]',
        'PAGES = [\n    "🎙️ Record",\n    "📚 Library",\n    "📊 Analysis",\n    "🔬 Clustering",\n    "❓ Pending",\n]'
    )

    req_p3 = """streamlit>=1.45.0
pytest>=8.0
librosa>=0.10.2
matplotlib>=3.8
numpy>=1.26
pandas>=2.1
"""

    return {
        "requirements.txt": req_p3,
        "config.py": config_p3,
        "audio_io.py": AUDIO_IO_PY,
        "db.py": db_p3,
        "features.py": p2["features.py"],
        "clustering.py": clustering_py,
        "recognizer.py": recognizer_py,
        "app.py": app_p3,
        "tests/__init__.py": TESTS_INIT,
        "tests/test_clip.py": TEST_CLIP_PY,
        "tests/test_features.py": p2["tests/test_features.py"],
        "tests/test_clustering.py": test_clustering_py,
        "tests/test_recognizer.py": test_recognizer_py,
    }

# ─────────────────────────────────────────────────────────────────
# 4. PHASE 4 SNAPSHOT (Phase 1 + 2 + 3 + 4)
# ─────────────────────────────────────────────────────────────────
def get_p4_files():
    with open(os.path.join(BASE_DIR, "clustering.py"), "r") as f:
        clustering_py = f.read()
    with open(os.path.join(BASE_DIR, "recognizer.py"), "r") as f:
        recognizer_py = f.read()
    with open(os.path.join(BASE_DIR, "features.py"), "r") as f:
        features_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_features.py"), "r") as f:
        test_features_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_clustering.py"), "r") as f:
        test_clustering_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_recognizer.py"), "r") as f:
        test_recognizer_py = f.read()
    with open(os.path.join(BASE_DIR, "tests", "test_words.py"), "r") as f:
        test_words_py = f.read()

    # Config Phase 4
    with open(os.path.join(BASE_DIR, "config.py"), "r") as f:
        config_full = f.read()
    # Strip Phase 5 additions from config
    config_p4 = config_full.split("# ── Sequences (Phase 5)")[0] + "# ── Schema version (bumped on every migration) ────────────────────\nSCHEMA_VERSION = 4\n"

    # DB Phase 4
    with open(os.path.join(BASE_DIR, "db.py"), "r") as f:
        db_full = f.read()
    db_p4 = db_full.split("# ── CRUD: phrases")[0].replace("SCHEMA_VERSION = 5", "SCHEMA_VERSION = 4").replace(
        "        # V5: sequences table\n        if cur_ver < 5:\n            conn.executescript(_MIGRATION_V5)\n", ""
    )

    # App Phase 4 (has Record, Library, Analysis, Clustering, Pending, Words, Test, Summary)
    with open(os.path.join(BASE_DIR, "app.py"), "r") as f:
        app_full = f.read()
    # Remove Sequences and Rules pages from app_full
    app_p4_part1 = app_full.split('#  PAGE: Sequences')[0]
    app_p4_part2 = '#  PAGE: Test' + app_full.split('#  PAGE: Test')[1]
    app_p4 = app_p4_part1 + app_p4_part2
    app_p4 = app_p4.replace(
        'PAGES = [\n    "🎙️ Record",\n    "📚 Library",\n    "📊 Analysis",\n    "🔬 Clustering",\n    "❓ Pending",\n    "📝 Words",\n    "🔗 Sequences",\n    "📏 Rules",\n    "✅ Test",\n    "📋 Summary",\n]',
        'PAGES = [\n    "🎙️ Record",\n    "📚 Library",\n    "📊 Analysis",\n    "🔬 Clustering",\n    "❓ Pending",\n    "📝 Words",\n    "✅ Test",\n    "📋 Summary",\n]'
    )

    req_p4 = """streamlit>=1.45.0
pytest>=8.0
librosa>=0.10.2
matplotlib>=3.8
numpy>=1.26
pandas>=2.1
"""

    return {
        "requirements.txt": req_p4,
        "config.py": config_p4,
        "audio_io.py": AUDIO_IO_PY,
        "db.py": db_p4,
        "features.py": features_py,
        "clustering.py": clustering_py,
        "recognizer.py": recognizer_py,
        "app.py": app_p4,
        "tests/__init__.py": TESTS_INIT,
        "tests/test_clip.py": TEST_CLIP_PY,
        "tests/test_features.py": test_features_py,
        "tests/test_clustering.py": test_clustering_py,
        "tests/test_recognizer.py": test_recognizer_py,
        "tests/test_words.py": test_words_py,
    }

def main():
    print("Building Phase 1 Zip...")
    create_zip("ucs_phase1.zip", {
        "requirements.txt": REQ_P1,
        "config.py": CONFIG_P1,
        "audio_io.py": AUDIO_IO_PY,
        "db.py": DB_P1,
        "app.py": APP_P1,
        "tests/__init__.py": TESTS_INIT,
        "tests/test_clip.py": TEST_CLIP_PY,
    })

    print("Building Phase 2 Zip (Phase 1 + 2)...")
    create_zip("ucs_phase2.zip", get_p2_files())

    print("Building Phase 3 Zip (Phase 1 + 2 + 3)...")
    create_zip("ucs_phase3.zip", get_p3_files())

    print("Building Phase 4 Zip (Phase 1 + 2 + 3 + 4)...")
    create_zip("ucs_phase4.zip", get_p4_files())

    print("\n✅ All 4 phase zip files successfully generated!")

if __name__ == "__main__":
    main()

