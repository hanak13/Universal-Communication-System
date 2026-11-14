"""
tests/test_clip.py — saves a clip to a temp DB and reads it back.
Run:  python -m pytest tests/
"""
import os
import sys
import tempfile
import wave

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import init_db, save_clip, get_clips, get_clip_by_id


def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 44100) -> bytes:
    """Generate a silent WAV file in memory and return its bytes."""
    import io
    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def test_save_and_read_clip():
    """Insert a clip row, read it back, verify fields."""
    # Use a temporary DB so tests never touch the real one
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_ucs.db")
        conn = init_db(db_path)

        # Save
        filename = "hello_alice_001.wav"
        clip_id = save_clip(
            conn,
            filename=filename,
            label="hello",
            speaker="alice",
            mic_type="usb",
            noise_level=0.02,
        )
        assert clip_id is not None
        assert clip_id > 0

        # Read back by id
        row = get_clip_by_id(conn, clip_id)
        assert row is not None
        assert row["filename"] == filename
        assert row["label"] == "hello"
        assert row["speaker"] == "alice"
        assert row["mic_type"] == "usb"
        assert row["noise_level"] == 0.02

        # Read back via filter
        rows = get_clips(conn, label="hello")
        assert len(rows) == 1
        assert rows[0]["id"] == clip_id

        # Filter that matches nothing
        empty = get_clips(conn, speaker="bob")
        assert len(empty) == 0

        conn.close()


def test_tables_exist():
    """Verify all required tables are created."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_ucs.db")
        conn = init_db(db_path)

        expected = {
            "clips", "words", "word_variants", "phrases",
            "rules", "pending_sounds", "history", "schema_version",
        }
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {r["name"] for r in rows}

        for tbl in expected:
            assert tbl in tables, f"Missing table: {tbl}"

        conn.close()


if __name__ == "__main__":
    test_save_and_read_clip()
    test_tables_exist()
    print("All tests passed ✓")

