"""
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
        raw_frames = b"\x00\x00" * n_frames

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
