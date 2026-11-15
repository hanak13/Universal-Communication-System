"""
Shared pytest fixtures and utilities for UCS test suite.
"""
import io
import wave
from pathlib import Path
import pytest

from db import init_db


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provide a fresh, isolated temporary database connection."""
    db_file = tmp_path / "test_ucs.db"
    conn = init_db(str(db_file))
    yield conn
    conn.close()


@pytest.fixture
def make_wav_bytes():
    """Factory fixture to generate silent or synthetic WAV bytes."""
    def _generator(duration_s: float = 0.5, sample_rate: int = 44100) -> bytes:
        n_frames = int(sample_rate * duration_s)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * n_frames)
        return buf.getvalue()
    return _generator
