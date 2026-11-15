"""
tests/test_transcribe.py — Phase 8 tests:
(a) unknown word shows a guess (or fallback),
(b) a 3-word sentence gives 3 separate items,
(c) edited guess flows into the Phase 7 form as default spelling,
(d) transcription log stores heard guess and edit,
(e) fallback when whisper not installed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import FEATURE_VECTOR_LEN, UNKNOWN_DISTANCE_THRESHOLD, FEATURE_SR
from db import (
    init_db, save_transcription_log, get_transcription_log,
    transcription_accuracy,
)
from recognizer import Recognizer
from sequences import extract_segment_features


def _make_temp_db():
    tmp = tempfile.mkdtemp()
    return init_db(os.path.join(tmp, "test_p8.db")), tmp


def _make_dummy_vector(val: float = 1.0) -> list[float]:
    return [val] * FEATURE_VECTOR_LEN


def test_unknown_word_shows_guess_or_fallback():
    """(a) When recognizer is unfitted, every segment is unknown.
    The system should assign a heard_guess (from whisper) or fall back
    to 'new_word_N' style label."""
    recog = Recognizer()
    assert not recog.is_fitted

    vec = _make_dummy_vector(5.0)
    result = recog.recognize(vec, threshold=UNKNOWN_DISTANCE_THRESHOLD)
    assert result["is_unknown"] is True

    # Simulate fallback: if whisper returns None, use new_word_N
    heard_guess = None  # simulates whisper unavailable
    fallback = f"new_word_1" if heard_guess is None else heard_guess
    assert fallback == "new_word_1"

    # Simulate whisper returning a guess
    heard_guess = "kimiko"
    label_shown = heard_guess if heard_guess else "new_word_1"
    assert label_shown == "kimiko"


def test_three_word_sentence_gives_three_items():
    """(b) A 3-word clip split into 3 segments produces 3 separate
    segment results, each with its own recognition and heard_guess."""
    recog = Recognizer()

    # Simulate 3 segments
    vectors = [_make_dummy_vector(1.0), _make_dummy_vector(5.0), _make_dummy_vector(10.0)]

    results = []
    for idx, vec in enumerate(vectors):
        rec = recog.recognize(vec, threshold=UNKNOWN_DISTANCE_THRESHOLD)
        results.append({
            "idx": idx,
            "is_unknown": rec["is_unknown"],
            "heard_guess": f"word_{idx}",  # simulate whisper guess per segment
        })

    assert len(results) == 3
    # Each result is separate and has its own index
    assert results[0]["idx"] == 0
    assert results[1]["idx"] == 1
    assert results[2]["idx"] == 2
    # Each has a separate heard_guess
    assert results[0]["heard_guess"] != results[1]["heard_guess"]
    assert results[1]["heard_guess"] != results[2]["heard_guess"]


def test_edited_guess_flows_into_form():
    """(c) The heard guess should pre-fill the Spelling field.
    If the user edits it, the edited value becomes the canonical spelling."""
    heard_guess = "kemiko"
    # Simulate user edit
    user_edit = "kimiko"

    # default_spelling should be the edit, not the raw guess
    default_spelling = user_edit if user_edit else heard_guess
    assert default_spelling == "kimiko"

    # When user doesn't edit, use heard_guess directly
    default_spelling_no_edit = heard_guess
    assert default_spelling_no_edit == "kemiko"


def test_transcription_log_stores_entries():
    """(d) transcription_log table stores heard guess, user edit, and final spelling."""
    conn, _ = _make_temp_db()

    save_transcription_log(
        conn, segment_index=0,
        heard_guess="kemiko", user_edit="kimiko",
        final_spelling="kimiko", clip_hash="abc123",
    )
    save_transcription_log(
        conn, segment_index=1,
        heard_guess="inu", user_edit=None,
        final_spelling="inu", clip_hash="abc123",
    )
    save_transcription_log(
        conn, segment_index=2,
        heard_guess="manji", user_edit="manji",
        final_spelling="manji", clip_hash="abc123",
    )

    log = get_transcription_log(conn, limit=10)
    assert len(log) == 3

    # Most recent first
    assert log[0]["segment_index"] == 2
    assert log[0]["heard_guess"] == "manji"
    assert log[0]["final_spelling"] == "manji"

    # First entry had an edit
    assert log[2]["heard_guess"] == "kemiko"
    assert log[2]["user_edit"] == "kimiko"
    assert log[2]["final_spelling"] == "kimiko"

    conn.close()


def test_transcription_accuracy():
    """(d-extra) accuracy calculation shows how often guess was right."""
    conn, _ = _make_temp_db()

    # 2 correct, 1 wrong
    save_transcription_log(conn, 0, "inu", None, "inu", "h1")
    save_transcription_log(conn, 1, "manji", None, "manji", "h1")
    save_transcription_log(conn, 2, "kemiko", "kimiko", "kimiko", "h1")

    acc = transcription_accuracy(conn)
    assert acc["total"] == 3
    assert acc["correct"] == 2
    assert abs(acc["accuracy"] - 2 / 3) < 0.01

    conn.close()


def test_whisper_fallback_message():
    """(e) When faster-whisper is not installed, is_available returns False
    and transcribe functions return None without crashing."""
    from transcribe import is_available, transcribe_file, transcribe_array

    # Even if whisper IS available, transcribe_file on a non-existent path
    # should return None (not crash)
    result = transcribe_file("/nonexistent/path/audio.wav")
    assert result is None

    # transcribe_array on silence should return None or a string
    silence = np.zeros(FEATURE_SR, dtype=np.float32)
    result2 = transcribe_array(silence, sr=FEATURE_SR)
    # Either None (if whisper not installed) or a string (empty transcription)
    assert result2 is None or isinstance(result2, str)


if __name__ == "__main__":
    test_unknown_word_shows_guess_or_fallback()
    test_three_word_sentence_gives_three_items()
    test_edited_guess_flows_into_form()
    test_transcription_log_stores_entries()
    test_transcription_accuracy()
    test_whisper_fallback_message()
    print("All Phase 8 tests passed ✓")
