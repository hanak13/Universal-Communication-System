"""
tests/test_quick_recognize.py — Phase 7 tests:
(a) empty database makes every segment unknown and the form appears for the first one,
(b) saving a word then re-running recognizes it,
(c) Skip keeps new_word_N in Pending,
(d) "Same as existing word" adds a clip to the existing word,
(e) undo reverts the last saved word.
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FEATURE_VECTOR_LEN, UNKNOWN_DISTANCE_THRESHOLD,
    DUPLICATE_SOUND_THRESHOLD,
)
from db import (
    init_db, save_clip, get_clips, save_features, get_features,
    save_word, get_word_by_canonical, get_word_by_id, get_all_words,
    save_pending_sound, get_all_pending_sounds, next_pending_name,
    undo_last_change,
)
from recognizer import Recognizer
from audio_io import build_filename, audio_array_to_wav_bytes


def _make_temp_db():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_p7.db")
    return init_db(db_path), tmp


def _make_dummy_vector(val: float = 1.0) -> list[float]:
    return [val] * FEATURE_VECTOR_LEN


def test_empty_database_makes_every_segment_unknown():
    """(a) Empty DB means recognizer is unfitted; all segments are unknown, first is unresolved."""
    conn, _ = _make_temp_db()

    # Empty DB
    clips = get_clips(conn)
    assert len(clips) == 0

    recog = Recognizer()
    assert not recog.is_fitted

    # 3 dummy segments in a sequence
    vectors = [_make_dummy_vector(1.0), _make_dummy_vector(5.0), _make_dummy_vector(10.0)]
    results = [recog.recognize(v, threshold=UNKNOWN_DISTANCE_THRESHOLD) for v in vectors]

    # Every segment is unknown
    for r in results:
        assert r["is_unknown"] is True
        assert r["label"] == "unknown"

    # In left-to-right processing, the first unknown segment is index 0
    first_unresolved_idx = next(i for i, r in enumerate(results) if r["is_unknown"])
    assert first_unresolved_idx == 0
    conn.close()


def test_saving_word_then_rerun_recognizes():
    """(b) Saving a word and adding its clip features allows immediate re-recognition."""
    conn, _ = _make_temp_db()

    v1 = _make_dummy_vector(2.0)
    v2 = _make_dummy_vector(8.0)

    # 1. Initially unrecognized
    recog = Recognizer()
    assert recog.recognize(v1)["is_unknown"] is True

    # 2. User saves segment 0 as "inu"
    wid = save_word(
        conn, canonical="inu", meaning="you",
        word_type="pronoun", status="unverified",
    )
    cid = save_clip(conn, filename="inu_speaker1_001.wav", label="inu", speaker="speaker1")
    save_features(conn, cid, v1, duration=0.5)

    # 3. Retrain recognizer (from 1 clip!)
    clips = get_clips(conn)
    clip_data = []
    for c in clips:
        f = get_features(conn, c["id"])
        if f:
            clip_data.append({"label": c["label"], "vector": f["vector"]})

    recog = Recognizer()
    recog.fit([d["vector"] for d in clip_data], [d["label"] for d in clip_data])
    assert recog.is_fitted

    # 4. Re-run recognition on the same clip: v1 is now recognized as "inu"
    res1 = recog.recognize(v1, threshold=UNKNOWN_DISTANCE_THRESHOLD)
    assert not res1["is_unknown"]
    assert res1["label"] == "inu"

    # v2 is still unknown
    res2 = recog.recognize(v2, threshold=UNKNOWN_DISTANCE_THRESHOLD)
    assert res2["is_unknown"] is True

    conn.close()


def test_skip_keeps_new_word_in_pending():
    """(c) Skip creates new_word_N in pending_sounds."""
    conn, _ = _make_temp_db()

    v = _make_dummy_vector(3.5)
    p_name = next_pending_name(conn)
    assert p_name == "new_word_1"

    save_pending_sound(conn, name=p_name, vector=v)

    pending = get_all_pending_sounds(conn)
    assert len(pending) == 1
    assert pending[0]["name"] == "new_word_1"
    assert pending[0]["vector"] == v

    # Next pending name increments
    assert next_pending_name(conn) == "new_word_2"
    conn.close()


def test_same_as_existing_word_adds_clip():
    """(d) 'Same as existing word' links audio clip and features to existing word."""
    conn, _ = _make_temp_db()

    # Create word "inu"
    save_word(conn, canonical="inu", meaning="you", word_type="pronoun")
    c1 = save_clip(conn, filename="inu_speaker1_001.wav", label="inu", speaker="speaker1")
    v1 = _make_dummy_vector(1.0)
    save_features(conn, c1, v1, duration=0.4)

    assert len(get_clips(conn, label="inu")) == 1

    # New audio segment arrives with slightly different pronunciation v2
    v2 = _make_dummy_vector(1.2)
    c2 = save_clip(conn, filename="inu_speaker1_002.wav", label="inu", speaker="speaker1")
    save_features(conn, c2, v2, duration=0.45)

    # Word now has 2 clips
    inu_clips = get_clips(conn, label="inu")
    assert len(inu_clips) == 2

    # Recognizer trained with both clips
    recog = Recognizer()
    recog.fit([v1, v2], ["inu", "inu"])
    assert not recog.recognize(v2)["is_unknown"]
    conn.close()


def test_undo_reverts_last_saved_word():
    """(e) Undo reverts the last saved word."""
    conn, _ = _make_temp_db()

    wid = save_word(conn, canonical="manji", meaning="are", word_type="verb")
    assert get_word_by_canonical(conn, "manji") is not None

    desc = undo_last_change(conn)
    assert desc is not None
    assert "words" in desc

    assert get_word_by_canonical(conn, "manji") is None
    conn.close()


if __name__ == "__main__":
    test_empty_database_makes_every_segment_unknown()
    test_saving_word_then_rerun_recognizes()
    test_skip_keeps_new_word_in_pending()
    test_same_as_existing_word_adds_clip()
    test_undo_reverts_last_saved_word()
    print("All Phase 7 tests passed ✓")

