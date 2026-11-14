"""
tests/test_recognizer.py — recognizer returns "unknown" for far vectors
and reuses pending names for nearby unknowns.
Run:  python -m pytest tests/ -v
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FEATURE_VECTOR_LEN, RECOGNIZE_THRESHOLD, PENDING_MERGE_THRESHOLD
from recognizer import Recognizer, find_or_create_pending_name


def _make_vector(base: float = 0.0, noise: float = 0.0) -> list[float]:
    """Generate a dummy feature vector centred at `base`."""
    import random
    random.seed(42)
    return [base + random.uniform(-noise, noise) for _ in range(FEATURE_VECTOR_LEN)]


def test_recognize_known():
    """Recognizer returns the correct label for a vector close to training data."""
    r = Recognizer()
    v_hello = [1.0] * FEATURE_VECTOR_LEN
    v_bye   = [10.0] * FEATURE_VECTOR_LEN

    r.fit(
        [v_hello, v_bye],
        ["hello", "bye"],
    )
    assert r.is_fitted

    result = r.recognize(v_hello)
    assert result["label"] == "hello"
    assert not result["is_unknown"]


def test_recognize_unknown():
    """Recognizer returns 'unknown' when the distance exceeds the threshold."""
    r = Recognizer()
    v_hello = [1.0] * FEATURE_VECTOR_LEN
    v_bye   = [10.0] * FEATURE_VECTOR_LEN

    r.fit(
        [v_hello, v_bye],
        ["hello", "bye"],
    )

    # A vector very far from both
    v_far = [9999.0] * FEATURE_VECTOR_LEN
    result = r.recognize(v_far)
    assert result["is_unknown"]
    assert result["label"] == "unknown"
    assert result["distance"] > RECOGNIZE_THRESHOLD


def test_recognize_top3():
    """Top-3 list has at most 3 distinct labels."""
    r = Recognizer()
    vecs = [
        [1.0] * FEATURE_VECTOR_LEN,
        [5.0] * FEATURE_VECTOR_LEN,
        [10.0] * FEATURE_VECTOR_LEN,
        [15.0] * FEATURE_VECTOR_LEN,
    ]
    labels = ["alpha", "beta", "gamma", "delta"]
    r.fit(vecs, labels)

    result = r.recognize(vecs[0])
    assert len(result["top3"]) <= 3
    # First entry should be the closest label
    assert result["top3"][0][0] == "alpha"


def test_pending_name_reuse():
    """A new unknown near an existing pending reuses its name."""
    v1 = [5.0] * FEATURE_VECTOR_LEN
    v2 = [5.01] * FEATURE_VECTOR_LEN  # very close to v1

    existing_pending = [
        {"name": "new_word_1", "vector": v1},
    ]

    name, reused = find_or_create_pending_name(
        v2, existing_pending, threshold=PENDING_MERGE_THRESHOLD,
    )
    assert reused is True
    assert name == "new_word_1"


def test_pending_name_new():
    """A far unknown gets a new name."""
    v1 = [5.0] * FEATURE_VECTOR_LEN
    v_far = [9999.0] * FEATURE_VECTOR_LEN

    existing_pending = [
        {"name": "new_word_1", "vector": v1},
    ]

    name, reused = find_or_create_pending_name(
        v_far, existing_pending, threshold=PENDING_MERGE_THRESHOLD,
    )
    assert reused is False
    assert name == "new_word_2"


def test_pending_name_empty():
    """First pending sound ever gets new_word_1."""
    v = [1.0] * FEATURE_VECTOR_LEN
    name, reused = find_or_create_pending_name(v, [])
    assert reused is False
    assert name == "new_word_1"


def test_recognizer_unfitted():
    """An unfitted recognizer returns unknown gracefully."""
    r = Recognizer()
    result = r.recognize([0.0] * FEATURE_VECTOR_LEN)
    assert result["is_unknown"]
    assert result["label"] == "unknown"


if __name__ == "__main__":
    test_recognize_known()
    test_recognize_unknown()
    test_recognize_top3()
    test_pending_name_reuse()
    test_pending_name_new()
    test_pending_name_empty()
    test_recognizer_unfitted()
    print("All recognizer tests passed ✓")
