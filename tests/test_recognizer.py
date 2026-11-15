"""
tests/test_recognizer.py — recognizer returns "unknown" for far vectors,
reuses pending names for nearby unknowns, and performs majority voting among K nearest neighbors.
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


def test_majority_voting_toy_case():
    """
    Toy case confirming majority voting fix (issue #1):
    Nearest sample is class A, but 2 of 3 nearest samples are class B (K=3).
    1-NN would pick A; majority voting picks B.
    """
    r = Recognizer(k=3)
    v_a = [1.0] * FEATURE_VECTOR_LEN
    v_b1 = [1.2] * FEATURE_VECTOR_LEN
    v_b2 = [1.3] * FEATURE_VECTOR_LEN
    # Query very close to A
    v_query = [1.05] * FEATURE_VECTOR_LEN

    r.fit([v_a, v_b1, v_b2], ["A", "B", "B"])
    res = r.recognize(v_query, threshold=50.0)

    # 1-NN would have returned "A"
    # KNN majority voting must return "B"
    assert res["label"] == "B"
    assert not res["is_unknown"]
    assert res["confidence"] == 2 / 3
    assert len(res["top3"]) == 2
    assert res["top3"][0].label == "B"
    assert res["top3"][0].votes == 2
    assert res["top3"][1].label == "A"
    assert res["top3"][1].votes == 1


def test_majority_voting_tie_break():
    """
    Tie breaking among equal votes:
    1) Primary: smaller sum of distances among voting samples.
    2) Secondary: closest single sample.
    3) Stable fallback: alphabetical label order.
    """
    r = Recognizer(k=2)
    v_a = [1.0] * FEATURE_VECTOR_LEN
    v_b = [3.0] * FEATURE_VECTOR_LEN
    v_query = [1.1] * FEATURE_VECTOR_LEN

    r.fit([v_a, v_b], ["A", "B"])
    res = r.recognize(v_query, threshold=50.0)
    assert res["label"] == "A"
    assert res["confidence"] == 1 / 2


def test_fewer_samples_than_k():
    """Works when training set has fewer samples than K."""
    r = Recognizer(k=5)
    v1 = [1.0] * FEATURE_VECTOR_LEN
    v2 = [2.0] * FEATURE_VECTOR_LEN
    r.fit([v1, v2], ["hello", "bye"])
    assert r.is_fitted
    assert r._k == 2

    res = r.recognize(v1, threshold=50.0)
    assert res["label"] == "hello"
    assert not res["is_unknown"]
    assert res["confidence"] == 1.0


def test_one_class_data():
    """Recognizer handles training data containing only a single class."""
    r = Recognizer(k=3)
    v1 = [1.0] * FEATURE_VECTOR_LEN
    v2 = [1.1] * FEATURE_VECTOR_LEN
    v3 = [1.2] * FEATURE_VECTOR_LEN
    r.fit([v1, v2, v3], ["solo", "solo", "solo"])
    assert r.is_fitted

    res = r.recognize([1.05] * FEATURE_VECTOR_LEN, threshold=50.0)
    assert res["label"] == "solo"
    assert not res["is_unknown"]
    assert res["confidence"] == 1.0
    assert len(res["top3"]) == 1
    assert res["top3"][0].label == "solo"
    assert res["top3"][0].votes == 3


def test_return_keys_unchanged():
    """Output dict retains exact expected keys and shapes."""
    r = Recognizer(k=3)
    v = [1.0] * FEATURE_VECTOR_LEN
    r.fit([v], ["word"])
    res = r.recognize(v, threshold=50.0)
    expected_keys = {"label", "distance", "top3", "is_unknown", "confidence"}
    assert set(res.keys()) == expected_keys


if __name__ == "__main__":
    test_recognize_known()
    test_recognize_unknown()
    test_recognize_top3()
    test_pending_name_reuse()
    test_pending_name_new()
    test_pending_name_empty()
    test_recognizer_unfitted()
    test_majority_voting_toy_case()
    test_majority_voting_tie_break()
    test_fewer_samples_than_k()
    test_one_class_data()
    test_return_keys_unchanged()
    print("All recognizer tests passed ✓")
