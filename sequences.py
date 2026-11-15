"""
UCS Sequences — split a long audio clip into word segments,
run each through feature extraction + recognizer.
No Streamlit imports.
"""
import os
from collections import Counter
import numpy as np
import librosa

from config import (
    FEATURE_SR, SILENCE_TOP_DB, SPLIT_TOP_DB_DEFAULT,
    MIN_SEGMENT_DURATION, FEATURE_VECTOR_LEN,
    RECOGNIZE_THRESHOLD, PENDING_MERGE_THRESHOLD,
)
from features import extract_features, extract_features_from_array


# ── Segment splitting ────────────────────────────────────────────

def split_audio(path: str, top_db: float = SPLIT_TOP_DB_DEFAULT) -> list[dict]:
    """
    Split a long audio file into non-silent segments.
    Returns list of {"start_sample", "end_sample", "start_sec", "end_sec", "y"}.
    """
    y, sr = librosa.load(path, sr=FEATURE_SR, mono=True)

    # Normalise
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    intervals = librosa.effects.split(y, top_db=top_db)

    segments = []
    for start, end in intervals:
        duration = (end - start) / sr
        if duration < MIN_SEGMENT_DURATION:
            continue
        segments.append({
            "start_sample": int(start),
            "end_sample": int(end),
            "start_sec": float(start / sr),
            "end_sec": float(end / sr),
            "y": y[start:end],
        })

    return segments


def extract_segment_features(segment_y: np.ndarray, sr: int = FEATURE_SR) -> tuple[list[float], dict]:
    """
    Extract features from a raw audio segment (numpy array).
    Reuses the core extraction pipeline from features module.
    """
    return extract_features_from_array(segment_y, sr=sr)


# ── Recognize a full sequence ────────────────────────────────────

def recognize_sequence(
    segments: list[dict],
    recognizer,
    pending_vectors: list[dict] | None = None,
) -> list[dict]:
    """
    Run each segment through the recognizer.
    Returns list of dicts: {label, distance, is_unknown, vector, start_sec, end_sec, ...}
    Unknown segments get new_word_N names; repeated similar unknowns reuse the same name.
    """
    from recognizer import find_or_create_pending_name

    if pending_vectors is None:
        pending_vectors = []

    results = []
    # Track unknowns within this sequence for reuse
    local_pending: list[dict] = list(pending_vectors)

    for seg in segments:
        vector, raw = extract_segment_features(seg["y"])
        rec = recognizer.recognize(vector)

        entry = {
            "label": rec["label"],
            "distance": rec["distance"],
            "is_unknown": rec["is_unknown"],
            "top3": rec["top3"],
            "vector": vector,
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "duration": raw["duration"],
        }

        if rec["is_unknown"]:
            # Find or create a pending name
            name, reused = find_or_create_pending_name(
                vector, local_pending,
            )
            entry["label"] = name
            entry["pending_name"] = name
            entry["pending_reused"] = reused
            # Add to local tracking
            if not reused:
                local_pending.append({"name": name, "vector": vector})

        results.append(entry)

    return results


# ── N-gram counting ──────────────────────────────────────────────

def count_ngrams(labels: list[str], n: int) -> dict[str, int]:
    """
    Count n-grams from a list of word labels.
    Returns {ngram_string: count} sorted by count descending.
    """
    if len(labels) < n:
        return {}
    grams = (" ".join(labels[i : i + n]) for i in range(len(labels) - n + 1))
    return dict(Counter(grams).most_common())


def compare_phrase_vs_words(
    phrase_meaning: str | None,
    word_meanings: list[str | None],
) -> dict:
    """
    Compare a phrase meaning with joined word meanings.
    Returns {"phrase_meaning", "joined_words", "match"}.
    """
    joined = " ".join(m or "?" for m in word_meanings)
    return {
        "phrase_meaning": phrase_meaning or "?",
        "joined_words": joined,
        "match": (phrase_meaning or "").strip().lower() == joined.strip().lower(),
    }
