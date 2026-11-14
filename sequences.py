"""
UCS Sequences — split a long audio clip into word segments,
run each through feature extraction + recognizer.
No Streamlit imports.
"""
import os
import numpy as np
import librosa

from config import (
    FEATURE_SR, SILENCE_TOP_DB, SPLIT_TOP_DB_DEFAULT,
    MIN_SEGMENT_DURATION, FEATURE_VECTOR_LEN,
    RECOGNIZE_THRESHOLD, PENDING_MERGE_THRESHOLD,
)
from features import extract_features


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
    Like extract_features() but works on a pre-split numpy array
    instead of a file path.
    """
    from config import (
        N_MFCC, SILENCE_TOP_DB, PITCH_FMIN, PITCH_FMAX,
        N_FFT, HOP_LENGTH,
    )

    y = segment_y
    duration = float(len(y) / sr)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Pitch
    f0, voiced, _ = librosa.pyin(
        y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
        hop_length=HOP_LENGTH,
    )
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) == 0:
        pitch_mean, pitch_std = 0.0, 0.0
    else:
        pitch_mean = float(np.mean(f0_valid))
        pitch_std = float(np.std(f0_valid))

    # Spectral centroid
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    vector = (
        mfcc_mean.tolist()
        + mfcc_std.tolist()
        + [pitch_mean, pitch_std]
        + [cent_mean, cent_std]
        + [duration]
    )

    raw = {
        "duration": duration,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "centroid_mean": cent_mean,
        "centroid_std": cent_std,
    }

    return vector, raw


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

    counts: dict[str, int] = {}
    for i in range(len(labels) - n + 1):
        gram = " ".join(labels[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


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
