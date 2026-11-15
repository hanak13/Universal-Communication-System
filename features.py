"""
UCS Feature Extraction — librosa-based audio analysis.
No Streamlit imports so this module can be reused in CLI / batch pipelines.

extract_features(path) → (vector: list[float], raw_dict: dict)
"""
import numpy as np
import librosa

from config import (
    FEATURE_SR,
    N_MFCC,
    SILENCE_TOP_DB,
    PITCH_FMIN,
    PITCH_FMAX,
    N_FFT,
    HOP_LENGTH,
    FEATURE_VECTOR_LEN,
)


# ── Core pipeline ─────────────────────────────────────────────────

def _load_and_prepare(path: str) -> tuple[np.ndarray, int]:
    """Load audio, resample to FEATURE_SR, normalise, trim silence."""
    y, sr = librosa.load(path, sr=FEATURE_SR, mono=True)

    # Normalise amplitude to [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)

    return y, sr


def _mfcc_stats(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) arrays of shape (N_MFCC,)."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)


def _pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (pitch_mean, pitch_std) via pyin, NaN-safe."""
    f0, voiced, _ = librosa.pyin(
        y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
        hop_length=HOP_LENGTH,
    )
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) == 0:
        return 0.0, 0.0
    return float(np.mean(f0_valid)), float(np.std(f0_valid))


def _spectral_centroid_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (centroid_mean, centroid_std)."""
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )[0]
    return float(np.mean(cent)), float(np.std(cent))


# ── Public API ────────────────────────────────────────────────────

def extract_features_from_array(y: np.ndarray, sr: int = FEATURE_SR) -> tuple[list[float], dict]:
    """
    Extract features directly from a 1D audio numpy array and return:
      vector – list of FEATURE_VECTOR_LEN floats (for DB storage / ML)
      raw    – dict with labelled values + numpy arrays for plotting
    """
    duration = float(len(y) / sr)

    mfcc_mean, mfcc_std = _mfcc_stats(y, sr)
    pitch_mean, pitch_std = _pitch_stats(y, sr)
    cent_mean, cent_std = _spectral_centroid_stats(y, sr)

    # Assemble the flat feature vector
    vector = (
        mfcc_mean.tolist()
        + mfcc_std.tolist()
        + [pitch_mean, pitch_std]
        + [cent_mean, cent_std]
        + [duration]
    )
    assert len(vector) == FEATURE_VECTOR_LEN, (
        f"Vector length {len(vector)} != expected {FEATURE_VECTOR_LEN}"
    )

    # Raw dict carries everything a UI needs for plots
    raw = {
        "y": y,
        "sr": sr,
        "duration": duration,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "centroid_mean": cent_mean,
        "centroid_std": cent_std,
        # Full matrices for heatmap / spectrogram
        "mfcc_matrix": librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        ),
        "stft": np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)),
    }

    return vector, raw


def extract_features(path: str) -> tuple[list[float], dict]:
    """
    Analyse an audio file and return (vector, raw).
    """
    y, sr = _load_and_prepare(path)
    return extract_features_from_array(y, sr)

