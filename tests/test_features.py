"""
tests/test_features.py — extract features from a generated sine wave.
Run:  python -m pytest tests/ -v
"""
import os
import sys
import tempfile
import wave
import struct
import math

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FEATURE_VECTOR_LEN


def _make_sine_wav(path: str, freq: float = 440.0, duration_s: float = 1.0,
                   sample_rate: int = 44100, amplitude: float = 0.8) -> str:
    """Generate a sine-wave WAV file on disk. Returns the path."""
    n_frames = int(sample_rate * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(n_frames):
            t = i / sample_rate
            val = amplitude * math.sin(2 * math.pi * freq * t)
            samples.append(struct.pack("<h", int(val * 32767)))
        wf.writeframes(b"".join(samples))
    return path


def test_extract_features_sine_wave():
    """extract_features on a 440 Hz sine returns a vector of fixed length."""
    from features import extract_features

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "sine_440.wav")
        _make_sine_wav(wav_path, freq=440.0, duration_s=1.0)

        vector, raw = extract_features(wav_path)

        # Vector is the right length
        assert isinstance(vector, list)
        assert len(vector) == FEATURE_VECTOR_LEN, (
            f"Expected {FEATURE_VECTOR_LEN}, got {len(vector)}"
        )

        # All entries are finite floats
        for i, v in enumerate(vector):
            assert isinstance(v, float), f"vector[{i}] is {type(v)}, not float"
            assert math.isfinite(v), f"vector[{i}] is not finite: {v}"

        # Duration should be close to 1 second (trim may shorten slightly)
        assert 0.5 < raw["duration"] < 1.5, f"Unexpected duration: {raw['duration']}"

        # Pitch mean should be near 440 Hz (within tolerance)
        assert 300 < raw["pitch_mean"] < 600, (
            f"Pitch mean {raw['pitch_mean']} not near 440 Hz"
        )

        # raw contains the numpy arrays for plotting
        assert "y" in raw
        assert "mfcc_matrix" in raw
        assert "stft" in raw


def test_extract_features_short_clip():
    """Very short clip (0.2 s) still produces a valid vector."""
    from features import extract_features

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "short.wav")
        _make_sine_wav(wav_path, freq=300.0, duration_s=0.2)

        vector, raw = extract_features(wav_path)
        assert len(vector) == FEATURE_VECTOR_LEN


if __name__ == "__main__":
    test_extract_features_sine_wave()
    test_extract_features_short_clip()
    print("All feature tests passed ✓")
