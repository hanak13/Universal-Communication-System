"""
UCS Transcribe — speech-to-text via faster-whisper (CPU, cached).
Falls back gracefully when the library is not installed.
No Streamlit imports.
"""
import os
import tempfile
import numpy as np

from config import FEATURE_SR

# ── Lazy-loaded model cache ────────────────────────────────────────

_whisper_model = None
_whisper_available: bool | None = None  # None = not yet checked


def _check_available() -> bool:
    """Check once whether faster-whisper can be imported."""
    global _whisper_available
    if _whisper_available is not None:
        return _whisper_available
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        _whisper_available = True
    except ImportError:
        _whisper_available = False
    return _whisper_available


def _get_model():
    """Load the Whisper model once and cache it in a module-level variable."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    if not _check_available():
        return None
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )
    return _whisper_model


def is_available() -> bool:
    """Return True if faster-whisper is installed and can be used."""
    return _check_available()


# ── Public API ─────────────────────────────────────────────────────

def transcribe_file(audio_path: str) -> str | None:
    """
    Transcribe a WAV file to text.
    Returns the full transcription string (lowercase, stripped) or None on failure.
    """
    model = _get_model()
    if model is None:
        return None
    try:
        segments, _info = model.transcribe(
            audio_path,
            language=None,       # auto-detect
            beam_size=3,
            word_timestamps=False,
            vad_filter=True,
        )
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
        full = " ".join(text_parts).strip().lower()
        return full if full else None
    except Exception:
        return None


def transcribe_array(y: np.ndarray, sr: int = FEATURE_SR) -> str | None:
    """
    Transcribe a numpy audio array to text.
    Writes to a temp file, transcribes, cleans up.
    """
    model = _get_model()
    if model is None:
        return None
    try:
        import soundfile as sf
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, y, sr)
            return transcribe_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception:
        return None


def transcribe_word(audio_path: str) -> str | None:
    """
    Transcribe a single-word audio file.
    Returns a lowercase heard-spelling guess (first word only), or None.
    """
    result = transcribe_file(audio_path)
    if result is None:
        return None
    # Take just the first word as a single-word guess
    words = result.split()
    return words[0] if words else None


def transcribe_segment(y: np.ndarray, sr: int = FEATURE_SR) -> str | None:
    """
    Transcribe a numpy array segment (single word).
    Returns lowercase heard-spelling guess or None.
    """
    result = transcribe_array(y, sr)
    if result is None:
        return None
    words = result.split()
    return words[0] if words else None
