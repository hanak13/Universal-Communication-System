"""
UCS Audio I/O — recording, saving, loading WAV files.
Filename convention:  label_speaker_take.wav
"""
import io
import os
import re
import wave

from config import AUDIO_DIR, SAMPLE_RATE, CHANNELS, FILENAME_TEMPLATE


_FNAME_RE = re.compile(
    r"^(?P<label>[a-zA-Z0-9_]+)_(?P<speaker>[a-zA-Z0-9_]+)_(?P<take>\d{3})\.wav$"
)


def parse_filename(fname: str) -> dict | None:
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return {
        "label": m.group("label"),
        "speaker": m.group("speaker"),
        "take": int(m.group("take")),
    }


def next_take(label: str, speaker: str) -> int:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    max_take = 0
    prefix = f"{label}_{speaker}_"
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(prefix) and f.endswith(".wav"):
            parts = parse_filename(f)
            if parts and parts["take"] > max_take:
                max_take = parts["take"]
    return max_take + 1


def build_filename(label: str, speaker: str, take: int | None = None) -> str:
    if take is None:
        take = next_take(label, speaker)
    return FILENAME_TEMPLATE.format(label=label, speaker=speaker, take=take)


def save_wav(
    audio_bytes: bytes,
    filename: str,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sampwidth: int = 2,
) -> str:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return path


def save_wav_from_file(uploaded_bytes: bytes, filename: str) -> str:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_bytes)
    return path


def load_wav(filename: str) -> bytes | None:
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def wav_duration(filename: str) -> float:
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate if rate else 0.0


def list_recordings() -> list[str]:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
