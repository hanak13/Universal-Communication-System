"""
UCS Configuration — all thresholds and paths in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ucs.db")
AUDIO_DIR = os.path.join(BASE_DIR, "recordings")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

# ── Audio defaults ─────────────────────────────────────────────────
SAMPLE_RATE = 44100          # Hz
CHANNELS = 1                 # mono
CHUNK_DURATION_S = 0.5       # seconds per chunk while recording
MAX_RECORD_S = 30            # hard cap on a single recording

# ── Filename format ────────────────────────────────────────────────
#   label_speaker_take.wav   (e.g.  hello_alice_003.wav)
FILENAME_TEMPLATE = "{label}_{speaker}_{take:03d}.wav"

# ── Feature extraction (Phase 2) ──────────────────────────────────
FEATURE_SR = 16000            # resample target for analysis (Hz)
N_MFCC = 20                  # number of MFCC coefficients
SILENCE_TOP_DB = 30           # librosa.effects.trim threshold (dB)
PITCH_FMIN = 50.0             # pyin minimum frequency (Hz)
PITCH_FMAX = 500.0            # pyin maximum frequency (Hz)
N_FFT = 2048                  # FFT window size
HOP_LENGTH = 512              # hop length for STFT / MFCC

# Derived: feature vector length
#   MFCC mean(20) + std(20) + pitch mean+std(2) + centroid mean+std(2) + duration(1)
FEATURE_VECTOR_LEN = N_MFCC * 2 + 2 + 2 + 1   # = 45

# ── Recognition / Clustering (Phase 3) ────────────────────────────
UNKNOWN_DISTANCE_THRESHOLD = 15.0  # distance above which a sound is unknown
RECOGNIZE_THRESHOLD = UNKNOWN_DISTANCE_THRESHOLD
PENDING_MERGE_THRESHOLD = 10.0  # distance to reuse an existing pending name
DUPLICATE_SOUND_THRESHOLD = 8.0  # distance below which sound is close to an existing word
KNN_NEIGHBORS = 3             # k for KNN recognizer
TSNE_PERPLEXITY = 5           # t-SNE perplexity (low for small datasets)
DBSCAN_EPS_DEFAULT = 5.0      # DBSCAN default eps
DBSCAN_MIN_SAMPLES_DEFAULT = 2  # DBSCAN default min_samples
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.json")
MIN_CLIPS_FOR_CLUSTERING = 4  # minimum clips to attempt clustering

# ── Words & Confidence (Phase 4) ─────────────────────────────────
LOW_EVIDENCE_THRESHOLD = 5    # fewer total tests → "low evidence" warning
WORD_TYPES = ["noun", "verb", "pronoun", "question_word", "adjective", "other"]

# ── Sequences (Phase 5) ────────────────────────────────────────────
SPLIT_TOP_DB_DEFAULT = 25       # librosa.effects.split top_db default
SPLIT_TOP_DB_MIN = 10
SPLIT_TOP_DB_MAX = 60
MIN_SEGMENT_LENGTH = 0.05       # seconds — minimum duration for a word segment
MIN_SEGMENT_DURATION = MIN_SEGMENT_LENGTH
MAX_SEGMENTS_PER_CLIP = 20      # maximum word segments processed per clip
SEQUENCE_AUDIO_DIR_NAME = "sequence_recordings"

# ── Transcription (Phase 8) ────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"        # faster-whisper model: "tiny", "base", "small"

# ── Schema version (bumped on every migration) ────────────────────
SCHEMA_VERSION = 6

