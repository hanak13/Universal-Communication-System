"""
UCS Configuration — Phase 1
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ucs.db")
AUDIO_DIR = os.path.join(BASE_DIR, "recordings")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_DURATION_S = 0.5
MAX_RECORD_S = 30
FILENAME_TEMPLATE = "{label}_{speaker}_{take:03d}.wav"
SCHEMA_VERSION = 1
