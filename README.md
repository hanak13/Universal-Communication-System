# Universal Communication System (UCS)

A Streamlit + SQLite app for recording, cataloguing, and replaying spoken words
and phrases of an unknown / new language.  Every meaning is a **hypothesis**
until the user marks it verified.

## Quick Start

```bash
# From a terminal inside this folder:
setup.bat            # creates venv, installs deps, runs tests

# Run the app:
.venv\Scripts\streamlit.exe run app.py
```

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | All paths, thresholds, constants |
| `db.py` | SQLite schema, migrations, CRUD |
| `audio_io.py` | WAV save/load, filename convention |
| `app.py` | Streamlit UI (Record + Library pages) |
| `tests/test_clip.py` | Save-and-read-back smoke test |
| `setup.bat` | One-click venv + test runner |

## Filename Convention

```
label_speaker_take.wav
e.g.  hello_alice_003.wav
```

## Database Tables

`clips` · `words` · `word_variants` · `phrases` · `rules` · `pending_sounds` · `history` · `schema_version`

Migrations are version-gated; `ucs.db` is backed up to `backups/` before each migration.

