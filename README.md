# Universal Communication System (UCS)

A modular Streamlit and SQLite application for recording, cataloguing, analysing, and decoding spoken words and sentences of newly encountered or undocumented languages.

> **Limitations & System Boundary**  
> UCS recognizes only sounds it has been explicitly taught and stored. All generated meanings, translations, and syntactic arrangements are treated strictly as **hypotheses**, not settled facts. A small vocabulary match does not prove comprehensive understanding of a natural language, and grammar rules require ongoing native-speaker verification.

---

## 📋 Requirements

Ensure your system meets the following prerequisites before installation:

- **Python 3.12**  
  *(Note: Python 3.13 and 3.14 are **not supported** because key audio dependencies such as `numba` and `llvmlite`, required by `librosa`, lack pre-built wheels and will fail to compile).*
- **Git** (for cloning the repository)
- **A functioning microphone** (built-in or USB headset)
- **A modern web browser** (Chrome, Edge, Firefox, or Safari)
- **Optional Tools:**
  - `ffmpeg`: Required only if analyzing compressed non-WAV audio files (`.mp3`, `.m4a`).
  - `audacity`: Useful for external audio inspection and manual trimming.

---

## 📥 Get the Code

Clone the repository and enter the project directory:

### Windows CMD
```cmd
git clone https://github.com/your-username/universal-communication-system.git
cd "universal-communication-system"
```

### Windows PowerShell
```powershell
git clone https://github.com/your-username/universal-communication-system.git
cd "universal-communication-system"
```

### Linux / macOS
```bash
git clone https://github.com/your-username/universal-communication-system.git
cd universal-communication-system
```

---

## 🐍 Create the Virtual Environment

Always isolate dependencies inside a Python 3.12 virtual environment.

### Windows CMD
```cmd
py -3.12 -m venv venv
venv\Scripts\activate.bat
```

### Windows PowerShell
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
python3.12 -m venv venv
source venv/bin/activate
```

---

## 📦 Install Dependencies

Once the virtual environment is activated, upgrade `pip` and install all required libraries:

### Windows CMD
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Included Packages Overview
The following libraries are installed via `requirements.txt`:
- `streamlit`: Web application framework and dashboard interface.
- `numpy`: Array computing, linear algebra, and pure-NumPy classifiers.
- `librosa`: Acoustic feature extraction (MFCCs, spectral centroid, silence splitting).
- `soundfile`: WAV reading and writing backend.
- `matplotlib`: Spectrogram, waveform, and confusion matrix visualizations.
- `pandas`: Tabular data organization and summary views.
- `joblib`: Model caching and serialization utilities.
- `pytest`: Automated test runner.
- `faster-whisper` *(Optional Extra)*: Offline speech-to-text model for automatic heard-spelling guesses. If omitted, UCS falls back gracefully to acoustic placeholders (`new_word_N`).

---

## 🚀 Run the Application

Start the Streamlit application server:

### Windows CMD
```cmd
streamlit run app.py
```
*(Alternative if script is not on PATH: `python -m streamlit run app.py`)*

### Windows PowerShell
```powershell
streamlit run app.py
```
*(Alternative if script is not on PATH: `python -m streamlit run app.py`)*

### Linux / macOS
```bash
streamlit run app.py
```
*(Alternative if script is not on PATH: `python3 -m streamlit run app.py`)*

Open your browser and navigate to:  
👉 **`http://localhost:8501`**

> **Microphone Permission:** When the browser opens, click **Allow** when prompted for microphone access. Audio recording requires permission on `localhost`.

---

## 🧪 Run Automated Tests

Execute the comprehensive test suite to verify that database migrations, feature extractors, recognizers, and evaluation modules are working properly:

### Windows CMD
```cmd
python -m pytest -v
```

### Windows PowerShell
```powershell
python -m pytest -v
```

### Linux / macOS
```bash
python3 -m pytest -v
```

All 54 unit tests across 9 test suites should pass cleanly.

---

## 🎯 First Use: 5-Step Walkthrough

Follow these steps to teach UCS your first words and verify an incoming sentence:

1. **Open the `💬 Talk` Dashboard:**  
   Navigate to `http://localhost:8501`. You will see the main conversation screen with a stats row and a large centered recording card.
2. **Record Word #1:**  
   In the **🎙️ Press to Record** tab, click the record button, speak a single foreign or target word clearly (e.g. *"inu"*), and stop the recording.
3. **Teach the Meaning Inline:**  
   Because the word is new, it appears as an unknown dashed chip (`❓ new_word_1`). The inline teaching form will open directly below it. Enter:
   - **Spelling:** `inu`
   - **English meaning (hypothesis):** `you`
   - **Word type:** `pronoun`
   - Click **💾 Save word**.
4. **Teach Words #2 and #3:**  
   Record two additional words in the same manner to build an initial dictionary:
   - Speak *"ising"* → Save with spelling `ising`, meaning `how`, type `question_word`.
   - Speak *"manji"* → Save with spelling `manji`, meaning `are`, type `verb`.
5. **Speak a Sentence & Test:**  
   Record a multi-word sequence: *"inu ising manji"*.  
   UCS splits the audio into word segments, shows filled primary chips (`[inu: you] [ising: how] [manji: are]`), and synthesizes the full sentence hypothesis (*"how are you"*). Click **👍 Right** to confirm and verify the hypothesis in your permanent database.

---

## 📂 Project Folder Structure

```text
Universal Communication System/
├── app.py                     # Streamlit frontend with 3-tier navigation (Talk, Teach, Review)
├── config.py                  # Global configurations, audio paths, thresholds, and parameters
├── db.py                      # SQLite schema, versioned migrations, CRUD helpers, and undo log
├── audio_io.py                # WAV recording, saving, loading, duration, and take numbering
├── features.py                # 45-D acoustic feature extraction (MFCCs, pitch/f0, centroid)
├── clustering.py              # Pure-NumPy KMeans, DBSCAN, PCA, t-SNE, and clustering metrics
├── recognizer.py              # Pure-NumPy KNN recognizer and unknown sound clustering
├── sequences.py               # Continuous audio silence-splitting, sequence decoding, n-grams
├── rule_engine.py             # Rule plugin discovery and grammar-assisted sentence arrangement
├── transcribe.py              # Speech-to-text heard-word transcription via faster-whisper
├── rules/                     # Drop-in modular directory for grammatical rules
│   ├── __init__.py            # Exports RulePlugin base class
│   ├── word_order.py          # Word-order transformation plugin
│   └── agreement.py           # Pronoun-verb agreement rule plugin
├── recordings/                # Single-word isolated WAV recordings (label_speaker_take.wav)
├── sequence_recordings/       # Continuous multi-word sequence WAV recordings
├── backups/                   # Automatic timestamped SQLite database backups
├── models/                    # Serialized feature scalers (scaler.json) and model artifacts
├── reports/                   # Exported evaluation reports written in Markdown
├── ui/                        # User interface assets
│   └── style.css              # Custom styling for chips, badges, and talk container
├── tests/                     # Automated pytest test suites
│   ├── conftest.py            # Shared pytest fixtures (tmp_db, make_wav_bytes)
│   ├── test_clip.py           # Clip storage and table existence smoke tests
│   ├── test_features.py       # Audio feature extraction and dimension tests
│   ├── test_clustering.py     # Scaler, KMeans, DBSCAN, and PCA tests
│   ├── test_recognizer.py     # KNN audio recognizer and pending sound matching tests
│   ├── test_words.py          # Word confidence, variants, and undo tests
│   ├── test_sequences.py      # Sequence splitting, n-grams, and rule application tests
│   ├── test_evaluation.py     # Speaker holdout split, classifier eval, and export tests
│   ├── test_quick_recognize.py# Zero-data cold start and inline teaching tests
│   └── test_transcribe.py     # Heard-spelling guesses and transcription log tests
├── requirements.txt           # Pinned dependency specification
├── ruff.toml                  # Linter and code formatting configuration
└── ucs.db                     # Primary SQLite database (auto-created on first run)
```

---

## 🧩 Adding Words and Rule Plug-ins

### Adding Words & Variants
1. Navigate to **🌱 Teach** → **📝 Words** (or use the inline form on **💬 Talk**).
2. For any recorded sound, specify its canonical spelling, meaning hypothesis, grammatical word type, and optional person/number (e.g. `2sg`).
3. To add alternate spellings (e.g. `eenu` for `inu`), expand the word card and enter the variant into the **Add a spelling variant** field. The recognizer will map both variants to the same canonical word.

### Adding a New Rule Plug-in
Rules use an extensible plug-in architecture:
1. Create a new `.py` file inside the `rules/` directory (e.g. `rules/tense_marker.py`).
2. Subclass `RulePlugin` and define the required methods:
   ```python
   from rules import RulePlugin

   class TenseMarkerRule(RulePlugin):
       RULE_TYPE = "tense_marker"

       @staticmethod
       def describe(definition: dict) -> str:
           return f"Tense rule: {definition.get('label', '')}"

       @staticmethod
       def apply(definition: dict, sequence_labels: list[str], word_lookup: dict) -> list[dict]:
           # Rule detection logic
           return [{"matched": True, "message": "Past tense detected"}]

       @staticmethod
       def suggest(definition: dict, sequence_labels: list[str], gap_index: int, word_lookup: dict) -> list[dict]:
           # Candidate suggestions for unknowns
           return []
   ```
3. Restart or refresh the app. UCS dynamically registers the plug-in without modifying the core engine.

---

## 💾 Backups and Reset

### Data Locations
- **Database:** `ucs.db` in the project root directory.
- **Audio Recordings:** WAV files stored in `recordings/` and `sequence_recordings/`.
- **Automatic Backups:** Stored in `backups/ucs_YYYYMMDD_HHMMSS.db` (auto-generated before schema migrations).

### How to Create a Manual Backup
Copy the database and audio files to a safe backup directory:
```bash
# Windows (PowerShell)
Copy-Item ucs.db -Destination "backups\ucs_manual.db"
Copy-Item -Recurse recordings -Destination "backups\recordings_backup"

# Linux / macOS
cp ucs.db backups/ucs_manual.db
cp -r recordings backups/recordings_backup
```

### How to Reset to a Fresh State
To wipe all data and start completely fresh:
1. Stop the Streamlit application (`Ctrl + C` in your terminal).
2. Delete `ucs.db`.
3. Delete all `.wav` files inside `recordings/` and `sequence_recordings/`.
4. Delete `models/scaler.json` if present.
5. Restart the app (`streamlit run app.py`). A clean database schema will be bootstrapped automatically.

---

## 🛠️ Troubleshooting

Find your exact error message or symptom below to resolve issues quickly:

### 1. PowerShell Script Execution Disabled
- **Error:**
  ```text
  venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
  ```
- **Cause:** Windows PowerShell disables script execution by default under the `Restricted` execution policy.
- **Fix (Option A - Recommended for PowerShell):** Bypass policy for the current terminal session only:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```
- **Fix (Option B):** Use Windows Command Prompt (`cmd.exe`) instead:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Fix (Option C):** Skip activation and run python directly from the virtual environment:
  ```powershell
  .\venv\Scripts\python.exe -m pip install -r requirements.txt
  .\venv\Scripts\python.exe -m streamlit run app.py
  ```

---

### 2. "activate is not recognized"
- **Error:**
  ```text
  'activate' is not recognized as an internal or external command, operable program or batch file.
  ```
- **Cause:** Running `activate` without the relative directory path or using incorrect slashes for your shell.
- **Fix:** Specify the full relative path for your operating system:
  - **CMD:** `venv\Scripts\activate.bat`
  - **PowerShell:** `.\venv\Scripts\Activate.ps1`
  - **Linux / macOS:** `source venv/bin/activate`

---

### 3. "py is not recognized" or "python is not recognized"
- **Error:**
  ```text
  'py' is not recognized as an internal or external command
  ```
- **Cause:** Python is not added to your system `PATH` environment variable.
- **Fix:**
  1. Verify which Python versions exist by running: `py -0` or `where python`.
  2. If Python is installed but not on PATH, call the absolute path to your Python 3.12 executable, for example:
     ```cmd
     "C:\Users\<YourUser>\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
     ```
  3. Alternatively, re-run the Python 3.12 installer, choose **Modify**, and check **"Add Python to environment variables"**.

---

### 4. Wrong Python Version in Virtual Environment
- **Error:**
  ```text
  RuntimeError: Python 3.13 is not supported due to numba/llvmlite incompatibility.
  ```
- **Cause:** Virtual environment was created with Python 3.13, 3.14, or 3.11 instead of 3.12.
- **Fix:** Check your version with `python --version`. If it is not 3.12.x, delete the `venv` directory and recreate it explicitly with Python 3.12:
  ```cmd
  deactivate
  rmdir /s /q venv
  py -3.12 -m venv venv
  venv\Scripts\activate.bat
  ```

---

### 5. "&&" Operator Not Working in PowerShell
- **Error:**
  ```text
  The token '&&' is not a valid statement separator in this version.
  ```
- **Cause:** Windows PowerShell 5.1 does not support the `&&` chaining operator (introduced in PowerShell 7+).
- **Fix:** Run the commands on separate lines:
  ```powershell
  py -3.12 -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

---

### 6. Pip Install Fails Building numba / llvmlite / librosa
- **Error:**
  ```text
  Building wheel for llvmlite (pyproject.toml) ... error
  ERROR: Failed building wheel for llvmlite
  ```
- **Cause:** You are running Python 3.13 or 3.14 where pre-compiled wheels for `llvmlite` and `numba` do not yet exist, requiring a local C++ compiler.
- **Fix:** Recreate your virtual environment with **Python 3.12** and update pip:
  ```cmd
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```

---

### 7. "streamlit is not recognized"
- **Error:**
  ```text
  'streamlit' is not recognized as an internal or external command
  ```
- **Cause:** The virtual environment is not activated, or packages were installed into the global Python environment.
- **Fix:** Activate your virtual environment first, or invoke Streamlit through the Python module flag:
  ```cmd
  python -m streamlit run app.py
  ```

---

### 8. Port 8501 Already in Use
- **Error:**
  ```text
  Port 8501 is already in use
  ```
- **Cause:** Another instance of Streamlit or another background process is using port 8501.
- **Fix:** Launch UCS on a different port:
  ```cmd
  streamlit run app.py --server.port 8502
  ```
  *(Or close the existing process running on port 8501 via Task Manager).*

---

### 9. Microphone Not Working or Recording Empty Audio
- **Symptoms:** The audio input widget shows no waveform or generates a zero-length clip.
- **Fix:**
  1. **Browser Permission:** Click the lock or settings icon in your browser's address bar and ensure **Microphone** is set to **Allow**.
  2. **Access via localhost:** Always open the dashboard at `http://localhost:8501` (browsers block microphone capture over insecure non-localhost IP addresses like `http://192.168.x.x`).
  3. **Audio Device Selection:** Ensure your desired microphone is set as the default recording device in your OS sound settings.
  4. **Exclusive Access:** Close Zoom, Teams, Discord, or Audacity if they are holding exclusive locks on the microphone.

---

### 10. soundfile / libsndfile Error on Linux
- **Error:**
  ```text
  OSError: cannot load library 'libsndfile.so': libsndfile.so: cannot open shared object file
  ```
- **Cause:** The underlying C-library `libsndfile` is missing on Debian/Ubuntu/Fedora systems.
- **Fix:** Install the system library:
  ```bash
  # Debian / Ubuntu / Mint
  sudo apt update && sudo apt install -y libsndfile1 ffmpeg

  # Fedora / RHEL
  sudo dnf install -y libsndfile ffmpeg

  # Arch Linux
  sudo pacman -S libsndfile ffmpeg
  ```

---

### 11. faster-whisper Model Download Slow or Fails
- **Symptoms:** The app freezes for several minutes on the first recording or throws a connection timeout.
- **Cause:** `faster-whisper` downloads the ~140 MB `"base"` model weights from Hugging Face on its very first run.
- **Fix:**
  1. Ensure you have an active internet connection on first execution.
  2. If running offline or behind a strict corporate proxy, UCS will automatically fall back to acoustic identifiers (`new_word_N`) without crashing. You can also pre-download the model in Python:
     ```python
     from faster_whisper import WhisperModel
     WhisperModel("base", device="cpu", compute_type="int8")
     ```

---

### 12. "database is locked" or Corrupted ucs.db
- **Error:**
  ```text
  sqlite3.OperationalError: database is locked
  ```
- **Cause:** Multiple Streamlit server instances or external SQLite browser tools are accessing `ucs.db` concurrently.
- **Fix:**
  1. Close all open browser tabs and terminate all running python/streamlit terminal processes.
  2. If the database was corrupted, restore from the latest timestamped backup in `backups/`:
     ```cmd
     copy backups\ucs_YYYYMMDD_HHMMSS.db ucs.db
     ```

---

### 13. Antivirus or Windows Long Path Issues
- **Error:**
  ```text
  FileNotFoundError: [Errno 2] No such file or directory
  ```
- **Cause:** Path lengths exceeding the Windows 260-character MAX_PATH limit when deeply nested in user profile folders.
- **Fix:** Move the project directory to a shorter path closer to the drive root, such as `C:\ucs`.

---

### 14. Git Line-Ending Warnings (CRLF vs LF)
- **Warning:**
  ```text
  warning: in the working copy of '...', LF will be replaced by CRLF the next time Git touches it
  ```
- **Cause:** Normal Git behavior when checking out files across Linux and Windows systems.
- **Fix:** This warning is harmless and can be ignored. To suppress it globally, configure Git autocrlf:
  ```bash
  git config --global core.autocrlf true
  ```
