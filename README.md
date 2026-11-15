# Universal Communication System (UCS)

A Streamlit + SQLite platform for recording, cataloguing, analysing, and decoding spoken words and phrases of an unknown or newly encountered language.

In UCS, every meaning is treated as a **hypothesis**, never a settled fact. All changes are tracked in a reversible history log with one-click undo.

---

## 🛠️ Setup Instructions

### Prerequisites
- **Python 3.12**

### Setup Virtual Environment & Dependencies

```bash
# 1. Open a terminal inside the project directory:
cd "c:\Users\pradh\Documents\Universal Communication System"

# 2. Create the virtual environment (if not already present):
py -3.12 -m venv .venv

# 3. Activate the virtual environment:
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# Or Command Prompt:
.venv\Scripts\activate.bat

# 4. Install required packages:
pip install streamlit librosa scikit-learn matplotlib numpy soundfile joblib pandas pytest faster-whisper
```

*(Note: On systems where Windows Device Guard restricts native C-extension DLLs such as scikit-learn, UCS automatically uses its built-in pure-NumPy classifiers and clustering modules without any degradation of functionality. `faster-whisper` is optional and provides automatic heard-spelling guesses; if unavailable, the app falls back gracefully to `new_word_N` placeholders).*

---

## 🚀 How to Run the App

Always execute using the virtual environment's Python interpreter:

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🎨 UI & Theming

UCS features a calm, modern, and accessible design system optimized for high readability during linguistic fieldwork and hypothesis evaluation.

### Navigation Architecture
The sidebar groups the 12 system pages into 3 functional workflow sections:
- **💬 Talk**:
  - `🏠 Dashboard`: Main conversation screen for recording/uploading clips, continuous word chip decoding, inline hypothesis capture, and sentence verification.
- **🌱 Teach**:
  - `🎙️ Record`: Record individual words or phrases.
  - `📚 Library`: Browse recorded audio clips with speaker/label filters.
  - `📝 Words`: Word dictionary with confidence meters, status badges, and spelling variants.
  - `❓ Pending`: Catalog of unassigned acoustic clusters (`new_word_N`) ready to be named.
  - `📏 Rules`: Grammar rule builder (word order and pronoun-verb agreement).
- **🔍 Review**:
  - `✅ Test`: Interactive hypothesis testing with Right/Wrong feedback.
  - `📋 Summary`: Overview table of vocabulary, tests, and evidence levels.
  - `📊 Analysis`: Waveform, spectrogram, and MFCC inspection.
  - `🔬 Clustering`: K-Means / DBSCAN acoustic clustering and 2D PCA/t-SNE projections.
  - `🔗 Sequences`: Continuous sequence splitting and rule-assisted decoding.
  - `📈 Evaluation`: Speaker holdout split, classifier comparison, noise robustness, and report export.

*(A persistent **↩️ Undo last change** button sits at the bottom of the sidebar with non-blocking toast feedback).*

### Theme Configuration (`.streamlit/config.toml`)
The application defines primary palette and layout properties in `.streamlit/config.toml`:
- **Primary Color:** `#2563EB` (calm royal blue)
- **Background:** Clean neutral background (`#F8FAFC` light / `#0F172A` dark)
- **Secondary Background:** Clean card background (`#FFFFFF` light / `#1E293B` dark)
- **Font:** Clean sans-serif system font stack (`Inter`, system-ui)
- **Base Radius:** `0.75rem` for rounded modern cards and buttons

To change theme colors:
1. Open `.streamlit/config.toml`.
2. Edit `primaryColor`, `backgroundColor`, or `secondaryBackgroundColor`.
3. To switch between light and dark mode, toggle `base = "light"` or `base = "dark"`, or select your preference in the Streamlit hamburger menu (Settings → Theme).

### Custom CSS Overrides (`ui/style.css`)
Loaded once and cached via `@st.cache_data _load_ui_styles()`:
- **`talk-container`**: Centers the main Talk workspace to a comfortable reading max-width (~900px) preventing horizontal eye strain.
- **Interactive Word Chips**:
  - **Known words:** Filled primary badge with meaning caption below (`.chip-known`).
  - **Unknown sounds:** Dashed accent border with "Teach this word" prompt (`.chip-unknown`).
  - **Low-confidence / Unverified:** Subtle warning chip with warning icon (`.chip-warning`).
- **Status Badges**: High-contrast, color-blind accessible badges (`.status-verified` and `.status-unverified`) using both distinct colors and explicit text/icons.
- **Buttons**: Visually distinguished green **Right** (`#16A34A`) and red **Wrong** (`#DC2626`) actions.

### Taking Demo Screenshots
1. Launch the app: `.venv\Scripts\python.exe -m streamlit run app.py`.
2. Navigate to **💬 Talk** (`🏠 Dashboard`).
3. Record or upload an audio sample (e.g. `tests/test_audio.wav`).
4. Capture screenshots of:
   - The compact metrics card row and recording card.
   - The segmented word chips (known vs unknown).
   - The inline teaching card under unresolved segments.
   - The highlighted sentence card with Right/Wrong buttons.
   - The **📈 Evaluation** page showing the confusion matrix and noise robustness bar chart.

---

## 📂 Project Folder Structure

```
Universal Communication System/
├── app.py                     # Streamlit web application with all pages
├── config.py                  # Global configurations, audio paths, thresholds
├── db.py                      # SQLite database schema, versioned migrations, CRUD
├── audio_io.py                # Audio saving, loading, and filename utilities
├── features.py                # 29-dimensional audio feature extraction (librosa/numpy)
├── clustering.py              # Pure-NumPy K-Means, DBSCAN, PCA, and metrics
├── recognizer.py              # Pure-NumPy nearest-centroid audio recognizer
├── sequences.py               # Long-clip splitting into word segments & decoding
├── rule_engine.py             # Modular plug-in engine for syntactic and semantic rules
├── evaluation.py              # Speaker-holdout split, multi-classifier eval, report export
├── transcribe.py              # Speech-to-text heard-spelling guesses (faster-whisper)
├── rules/                     # Rule plug-ins directory (drop-in modular architecture)
│   ├── __init__.py            # Base RulePlugin interface and registry loader
│   ├── word_order.py          # Word-order mapping rule plug-in
│   └── agreement.py           # Pronoun-verb agreement rule plug-in
├── models/                    # Serialized feature scalers and model artifacts
├── recordings/                # Single-word and short-clip WAV audio storage
├── sequence_audio/            # Continuous sequence WAV audio storage
├── reports/                   # Exported evaluation Markdown reports
├── tests/                     # Comprehensive test suite (pytest)
│   ├── test_clip.py           # DB and audio storage smoke tests
│   ├── test_features.py       # Audio feature extraction tests
│   ├── test_clustering.py     # Scaler, K-Means, DBSCAN, and PCA tests
│   ├── test_recognizer.py     # Nearest-centroid classifier & pending sounds
│   ├── test_words.py          # Word creation, confidence, variants, and undo
│   ├── test_sequences.py      # Sequence splitting, rules, n-grams, arrangement
│   ├── test_evaluation.py     # Classifier comparisons, noise robustness, report export
│   ├── test_quick_recognize.py# Zero-data cold start, inline meaning capture & undo
│   └── test_transcribe.py     # Heard-spelling guesses, word chips & transcription log
└── requirements.txt           # Python dependency specification
```

---

## 🎬 How to Demo (Walkthrough)

1. **🏠 Dashboard**:
   - The central command station. Record or upload an audio clip.
   - Live prediction displays recognized text, word-by-word hypothesis, phrase-level meaning, confidence score, and low-evidence/unverified warnings.
   - Click **👍 Right** or **👎 Wrong** directly on the dashboard to confirm or refine hypotheses.
   - View currently pending unknown sounds at a glance.

2. **🎙️ Record & 📚 Library**:
   - Record clips under different speakers (`speaker1`, `speaker2`) and labels.
   - Inspect recordings, playback audio, and verify database consistency.

3. **📊 Analysis & 🔬 Clustering**:
   - Extract 29-D feature vectors (MFCCs, spectral centroid, rolloff, zero-crossing rate, RMS energy).
   - Visualize feature distributions and run 2D PCA projections with K-Means or DBSCAN clustering to discover phonetic groupings.

4. **❓ Pending & 📝 Words**:
   - Sounds that fall beyond the recognition distance threshold are assigned pending identifiers (`new_word_1`, `new_word_2`).
   - Group identical repeated unknown sounds together.
   - Convert a pending sound into a canonical word: define its spelling, canonical meaning hypothesis, grammatical word type, and spelling variants (e.g., `inu`/`eenu`).

5. **🔗 Sequences & Sentence Decoding**:
   - Upload or record a continuous multi-word sentence (e.g., `"inu ising manji"`).
   - UCS splits the sentence on silence boundaries using `librosa.effects.split`.
   - Recognizes each segment, preserving unknown sounds as placeholders.
   - Displays both word-by-word literal hypotheses and phrase-level idiomatic meaning.
   - Suggests candidates to fill unknown gaps using grammar rules.

6. **📏 Rules**:
   - View and manage grammatical rules (word-order patterns, pronoun-verb agreement).
   - Toggle rules as verified/unverified hypotheses.

7. **✅ Test & 📋 Summary**:
   - Test words interactively with Right/Wrong feedback.
   - Track hypothesis confidence (`confirmations / total_tests`).
   - View warning indicators for words with low evidence (< 5 tests).

8. **📈 Evaluation**:
   - Run speaker-wise holdout evaluation holding out an entire speaker.
   - Compare performance across pure-NumPy KNN, Linear SVM, and Random Forest classifiers.
   - Inspect the confusion matrix heatmap and noise robustness degradation curve (0–30 dB SNR).
   - Export an evaluation report with one click to `reports/`.

---

## 🧩 How to Add Words and Rule Plug-ins

### Adding Words & Variants
1. Navigate to the **❓ Pending** page or **🏠 Dashboard**.
2. When an unknown sound is recorded, it appears under a placeholder like `new_word_1`.
3. Click on the pending sound to assign:
   - Canonical spelling (e.g., `inu`)
   - Spelling variants (e.g., `eenu, innu`)
   - Hypothesised meaning (e.g., `you`)
   - Word type (`pronoun`, `verb`, `noun`, `question word`, `other`)
   - Person/number (e.g., `2s`)
4. Saving automatically migrates the clips and updates the recognizer.

### Adding a New Rule Plug-in
Rules use an extensible plug-in architecture located in `rules/`:
1. Create a new Python file in the `rules/` directory (e.g., `rules/tense_marker.py`).
2. Subclass `RulePlugin` from `rules`:
   ```python
   from rules import RulePlugin

   class TenseMarkerRule(RulePlugin):
       rule_type = "tense_marker"

       def apply(self, words: list[dict], rules_data: list[dict]) -> dict:
           # Implement your rule logic here
           return {"notes": ["Detected tense marker pattern"]}
   ```
3. Restart or reload the app. UCS automatically discovers and registers all `RulePlugin` subclasses in the `rules/` directory without requiring code changes to the core engine.

---

## 🧪 Running Automated Tests

Run the full test suite with:

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected output: 49 tests across 9 test suites passing cleanly.
