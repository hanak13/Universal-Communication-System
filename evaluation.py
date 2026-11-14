"""
UCS Evaluation — speaker-wise split, classifier comparison, metrics.
No Streamlit imports so it can be used from CLI or tests.

Classifiers:
  • KNN   — pure-numpy (reuses recognizer.py logic)
  • SVM   — one-vs-rest linear SVM (pure numpy gradient descent)
  • RF    — bagged decision stumps (pure numpy)

All pure-numpy because scikit-learn .pyd DLLs are blocked by Device Guard
on the project machine.
"""
import json
import os
import numpy as np
from datetime import datetime, timezone
from collections import Counter

from config import (
    FEATURE_VECTOR_LEN, RECOGNIZE_THRESHOLD, BASE_DIR,
)
from clustering import SimpleScaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pure-numpy classifiers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KNNClassifier:
    """K-Nearest Neighbours (pure numpy)."""
    def __init__(self, k: int = 3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X - x, axis=1)
            nearest = np.argsort(dists)[:self.k]
            counts = Counter(self.y[nearest].tolist())
            preds.append(counts.most_common(1)[0][0])
        return np.array(preds)


class LinearSVM:
    """One-vs-rest linear SVM (hinge loss + SGD, pure numpy)."""
    def __init__(self, lr: float = 0.01, epochs: int = 200, C: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.models = {}  # label -> (w, b)

    def fit(self, X: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        for cls in classes:
            binary_y = np.where(y == cls, 1.0, -1.0)
            w, b = self._fit_binary(X, binary_y)
            self.models[cls] = (w, b)

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            scores = X @ w + b
            margins = y * scores
            mask = margins < 1  # hinge loss active
            dw = -self.C * (y[mask][:, None] * X[mask]).mean(axis=0) if mask.any() else np.zeros(d)
            db = -self.C * y[mask].mean() if mask.any() else 0.0
            w -= self.lr * (w / n + dw)
            b -= self.lr * db
        return w, b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = {}
        for cls, (w, b) in self.models.items():
            scores[cls] = X @ w + b
        preds = []
        classes = list(self.models.keys())
        for i in range(X.shape[0]):
            best = max(classes, key=lambda c: scores[c][i])
            preds.append(best)
        return np.array(preds)


class TreeNode:
    """Single node in a decision tree (pure numpy)."""
    def __init__(self, depth: int = 0, max_depth: int = 3):
        self.depth = depth
        self.max_depth = max_depth
        self.feat = None
        self.thresh = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState, max_features: int):
        n, d = X.shape
        counts = Counter(y.tolist())
        self.label = counts.most_common(1)[0][0]

        if self.depth >= self.max_depth or len(counts) <= 1 or n < 2:
            return

        feats = rng.choice(d, min(max_features, d), replace=False)
        best_feat, best_thresh, best_gain = None, None, -1.0

        p = np.array(list(counts.values())) / n
        base_gini = 1.0 - np.sum(p ** 2)

        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            if len(vals) > 12:
                step = max(1, len(vals) // 10)
                candidates = vals[::step][:-1]
            else:
                candidates = vals[:-1]

            for t in candidates:
                left_mask = X[:, f] <= t
                n_l = int(np.sum(left_mask))
                n_r = n - n_l
                if n_l == 0 or n_r == 0:
                    continue

                _, l_cnt = np.unique(y[left_mask], return_counts=True)
                _, r_cnt = np.unique(y[~left_mask], return_counts=True)
                gini_l = 1.0 - np.sum((l_cnt / n_l) ** 2)
                gini_r = 1.0 - np.sum((r_cnt / n_r) ** 2)
                gain = base_gini - ((n_l / n) * gini_l + (n_r / n) * gini_r)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = float(t)

        if best_feat is not None and best_gain > 0:
            self.feat = best_feat
            self.thresh = best_thresh
            left_mask = X[:, best_feat] <= best_thresh
            self.left = TreeNode(depth=self.depth + 1, max_depth=self.max_depth)
            self.left.fit(X[left_mask], y[left_mask], rng, max_features)
            self.right = TreeNode(depth=self.depth + 1, max_depth=self.max_depth)
            self.right.fit(X[~left_mask], y[~left_mask], rng, max_features)

    def predict_one(self, x: np.ndarray):
        if self.feat is None or self.left is None or self.right is None:
            return self.label
        if x[self.feat] <= self.thresh:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)


class SimpleRandomForest:
    """Bagged decision trees (pure numpy)."""
    def __init__(self, n_trees: int = 30, max_depth: int = 3, max_features: int | None = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(d)))
        rng = np.random.RandomState(42)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, n, replace=True)
            Xb, yb = X[idx], y[idx]
            tree = TreeNode(depth=0, max_depth=self.max_depth)
            tree.fit(Xb, yb, rng, self.max_features)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            votes = [t.predict_one(x) for t in self.trees]
            preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(preds)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data preparation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_data(conn) -> dict:
    """Load all clips with features and speaker info. Returns structured dict."""
    from db import get_clips, get_features
    all_clips = get_clips(conn)
    data = []
    for c in all_clips:
        feat = get_features(conn, c["id"])
        if feat:
            data.append({
                "clip_id": c["id"],
                "filename": c["filename"],
                "label": c["label"],
                "speaker": c["speaker"],
                "vector": feat["vector"],
            })
    return {
        "items": data,
        "n_clips": len(data),
        "n_labels": len(set(d["label"] for d in data)) if data else 0,
        "n_speakers": len(set(d["speaker"] for d in data)) if data else 0,
        "speakers": sorted(set(d["speaker"] for d in data)) if data else [],
        "labels": sorted(set(d["label"] for d in data)) if data else [],
    }


def speaker_split(data: dict, holdout_speaker: str) -> tuple:
    """
    Split data by speaker: one speaker fully held out for testing.
    Returns (X_train, y_train, X_test, y_test, warnings).
    """
    warnings = []
    items = data["items"]

    train_items = [d for d in items if d["speaker"] != holdout_speaker]
    test_items = [d for d in items if d["speaker"] == holdout_speaker]

    if not train_items:
        warnings.append(f"No training data after holding out speaker '{holdout_speaker}'.")
        return None, None, None, None, warnings
    if not test_items:
        warnings.append(f"Speaker '{holdout_speaker}' has no clips for testing.")
        return None, None, None, None, warnings

    train_labels = set(d["label"] for d in train_items)
    test_labels = set(d["label"] for d in test_items)
    unseen = test_labels - train_labels
    if unseen:
        warnings.append(f"Test labels not in training set: {unseen}. These will always be wrong.")

    if len(train_items) < 4:
        warnings.append(f"Only {len(train_items)} training clips — too small for a reliable split.")
    if len(set(d["label"] for d in train_items)) < 2:
        warnings.append("Only 1 label in training set — classifier cannot learn.")

    X_train = np.array([d["vector"] for d in train_items], dtype=np.float64)
    y_train = np.array([d["label"] for d in train_items])
    X_test = np.array([d["vector"] for d in test_items], dtype=np.float64)
    y_test = np.array([d["label"] for d in test_items])

    return X_train, y_train, X_test, y_test, warnings


def random_split(data: dict, test_frac: float = 0.25, seed: int = 42) -> tuple:
    """Random train/test split (fallback when only 1 speaker)."""
    warnings = []
    items = data["items"]
    if len(items) < 4:
        warnings.append(f"Only {len(items)} clips — too small for a reliable split.")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(items))
    split = max(1, int(len(items) * (1 - test_frac)))

    train_items = [items[i] for i in idx[:split]]
    test_items = [items[i] for i in idx[split:]]

    if not test_items:
        warnings.append("Not enough data for a test set.")
        return None, None, None, None, warnings

    X_train = np.array([d["vector"] for d in train_items], dtype=np.float64)
    y_train = np.array([d["label"] for d in train_items])
    X_test = np.array([d["vector"] for d in test_items], dtype=np.float64)
    y_test = np.array([d["label"] for d in test_items])

    return X_train, y_train, X_test, y_test, warnings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, per-class precision/recall, confusion matrix."""
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    n = len(y_true)

    # Overall accuracy
    accuracy = float(np.mean(y_true == y_pred)) if n > 0 else 0.0

    # Confusion matrix
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # Per-class precision and recall
    per_class = {}
    for label in labels:
        i = label_to_idx[label]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "support": int(cm[i, :].sum()),
        }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_total": n,
    }


def add_noise(X: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to feature vectors at a given SNR level."""
    rng = np.random.RandomState(seed)
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(*X.shape) * np.sqrt(noise_power)
    return X + noise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full evaluation pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_evaluation(conn) -> dict:
    """
    Run the full evaluation: speaker-wise split, classifier comparison,
    noise robustness, dataset statistics.
    Returns a dict with everything needed for the report.
    """
    from db import get_all_words, get_all_phrases, get_all_pending_sounds, get_recent_history

    data = prepare_data(conn)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_clips": data["n_clips"],
            "n_labels": data["n_labels"],
            "n_speakers": data["n_speakers"],
            "speakers": data["speakers"],
            "labels": data["labels"],
        },
        "vocabulary": {
            "n_words": len(get_all_words(conn)),
            "n_phrases": len(get_all_phrases(conn)),
            "n_pending": len(get_all_pending_sounds(conn)),
        },
        "warnings": [],
        "classifiers": {},
        "noise_robustness": {},
        "cross_speaker": {},
    }

    if data["n_clips"] < 4:
        result["warnings"].append(
            f"Dataset has only {data['n_clips']} clip(s) — too small for reliable evaluation."
        )
        return result

    if data["n_labels"] < 2:
        result["warnings"].append("Only 1 label — classifiers need at least 2.")
        return result

    # ── Split ──────────────────────────────────────────────────────
    if data["n_speakers"] >= 2:
        holdout = data["speakers"][-1]  # hold out last alphabetically
        X_train, y_train, X_test, y_test, split_warnings = speaker_split(data, holdout)
        result["split_method"] = f"speaker holdout ({holdout})"
    else:
        X_train, y_train, X_test, y_test, split_warnings = random_split(data)
        result["split_method"] = "random 75/25 (single speaker)"
        result["warnings"].append("Only 1 speaker — using random split instead of speaker holdout.")

    result["warnings"].extend(split_warnings)

    if X_train is None or X_test is None:
        return result

    # ── Scale ──────────────────────────────────────────────────────
    scaler = SimpleScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    result["split_sizes"] = {
        "train": len(y_train),
        "test": len(y_test),
    }

    # ── Train & evaluate classifiers ──────────────────────────────
    classifiers = {
        "KNN (k=3)": KNNClassifier(k=3),
        "Linear SVM": LinearSVM(lr=0.005, epochs=300, C=1.0),
        "Random Forest (30 trees)": SimpleRandomForest(n_trees=30),
    }

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            metrics = compute_metrics(y_test, y_pred)
            result["classifiers"][name] = metrics
        except Exception as e:
            result["classifiers"][name] = {"error": str(e)}

    # ── Cross-speaker accuracy (if multiple speakers) ─────────────
    if data["n_speakers"] >= 2:
        for spk in data["speakers"]:
            Xtr, ytr, Xte, yte, _ = speaker_split(data, spk)
            if Xtr is None or Xte is None or len(set(ytr)) < 2:
                result["cross_speaker"][spk] = {"accuracy": None, "note": "insufficient data"}
                continue
            sc = SimpleScaler().fit(Xtr)
            clf = KNNClassifier(k=min(3, len(Xtr)))
            clf.fit(sc.transform(Xtr), ytr)
            y_p = clf.predict(sc.transform(Xte))
            result["cross_speaker"][spk] = {
                "accuracy": float(np.mean(yte == y_p)),
                "n_test": len(yte),
            }

    # ── Noise robustness ──────────────────────────────────────────
    snr_levels = [30, 20, 10, 5, 0]
    knn = KNNClassifier(k=3)
    knn.fit(X_train_s, y_train)  # train on clean

    for snr in snr_levels:
        X_noisy = scaler.transform(add_noise(X_test, snr))
        y_noisy_pred = knn.predict(X_noisy)
        acc = float(np.mean(y_test == y_noisy_pred))
        result["noise_robustness"][f"SNR {snr} dB"] = acc

    # ── Confirmation stats ────────────────────────────────────────
    words = get_all_words(conn)
    total_right = sum(w.get("confirmations", 0) for w in words)
    total_wrong = sum(w.get("rejections", 0) for w in words)
    total_tests = total_right + total_wrong
    result["confirmation"] = {
        "total_right": total_right,
        "total_wrong": total_wrong,
        "total_tests": total_tests,
        "right_share": total_right / total_tests if total_tests > 0 else 0.0,
    }

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LIMITATIONS_TEXT = """## Limitations

1. **Recognising a sound is not understanding meaning.**
   The system matches audio features to stored examples. It does not comprehend
   semantics, context, or intent behind a word.

2. **Meaning changes with context.**
   A word's meaning can vary depending on surrounding words, intonation, and
   social setting. The system treats each word's meaning as a static hypothesis.

3. **A small dataset can be memorised.**
   With few clips the classifier can achieve high accuracy simply by memorising
   examples rather than learning generalisable patterns. High accuracy on a tiny
   dataset should not be mistaken for robust performance.

4. **The system knows only the words it was taught.**
   Any sound not in the training set is labelled "unknown". The vocabulary is
   limited to what the user has recorded and labelled.

5. **Success on a small taught vocabulary does not prove understanding of a real
   unknown language.**
   Distinguishing 5–20 pre-recorded words is far simpler than transcribing
   continuous speech in an under-documented language with thousands of words,
   tonal variation, dialect differences, and coarticulation.

6. **Grammar rules come from a small amount of user-supplied data and need a
   native-speaker check.**
   The word-order and agreement rules are hypotheses entered by the user, not
   learned from a large corpus. They may be incomplete, incorrect, or
   over-generalised.
"""


def generate_report(eval_result: dict) -> str:
    """Generate a Markdown evaluation report."""
    r = eval_result
    lines = []
    lines.append("# UCS Evaluation Report")
    lines.append(f"\n**Generated:** {r.get('timestamp', 'unknown')}\n")

    # Dataset
    ds = r.get("dataset", {})
    lines.append("## Dataset Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | :--- |")
    lines.append(f"| Clips | {ds.get('n_clips', 0)} |")
    lines.append(f"| Labels (distinct sounds) | {ds.get('n_labels', 0)} |")
    lines.append(f"| Speakers | {ds.get('n_speakers', 0)} |")
    lines.append(f"| Labels | {', '.join(ds.get('labels', []))} |")
    lines.append(f"| Speakers | {', '.join(ds.get('speakers', []))} |")

    # Vocabulary
    vocab = r.get("vocabulary", {})
    lines.append(f"\n## Vocabulary\n")
    lines.append(f"- **Words:** {vocab.get('n_words', 0)}")
    lines.append(f"- **Phrases:** {vocab.get('n_phrases', 0)}")
    lines.append(f"- **Pending (unknown sounds):** {vocab.get('n_pending', 0)}")

    # Warnings
    warnings = r.get("warnings", [])
    if warnings:
        lines.append(f"\n## ⚠️ Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")

    # Split info
    lines.append(f"\n## Train / Test Split\n")
    lines.append(f"- **Method:** {r.get('split_method', 'N/A')}")
    ss = r.get("split_sizes", {})
    if ss:
        lines.append(f"- **Train clips:** {ss.get('train', 0)}")
        lines.append(f"- **Test clips:** {ss.get('test', 0)}")

    # Classifier comparison
    clfs = r.get("classifiers", {})
    if clfs:
        lines.append(f"\n## Classifier Comparison\n")
        lines.append("| Classifier | Accuracy |")
        lines.append("| :--- | :--- |")
        for name, m in clfs.items():
            if "error" in m:
                lines.append(f"| {name} | ERROR: {m['error']} |")
            else:
                lines.append(f"| {name} | {m['accuracy']:.1%} |")

        # Per-class detail for best classifier
        best_name = max(
            (n for n in clfs if "accuracy" in clfs[n]),
            key=lambda n: clfs[n]["accuracy"],
            default=None,
        )
        if best_name:
            lines.append(f"\n### Best Classifier: {best_name}\n")
            pc = clfs[best_name].get("per_class", {})
            if pc:
                lines.append("| Label | Precision | Recall | Support |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for label, vals in sorted(pc.items()):
                    lines.append(
                        f"| {label} | {vals['precision']:.2f} | "
                        f"{vals['recall']:.2f} | {vals['support']} |"
                    )

    # Cross-speaker accuracy
    cs = r.get("cross_speaker", {})
    if cs:
        lines.append(f"\n## Cross-Speaker Accuracy (KNN)\n")
        lines.append("| Held-out Speaker | Accuracy | Test Clips |")
        lines.append("| :--- | :--- | :--- |")
        for spk, info in sorted(cs.items()):
            acc = info.get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            n_test = info.get("n_test", "—")
            lines.append(f"| {spk} | {acc_str} | {n_test} |")

    # Noise robustness
    nr = r.get("noise_robustness", {})
    if nr:
        lines.append(f"\n## Noise Robustness (KNN on noisy test features)\n")
        lines.append("| SNR Level | Accuracy |")
        lines.append("| :--- | :--- |")
        for level, acc in nr.items():
            lines.append(f"| {level} | {acc:.1%} |")

    # Confirmation stats
    conf = r.get("confirmation", {})
    if conf.get("total_tests", 0) > 0:
        lines.append(f"\n## User Confirmation\n")
        lines.append(f"- **Total tests:** {conf['total_tests']}")
        lines.append(f"- **Right:** {conf['total_right']} ({conf['right_share']:.0%})")
        lines.append(f"- **Wrong:** {conf['total_wrong']}")

    # Limitations (always included)
    lines.append(f"\n{LIMITATIONS_TEXT}")

    lines.append("\n---\n*Report generated by UCS Evaluation Module.*\n")

    return "\n".join(lines)


def export_report(conn, output_path: str | None = None) -> str:
    """Run evaluation and write the Markdown report to disk."""
    eval_result = run_evaluation(conn)
    report_md = generate_report(eval_result)

    if output_path is None:
        reports_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"evaluation_{stamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return output_path

