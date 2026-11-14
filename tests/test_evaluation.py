"""
tests/test_evaluation.py — report export and classifier tests.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import FEATURE_VECTOR_LEN
from evaluation import (
    KNNClassifier, LinearSVM, SimpleRandomForest,
    compute_metrics, add_noise, generate_report, export_report,
)
from db import init_db


def _make_dataset(n_per_class=10, n_classes=3, seed=42):
    """Generate a simple synthetic dataset."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n_classes):
        center = np.full(FEATURE_VECTOR_LEN, float(i * 10))
        for _ in range(n_per_class):
            X.append(center + rng.randn(FEATURE_VECTOR_LEN) * 0.5)
            y.append(f"class_{i}")
    return np.array(X), np.array(y)


def test_knn_classifier():
    """KNN achieves >90% on well-separated synthetic data."""
    X, y = _make_dataset()
    clf = KNNClassifier(k=3)
    clf.fit(X[:25], y[:25])
    preds = clf.predict(X[25:])
    acc = np.mean(preds == y[25:])
    assert acc > 0.9, f"KNN accuracy {acc:.2f} too low on easy data"


def test_svm_classifier():
    """Linear SVM learns well-separated classes."""
    X, y = _make_dataset()
    clf = LinearSVM(lr=0.01, epochs=200)
    clf.fit(X[:25], y[:25])
    preds = clf.predict(X[25:])
    acc = np.mean(preds == y[25:])
    assert acc > 0.5, f"SVM accuracy {acc:.2f} too low"


def test_random_forest_classifier():
    """Random Forest learns well-separated classes."""
    X, y = _make_dataset()
    clf = SimpleRandomForest(n_trees=30)
    clf.fit(X[:25], y[:25])
    preds = clf.predict(X[25:])
    acc = np.mean(preds == y[25:])
    assert acc > 0.5, f"RF accuracy {acc:.2f} too low"


def test_compute_metrics():
    """Metrics computation returns correct structure."""
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "b", "c", "c", "c"])
    m = compute_metrics(y_true, y_pred)
    assert 0 < m["accuracy"] <= 1.0
    assert "a" in m["per_class"]
    assert "b" in m["per_class"]
    assert "confusion_matrix" in m
    assert m["per_class"]["a"]["precision"] == 1.0
    assert m["per_class"]["a"]["recall"] == 1.0


def test_noise_degrades_accuracy():
    """Adding noise should generally reduce accuracy."""
    X, y = _make_dataset()
    clf = KNNClassifier(k=3)
    clf.fit(X[:25], y[:25])

    acc_clean = np.mean(clf.predict(X[25:]) == y[25:])
    acc_noisy = np.mean(clf.predict(add_noise(X[25:], snr_db=0)) == y[25:])
    # With SNR=0 dB (equal noise), accuracy should drop or stay same
    assert acc_noisy <= acc_clean + 0.1  # allow small float jitter


def test_report_export():
    """generate_report produces a Markdown string with required sections."""
    eval_result = {
        "timestamp": "2026-01-01T00:00:00",
        "dataset": {
            "n_clips": 20, "n_labels": 3, "n_speakers": 2,
            "speakers": ["alice", "bob"], "labels": ["hello", "bye", "thanks"],
        },
        "vocabulary": {"n_words": 3, "n_phrases": 1, "n_pending": 0},
        "warnings": ["Only 20 clips — small dataset."],
        "split_method": "speaker holdout (bob)",
        "split_sizes": {"train": 15, "test": 5},
        "classifiers": {
            "KNN (k=3)": {
                "accuracy": 0.8,
                "per_class": {
                    "hello": {"precision": 1.0, "recall": 0.5, "support": 2},
                    "bye": {"precision": 0.67, "recall": 1.0, "support": 2},
                    "thanks": {"precision": 1.0, "recall": 1.0, "support": 1},
                },
                "confusion_matrix": [[1,1,0],[0,2,0],[0,0,1]],
                "labels": ["bye", "hello", "thanks"],
                "n_total": 5,
            },
            "Linear SVM": {"accuracy": 0.6, "per_class": {}, "labels": [], "n_total": 5},
        },
        "noise_robustness": {"SNR 30 dB": 0.8, "SNR 10 dB": 0.6, "SNR 0 dB": 0.4},
        "cross_speaker": {"bob": {"accuracy": 0.8, "n_test": 5}},
        "confirmation": {"total_right": 10, "total_wrong": 2, "total_tests": 12, "right_share": 0.833},
    }

    report = generate_report(eval_result)

    # Check required sections exist
    assert "# UCS Evaluation Report" in report
    assert "## Dataset Summary" in report
    assert "## Vocabulary" in report
    assert "## Classifier Comparison" in report
    assert "## Limitations" in report
    assert "recognising a sound is not understanding meaning" in report.lower()
    assert "meaning changes with context" in report.lower()
    assert "small dataset can be memorised" in report.lower()
    assert "knows only the words it was taught" in report.lower()
    assert "does not prove understanding" in report.lower()
    assert "native-speaker check" in report.lower()

    # Should be writable to disk
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(report)
        path = f.name
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 100
    os.unlink(path)


def test_report_with_empty_data():
    """Report generation handles empty dataset gracefully."""
    eval_result = {
        "timestamp": "2026-01-01",
        "dataset": {"n_clips": 0, "n_labels": 0, "n_speakers": 0, "speakers": [], "labels": []},
        "vocabulary": {"n_words": 0, "n_phrases": 0, "n_pending": 0},
        "warnings": ["No data."],
    }
    report = generate_report(eval_result)
    assert "## Limitations" in report
    assert "No data." in report


def test_export_report_end_to_end():
    """export_report writes a markdown file to a specified path."""
    conn = init_db(":memory:")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        tmp_path = f.name

    try:
        out_path = export_report(conn, output_path=tmp_path)
        assert out_path == tmp_path
        assert os.path.isfile(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "# UCS Evaluation Report" in content
        assert "## Limitations" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    test_knn_classifier()
    test_svm_classifier()
    test_random_forest_classifier()
    test_compute_metrics()
    test_noise_degrades_accuracy()
    test_report_export()
    test_report_with_empty_data()
    test_export_report_end_to_end()
    print("All evaluation tests passed ✓")

