"""
UCS Recognizer — pure-numpy KNN matcher from labelled clips.
No scikit-learn dependency.
"""
import numpy as np

from config import (
    RECOGNIZE_THRESHOLD,
    PENDING_MERGE_THRESHOLD,
    KNN_NEIGHBORS,
)
from clustering import fit_scaler, load_scaler, SimpleScaler


class Recognizer:
    """
    Fit on labelled (vector, label) pairs.
    recognize(vector) → {label, distance, top3, is_unknown}
    """

    def __init__(self):
        self.scaler: SimpleScaler | None = None
        self.X: np.ndarray | None = None
        self.labels: list[str] = []
        self._k: int = KNN_NEIGHBORS
        self._fitted = False

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        if len(vectors) < 1 or len(set(labels)) < 1:
            self._fitted = False
            return

        X = np.array(vectors, dtype=np.float64)
        self.scaler = fit_scaler(X)
        self.X = self.scaler.transform(X)
        self.labels = list(labels)
        self._k = min(KNN_NEIGHBORS, len(vectors))
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def recognize(self, vector: list[float], threshold: float | None = None) -> dict:
        if not self._fitted:
            return {"label": "unknown", "distance": float("inf"),
                    "top3": [], "is_unknown": True}

        thresh = threshold if threshold is not None else RECOGNIZE_THRESHOLD
        x = np.array(vector, dtype=np.float64).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # Compute distances to all training points
        dists = np.linalg.norm(self.X - x_scaled, axis=1)
        nearest_idx = np.argsort(dists)[:self._k]

        # Top-3 unique labels
        top3 = []
        seen = set()
        for idx in nearest_idx:
            lbl = self.labels[idx]
            if lbl not in seen:
                top3.append((lbl, float(dists[idx])))
                seen.add(lbl)
            if len(top3) >= 3:
                break

        best_label = top3[0][0] if top3 else "unknown"
        best_dist = top3[0][1] if top3 else float("inf")

        return {
            "label": "unknown" if best_dist > thresh else best_label,
            "distance": best_dist,
            "top3": top3,
            "is_unknown": best_dist > thresh,
        }


def find_or_create_pending_name(
    vector: list[float],
    pending_vectors: list[dict],
    threshold: float | None = None,
) -> tuple[str, bool]:
    thresh = threshold if threshold is not None else PENDING_MERGE_THRESHOLD
    v = np.array(vector, dtype=np.float64)

    best_name = None
    best_dist = float("inf")

    for pend in pending_vectors:
        if pend.get("vector") is None:
            continue
        pv = np.array(pend["vector"], dtype=np.float64)
        dist = float(np.linalg.norm(v - pv))
        if dist < best_dist:
            best_dist = dist
            best_name = pend["name"]

    if best_name is not None and best_dist <= thresh:
        return best_name, True

    max_n = 0
    for pend in pending_vectors:
        name = pend.get("name", "")
        if name.startswith("new_word_"):
            try:
                n = int(name.split("_")[-1])
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    return f"new_word_{max_n + 1}", False
