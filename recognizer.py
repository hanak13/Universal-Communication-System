"""
UCS Recognizer — pure-numpy KNN matcher with majority voting.
No scikit-learn dependency (safe under Windows Device Guard).
"""
from collections import namedtuple
import numpy as np

from config import (
    RECOGNIZE_THRESHOLD,
    PENDING_MERGE_THRESHOLD,
    KNN_NEIGHBORS,
)
from clustering import fit_scaler, load_scaler, SimpleScaler


Candidate = namedtuple("Candidate", ["label", "distance", "votes"])


class Recognizer:
    """
    K-Nearest Neighbors classifier with majority voting and distance tie-breaking.
    Fit on labelled (vector, label) pairs.
    recognize(vector) → {label, distance, top3, is_unknown, confidence}
    """

    def __init__(self, k: int = KNN_NEIGHBORS):
        self.scaler: SimpleScaler | None = None
        self.X: np.ndarray | None = None
        self.labels: list[str] = []
        self._k_config: int = k
        self._k: int = k
        self._fitted: bool = False

    def fit(self, vectors: list[list[float]], labels: list[str]) -> None:
        if len(vectors) < 1 or len(labels) < 1:
            self._fitted = False
            return

        X = np.array(vectors, dtype=np.float64)
        self.scaler = fit_scaler(X)
        self.X = self.scaler.transform(X)
        self.labels = list(labels)
        # K is min(configured K, number of training samples)
        self._k = max(1, min(self._k_config, len(vectors)))
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def recognize(self, vector: list[float], threshold: float | None = None) -> dict:
        if not self._fitted or self.X is None or len(self.labels) == 0:
            return {
                "label": "unknown",
                "distance": float("inf"),
                "top3": [],
                "is_unknown": True,
                "confidence": 0.0,
            }

        thresh = threshold if threshold is not None else RECOGNIZE_THRESHOLD
        x = np.array(vector, dtype=np.float64).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # 1. Compute Euclidean distances to all training points
        dists = np.linalg.norm(self.X - x_scaled, axis=1)

        # 2. Take K nearest samples overall (K = min(config K, number of training samples))
        k = min(self._k, len(self.labels))
        nearest_idx = np.argsort(dists)[:k]

        # 3. Aggregate voting stats per label among the K nearest neighbors
        # label -> list of distances
        label_dists: dict[str, list[float]] = {}
        for idx in nearest_idx:
            lbl = self.labels[idx]
            label_dists.setdefault(lbl, []).append(float(dists[idx]))

        # 4. Rank candidates by:
        #    a) Most votes (descending)
        #    b) Tie-breaker 1: smaller sum of distances among voting samples (ascending)
        #    c) Tie-breaker 2: closest single sample (ascending)
        #    d) Deterministic tie-breaker: label string
        candidates: list[Candidate] = []
        for lbl, d_list in label_dists.items():
            votes = len(d_list)
            sum_d = float(sum(d_list))
            min_d = float(min(d_list))
            mean_d = float(sum_d / votes)
            candidates.append(Candidate(label=lbl, distance=mean_d, votes=votes))

        candidates.sort(
            key=lambda c: (
                -c.votes,                      # highest vote count first
                sum(label_dists[c.label]),     # smaller sum of distances
                min(label_dists[c.label]),     # closest individual sample
                c.label                        # stable tie-breaker
            )
        )

        winner = candidates[0]
        # The returned distance is the mean distance of the winning label's voting samples
        # among the K nearest neighbors. This represents the average acoustic distance
        # to the cluster of supporting exemplars, rather than relying on a single outlier sample.
        winning_dist = winner.distance
        confidence = float(winner.votes / k) if k > 0 else 0.0

        # 5. Apply unknown distance threshold consistently to the winning label's mean distance
        is_unknown = winning_dist > thresh
        predicted_label = "unknown" if is_unknown else winner.label

        # 6. Top-3 candidates ranked by votes and distance
        top3 = candidates[:3]

        return {
            "label": predicted_label,
            "distance": winning_dist,
            "top3": top3,
            "is_unknown": is_unknown,
            "confidence": confidence,
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
