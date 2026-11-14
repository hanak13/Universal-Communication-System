"""
UCS Clustering — pure-numpy KMeans, DBSCAN, PCA, t-SNE, scaler.
No scikit-learn dependency.
"""
import os
import json
import numpy as np

from config import SCALER_PATH, TSNE_PERPLEXITY, MIN_CLIPS_FOR_CLUSTERING


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  StandardScaler (pure numpy, saved as JSON)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimpleScaler:
    """Zero-mean, unit-variance scaler (drop-in for StandardScaler)."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SimpleScaler":
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid div-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "mean": self.mean_.tolist(),
                "std": self.std_.tolist(),
            }, f)

    @classmethod
    def load(cls, path: str) -> "SimpleScaler":
        with open(path, "r") as f:
            d = json.load(f)
        s = cls()
        s.mean_ = np.array(d["mean"])
        s.std_ = np.array(d["std"])
        return s


def fit_scaler(vectors: np.ndarray) -> SimpleScaler:
    """Fit and persist a scaler. Returns it."""
    scaler = SimpleScaler()
    scaler.fit(vectors)
    scaler.save(SCALER_PATH)
    return scaler


def load_scaler() -> SimpleScaler | None:
    path = SCALER_PATH
    if os.path.isfile(path):
        return SimpleScaler.load(path)
    return None


def scale_vectors(vectors: np.ndarray, scaler: SimpleScaler | None = None) -> np.ndarray:
    if scaler is None:
        scaler = load_scaler()
    if scaler is None:
        scaler = fit_scaler(vectors)
    return scaler.transform(vectors)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KMeans (pure numpy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Return cluster labels via KMeans."""
    rng = np.random.RandomState(42)
    n = X.shape[0]
    # Init: pick k random points
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for j in range(len(centroids)):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)

    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DBSCAN (pure numpy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Return cluster labels (-1 = noise)."""
    n = X.shape[0]
    # Pairwise distance matrix
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = np.where(dist_matrix[i] <= eps)[0]

        if len(neighbours) < min_samples:
            continue  # noise for now

        # Expand cluster
        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = np.where(dist_matrix[q] <= eps)[0]
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(q_neighbours.tolist())
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dimensionality reduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reduce_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA via SVD (pure numpy)."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n = min(n_components, Vt.shape[0])
    return X_centered @ Vt[:n].T


def reduce_tsne(X: np.ndarray, n_components: int = 2,
                lr: float = 200.0, n_iter: int = 500) -> np.ndarray:
    """Simple exact t-SNE (fine for < 200 points)."""
    n = X.shape[0]
    perp = min(TSNE_PERPLEXITY, max(1, n - 1))

    # Pairwise distances
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    D = np.maximum(D, 0)

    # Compute P (joint probabilities)
    P = np.zeros((n, n))
    for i in range(n):
        # Binary search for sigma to match perplexity
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2
            pi = np.exp(-D[i] / (2 * sigma ** 2))
            pi[i] = 0
            s = pi.sum()
            if s == 0:
                lo = sigma
                continue
            pi /= s
            entropy = -np.sum(pi[pi > 0] * np.log2(pi[pi > 0]))
            if entropy > np.log2(perp):
                hi = sigma
            else:
                lo = sigma
        P[i] = pi

    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    # Init Y randomly
    rng = np.random.RandomState(42)
    Y = rng.randn(n, n_components) * 1e-4
    velocity = np.zeros_like(Y)
    momentum = 0.5

    for it in range(n_iter):
        # Q distribution
        sq_y = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + sq_y[:, None] + sq_y[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = num / num.sum()
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ[i] * num[i])[:, None] * diff, axis=0)

        if it > 100:
            momentum = 0.8
        velocity = momentum * velocity - lr * grad
        Y += velocity

    return Y


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _contingency(true_labels: list, pred_labels: np.ndarray) -> np.ndarray:
    """Build a contingency matrix."""
    t_unique = sorted(set(true_labels))
    p_unique = sorted(set(int(x) for x in pred_labels))
    t_map = {v: i for i, v in enumerate(t_unique)}
    p_map = {v: i for i, v in enumerate(p_unique)}
    mat = np.zeros((len(t_unique), len(p_unique)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        mat[t_map[t], p_map[int(p)]] += 1
    return mat


def _entropy(labels) -> float:
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-12))


def clustering_metrics(true_labels: list[str], cluster_labels: np.ndarray) -> dict:
    """Return ARI and homogeneity."""
    n = len(true_labels)
    cont = _contingency(true_labels, cluster_labels)

    # ARI
    sum_comb_rows = sum(r * (r - 1) / 2 for r in cont.sum(axis=1))
    sum_comb_cols = sum(c * (c - 1) / 2 for c in cont.sum(axis=0))
    sum_comb_cells = sum(cont[i, j] * (cont[i, j] - 1) / 2
                         for i in range(cont.shape[0])
                         for j in range(cont.shape[1]))
    total_comb = n * (n - 1) / 2
    expected = sum_comb_rows * sum_comb_cols / total_comb if total_comb > 0 else 0
    max_idx = (sum_comb_rows + sum_comb_cols) / 2
    if max_idx == expected:
        ari = 1.0
    else:
        ari = (sum_comb_cells - expected) / (max_idx - expected)

    # Homogeneity: 1 - H(C|K) / H(C)
    h_c = _entropy(true_labels)
    if h_c == 0:
        homogeneity = 1.0
    else:
        h_c_k = 0.0
        for j in range(cont.shape[1]):
            col_sum = cont[:, j].sum()
            if col_sum == 0:
                continue
            p = cont[:, j] / col_sum
            h_c_k -= (col_sum / n) * np.sum(p * np.log(p + 1e-12))
        homogeneity = max(0.0, min(1.0, float(1.0 - h_c_k / h_c)))

    return {"ari": max(-1.0, min(1.0, float(ari))), "homogeneity": homogeneity}


def cluster_vs_label_table(
    labels: list[str], clusters: np.ndarray
) -> dict[str, dict[int, int]]:
    table: dict[str, dict[int, int]] = {}
    for lbl, cl in zip(labels, clusters):
        if lbl not in table:
            table[lbl] = {}
        table[lbl][int(cl)] = table[lbl].get(int(cl), 0) + 1
    return table


def find_outliers(
    labels: list[str], clusters: np.ndarray, filenames: list[str],
) -> list[dict]:
    cross = cluster_vs_label_table(labels, clusters)
    majority: dict[str, int] = {}
    for lbl, counts in cross.items():
        majority[lbl] = max(counts, key=counts.get)

    outliers = []
    for lbl, cl, fn in zip(labels, clusters, filenames):
        if int(cl) != majority.get(lbl, -999):
            outliers.append({"filename": fn, "label": lbl, "cluster": int(cl),
                             "expected_cluster": majority[lbl]})
    return outliers
