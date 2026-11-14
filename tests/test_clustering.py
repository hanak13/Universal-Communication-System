"""
tests/test_clustering.py — verify pure numpy KMeans, DBSCAN, PCA, scaler, and metrics.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clustering import (
    SimpleScaler,
    run_kmeans,
    run_dbscan,
    reduce_pca,
    clustering_metrics,
    cluster_vs_label_table,
    find_outliers,
)


def test_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = SimpleScaler()
    scaled = scaler.fit_transform(X)
    assert np.allclose(np.mean(scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(scaled, axis=0), [1.0, 1.0])


def test_kmeans():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ])
    labels = run_kmeans(X, k=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_dbscan():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [100.0, 100.0],  # noise / separated
    ])
    labels = run_dbscan(X, eps=0.5, min_samples=2)
    assert labels[0] == labels[1]
    assert labels[0] >= 0
    assert labels[2] == -1  # noise


def test_pca():
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0],
    ])
    pca = reduce_pca(X, n_components=2)
    assert pca.shape == (4, 2)


def test_metrics_and_outliers():
    labels = ["a", "a", "b", "b"]
    clusters = np.array([0, 0, 1, 1])
    m = clustering_metrics(labels, clusters)
    assert np.isclose(m["ari"], 1.0)
    assert np.isclose(m["homogeneity"], 1.0)

    table = cluster_vs_label_table(labels, clusters)
    assert table["a"][0] == 2
    assert table["b"][1] == 2

    outliers = find_outliers(["a", "a", "a"], np.array([0, 0, 1]), ["f1", "f2", "f3"])
    assert len(outliers) == 1
    assert outliers[0]["filename"] == "f3"

