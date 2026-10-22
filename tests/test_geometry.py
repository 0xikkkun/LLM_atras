import subprocess
import sys

import numpy as np
import pytest

from llm_atras import (
    align, cka_distance, graph_distances, knn_distance, metric_audit,
    normalize, pairwise_from_features, rank_distance,
)


def test_core_import_does_not_load_torch():
    subprocess.run([sys.executable, "-c", "import llm_atras,sys; assert 'torch' not in sys.modules"], check=True)


def test_rotation_and_translation_are_recovered_without_scale_change():
    X = np.random.default_rng(1).normal(size=(40, 2))
    Q = np.array([[0.0, -1.0], [1.0, 0.0]])
    Y, R, residual = align(X @ Q + 10, X, rotation_only=True)
    assert residual < 1e-12
    assert np.linalg.det(R) == pytest.approx(1)
    np.testing.assert_allclose(pairwise_from_features(Y), pairwise_from_features(X), atol=1e-12)


def test_reflection_is_optional():
    X = np.random.default_rng(2).normal(size=(40, 2))
    reflected = X @ np.diag([1, -1])
    _, R, free = align(reflected, X)
    _, rotation, constrained = align(reflected, X, rotation_only=True)
    assert free < 1e-12
    assert np.linalg.det(R) < 0
    assert np.linalg.det(rotation) > 0
    assert constrained > 0.1


def test_isotropic_fisher_metric_scales_shortest_paths():
    X = np.random.default_rng(3).normal(size=(30, 3))
    metrics = np.broadcast_to(4 * np.eye(3), (len(X), 3, 3))
    graphs, components = graph_distances(X, metrics, k=8)
    assert components == 1
    np.testing.assert_allclose(graphs["fisher"]["S"], 2 * graphs["euclidean"]["S"], atol=1e-10)
    np.testing.assert_array_equal(graphs["fisher"]["a"], graphs["euclidean"]["a"])
    np.testing.assert_allclose(normalize(graphs["fisher"]["S"]), normalize(graphs["euclidean"]["S"]))


def test_disconnected_graph_is_reported_without_imputing_distances():
    X = np.array([[0., 0.], [.1, 0.], [10., 0.], [10.1, 0.]])
    metrics = np.broadcast_to(np.eye(2), (4, 2, 2))
    graphs, components = graph_distances(X, metrics, k=1)
    assert graphs is None and components == 2


def test_cka_is_invariant_to_rotation_and_uniform_scale():
    X = np.random.default_rng(4).normal(size=(40, 3))
    Q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    np.testing.assert_allclose(cka_distance([X, 7 * X @ Q]), 0, atol=1e-12)


def test_rank_distance_discards_monotonic_rescaling():
    X = np.random.default_rng(6).normal(size=(20, 3))
    S = pairwise_from_features(X)
    np.testing.assert_allclose(rank_distance([S, S ** 3]), 0, atol=1e-12)


def test_neighbor_distance_uses_actual_neighbor_count_for_small_inputs():
    S = pairwise_from_features(np.array([[0., 0.], [1., 0.], [3., 0.]]))
    np.testing.assert_allclose(knn_distance([S, S], k=10), 0)


def test_triangle_violation_is_detected():
    D = np.array([[0., 1., 3.], [1., 0., 1.], [3., 1., 0.]])
    result = metric_audit(D)
    assert result["triangle_violations"] == 2
    assert result["triangle_max_excess"] == pytest.approx(1)


def test_zero_distance_scale_and_invalid_topology_are_rejected():
    with pytest.raises(ValueError):
        normalize(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        graph_distances(np.zeros((3, 2)), np.zeros((3, 2, 2)), k=3)
