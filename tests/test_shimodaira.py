import numpy as np
from llm_atras.shimodaira import likelihood_coordinates


def test_q_distances_equal_variance_of_likelihood_difference():
    L = np.random.default_rng(6).normal(size=(5, 21))
    _, Q, _ = likelihood_coordinates(L, clip=False)
    assert np.allclose(Q.mean(0), 0)
    assert np.allclose(Q.mean(1), 0)
    for i in range(5):
        for j in range(5):
            assert np.isclose(np.mean((Q[i]-Q[j])**2), np.var(L[i]-L[j]))


def test_global_clipping_precedes_centering():
    L = np.array([[-100., -2., -1.], [-3., -2., -1.]])
    clipped, Q, threshold = likelihood_coordinates(L)
    assert threshold == np.percentile(L, 2)
    assert clipped.min() == threshold
    assert np.allclose(Q.mean(0), 0)
    assert np.allclose(Q.mean(1), 0)


def test_q_ignores_per_model_likelihood_offsets_without_clipping():
    L = np.random.default_rng(7).normal(size=(4, 13))
    _, Q, _ = likelihood_coordinates(L, clip=False)
    _, shifted, _ = likelihood_coordinates(L+np.arange(4)[:, None]*3, clip=False)
    assert np.allclose(Q, shifted)
