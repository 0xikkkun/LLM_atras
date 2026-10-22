"""Likelihood coordinates from Oyama et al. (ACL 2025), Sections 2.3/3.3.

Q distances approximate sqrt(2 KL) locally; L distances retain mean likelihood.
The global clipping threshold must be frozen for out-of-sample comparisons.
"""
import numpy as np


def likelihood_coordinates(log_likelihood, clip=True):
    L = np.asarray(log_likelihood, dtype=np.float64)
    if L.ndim != 2 or min(L.shape) < 2 or not np.isfinite(L).all():
        raise ValueError('Expected a finite models-by-texts likelihood matrix')
    threshold = float(np.percentile(L, 2)) if clip else None
    L = np.maximum(L, threshold) if clip else L.copy()
    Q = L - L.mean(axis=1, keepdims=True) - L.mean(axis=0, keepdims=True) + L.mean()
    return L, Q, threshold
