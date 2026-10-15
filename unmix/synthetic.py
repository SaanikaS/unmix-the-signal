"""Synthetic, non-Gaussian test sources. Useful as a sanity check without audio files."""
import numpy as np
from scipy import signal


def make_synthetic_sources(n_sources, sr, duration, rng):
    """Sine, sawtooth, square, and Laplacian noise, cycled to n_sources rows.

    Returns zero-mean, unit-variance signals of shape (n_sources, T).
    """
    t = np.arange(int(sr * duration)) / sr
    makers = [
        lambda: np.sin(2 * np.pi * 440 * t),
        lambda: signal.sawtooth(2 * np.pi * 220 * t),
        lambda: signal.square(2 * np.pi * 330 * t),
        lambda: rng.laplace(size=t.size),
    ]
    S = np.stack([makers[i % len(makers)]() for i in range(n_sources)])
    S = S - S.mean(axis=1, keepdims=True)
    S = S / S.std(axis=1, keepdims=True)
    return S
