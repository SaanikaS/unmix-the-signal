"""Create mixtures: random mixing matrices and additive noise at a target SNR."""
import numpy as np


def make_mixing_matrix(n_sources, n_mixtures, rng, max_condition=10.0):
    """Random Gaussian mixing matrix, rejecting ill-conditioned draws."""
    while True:
        A = rng.standard_normal((n_mixtures, n_sources))
        if np.linalg.cond(A) < max_condition:
            return A


def mix(S, A):
    """Instantaneous linear mixing: X = A S. S: (n_sources, T) -> X: (n_mixtures, T)."""
    return A @ S


def add_noise(X, snr_db, rng):
    """Add white Gaussian noise so that the mean power SNR equals snr_db."""
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return X + rng.standard_normal(X.shape) * np.sqrt(noise_power)
