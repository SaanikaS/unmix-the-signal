import numpy as np

from unmix.ica import ica_separate, pca_separate, whiten
from unmix.metrics import evaluate, si_sdr
from unmix.mixing import make_mixing_matrix, mix


def test_whitening_gives_identity_covariance():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 20000))
    X = rng.standard_normal((3, 3)) @ S          # correlated mixtures
    Z, _, _ = whiten(X)
    C = Z @ Z.T / Z.shape[1]
    assert np.allclose(C, np.eye(3), atol=1e-2)


def test_fastica_recovers_laplacian_sources():
    rng = np.random.default_rng(1)
    S = rng.laplace(size=(3, 20000))
    S = S / S.std(axis=1, keepdims=True)
    A = make_mixing_matrix(3, 3, rng)
    X = mix(S, A)

    S_hat, _, _ = ica_separate(X, n_sources=3, random_state=0)
    result = evaluate(S, S_hat)
    assert result["mean_correlation"] > 0.95
    assert result["mean_si_sdr"] > 15


def test_ica_beats_pca_on_non_gaussian_sources():
    rng = np.random.default_rng(2)
    S = rng.laplace(size=(3, 20000))
    S = S / S.std(axis=1, keepdims=True)
    X = mix(S, make_mixing_matrix(3, 3, rng))

    ica = evaluate(S, ica_separate(X, n_sources=3, random_state=0)[0])
    pca = evaluate(S, pca_separate(X, n_sources=3))
    assert ica["mean_si_sdr"] > pca["mean_si_sdr"] + 5


def test_si_sdr_is_scale_and_sign_invariant():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(5000)
    est = s + 0.1 * rng.standard_normal(5000)   # imperfect estimate, nonzero distortion
    base = si_sdr(s, est)
    assert abs(si_sdr(s, 3.7 * est) - base) < 1e-6
    assert abs(si_sdr(s, -0.5 * est) - base) < 1e-6
