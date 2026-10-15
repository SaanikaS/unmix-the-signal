"""PCA whitening and FastICA, implemented from scratch with NumPy.

Shapes convention: data matrices are (n_signals, n_samples).
"""
import numpy as np


def center(X):
    """Subtract the per-signal mean. Returns (centered, mean)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def whiten(X, n_components=None, eps=1e-12):
    """PCA whitening via eigendecomposition of the covariance matrix.

    With C = E D E^T (eigenvalues D, eigenvectors E), the whitening matrix is
    W = D^(-1/2) E^T. The whitened signals Z = W X have identity covariance,
    so they are uncorrelated with unit variance. This removes second-order
    structure and leaves ICA only to find the remaining rotation.

    Returns (Z, W, mean). If n_components < n_signals, only the top
    components are kept (dimensionality reduction).
    """
    Xc, mean = center(X)
    n_features, n_samples = Xc.shape
    C = (Xc @ Xc.T) / n_samples

    eigvals, eigvecs = np.linalg.eigh(C)          # ascending order
    order = np.argsort(eigvals)[::-1]             # sort descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    k = n_components or n_features
    eigvals, eigvecs = eigvals[:k], eigvecs[:, :k]
    if np.any(eigvals <= eps):
        raise ValueError("Covariance is (near-)singular; reduce n_components.")

    W = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    Z = W @ Xc
    return Z, W, mean


def pca_separate(X, n_sources=None):
    """PCA baseline: the whitened principal-component scores.

    These outputs are uncorrelated but generally NOT independent, so they
    usually fail to separate sources. This is the comparison point for ICA.
    """
    Z, _, _ = whiten(X, n_components=n_sources)
    return Z


def _sym_decorrelate(W):
    """Symmetric decorrelation: W <- (W W^T)^(-1/2) W."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(s)) @ U.T @ W


def _nonlinearity(u, kind):
    """Return g(u) and its derivative g'(u).

    tanh: suits super-Gaussian sources (speech, music).
    cube: suits sub-Gaussian sources.
    """
    if kind == "tanh":
        t = np.tanh(u)
        return t, 1.0 - t ** 2
    if kind == "cube":
        return u ** 3, 3.0 * u ** 2
    raise ValueError(f"Unknown nonlinearity: {kind}")


def fastica(Z, nonlinearity="tanh", max_iter=500, tol=1e-6, random_state=None):
    """Symmetric FastICA on whitened data Z of shape (k, n_samples).

    Each iteration applies the fixed-point update
        w <- E[z g(w^T z)] - E[g'(w^T z)] w
    to all rows at once, then symmetrically decorrelates them. Converged when
    |<w_new, w_old>| is 1 for every row.

    Returns (W, n_iter) where the estimated sources are S = W @ Z.
    """
    rng = np.random.default_rng(random_state)
    k, T = Z.shape
    W = _sym_decorrelate(rng.standard_normal((k, k)))

    for it in range(1, max_iter + 1):
        WZ = W @ Z
        g, dg = _nonlinearity(WZ, nonlinearity)
        W_new = (g @ Z.T) / T - dg.mean(axis=1)[:, None] * W
        W_new = _sym_decorrelate(W_new)

        converged = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)) < tol
        W = W_new
        if converged:
            break
    return W, it


def ica_separate(X, n_sources=None, nonlinearity="tanh", random_state=None,
                 max_iter=500, tol=1e-6):
    """Full ICA pipeline: center -> whiten -> FastICA.

    X: mixtures, shape (n_mixtures, n_samples), n_mixtures >= n_sources.
    Returns (S_hat, unmixing, n_iter):
      S_hat    estimated sources, shape (n_sources, n_samples)
      unmixing matrix U with S_hat = U @ (X - mean)
      n_iter   FastICA iterations used
    """
    Z, W_white, _ = whiten(X, n_components=n_sources)
    W, n_iter = fastica(Z, nonlinearity=nonlinearity, max_iter=max_iter,
                        tol=tol, random_state=random_state)
    S_hat = W @ Z
    unmixing = W @ W_white
    return S_hat, unmixing, n_iter
