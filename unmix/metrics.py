"""Evaluation that is invariant to ICA's permutation and scale/sign ambiguity."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _center_normalize_rows(X):
    X = X - X.mean(axis=1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def correlation_matrix(S, S_hat):
    """|Pearson correlation| between every reference S[i] and estimate S_hat[j]."""
    return np.abs(_center_normalize_rows(S) @ _center_normalize_rows(S_hat).T)


def match_sources(S, S_hat):
    """Hungarian matching. order[i] is the index of the estimate that best matches S[i]."""
    C = correlation_matrix(S, S_hat)
    _, order = linear_sum_assignment(-C)
    return order


def si_sdr(reference, estimate):
    """Scale-invariant SDR in dB.

    Project the estimate onto the reference to get the target component; the
    residual is distortion. Sign and scale of the estimate do not matter.
    """
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    distortion = est - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) /
                                 (np.dot(distortion, distortion) + 1e-12)))


def evaluate(S, S_hat):
    """Match estimates to references and report per-source and mean metrics."""
    if S.shape[0] != S_hat.shape[0]:
        raise ValueError("Need the same number of references and estimates.")
    order = match_sources(S, S_hat)
    C = correlation_matrix(S, S_hat)
    corr = [float(C[i, order[i]]) for i in range(len(S))]
    sdr = [si_sdr(S[i], S_hat[order[i]]) for i in range(len(S))]
    return {
        "order": order.tolist(),
        "correlation": corr,
        "si_sdr": sdr,
        "mean_correlation": float(np.mean(corr)),
        "mean_si_sdr": float(np.mean(sdr)),
    }
