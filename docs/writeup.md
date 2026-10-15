# Unmix the Signal (ICA): Write-up

## Goal

Recover clean audio sources from mixed recordings using Independent Component Analysis, implemented from scratch.

## 1. Why ICA over PCA

PCA finds uncorrelated directions, meaning the second-order statistics (covariance) of the data are diagonal. Uncorrelated is weaker than independent. Two signals can be uncorrelated and still be strongly dependent, for example one being the square of the other. PCA is therefore guaranteed to remove only second-order dependence. ICA requires full statistical independence, which constrains higher-order statistics as well. For Gaussian sources this is impossible: any rotation of independent Gaussians is still independent, so the mixture is not identifiable. For non-Gaussian sources the rotation is unique up to permutation and scale, which is why ICA can separate them.

*Result to report:* SI-SDR for ICA vs PCA (see `results/summary.md`).

## 2. What whitening does and why it simplifies ICA

Whitening centers the data, then rotates and rescales it so the covariance is the identity. It does two things:

1. It removes all second-order correlation, so the remaining problem is a pure rotation.
2. It reduces the search space. An arbitrary mixing matrix has n² parameters, but after whitening the unmixing matrix is orthogonal, which has n(n−1)/2 free parameters.

Whitening is computed from the covariance `C = E D Eᵀ` as `W = D^(-1/2) Eᵀ`. It also acts as dimensionality reduction when there are more mixtures than sources.

## 3. Why non-Gaussianity enables separation and how the CLT relates

A sum of independent random variables is closer to Gaussian than any of the components (the central limit theorem). A mixture is therefore "more Gaussian" than its sources. ICA reverses this: it searches for the rotation that makes each output as non-Gaussian as possible. Kurtosis or negentropy measure non-Gaussianity, and at the optimum each output is a single source.

FastICA maximizes an approximation of negentropy using the nonlinearity `g = tanh` (super-Gaussian sources such as speech and music) or `g = u³` (sub-Gaussian sources).

## 4. How experiments and metrics were chosen

- **Sources:** speech, music, and ambient signals, so the sources differ in statistics and are non-Gaussian.
- **Mixing:** random Gaussian matrices, rejected if ill-conditioned, so that separation is not trivially hard.
- **Noise:** additive white Gaussian noise at 30, 20, 10, and 0 dB SNR, to measure robustness.
- **Metrics:**
  - Absolute correlation after Hungarian matching: a simple, interpretable check of recovery.
  - SI-SDR: a scale-invariant distortion measure in dB, standard in source separation. It handles the scale and sign ambiguity of ICA.
- **Baseline:** PCA scores, to show the value of higher-order statistics.

## Limitations and next steps

- Instantaneous mixing only. Real room recordings need convolutive or frequency-domain ICA.
- Number of sources assumed known.
- Real-world multi-device recordings and a mini-Shazam (song identification) are natural extensions.
