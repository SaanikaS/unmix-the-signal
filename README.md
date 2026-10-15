# Unmix the Signal (ICA)

Recover clean audio sources from mixed recordings using **Independent Component Analysis (ICA)**, implemented from scratch with NumPy.

The "cocktail party problem": several people talk at once, several microphones each record a different blend of them. Given only the blends, we recover the individual voices/instruments.

- PCA whitening implemented from scratch (covariance + eigendecomposition)
- FastICA implemented from scratch (no `sklearn.decomposition.FastICA`)
- Mixing, noise injection, and WAV import/export
- Evaluation with absolute correlation and scale-invariant SDR (SI-SDR), handling permutation and scale ambiguity
- PCA-vs-ICA comparison across SNR levels

## Repository layout

```
unmix-the-signal/
├── README.md
├── requirements.txt
├── unmix/                  # the library
│   ├── __init__.py
│   ├── ica.py              # centering, PCA whitening, FastICA
│   ├── mixing.py           # random mixing matrices, mixing, noise at a target SNR
│   ├── metrics.py          # source matching, correlation, SI-SDR
│   ├── audio_io.py         # load/resample/save WAV
│   └── synthetic.py        # synthetic test sources (sanity check)
├── scripts/
│   └── run_experiment.py   # full pipeline: mix -> separate -> evaluate -> plots + audio
├── tests/
│   └── test_ica.py
├── data/
│   └── sources/            # put your clean source .wav files here (see its README)
├── docs/
│   └── writeup.md          # results and the four "be ready to explain" topics
└── results/                # generated: metrics, plots, mixtures, separated audio
```

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Run the unit tests:

```bash
python -m pytest -v
```

Run the full experiment on synthetic sources (no audio files needed):

```bash
python -m scripts.run_experiment
```

Run it on your own clean sources:

```bash
python -m scripts.run_experiment --sources-dir data/sources --n-sources 3 --n-mixtures 3
```

Outputs are written to `results/`:

- `results/summary.md`: table of correlation and SI-SDR for ICA and PCA at each SNR
- `results/metrics.json`: full per-source metrics
- `results/snr_comparison.png`: ICA vs PCA bar chart
- `results/waveforms.png`: true sources vs ICA estimates (clean condition)
- `results/audio/reference/`: the clean sources
- `results/audio/<condition>/mixture_*.wav`: the mixed recordings
- `results/audio/<condition>/ica_separated_*.wav`: the separated audio (playable)
- `results/audio/<condition>/pca_separated_*.wav`: PCA output, for comparison

## Choosing sources

Pick N clean sources, one per category, for example:

- Speech: a public-domain or CC-licensed recording of a voice
- Music: a short instrumental clip
- Ambient: rain, wind, or room tone

Requirements: mono or stereo WAV, each source at least as long as `--duration`. Put them in `data/sources/` and keep them to CC0, CC-BY, or your own recordings. Don't commit copyrighted audio. `.gitignore` excludes `data/sources/*.wav` by default.

Sources at different sample rates are resampled to `--sr` automatically.

## Method

1. **Center** each mixed signal (subtract its mean).
2. **Whiten** with PCA: eigendecompose the covariance `C = E D Eᵀ` and use `W = D^(-1/2) Eᵀ`. The whitened data has identity covariance.
3. **FastICA** on the whitened data: find an orthogonal rotation `W` that maximizes non-Gaussianity of each output, using the fixed-point update `w ← E[z g(wᵀz)] − E[g′(wᵀz)] w`, followed by symmetric decorrelation `W ← (WWᵀ)^(-1/2) W`.
4. The total unmixing matrix is `W · W_white`, and the sources are `Ŝ = W · W_white · X`.

## Evaluation

ICA recovers sources only up to **permutation** (which output is which source) and **scale/sign**. The evaluation handles this:

- **Matching:** the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) pairs estimates with true sources by absolute correlation.
- **Correlation:** absolute Pearson correlation of each matched pair (sign-invariant).
- **SI-SDR:** scale-invariant signal-to-distortion ratio in dB. The estimate is projected onto the reference to get the target component, and the rest counts as distortion. Higher is better.

## Results

See `results/summary.md` after running the experiment, and `docs/writeup.md` for discussion.

## Limitations

- Assumes instantaneous (linear, static) mixing. Real room recordings have reverberation and delays (convolutive mixing), which this does not model.
- Assumes at most one Gaussian source, and that the number of sources is known.
- The tanh nonlinearity suits super-Gaussian sources (speech, music). Use `nonlinearity="cube"` for sub-Gaussian sources.

## References

- Hyvärinen, A. & Oja, E. (2000). *Independent component analysis: algorithms and applications.* Neural Networks.
- Hyvärinen, A. (1999). *Fast and robust fixed-point algorithms for independent component analysis.* IEEE Transactions on Neural Networks.
