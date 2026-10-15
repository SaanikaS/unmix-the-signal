"""WAV loading (mono, common sample rate and length) and saving (playable)."""
import pathlib

import numpy as np
import soundfile as sf
from scipy.signal import resample


def load_mono_wavs(directory, sr, duration, n_sources):
    """Load the first n_sources .wav files (sorted by name) as mono at rate sr.

    Stereo is averaged to mono, resampled if needed, and all signals are
    trimmed to a common length of min(duration, shortest file).
    Returns (signals of shape (n_sources, T), file names).
    """
    files = sorted(pathlib.Path(directory).glob("*.wav"))[:n_sources]
    if len(files) < n_sources:
        raise ValueError(f"Need {n_sources} .wav files in {directory}, found {len(files)}.")

    chans = []
    for f in files:
        x, fs = sf.read(f, dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if fs != sr:
            x = resample(x, int(round(len(x) * sr / fs)))
        chans.append(x)

    L = min([int(sr * duration)] + [len(c) for c in chans])
    return np.stack([c[:L] for c in chans]), [f.name for f in files]


def save_wav(path, x, sr):
    """Save as 16-bit PCM, peak-normalized to 0.9 so it plays at a safe level."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.9
    sf.write(path, x, sr, subtype="PCM_16")
