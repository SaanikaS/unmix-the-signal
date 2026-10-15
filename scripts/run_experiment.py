"""Mix sources, separate them with ICA and PCA at several SNRs, then evaluate and save everything.

Usage (from the repo root):
    python -m scripts.run_experiment                           # synthetic sources
    python -m scripts.run_experiment --sources-dir data/sources
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from unmix.audio_io import load_mono_wavs, save_wav  # noqa: E402
from unmix.ica import ica_separate, pca_separate  # noqa: E402
from unmix.metrics import evaluate  # noqa: E402
from unmix.mixing import add_noise, make_mixing_matrix, mix  # noqa: E402
from unmix.synthetic import make_synthetic_sources  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources-dir", type=pathlib.Path, default=None,
                   help="Folder of clean .wav sources. Omit to use synthetic sources.")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    p.add_argument("--sr", type=int, default=16000, help="working sample rate (Hz)")
    p.add_argument("--duration", type=float, default=10.0, help="seconds per source")
    p.add_argument("--n-sources", type=int, default=3)
    p.add_argument("--n-mixtures", type=int, default=3, help="must be >= n-sources")
    p.add_argument("--snr", type=float, nargs="+",
                   default=[float("inf"), 30, 20, 10, 0],
                   help="SNR levels in dB; 'inf' means noiseless")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def label(snr):
    return "clean" if np.isinf(snr) else f"snr_{snr:g}db"


def write_table(summary, path):
    lines = [
        "| Condition | ICA correlation | ICA SI-SDR (dB) | PCA correlation | PCA SI-SDR (dB) |",
        "|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['condition']} | {r['ica']['mean_correlation']:.3f} | "
            f"{r['ica']['mean_si_sdr']:.2f} | {r['pca']['mean_correlation']:.3f} | "
            f"{r['pca']['mean_si_sdr']:.2f} |"
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)


def plot_comparison(summary, path):
    labels = [r["condition"] for r in summary]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, [r["ica"]["mean_si_sdr"] for r in summary], width, label="ICA")
    ax.bar(x + width / 2, [r["pca"]["mean_si_sdr"] for r in summary], width, label="PCA")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean SI-SDR (dB)")
    ax.set_title("Separation quality vs noise level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_waveforms(S, S_hat_matched, path, n=2000):
    k = len(S)
    fig, axes = plt.subplots(k, 2, figsize=(10, 2.2 * k), squeeze=False)
    for i in range(k):
        axes[i, 0].plot(S[i, :n])
        axes[i, 0].set_title(f"True source {i + 1}")
        axes[i, 1].plot(S_hat_matched[i, :n])
        axes[i, 1].set_title(f"ICA estimate {i + 1} (sign/scale arbitrary)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.n_mixtures < args.n_sources:
        raise SystemExit("--n-mixtures must be >= --n-sources")

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    audio = args.out / "audio"

    # 1. Sources
    if args.sources_dir:
        S, names = load_mono_wavs(args.sources_dir, args.sr, args.duration, args.n_sources)
    else:
        S = make_synthetic_sources(args.n_sources, args.sr, args.duration, rng)
        names = [f"synthetic_{i + 1}" for i in range(args.n_sources)]
    S = S / np.abs(S).max(axis=1, keepdims=True)
    for i, name in enumerate(names):
        save_wav(audio / "reference" / f"source_{i + 1}_{pathlib.Path(name).stem}.wav",
                 S[i], args.sr)

    # 2. Mix (noiseless)
    A = make_mixing_matrix(args.n_sources, args.n_mixtures, rng)
    X_clean = mix(S, A)

    # 3. Separate and evaluate at each SNR
    summary = []
    first_matched = None
    for snr in args.snr:
        X = X_clean if np.isinf(snr) else add_noise(X_clean, snr, rng)
        X = X / np.abs(X).max()

        ica_S, _, n_iter = ica_separate(X, n_sources=args.n_sources, random_state=args.seed)
        pca_S = pca_separate(X, n_sources=args.n_sources)

        ica_eval = evaluate(S, ica_S)
        pca_eval = evaluate(S, pca_S)

        cond_dir = audio / label(snr)
        for j in range(X.shape[0]):
            save_wav(cond_dir / f"mixture_{j + 1}.wav", X[j], args.sr)
        for i, idx in enumerate(ica_eval["order"]):
            save_wav(cond_dir / f"ica_separated_{i + 1}.wav", ica_S[idx], args.sr)
        for i, idx in enumerate(pca_eval["order"]):
            save_wav(cond_dir / f"pca_separated_{i + 1}.wav", pca_S[idx], args.sr)

        if first_matched is None:
            first_matched = ica_S[ica_eval["order"]]

        summary.append({
            "condition": label(snr),
            "snr_db": None if np.isinf(snr) else float(snr),
            "ica_iterations": int(n_iter),
            "ica": ica_eval,
            "pca": pca_eval,
        })
        print(f"{label(snr):>12}  ICA SI-SDR {ica_eval['mean_si_sdr']:6.2f} dB  "
              f"PCA SI-SDR {pca_eval['mean_si_sdr']:6.2f} dB  (FastICA iters: {n_iter})")

    # 4. Save results
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_table(summary, args.out / "summary.md")
    plot_comparison(summary, args.out / "snr_comparison.png")
    plot_waveforms(S, first_matched, args.out / "waveforms.png")
    print(f"\nResults written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
