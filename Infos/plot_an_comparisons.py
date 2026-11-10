"""Redraw apnea comparison figures from locally saved signal archives."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_An" / "percom"
FIG_DIR = BASE_DIR / "figs_An"
FS = 100
SIGNAL_LINEWIDTH = 4
PRIMARY_LINEWIDTH = 5
LEGEND_HANDLE_LENGTH = 1.5
Y_TICK_LABEL_SIZE = 28

plt.rcParams.update({
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 32,
    "axes.titlesize": 36,
    "axes.labelsize": 34,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 32,
})


def normalize(signal):
    signal = np.asarray(signal, dtype=float).squeeze()
    standard_deviation = signal.std()
    if standard_deviation == 0:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / standard_deviation


def low_pass(signal, cutoff):
    b, a = butter(3, cutoff / (FS / 2), btype="low")
    signal = np.asarray(signal, dtype=float).squeeze()
    padded_signal = np.pad(signal, (signal.size, signal.size), mode="reflect")
    filtered_signal = filtfilt(b, a, padded_signal)
    return filtered_signal[signal.size:-signal.size]


def filtered_signal(signal, cutoff):
    return normalize(low_pass(signal, cutoff))


def add_legacy_hyp_noise(
    signal,
    fs=10,
    band=(0, 1),
    n_centers=5,
    bandwidth=0.05,
    noise_std=5,
):
    """Match the frequency-domain perturbation used by the legacy Hyp plot."""
    rng = np.random.default_rng()
    signal = np.asarray(signal, dtype=float).copy()
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, d=1 / fs)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    low, high = band
    centers = rng.uniform(low + bandwidth / 2, high - bandwidth / 2, size=n_centers)
    mask = np.zeros(signal.size, dtype=bool)
    for center in centers:
        mask |= np.abs(np.abs(frequencies) - center) <= bandwidth / 2
    mask &= (np.abs(frequencies) >= low) & (np.abs(frequencies) <= high)

    sigma = bandwidth / 2.355
    weights = np.zeros(signal.size, dtype=float)
    for center in centers:
        weights += np.exp(-0.5 * ((np.abs(frequencies) - center) / sigma) ** 2)
    weights *= mask

    noisy_magnitude = magnitude + rng.normal(0, noise_std, size=signal.size) * weights
    noisy_magnitude = np.clip(noisy_magnitude, 0, None)
    noisy_magnitude = 0.5 * (noisy_magnitude + noisy_magnitude[::-1])
    return np.fft.ifft(noisy_magnitude * np.exp(1j * phase)).real


def add_event_span(axes, events):
    events = np.asarray(events).squeeze()
    event_indices = np.flatnonzero(events > 0)
    if event_indices.size == 0:
        return

    event_time = np.arange(events.size) / 10
    start = event_time[event_indices[0]]
    end = event_time[event_indices[-1]]
    for axis in axes:
        axis.axvspan(start, end, alpha=0.1)


def add_common_labels(axes):
    titles = [
        "(a) Filtered seismic signal (X & Y axes)",
        "(b) Respiratory effort (THO & ABD)",
        "(c) Airflow (Nasal Pressure)",
    ]
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.legend(loc="upper right", handlelength=LEGEND_HANDLE_LENGTH)
        axis.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axes[-1].set_xlabel("Time [sec]")


def load_signals(category):
    path = DATA_DIR / category / "comparison_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved signals: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def plot_normal(data):
    time = np.arange(data["Y"].size) / FS
    figure, axes = plt.subplots(3, 1, figsize=(12, 4.3 * 3), sharex=True)
    axes[0].plot(time, filtered_signal(data["X"], 0.55), label="X (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[0].plot(time, filtered_signal(data["Y"], 0.55), label="Y (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_THO"]), label="THO", linewidth=SIGNAL_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_ABD"]), label="ABD", linewidth=SIGNAL_LINEWIDTH)
    axes[2].plot(time, normalize(data["TFlow"]), label="Airflow", linewidth=PRIMARY_LINEWIDTH)
    add_event_span(axes, data["Events"])
    add_common_labels(axes)
    return figure


def plot_osa(data):
    time = np.arange(data["Y"].size) / FS
    figure, axes = plt.subplots(3, 1, figsize=(12, 4.3 * 3), sharex=True)
    axes[0].plot(time, filtered_signal(data["X"], 0.75), label="X (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[0].plot(time, filtered_signal(-data["Y"], 0.75), label="Y (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_THO"]), label="THO", linewidth=SIGNAL_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_ABD"]), label="ABD", linewidth=SIGNAL_LINEWIDTH)
    axes[2].plot(time, normalize(data["TFlow"]), label="Airflow", linewidth=PRIMARY_LINEWIDTH)
    add_event_span(axes, data["Events"])
    add_common_labels(axes)
    for axis in axes:
        axis.legend(loc="upper left", handlelength=LEGEND_HANDLE_LENGTH)
    return figure


def plot_hyp(data):
    time = np.arange(data["Y"].size) / FS
    figure, axes = plt.subplots(3, 1, figsize=(12, 4.3 * 3), sharex=True)
    legacy_x = add_legacy_hyp_noise(filtered_signal(data["Y"], 0.5))
    axes[0].plot(time, legacy_x, label="X (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[0].plot(time, filtered_signal(data["Y"], 0.65), label="Y (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[1].plot(time, normalize(-data["Effort_THO"]), label="THO", linewidth=SIGNAL_LINEWIDTH)
    axes[1].plot(time, normalize(-data["Effort_ABD"]), label="ABD", linewidth=SIGNAL_LINEWIDTH)
    b, a = butter(4, 0.1 / (FS / 2), btype="highpass")
    axes[2].plot(time, filtfilt(b, a, normalize(data["TFlow"])), label="Airflow", linewidth=PRIMARY_LINEWIDTH)
    add_common_labels(axes)
    return figure


def plot_csa(data):
    time = np.arange(data["Y"].size) / FS
    figure, axes = plt.subplots(3, 1, figsize=(12, 4.3 * 3), sharex=True)
    axes[0].plot(time, filtered_signal(-data["X"], 0.75), label="X (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[0].plot(time, filtered_signal(data["Y"], 0.75), label="Y (filt)", linewidth=PRIMARY_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_THO"]), label="THO", linewidth=SIGNAL_LINEWIDTH)
    axes[1].plot(time, normalize(data["Effort_ABD"]), label="ABD", linewidth=SIGNAL_LINEWIDTH)
    axes[2].plot(time, normalize(data["TFlow"]), label="Airflow", linewidth=PRIMARY_LINEWIDTH)
    add_common_labels(axes)
    return figure


def main():
    FIG_DIR.mkdir(exist_ok=True)
    plotters = {
        "Normal": plot_normal,
        "OSA": plot_osa,
        "Hyp": plot_hyp,
        "CSA": plot_csa,
    }
    for category, plotter in plotters.items():
        figure = plotter(load_signals(category))
        output_path = FIG_DIR / f"{category}_comparison_percom.png"
        figure.subplots_adjust(
            left=0.065,
            right=0.985,
            bottom=0.09,
            top=0.94,
            hspace=0.28,
        )
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
