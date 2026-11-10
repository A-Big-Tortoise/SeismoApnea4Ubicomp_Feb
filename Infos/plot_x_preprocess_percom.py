"""Plot locally saved X-axis signals for the Normal and CSA examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_an_comparisons import filtered_signal


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_An" / "percom"
FIG_DIR = BASE_DIR / "fig_preprocess"


def thousands_formatter(value, _position):
    if value == 0:
        return "0"
    return f"{value / 1000:g}k"

plt.rcParams.update({
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 32,
    "axes.titlesize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 28,
    "legend.fontsize": 32,
})


def plot_example(category, output_name, invert_x=False, raw_offset=0):
    data_path = DATA_DIR / category / "comparison_data.npz"
    with np.load(data_path, allow_pickle=False) as data:
        raw_x = np.asarray(data["X"], dtype=float)

    if invert_x:
        raw_x = -raw_x

    time = np.arange(raw_x.size) / 100
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.3), sharex=True)
    axes[0].plot(time, raw_x + raw_offset, label="X (raw)", linewidth=5)
    axes[1].plot(time, filtered_signal(raw_x, 0.55 if category == "Normal" else 0.75), label="X (filt)", linewidth=5)

    axes[0].set_title("(a) Raw X-axis seismic signal")
    axes[1].set_title("(b) Filtered X-axis signal")
    for axis in axes:
        axis.legend(loc="upper left", handlelength=1.2)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.tick_params(axis="y", labelsize=28)
    axes[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.93, hspace=0.35)
    figure.savefig(FIG_DIR / output_name, dpi=300)
    plt.close(figure)


def main():
    plot_example("CSA", "X_preprocess_large_CSA_percom.png", invert_x=True, raw_offset=4000)
    plot_example("Normal", "X_preprocess_large_percom.png", raw_offset=40000)


if __name__ == "__main__":
    main()
