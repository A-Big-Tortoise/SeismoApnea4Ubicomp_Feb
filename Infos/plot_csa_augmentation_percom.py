"""Plot the CSA signal augmentations from locally saved data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_an_comparisons import add_legacy_hyp_noise, filtered_signal


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_An" / "percom" / "CSA" / "comparison_data.npz"
OUTPUT_PATH = BASE_DIR / "fig_preprocess" / "CSA_aug_percom.png"

plt.rcParams.update({
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 32,
    "axes.titlesize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 28,
})


def main():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        filtered_x = filtered_signal(-data["X"], 0.75)

    time = np.arange(filtered_x.size) / 100
    figure, axes = plt.subplots(2, 2, figsize=(22, 7.3), sharex=True, sharey=True)

    axes[0, 0].plot(time, filtered_x, linewidth=5)
    axes[0, 1].plot(time, -filtered_x, linewidth=5)
    axes[1, 0].plot(time, filtered_x[::-1], linewidth=5)
    axes[1, 1].plot(
        time,
        add_legacy_hyp_noise(filtered_x, band=(0.3, 0.75), noise_std=0.1),
        linewidth=5,
    )

    titles = [
        "(a) Filtered X-axis signal",
        "(b) Inverted X-axis signal",
        "(c) Flipped X-axis signal",
        "(d) Frequency-perturbed X-axis signal",
    ]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    figure.subplots_adjust(left=0.045, right=0.99, bottom=0.08, top=0.92, hspace=0.36, wspace=0.035)
    figure.savefig(OUTPUT_PATH, dpi=300)
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
