#!/usr/bin/env python3
"""
main.py
-------
Figure generator and measurement runner for BCG-based apnea detection.

Primary output: saved plots in plots/ for visual analysis.
Secondary output: metrics (bACC, F1, confusion matrix) on test set.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tool

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET    = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR  = os.path.join(DATASET, 'train', 'data')
TEST_DIR   = os.path.join(DATASET, 'test', 'data')
PLOT_DIR   = os.path.join(BASE_DIR, 'plots')
METADATA   = os.path.join(DATASET, 'metadata.csv')
GT_PATH    = os.path.join(DATASET, 'test', 'ground_truth.json')

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_train_example(idx):
    """Load train example by index."""
    path = os.path.join(TRAIN_DIR, f'example_{idx:03d}.npz')
    return tool.load_example(path)


def load_test_example(idx):
    """Load test example by index."""
    path = os.path.join(TEST_DIR, f'example_{idx:03d}.npz')
    return tool.load_example(path)


def load_ground_truth():
    """Load test ground truth labels."""
    with open(GT_PATH) as f:
        return json.load(f)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_train_sample(idx, save=True):
    """Plot a training sample with all available channels and annotations.

    Layout:
      Row 1: BCG X-axis + envelope
      Row 2: BCG Y-axis + envelope
      Row 3: Thoracic effort (THO) + envelope
      Row 4: Abdominal effort (ABD) + envelope
      Row 5: Event timeline (apnea events)
    """
    ex = load_train_example(idx)
    t = tool.time_axis(len(ex['X']))
    label = int(ex['apnea_label'])
    label_str = 'APNEA' if label == 1 else 'NORMAL'

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 0.5], hspace=0.35)

    channels = [
        ('X', 'BCG X-axis (anteroposterior)'),
        ('Y', 'BCG Y-axis (lateral)'),
        ('THO', 'Thoracic effort'),
        ('ABD', 'Abdominal effort'),
    ]

    for row, (ch_key, ch_name) in enumerate(channels):
        ax = fig.add_subplot(gs[row])
        sig = ex[ch_key]
        env = tool.amplitude_envelope(sig)
        ax.plot(t, sig, alpha=0.7, linewidth=0.8, label=ch_name)
        ax.plot(t, env, 'r-', alpha=0.6, linewidth=1.2, label='Envelope')
        ax.plot(t, -env, 'r-', alpha=0.6, linewidth=1.2)
        ax.set_ylabel(ch_name, fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Event timeline
    ax_ev = fig.add_subplot(gs[4])
    event_ts = ex['event_ts']
    t_ev = np.arange(len(event_ts)) / 10.0  # 10 Hz
    ax_ev.fill_between(t_ev, 0, event_ts, alpha=0.5, color='red', label='Apnea event')
    ax_ev.set_ylabel('Event', fontsize=9)
    ax_ev.set_xlabel('Time (s)', fontsize=10)
    ax_ev.set_xlim(t[0], t[-1])
    ax_ev.set_ylim(-0.5, 5)
    ax_ev.legend(loc='upper right', fontsize=8)
    ax_ev.grid(True, alpha=0.3)

    fig.suptitle(f'Train #{idx:03d} — {label_str} (patient {int(ex["patient_id"])})',
                 fontsize=14, fontweight='bold')

    if save:
        fname = f'train_sample{idx:03d}_{label_str.lower()}.png'
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        return fname
    return fig


def plot_test_sample(idx, save=True):
    """Plot a test sample (X and Y channels only).

    Layout:
      Row 1: BCG X-axis + envelope
      Row 2: BCG Y-axis + envelope
      Row 3: X-axis sliding variance
      Row 4: Y-axis sliding variance
    """
    ex = load_test_example(idx)
    t = tool.time_axis(len(ex['X']))

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 0.6, 0.6], hspace=0.35)

    for row, (ch_key, ch_name) in enumerate([('X', 'BCG X-axis'), ('Y', 'BCG Y-axis')]):
        ax = fig.add_subplot(gs[row])
        sig = ex[ch_key]
        env = tool.amplitude_envelope(sig)
        ax.plot(t, sig, alpha=0.7, linewidth=0.8, label=ch_name)
        ax.plot(t, env, 'r-', alpha=0.6, linewidth=1.2, label='Envelope')
        ax.plot(t, -env, 'r-', alpha=0.6, linewidth=1.2)
        ax.set_ylabel(ch_name, fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for row_offset, (ch_key, ch_name) in enumerate([('X', 'X variance'), ('Y', 'Y variance')]):
        ax = fig.add_subplot(gs[2 + row_offset])
        sig = ex[ch_key]
        sv = tool.sliding_variance(sig, window_s=5.0)
        ax.plot(t, sv, color='orange', linewidth=1.0, label=ch_name)
        ax.set_ylabel(ch_name, fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax.set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(f'Test #{idx:03d}', fontsize=14, fontweight='bold')

    if save:
        fname = f'test_sample{idx:03d}.png'
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        return fname
    return fig


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute bACC, F1, and confusion matrix.
    Positive class = 1 (apnea)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bacc = (sensitivity + specificity) / 2.0

    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    return {
        'bacc': bacc, 'f1': f1,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'sensitivity': sensitivity, 'specificity': specificity
    }


def print_metrics(metrics):
    """Print metrics in required format."""
    print(f"test_bacc:  {metrics['bacc']:.2f}")
    print(f"test_f1:    {metrics['f1']:.2f}")
    print(f"test_cm:    TP={metrics['TP']} FP={metrics['FP']} TN={metrics['TN']} FN={metrics['FN']}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BCG Apnea Detection — Figure Generator')
    parser.add_argument('--plot-train', action='store_true', help='Plot all training samples')
    parser.add_argument('--plot-test', action='store_true', help='Plot all test samples')
    parser.add_argument('--sample', type=int, default=None, help='Plot a specific sample index')
    args = parser.parse_args()

    if args.plot_train:
        print('Plotting training samples...')
        for i in range(50):
            fname = plot_train_sample(i)
            print(f'  Saved: {fname}')

    if args.plot_test:
        print('Plotting test samples...')
        for i in range(50, 550):
            fname = plot_test_sample(i)
            print(f'  Saved: {fname}')

    if args.sample is not None:
        idx = args.sample
        if idx < 50:
            plot_train_sample(idx)
            print(f'Plotted train sample {idx}')
        else:
            plot_test_sample(idx)
            print(f'Plotted test sample {idx}')

    if not (args.plot_train or args.plot_test or args.sample is not None):
        print('Usage: python main.py --plot-train | --plot-test | --sample IDX')
