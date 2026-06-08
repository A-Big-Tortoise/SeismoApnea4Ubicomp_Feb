#!/usr/bin/env python3
"""
generate_llm_data.py
--------------------
Generates few-shot training figures and test figures for LLM-based apnea
detection from seismic/BCG signals.

Data column layout (fold_data_p109_MTL_60s, 31326 cols per row):
  [0:6000]        X-axis BCG   (100 Hz, 60 s = 6000 samples)
  [6000:12000]    Y-axis BCG
  [12000:18000]   Z-axis BCG
  [18000:24000]   Thoracic belt (THO)
  [24000:30000]   Abdominal belt (ABD)
  [30000:30060]   Body-position channel (all 0 in this dataset)
  [30060:30120]   Sleep-stage time series  (1 Hz, 60 values)
  [30120:30720]   Apnea-event time series  (10 Hz, 600 values)
  [30720:31320]   Reserved/zeros
  [31320]         MAC address
  [31321]         Unix timestamp
  [31322]         Patient ID
  [31323]         Room
  [-2] = [31324]  Binary stage label   (0=Sleep, 1=Wake)
  [-1] = [31325]  Binary apnea label   (0=Normal, 1=Apnea/Hypopnea)

Stage codes (1 Hz time series):
  0=N1, 1=N2, 2=N3, 3=REM, 4=Wake  (−1=Unscored, not present here)

Event codes (10 Hz time series):
  0=Normal, 1=Obstructive Apnea, 2=Central Apnea,
  3=Hypopnea, 4=Mixed Apnea

Output:
  LLM_Apnea/
  ├── few_shot/
  │   ├── figures/   PNG: X, Y, THO, ABD + event + stage panels (with labels)
  │   └── data/      .npz: processed signals, event_ts, stage_ts, metadata
  ├── test/
  │   ├── figures/   PNG: X, Y only (no labels shown)
  │   └── data/      .npz: processed X, Y only (labels saved separately)
  ├── test_ground_truth.json
  ├── metadata.csv
  └── aasm_reference.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'Data', 'fold_data_p109_MTL_60s')
OUT_DIR   = os.path.join(ROOT, 'LLM_Apnea')

# ── Signal processing constants ─────────────────────────────────────────────
FS_RAW    = 100     # Hz (raw sampling rate)
FS_DOWN   = 10      # Hz (target after downsampling)
SEQ_S     = 60      # window length in seconds
N_RAW     = FS_RAW * SEQ_S      # 6000 samples per signal
N_DOWN    = FS_DOWN * SEQ_S     # 600 samples after downsample
CROP_S    = 5                   # crop start after downsample (remove filter edge)
CROP_E    = 595                 # crop end  → 590 usable samples (59 s)
LP_HZ     = 1.0                 # low-pass cutoff for BCG respiratory band

# ── Column layout constants ─────────────────────────────────────────────────
COL_X_END   = N_RAW            # 6000
COL_Y_END   = 2 * N_RAW        # 12000
COL_Z_END   = 3 * N_RAW        # 18000
COL_THO_END = 4 * N_RAW        # 24000
COL_ABD_END = 5 * N_RAW        # 30000
COL_STAGE_S = 30060
COL_STAGE_E = 30120            # 60 values @ 1 Hz
COL_EVENT_S = 30120
COL_EVENT_E = 30720            # 600 values @ 10 Hz
COL_ID      = -4               # patient ID
COL_STAGE_L = -2               # binary stage label
COL_APNEA_L = -1               # binary apnea label

# ── Sampling counts ─────────────────────────────────────────────────────────
N_FEW_APNEA  = 25   # few-shot apnea examples (from fold 1)
N_FEW_NORMAL = 25   # few-shot normal examples (from fold 1)
N_TEST_APNEA = 25   # test apnea examples (from folds 2-4)
N_TEST_NORMAL= 25   # test normal examples (from folds 2-4)

SEED = 42

# ── Label / colour maps ─────────────────────────────────────────────────────
EVENT_MAP = {0: 'Normal', 1: 'Hypopnea',
             2: 'OSA (Obstructive)', 3: 'CSA (Central)', 4: 'Mixed Apnea'}
EVENT_COLOR = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728',
               3: '#9467bd', 4: '#8c564b'}
STAGE_MAP   = {-1: 'Unscored', 0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'Wake'}
STAGE_COLOR = {-1: '#7f7f7f', 0: '#17becf', 1: '#1f77b4',
               2: '#aec7e8',  3: '#bcbd22', 4: '#ff7f0e'}

AASM_REFERENCE = {
    "source": "AASM Manual for the Scoring of Sleep and Associated Events v2.6",
    "apnea_criteria": {
        "Obstructive Apnea (OA)": (
            "Drop in peak signal excursion by ≥90% from baseline for ≥10 s, "
            "with continued or increased inspiratory effort throughout."
        ),
        "Central Apnea (CA)": (
            "Drop in peak signal excursion by ≥90% from baseline for ≥10 s, "
            "with absent inspiratory effort throughout."
        ),
        "Mixed Apnea (MA)": (
            "Drop in peak signal excursion by ≥90% from baseline for ≥10 s, "
            "with absent effort in the initial portion and resumed effort in "
            "the latter portion."
        ),
        "Hypopnea (H)": (
            "Drop in peak signal excursion by ≥30% from baseline for ≥10 s, "
            "associated with ≥3% oximetric desaturation or an arousal."
        )
    },
    "severity_AHI": {
        "Normal": "AHI < 5 events/hour",
        "Mild OSA": "5 ≤ AHI < 15 events/hour",
        "Moderate OSA": "15 ≤ AHI < 30 events/hour",
        "Severe OSA": "AHI ≥ 30 events/hour"
    },
    "sleep_stages": {
        "Wake (W)":  "Alpha activity >50% of epoch; blinks; voluntary eye movements.",
        "N1":        "Low-amplitude mixed-frequency EEG; vertex sharp waves; slow rolling eye movements.",
        "N2":        "Sleep spindles (11-16 Hz, ≥0.5 s) and/or K-complexes.",
        "N3":        "Slow-wave activity (0.5-2 Hz, peak-to-peak ≥75 µV) ≥20% of the epoch.",
        "REM (R)":   "Low-amplitude mixed-frequency EEG; rapid eye movements; chin EMG atonia."
    },
    "stage_codes_this_dataset": {
        "-1": "Unscored", "0": "N3", "1": "N2", "2": "N1", "3": "REM", "4": "Wake"
    },
    "event_codes_this_dataset": {
        "0": "Normal", "1": "Hypopnea",
        "2": "OSA (Obstructive Apnea)", "3": "CSA (Central Apnea)", "4": "Mixed Apnea"
    },
    "sensor": (
        "Bed-mounted seismic sensor (BCG – ballistocardiography). "
        "X and Y axes capture respiratory body movements at 100 Hz. "
        "After 1 Hz low-pass filtering and downsampling to 10 Hz, "
        "the respiratory oscillation (~0.1-0.5 Hz) becomes clearly visible. "
        "During apnea, both amplitude and regularity of oscillations diminish."
    ),
    "few_shot_instructions": (
        "You are given labelled 60-second BCG windows. "
        "Each window shows 4 processed signals (X, Y, THO, ABD), "
        "a per-second sleep-stage timeline, and a 10 Hz apnea-event timeline. "
        "Learn to distinguish apnea windows (irregular/absent oscillations, "
        "attenuated THO/ABD effort) from normal windows "
        "(regular oscillations, clear respiratory effort in THO/ABD). "
        "In the test set you receive only X and Y – predict the apnea label."
    )
}


# ── Signal processing helpers ───────────────────────────────────────────────

def lowpass_filter(sig_2d, cutoff_hz, fs, order=4):
    """Butterworth low-pass filter with reflect padding."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    N = sig_2d.shape[1]
    padded = np.pad(sig_2d, ((0, 0), (N, N)), mode='reflect')
    filtered = filtfilt(b, a, padded, axis=1)
    return filtered[:, N:-N]


def process_bcg(raw_2d):
    """LP filter → downsample 100→10 Hz → crop → z-score normalise."""
    filtered   = lowpass_filter(raw_2d, LP_HZ, FS_RAW)
    downsampled = resample_poly(filtered, 1, FS_RAW // FS_DOWN, axis=1)
    cropped    = downsampled[:, CROP_S:CROP_E]               # [N, 590]
    mean       = cropped.mean(axis=1, keepdims=True)
    std        = cropped.std(axis=1, keepdims=True) + 1e-8
    return (cropped - mean) / std


def extract_all(data):
    """Extract every channel from raw fold_data rows."""
    X   = process_bcg(data[:, :COL_X_END])
    Y   = process_bcg(data[:, COL_X_END:COL_Y_END])
    THO = process_bcg(data[:, COL_THO_END - N_RAW: COL_THO_END])
    ABD = process_bcg(data[:, COL_ABD_END - N_RAW: COL_ABD_END])
    stage_ts = data[:, COL_STAGE_S:COL_STAGE_E].astype(int)   # [N, 60]  @ 1 Hz
    event_ts = data[:, COL_EVENT_S:COL_EVENT_E].astype(int)   # [N, 600] @ 10 Hz
    stage_label = data[:, COL_STAGE_L].astype(int)
    apnea_label = data[:, COL_APNEA_L].astype(int)
    patient_id  = data[:, COL_ID].astype(int)
    return X, Y, THO, ABD, stage_ts, event_ts, stage_label, apnea_label, patient_id


# ── Sampling helpers ─────────────────────────────────────────────────────────

def sample_balanced(data, n_apnea, n_normal, rng, prefer_sleep=True):
    """Return indices of n_apnea + n_normal rows from data."""
    apnea_mask  = data[:, COL_APNEA_L] == 1
    normal_mask = data[:, COL_APNEA_L] == 0
    if prefer_sleep:
        apnea_mask  = apnea_mask  & (data[:, COL_STAGE_L] == 0)
        normal_mask = normal_mask & (data[:, COL_STAGE_L] == 0)
    apnea_idx  = np.where(apnea_mask)[0]
    normal_idx = np.where(normal_mask)[0]

    # For apnea: prefer windows where the event is well-centred (not at edge)
    def event_center(idx):
        ev = data[idx, COL_EVENT_S:COL_EVENT_E]
        nz = np.where(ev != 0)[0]
        return abs(nz.mean() - 300) if len(nz) else 9999

    apnea_scores = np.array([event_center(i) for i in apnea_idx])
    sorted_apnea = apnea_idx[np.argsort(apnea_scores)]
    # Draw from the best-scored half, then shuffle to get variety
    pool_a = sorted_apnea[:max(n_apnea * 3, 150)]
    chosen_a = rng.choice(pool_a, size=n_apnea, replace=False)
    chosen_n = rng.choice(normal_idx, size=n_normal, replace=False)
    return chosen_a, chosen_n


# ── Figure helpers ───────────────────────────────────────────────────────────

def _time_axes(n_samples, fs):
    return np.arange(n_samples) / fs


def plot_few_shot(X, Y, THO, ABD, event_ts, stage_ts,
                  apnea_label, stage_label, patient_id,
                  save_path, example_id):
    """6-panel figure for a few-shot training example."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=False)
    fig.subplots_adjust(hspace=0.55, left=0.09, right=0.97, top=0.93, bottom=0.05)

    # Determine dominant stage for title
    unique_s, counts_s = np.unique(stage_ts, return_counts=True)
    dom_stage = STAGE_MAP.get(int(unique_s[np.argmax(counts_s)]), '?')

    # Determine dominant event type
    event_nz  = event_ts[event_ts != 0]
    if len(event_nz):
        unique_e, counts_e = np.unique(event_nz, return_counts=True)
        dom_event = EVENT_MAP.get(int(unique_e[np.argmax(counts_e)]), '?')
    else:
        dom_event = 'Normal'

    label_str  = 'APNEA' if apnea_label == 1 else 'NORMAL'
    stage_str  = 'Sleep' if stage_label == 0 else 'Wake'
    title = (f'Few-Shot Example {example_id:03d}  |  Patient {patient_id}  |  '
             f'Stage: {stage_str} ({dom_stage})  |  Label: {label_str}  |  '
             f'Event type: {dom_event}')
    fig.suptitle(title, fontsize=10, fontweight='bold',
                 color='#d62728' if apnea_label == 1 else '#2ca02c')

    t_bcg   = _time_axes(len(X), FS_DOWN)     # 0-59 s, 590 points
    t_event = _time_axes(600, FS_DOWN)         # 0-60 s, 600 points
    t_stage = _time_axes(60, 1)                # 0-60 s, 60 points

    # ── Panel 1: X ──────────────────────────────────────
    axes[0].plot(t_bcg, X, color='#1f77b4', lw=0.8)
    axes[0].set_ylabel('X (norm.)', fontsize=8)
    axes[0].set_title('BCG X-axis (seismic, LP 1 Hz, 10 Hz, z-score)', fontsize=8)
    _shade_events(axes[0], t_event, event_ts, alpha=0.12)

    # ── Panel 2: Y ──────────────────────────────────────
    axes[1].plot(t_bcg, Y, color='#ff7f0e', lw=0.8)
    axes[1].set_ylabel('Y (norm.)', fontsize=8)
    axes[1].set_title('BCG Y-axis (seismic, LP 1 Hz, 10 Hz, z-score)', fontsize=8)
    _shade_events(axes[1], t_event, event_ts, alpha=0.12)

    # ── Panel 3: THO ─────────────────────────────────────
    axes[2].plot(t_bcg, THO, color='#2ca02c', lw=0.8)
    axes[2].set_ylabel('THO (norm.)', fontsize=8)
    axes[2].set_title('Thoracic Belt (THO) – respiratory effort (LP 1 Hz, 10 Hz)', fontsize=8)
    _shade_events(axes[2], t_event, event_ts, alpha=0.12)

    # ── Panel 4: ABD ─────────────────────────────────────
    axes[3].plot(t_bcg, ABD, color='#9467bd', lw=0.8)
    axes[3].set_ylabel('ABD (norm.)', fontsize=8)
    axes[3].set_title('Abdominal Belt (ABD) – respiratory effort (LP 1 Hz, 10 Hz)', fontsize=8)
    _shade_events(axes[3], t_event, event_ts, alpha=0.12)

    # ── Panel 5: Event time series ───────────────────────
    ax5 = axes[4]
    for code, color in EVENT_COLOR.items():
        mask = event_ts == code
        if mask.any():
            ax5.fill_between(t_event, code, where=mask, step='post',
                             color=color, alpha=0.85, label=EVENT_MAP[code])
    ax5.set_yticks(list(EVENT_MAP.keys()))
    ax5.set_yticklabels([EVENT_MAP[k] for k in EVENT_MAP], fontsize=6)
    ax5.set_ylabel('Event', fontsize=8)
    ax5.set_title('Apnea Event Labels (10 Hz, PSG annotation)', fontsize=8)
    ax5.set_ylim(-0.5, 4.5)
    ax5.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.6)

    # ── Panel 6: Stage time series ───────────────────────
    ax6 = axes[5]
    for code, color in STAGE_COLOR.items():
        mask = stage_ts == code
        if mask.any():
            ax6.fill_between(t_stage, code, where=mask, step='post',
                             color=color, alpha=0.85, label=STAGE_MAP[code])
    ax6.set_yticks(list(STAGE_COLOR.keys()))
    ax6.set_yticklabels([STAGE_MAP[k] for k in STAGE_COLOR], fontsize=6)
    ax6.set_ylabel('Stage', fontsize=8)
    ax6.set_xlabel('Time (s)', fontsize=8)
    ax6.set_title('Sleep Stage Labels (1 Hz, PSG annotation)', fontsize=8)
    ax6.set_ylim(-1.5, 4.5)
    ax6.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.6)

    for ax in axes:
        ax.set_xlim(0, 60)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(axis='x', alpha=0.3, ls='--')
        _add_aasm_strip(ax, event_ts, t_event)

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_test(X, Y, save_path, example_id):
    """2-panel figure for a test example (X, Y only – no labels)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.subplots_adjust(hspace=0.4, left=0.09, right=0.97, top=0.88, bottom=0.10)

    t_bcg = _time_axes(len(X), FS_DOWN)

    fig.suptitle(f'Test Example {example_id:03d}  |  Predict: is this an apnea event?',
                 fontsize=10, fontweight='bold', color='#333333')

    axes[0].plot(t_bcg, X, color='#1f77b4', lw=0.8)
    axes[0].set_ylabel('X (norm.)', fontsize=8)
    axes[0].set_title('BCG X-axis (LP 1 Hz, 10 Hz, z-score)', fontsize=8)

    axes[1].plot(t_bcg, Y, color='#ff7f0e', lw=0.8)
    axes[1].set_ylabel('Y (norm.)', fontsize=8)
    axes[1].set_xlabel('Time (s)', fontsize=8)
    axes[1].set_title('BCG Y-axis (LP 1 Hz, 10 Hz, z-score)', fontsize=8)

    for ax in axes:
        ax.set_xlim(0, 59)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(axis='x', alpha=0.3, ls='--')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _shade_events(ax, t_event, event_ts, alpha=0.15):
    """Shade the time axis of a BCG panel where apnea events occur."""
    apnea_mask = event_ts != 0
    if not apnea_mask.any():
        return
    in_event = False
    start = 0
    for i, v in enumerate(apnea_mask):
        if v and not in_event:
            start = t_event[i]
            in_event = True
        elif not v and in_event:
            ax.axvspan(start, t_event[i - 1], color='#d62728', alpha=alpha, lw=0)
            in_event = False
    if in_event:
        ax.axvspan(start, t_event[-1], color='#d62728', alpha=alpha, lw=0)


def _add_aasm_strip(ax, event_ts, t_event):
    """Tiny top-of-axis strip colouring apnea spans (visual cue)."""
    # This is already handled by _shade_events; no additional action needed.
    pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print('Loading fold 1 (few-shot source)...')
    fold1 = np.load(os.path.join(DATA_PATH, 'fold1.npy'))

    print('Loading folds 2-4 (test source)...')
    fold234 = np.concatenate([
        np.load(os.path.join(DATA_PATH, f'fold{i}.npy')) for i in [2, 3, 4]
    ], axis=0)

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    print('Sampling examples...')
    few_apnea_idx, few_normal_idx = sample_balanced(
        fold1, N_FEW_APNEA, N_FEW_NORMAL, rng, prefer_sleep=True)
    test_apnea_idx, test_normal_idx = sample_balanced(
        fold234, N_TEST_APNEA, N_TEST_NORMAL, rng, prefer_sleep=True)

    few_idx   = np.concatenate([few_apnea_idx,  few_normal_idx])
    test_idx  = np.concatenate([test_apnea_idx, test_normal_idx])

    few_data  = fold1[few_idx]
    test_data = fold234[test_idx]

    # ── 3. Process signals ───────────────────────────────────────────────────
    print('Processing few-shot signals...')
    fX, fY, fTHO, fABD, fStageTS, fEventTS, fSL, fAL, fPID = extract_all(few_data)

    print('Processing test signals...')
    tX, tY, _, _, _, tEventTS, tSL, tAL, tPID = extract_all(test_data)

    # ── 4. Generate few-shot figures & data files ────────────────────────────
    metadata_rows = []
    print('Generating few-shot figures...')
    for i in tqdm(range(len(few_idx)), desc='Few-shot'):
        label_str = 'apnea' if fAL[i] == 1 else 'normal'
        fig_name  = f'example_{i:03d}_{label_str}.png'
        fig_path  = os.path.join(OUT_DIR, 'few_shot', 'figures', fig_name)
        data_path = os.path.join(OUT_DIR, 'few_shot', 'data', f'example_{i:03d}.npz')

        plot_few_shot(
            X=fX[i], Y=fY[i], THO=fTHO[i], ABD=fABD[i],
            event_ts=fEventTS[i], stage_ts=fStageTS[i],
            apnea_label=int(fAL[i]), stage_label=int(fSL[i]),
            patient_id=int(fPID[i]),
            save_path=fig_path, example_id=i)

        # Determine dominant event type
        ev_nz = fEventTS[i][fEventTS[i] != 0]
        if len(ev_nz):
            uc, cnt = np.unique(ev_nz, return_counts=True)
            dom_ev = int(uc[np.argmax(cnt)])
        else:
            dom_ev = 0

        np.savez_compressed(data_path,
            X=fX[i], Y=fY[i], THO=fTHO[i], ABD=fABD[i],
            event_ts=fEventTS[i], stage_ts=fStageTS[i],
            apnea_label=int(fAL[i]), stage_label=int(fSL[i]),
            patient_id=int(fPID[i]),
            fs_bcg=FS_DOWN, fs_event=FS_DOWN, fs_stage=1)

        metadata_rows.append(dict(
            example_id=i, split='few_shot',
            figure=fig_name,
            apnea_label=int(fAL[i]), stage_label=int(fSL[i]),
            patient_id=int(fPID[i]),
            dominant_event=EVENT_MAP.get(dom_ev, 'Normal'),
            dominant_stage=STAGE_MAP.get(
                int(np.bincount(fStageTS[i].clip(min=0)).argmax()), '?'),
            apnea_duration_s=float((fEventTS[i] != 0).sum() / FS_DOWN)
        ))

    # ── 5. Generate test figures & data files ────────────────────────────────
    ground_truth = {}
    print('Generating test figures...')
    for j in tqdm(range(len(test_idx)), desc='Test'):
        i_global  = len(few_idx) + j
        label_str = 'apnea' if tAL[j] == 1 else 'normal'
        fig_name  = f'example_{i_global:03d}.png'
        fig_path  = os.path.join(OUT_DIR, 'test', 'figures', fig_name)
        data_path = os.path.join(OUT_DIR, 'test', 'data', f'example_{i_global:03d}.npz')

        plot_test(X=tX[j], Y=tY[j], save_path=fig_path, example_id=i_global)

        np.savez_compressed(data_path, X=tX[j], Y=tY[j],
                            fs_bcg=FS_DOWN)

        ground_truth[f'example_{i_global:03d}'] = dict(
            apnea_label=int(tAL[j]), stage_label=int(tSL[j]),
            patient_id=int(tPID[j]),
            dominant_event=EVENT_MAP.get(
                int(tEventTS[j][tEventTS[j] != 0][0]) if (tEventTS[j] != 0).any() else 0, 'Normal'),
            apnea_duration_s=float((tEventTS[j] != 0).sum() / FS_DOWN)
        )
        metadata_rows.append(dict(
            example_id=i_global, split='test',
            figure=fig_name,
            apnea_label=int(tAL[j]), stage_label=int(tSL[j]),
            patient_id=int(tPID[j]),
            dominant_event='HIDDEN', dominant_stage='HIDDEN',
            apnea_duration_s='HIDDEN'
        ))

    # ── 6. Save metadata and reference files ─────────────────────────────────
    print('Saving metadata...')
    pd.DataFrame(metadata_rows).to_csv(
        os.path.join(OUT_DIR, 'metadata.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'test_ground_truth.json'), 'w') as f:
        json.dump(ground_truth, f, indent=2)

    with open(os.path.join(OUT_DIR, 'aasm_reference.json'), 'w') as f:
        json.dump(AASM_REFERENCE, f, indent=2, ensure_ascii=False)

    # ── 7. Summary ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print(f'Done. Output written to: {OUT_DIR}')
    print(f'  Few-shot figures : {len(few_idx):3d}  '
          f'(apnea={N_FEW_APNEA}, normal={N_FEW_NORMAL})')
    print(f'  Test figures     : {len(test_idx):3d}  '
          f'(apnea={N_TEST_APNEA}, normal={N_TEST_NORMAL})')
    print('  Files saved:')
    print('    few_shot/figures/   — 50 figures with full signal + label panels')
    print('    few_shot/data/      — 50 .npz files with X,Y,THO,ABD,event_ts,stage_ts')
    print('    test/figures/       — 50 figures with X, Y only')
    print('    test/data/          — 50 .npz files with X, Y only')
    print('    metadata.csv        — summary table')
    print('    test_ground_truth.json')
    print('    aasm_reference.json')
    print('='*60)


if __name__ == '__main__':
    main()
