#!/usr/bin/env python3
"""
prepare.py
----------
Generates train / test .npz data AND figures for LLM-based apnea detection.

Usage:
  python prepare.py --train 50 --test 500

- Train: balanced (50/50 apnea/normal), includes X, Y, THO, ABD, event_ts, stage_ts, labels
         + 6-panel PNG figures (X, Y, THO, ABD, event timeline, stage timeline)
- Test:  random sampling (no balancing), includes X, Y only
         + 2-panel PNG figures (X, Y only)

Data source:
  fold1 → train,  folds 2-4 → test

Output:
  LLM_Apnea/v5/dataset/
  ├── train/
  │   ├── figures/   PNG: X, Y, THO, ABD + event + stage panels (with labels)
  │   └── data/      .npz: processed signals, event_ts, stage_ts, metadata
  ├── test/
  │   ├── figures/   PNG: X, Y only (no labels shown)
  │   └── data/      .npz: processed X, Y only (labels saved separately)
  ├── test/ground_truth.json
  └── metadata.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'Data', 'fold_data_p109_MTL_60s')
OUT_DIR   = os.path.join(ROOT, 'LLM_Apnea', 'v5', 'dataset')

# ── Signal processing constants ─────────────────────────────────────────────
FS_RAW    = 100     # Hz (raw sampling rate)
FS_DOWN   = 10      # Hz (target after downsampling)
SEQ_S     = 60      # window length in seconds
N_RAW     = FS_RAW * SEQ_S      # 6000 samples per signal
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

SEED = 42

# ── Label / colour maps ─────────────────────────────────────────────────────
EVENT_MAP = {0: 'Normal', 1: 'Hypopnea',
             2: 'OSA (Obstructive)', 3: 'CSA (Central)', 4: 'Mixed Apnea'}
EVENT_COLOR = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728',
               3: '#9467bd', 4: '#8c564b'}
STAGE_MAP   = {-1: 'Unscored', 0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'Wake'}
STAGE_COLOR = {-1: '#7f7f7f', 0: '#17becf', 1: '#1f77b4',
               2: '#aec7e8',  3: '#bcbd22', 4: '#ff7f0e'}


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

def sample_balanced(data, n_total, rng, prefer_sleep=True):
    """Return indices of n_total rows, balanced 50/50 apnea/normal.
    For apnea windows, prefer well-centred events."""
    n_apnea  = n_total // 2
    n_normal = n_total - n_apnea

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
    pool_a = sorted_apnea[:max(n_apnea * 3, 150)]
    chosen_a = rng.choice(pool_a, size=n_apnea, replace=False)
    chosen_n = rng.choice(normal_idx, size=n_normal, replace=False)
    return np.concatenate([chosen_a, chosen_n])


def sample_random(data, n_total, rng):
    """Return indices of n_total rows, randomly sampled (no balancing).
    Only exclude windows with apnea label -1 (ambiguous)."""
    valid_mask = data[:, COL_APNEA_L] != -1
    valid_idx = np.where(valid_mask)[0]
    return rng.choice(valid_idx, size=n_total, replace=False)


def _get_dominant_event(event_ts_row):
    """Get dominant event type string for a single window."""
    ev_nz = event_ts_row[event_ts_row != 0]
    if len(ev_nz):
        uc, cnt = np.unique(ev_nz, return_counts=True)
        return EVENT_MAP.get(int(uc[np.argmax(cnt)]), 'Normal')
    return 'Normal'


def _get_dominant_stage(stage_ts_row):
    """Get dominant stage string for a single window."""
    return STAGE_MAP.get(
        int(np.bincount(stage_ts_row.clip(min=0)).argmax()), '?')


# ── Figure helpers ───────────────────────────────────────────────────────────

def _time_axes(n_samples, fs):
    return np.arange(n_samples) / fs


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


def plot_train(X, Y, THO, ABD, event_ts, stage_ts,
               apnea_label, stage_label, patient_id,
               save_path, example_id):
    """6-panel figure for a training example (X, Y, THO, ABD + event + stage)."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=False)
    fig.subplots_adjust(hspace=0.55, left=0.09, right=0.97, top=0.93, bottom=0.05)

    # Determine dominant stage for title
    unique_s, counts_s = np.unique(stage_ts, return_counts=True)
    dom_stage = STAGE_MAP.get(int(unique_s[np.argmax(counts_s)]), '?')

    # Determine dominant event type
    event_nz = event_ts[event_ts != 0]
    if len(event_nz):
        unique_e, counts_e = np.unique(event_nz, return_counts=True)
        dom_event = EVENT_MAP.get(int(unique_e[np.argmax(counts_e)]), '?')
    else:
        dom_event = 'Normal'

    label_str = 'APNEA' if apnea_label == 1 else 'NORMAL'
    stage_str = 'Sleep' if stage_label == 0 else 'Wake'
    title = (f'Train Example {example_id:03d}  |  Patient {patient_id}  |  '
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


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare train/test data for LLM apnea detection')
    parser.add_argument('--train', type=int, default=50,
                        help='Number of train examples (balanced apnea/normal)')
    parser.add_argument('--test', type=int, default=500,
                        help='Number of test examples (random, no balancing)')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print('Loading fold 1 (train source)...')
    fold1 = np.load(os.path.join(DATA_PATH, 'fold1.npy'))

    print('Loading folds 2-4 (test source)...')
    fold234 = np.concatenate([
        np.load(os.path.join(DATA_PATH, f'fold{i}.npy')) for i in [2, 3, 4]
    ], axis=0)

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    print('Sampling examples...')
    train_idx = sample_balanced(fold1, args.train, rng, prefer_sleep=True)
    test_idx  = sample_random(fold234, args.test, rng)

    train_data = fold1[train_idx]
    test_data  = fold234[test_idx]

    # ── 3. Process signals ───────────────────────────────────────────────────
    print('Processing train signals...')
    trX, trY, trTHO, trABD, trStageTS, trEventTS, trSL, trAL, trPID = extract_all(train_data)

    print('Processing test signals...')
    tX, tY, _, _, _, tEventTS, tSL, tAL, tPID = extract_all(test_data)

    # ── 4. Save train data + figures (with THO/ABD) ──────────────────────────
    os.makedirs(os.path.join(OUT_DIR, 'train', 'data'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'train', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'test',  'data'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'test',  'figures'), exist_ok=True)

    metadata_rows = []
    print('Saving train data & generating figures...')
    for i in tqdm(range(len(train_idx)), desc='Train'):
        label_str = 'apnea' if trAL[i] == 1 else 'normal'
        data_path = os.path.join(OUT_DIR, 'train', 'data', f'example_{i:03d}.npz')
        fig_name  = f'example_{i:03d}_{label_str}.png'
        fig_path  = os.path.join(OUT_DIR, 'train', 'figures', fig_name)

        np.savez_compressed(data_path,
            X=trX[i], Y=trY[i], THO=trTHO[i], ABD=trABD[i],
            event_ts=trEventTS[i], stage_ts=trStageTS[i],
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            fs_bcg=FS_DOWN, fs_event=FS_DOWN, fs_stage=1)

        plot_train(
            X=trX[i], Y=trY[i], THO=trTHO[i], ABD=trABD[i],
            event_ts=trEventTS[i], stage_ts=trStageTS[i],
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            save_path=fig_path, example_id=i)

        metadata_rows.append(dict(
            example_id=i, split='train',
            figure=fig_name,
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            dominant_event=_get_dominant_event(trEventTS[i]),
            dominant_stage=_get_dominant_stage(trStageTS[i]),
            apnea_duration_s=float((trEventTS[i] != 0).sum() / FS_DOWN)
        ))

    # ── 5. Save test data + figures (X, Y only) ──────────────────────────────
    test_ground_truth = {}
    print('Saving test data & generating figures...')
    for k in tqdm(range(len(test_idx)), desc='Test'):
        eid = len(train_idx) + k
        fig_name  = f'example_{eid:03d}.png'
        data_path = os.path.join(OUT_DIR, 'test', 'data', f'example_{eid:03d}.npz')
        fig_path  = os.path.join(OUT_DIR, 'test', 'figures', fig_name)

        np.savez_compressed(data_path, X=tX[k], Y=tY[k], fs_bcg=FS_DOWN)

        plot_test(X=tX[k], Y=tY[k], save_path=fig_path, example_id=eid)

        test_ground_truth[f'example_{eid:03d}'] = dict(
            apnea_label=int(tAL[k]), stage_label=int(tSL[k]),
            patient_id=int(tPID[k]),
            dominant_event=_get_dominant_event(tEventTS[k]),
            apnea_duration_s=float((tEventTS[k] != 0).sum() / FS_DOWN)
        )
        metadata_rows.append(dict(
            example_id=eid, split='test',
            figure=fig_name,
            apnea_label=int(tAL[k]), stage_label=int(tSL[k]),
            patient_id=int(tPID[k]),
            dominant_event='HIDDEN', dominant_stage='HIDDEN',
            apnea_duration_s='HIDDEN'
        ))

    # ── 6. Save metadata and ground truth ─────────────────────────────────────
    print('Saving metadata...')
    pd.DataFrame(metadata_rows).to_csv(
        os.path.join(OUT_DIR, 'metadata.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'test', 'ground_truth.json'), 'w') as f:
        json.dump(test_ground_truth, f, indent=2)

    # ── 7. Summary ─────────────────────────────────────────────────────────
    n_train_apnea  = sum(1 for r in metadata_rows if r['split'] == 'train' and r['apnea_label'] == 1)
    n_train_normal = sum(1 for r in metadata_rows if r['split'] == 'train' and r['apnea_label'] == 0)
    n_test_apnea   = int((tAL == 1).sum())
    n_test_normal  = int((tAL == 0).sum())

    print('\n' + '='*60)
    print(f'Done. Output written to: {OUT_DIR}')
    print(f'  Train : {len(train_idx):4d}  (apnea={n_train_apnea}, normal={n_train_normal})  [X,Y,THO,ABD + labels + figures]')
    print(f'  Test  : {len(test_idx):4d}  (apnea={n_test_apnea}, normal={n_test_normal})  [X,Y only, random + figures]')
    print('  Files saved:')
    print('    train/figures/  — 6-panel figures with X,Y,THO,ABD + event + stage panels')
    print('    train/data/     — .npz files with X,Y,THO,ABD,event_ts,stage_ts')
    print('    test/figures/   — 2-panel figures with X, Y only')
    print('    test/data/      — .npz files with X, Y only')
    print('    metadata.csv    — summary table')
    print('    test/ground_truth.json')
    print('='*60)


if __name__ == '__main__':
    main()
