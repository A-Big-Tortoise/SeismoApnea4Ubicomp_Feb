#!/usr/bin/env python3
"""
prepare.py
----------
Generates train / test .npz data for LLM-based apnea detection.

Usage:
  python prepare.py --train 50 --test 500

- Train: balanced (50/50 apnea/normal), includes X, Y, THO, ABD, event_ts, stage_ts, labels
- Test:  random sampling (no balancing), includes X, Y only

Data source:
  fold1 → train,  folds 2-4 → test
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'Data', 'fold_data_p109_MTL_60s')
OUT_DIR   = os.path.join(ROOT, 'LLM_Apnea', 'v3', 'dataset')

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

# ── Label maps ──────────────────────────────────────────────────────────────
EVENT_MAP = {0: 'Normal', 1: 'Hypopnea',
             2: 'OSA (Obstructive)', 3: 'CSA (Central)', 4: 'Mixed Apnea'}
STAGE_MAP = {-1: 'Unscored', 0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'Wake'}


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

    # ── 4. Save train data (with THO/ABD) ─────────────────────────────────────
    os.makedirs(os.path.join(OUT_DIR, 'train', 'data'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'test',  'data'), exist_ok=True)

    metadata_rows = []
    print('Saving train data...')
    for i in tqdm(range(len(train_idx)), desc='Train'):
        data_path = os.path.join(OUT_DIR, 'train', 'data', f'example_{i:03d}.npz')
        np.savez_compressed(data_path,
            X=trX[i], Y=trY[i], THO=trTHO[i], ABD=trABD[i],
            event_ts=trEventTS[i], stage_ts=trStageTS[i],
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            fs_bcg=FS_DOWN, fs_event=FS_DOWN, fs_stage=1)

        metadata_rows.append(dict(
            example_id=i, split='train',
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            dominant_event=_get_dominant_event(trEventTS[i]),
            dominant_stage=_get_dominant_stage(trStageTS[i]),
            apnea_duration_s=float((trEventTS[i] != 0).sum() / FS_DOWN)
        ))

    # ── 5. Save test data (X, Y only) ────────────────────────────────────────
    test_ground_truth = {}
    print('Saving test data...')
    for k in tqdm(range(len(test_idx)), desc='Test'):
        eid = len(train_idx) + k
        data_path = os.path.join(OUT_DIR, 'test', 'data', f'example_{eid:03d}.npz')
        np.savez_compressed(data_path, X=tX[k], Y=tY[k], fs_bcg=FS_DOWN)

        test_ground_truth[f'example_{eid:03d}'] = dict(
            apnea_label=int(tAL[k]), stage_label=int(tSL[k]),
            patient_id=int(tPID[k]),
            dominant_event=_get_dominant_event(tEventTS[k]),
            apnea_duration_s=float((tEventTS[k] != 0).sum() / FS_DOWN)
        )
        metadata_rows.append(dict(
            example_id=eid, split='test',
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
    print(f'  Train : {len(train_idx):4d}  (apnea={n_train_apnea}, normal={n_train_normal})  [X,Y,THO,ABD + labels]')
    print(f'  Test  : {len(test_idx):4d}  (apnea={n_test_apnea}, normal={n_test_normal})  [X,Y only, random]')
    print('='*60)


if __name__ == '__main__':
    main()
