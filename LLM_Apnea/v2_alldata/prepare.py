#!/usr/bin/env python3
"""
prepare.py
----------
Generates train / val / test .npy bulk data for LLM-based apnea detection.

Usage:
  python prepare.py

Uses ALL data from each fold (filtering out ambiguous apnea_label == -1).

Data source:
  folds 1-2 → train,  fold3 → val,  fold4 → test
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'Data', 'fold_data_p109_MTL_60s')
OUT_DIR   = os.path.join(ROOT, 'LLM_Apnea', 'v2_alldata', 'dataset')

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


def _filter_valid(data):
    """Remove rows with ambiguous apnea label (-1)."""
    valid_mask = data[:, COL_APNEA_L] != -1
    return data[valid_mask]


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

def main():
    # ── 1. Load data & filter out ambiguous labels ───────────────────────────
    print('Loading folds 1-2 (train source)...')
    train_raw = _filter_valid(np.concatenate([
        np.load(os.path.join(DATA_PATH, f'fold{i}.npy')) for i in [1, 2]
    ], axis=0))

    print('Loading fold 3 (val source)...')
    val_raw = _filter_valid(np.load(os.path.join(DATA_PATH, 'fold3.npy')))

    print('Loading fold 4 (test source)...')
    test_raw = _filter_valid(np.load(os.path.join(DATA_PATH, 'fold4.npy')))

    print(f'Samples after filtering: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}')

    # ── 2. Process signals ───────────────────────────────────────────────────
    print('Processing train signals...')
    trX, trY, trTHO, trABD, trStageTS, trEventTS, trSL, trAL, trPID = extract_all(train_raw)

    print('Processing val signals...')
    vX, vY, vTHO, vABD, vStageTS, vEventTS, vSL, vAL, vPID = extract_all(val_raw)

    print('Processing test signals...')
    tX, tY, tTHO, tABD, tStageTS, tEventTS, tSL, tAL, tPID = extract_all(test_raw)

    # ── 4. Save all data as bulk .npy files ─────────────────────────────────────
    os.makedirs(os.path.join(OUT_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'val'),   exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'test'),  exist_ok=True)

    # Train: X, Y, THO, ABD, event_ts, stage_ts, labels
    print('Saving train data...')
    np.save(os.path.join(OUT_DIR, 'train', 'X.npy'), trX)
    np.save(os.path.join(OUT_DIR, 'train', 'Y.npy'), trY)
    np.save(os.path.join(OUT_DIR, 'train', 'THO.npy'), trTHO)
    np.save(os.path.join(OUT_DIR, 'train', 'ABD.npy'), trABD)
    np.save(os.path.join(OUT_DIR, 'train', 'event_ts.npy'), trEventTS)
    np.save(os.path.join(OUT_DIR, 'train', 'stage_ts.npy'), trStageTS)
    np.save(os.path.join(OUT_DIR, 'train', 'apnea_label.npy'), trAL)
    np.save(os.path.join(OUT_DIR, 'train', 'stage_label.npy'), trSL)
    np.save(os.path.join(OUT_DIR, 'train', 'patient_id.npy'), trPID)

    # Val: X, Y + labels
    print('Saving val data...')
    np.save(os.path.join(OUT_DIR, 'val', 'X.npy'), vX)
    np.save(os.path.join(OUT_DIR, 'val', 'Y.npy'), vY)
    np.save(os.path.join(OUT_DIR, 'val', 'apnea_label.npy'), vAL)
    np.save(os.path.join(OUT_DIR, 'val', 'stage_label.npy'), vSL)
    np.save(os.path.join(OUT_DIR, 'val', 'patient_id.npy'), vPID)
    np.save(os.path.join(OUT_DIR, 'val', 'event_ts.npy'), vEventTS)

    # Test: X, Y + labels
    print('Saving test data...')
    np.save(os.path.join(OUT_DIR, 'test', 'X.npy'), tX)
    np.save(os.path.join(OUT_DIR, 'test', 'Y.npy'), tY)
    np.save(os.path.join(OUT_DIR, 'test', 'apnea_label.npy'), tAL)
    np.save(os.path.join(OUT_DIR, 'test', 'stage_label.npy'), tSL)
    np.save(os.path.join(OUT_DIR, 'test', 'patient_id.npy'), tPID)
    np.save(os.path.join(OUT_DIR, 'test', 'event_ts.npy'), tEventTS)

    # ── 5. Save metadata CSV ─────────────────────────────────────────────────
    print('Saving metadata...')
    metadata_rows = []
    for i in range(len(trAL)):
        metadata_rows.append(dict(
            example_id=i, split='train',
            apnea_label=int(trAL[i]), stage_label=int(trSL[i]),
            patient_id=int(trPID[i]),
            dominant_event=_get_dominant_event(trEventTS[i]),
            dominant_stage=_get_dominant_stage(trStageTS[i]),
            apnea_duration_s=float((trEventTS[i] != 0).sum() / FS_DOWN)
        ))
    for j in range(len(vAL)):
        metadata_rows.append(dict(
            example_id=j, split='val',
            apnea_label=int(vAL[j]), stage_label=int(vSL[j]),
            patient_id=int(vPID[j]),
            dominant_event=_get_dominant_event(vEventTS[j]),
            dominant_stage=_get_dominant_stage(vStageTS[j]),
            apnea_duration_s=float((vEventTS[j] != 0).sum() / FS_DOWN)
        ))
    for k in range(len(tAL)):
        metadata_rows.append(dict(
            example_id=k, split='test',
            apnea_label=int(tAL[k]), stage_label=int(tSL[k]),
            patient_id=int(tPID[k]),
            dominant_event=_get_dominant_event(tEventTS[k]),
            dominant_stage=_get_dominant_stage(tStageTS[k]),
            apnea_duration_s=float((tEventTS[k] != 0).sum() / FS_DOWN)
        ))
    pd.DataFrame(metadata_rows).to_csv(
        os.path.join(OUT_DIR, 'metadata.csv'), index=False)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    n_train_apnea  = sum(1 for r in metadata_rows if r['split'] == 'train' and r['apnea_label'] == 1)
    n_train_normal = sum(1 for r in metadata_rows if r['split'] == 'train' and r['apnea_label'] == 0)
    n_val_apnea    = sum(1 for r in metadata_rows if r['split'] == 'val' and r['apnea_label'] == 1)
    n_val_normal   = sum(1 for r in metadata_rows if r['split'] == 'val' and r['apnea_label'] == 0)
    n_test_apnea   = int((tAL == 1).sum())
    n_test_normal  = int((tAL == 0).sum())

    print('\n' + '='*60)
    print(f'Done. Output written to: {OUT_DIR}')
    print(f'  Train : {len(trAL):4d}  (apnea={n_train_apnea}, normal={n_train_normal})  [X,Y,THO,ABD + labels .npy]')
    print(f'  Val   : {len(vAL):4d}  (apnea={n_val_apnea}, normal={n_val_normal})  [X,Y + labels .npy]')
    print(f'  Test  : {len(tAL):4d}  (apnea={n_test_apnea}, normal={n_test_normal})  [X,Y + labels .npy, random]')
    print('='*60)


if __name__ == '__main__':
    main()
