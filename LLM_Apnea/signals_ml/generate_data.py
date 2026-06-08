#!/usr/bin/env python3
"""
generate_llm_data.py
--------------------
Splits the dataset into train (fold1) and test (fold2), processes signals,
and saves to .npz files.

Output:
  LLM_Apnea/signals_ml/
  ├── train.npz   (from fold1)
  └── test.npz    (from fold2)

Each .npz contains:
  X, Y        — processed BCG signals [N, 590] @ 10 Hz, z-scored
  apnea_label — binary apnea label [N]  (0=Normal, 1=Apnea/Hypopnea)
  stage_label — binary stage label [N]  (0=Sleep,  1=Wake)
  patient_id  — patient ID [N]
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Data/fold_data_p109_MTL_60s'
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Signal processing constants ─────────────────────────────────────────────
FS_RAW  = 100
FS_DOWN = 10
SEQ_S   = 60
N_RAW   = FS_RAW * SEQ_S   # 6000
CROP_S  = 5
CROP_E  = 595               # → 590 usable samples
LP_HZ   = 1.0

# ── Column layout ────────────────────────────────────────────────────────────
COL_X_END   = N_RAW
COL_Y_END   = 2 * N_RAW
COL_ID      = -4
COL_STAGE_L = -2
COL_APNEA_L = -1


def lowpass_filter(sig_2d, cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    N = sig_2d.shape[1]
    padded = np.pad(sig_2d, ((0, 0), (N, N)), mode='reflect')
    return filtfilt(b, a, padded, axis=1)[:, N:-N]


def process_bcg(raw_2d):
    filtered    = lowpass_filter(raw_2d, LP_HZ, FS_RAW)
    downsampled = resample_poly(filtered, 1, FS_RAW // FS_DOWN, axis=1)
    cropped     = downsampled[:, CROP_S:CROP_E]
    mean        = cropped.mean(axis=1, keepdims=True)
    std         = cropped.std(axis=1, keepdims=True) + 1e-8
    return (cropped - mean) / std


def extract_all(data):
    X   = process_bcg(data[:, :COL_X_END])
    Y   = process_bcg(data[:, COL_X_END:COL_Y_END])
    apnea_label = data[:, COL_APNEA_L].astype(int)
    stage_label = data[:, COL_STAGE_L].astype(int)
    patient_id  = data[:, COL_ID].astype(int)
    return X, Y, apnea_label, stage_label, patient_id


def save_split(data, out_path, split_name):
    print(f'Processing {split_name} ({len(data)} windows)...')
    X, Y, apnea_label, stage_label, patient_id = extract_all(data)
    np.savez_compressed(
        out_path,
        X=X, Y=Y,
        apnea_label=apnea_label, stage_label=stage_label,
        patient_id=patient_id,
        fs_bcg=FS_DOWN,
    )
    n_apnea  = (apnea_label == 1).sum()
    n_normal = (apnea_label == 0).sum()
    print(f'  Saved to {out_path}')
    print(f'  apnea={n_apnea}, normal={n_normal}, total={len(apnea_label)}')
    return dict(
        split=split_name,
        n_total=int(len(apnea_label)),
        n_apnea=int(n_apnea),
        n_normal=int(n_normal),
        apnea_ratio=round(float(n_apnea) / len(apnea_label), 4),
        n_sleep=int((stage_label == 0).sum()),
        n_wake=int((stage_label == 1).sum()),
        patients=sorted(int(p) for p in np.unique(patient_id)),
        n_patients=int(np.unique(patient_id).shape[0]),
        signal_shape=list(X.shape),
        fs_bcg=FS_DOWN,
        window_s=SEQ_S,
    )


def write_description(train_stats, test_stats):
    lines = [
        "# Dataset Description",
        "",
        "## Overview",
        f"- **Source**: `{DATA_PATH}`",
        f"- **Train**: fold1 &nbsp;|&nbsp; **Test**: fold2",
        f"- **Signals**: BCG X-axis and Y-axis only",
        f"- **Processing**: 1 Hz low-pass → downsample 100→10 Hz → z-score → crop to 590 samples (59 s)",
        f"- **Window length**: {SEQ_S} s (590 usable samples @ {FS_DOWN} Hz after crop)",
        "",
        "## Train Split (fold1)",
    ]
    for k, v in train_stats.items():
        if k != 'split':
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Test Split (fold2)",
    ]
    for k, v in test_stats.items():
        if k != 'split':
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Array Layout (per .npz)",
        "| Key | Shape | dtype | Description |",
        "|-----|-------|-------|-------------|",
        f"| `X` | [N, 590] | float64 | BCG X-axis, z-scored @ {FS_DOWN} Hz |",
        f"| `Y` | [N, 590] | float64 | BCG Y-axis, z-scored @ {FS_DOWN} Hz |",
        "| `apnea_label` | [N] | int | 0=Normal, 1=Apnea/Hypopnea |",
        "| `stage_label` | [N] | int | 0=Sleep, 1=Wake |",
        "| `patient_id`  | [N] | int | Patient identifier |",
        f"| `fs_bcg`      | scalar | int | Sampling rate = {FS_DOWN} Hz |",
    ]
    out_path = os.path.join(OUT_DIR, 'dataset_description.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nDataset description saved to {out_path}')


def main():
    print('Loading fold1 (train)...')
    fold1 = np.load(os.path.join(DATA_PATH, 'fold1.npy'))

    print('Loading fold2 (test)...')
    fold2 = np.load(os.path.join(DATA_PATH, 'fold2.npy'))

    train_stats = save_split(fold1, os.path.join(OUT_DIR, 'train.npz'), 'train')
    test_stats  = save_split(fold2, os.path.join(OUT_DIR, 'test.npz'),  'test')

    write_description(train_stats, test_stats)
    print('\nDone.')


if __name__ == '__main__':
    main()
