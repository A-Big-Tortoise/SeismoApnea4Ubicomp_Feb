# Dataset Description

## Overview
- **Source**: `/home/jiayu/SeismoApnea4Ubicomp_Feb/Data/fold_data_p109_MTL_60s`
- **Train**: fold1 &nbsp;|&nbsp; **Test**: fold2
- **Signals**: BCG X-axis and Y-axis only
- **Processing**: 1 Hz low-pass → downsample 100→10 Hz → z-score → crop to 590 samples (59 s)
- **Window length**: 60 s (590 usable samples @ 10 Hz after crop)

## Train Split (fold1)
- **n_total**: 37510
- **n_apnea**: 3866
- **n_normal**: 33644
- **apnea_ratio**: 0.1031
- **n_sleep**: 29720
- **n_wake**: 7790
- **patients**: [4, 9, 23, 26, 30, 35, 39, 48, 50, 53, 56, 59, 63, 84, 90, 93, 99, 102, 108, 114, 118, 136, 138, 143, 144, 150, 151, 155]
- **n_patients**: 28
- **signal_shape**: [37510, 590]
- **fs_bcg**: 10
- **window_s**: 60

## Test Split (fold2)
- **n_total**: 27111
- **n_apnea**: 4176
- **n_normal**: 22935
- **apnea_ratio**: 0.154
- **n_sleep**: 19902
- **n_wake**: 7209
- **patients**: [12, 16, 18, 25, 28, 29, 31, 36, 41, 45, 51, 58, 60, 66, 68, 71, 97, 101, 104, 105, 107, 110, 111, 116, 127, 131, 154]
- **n_patients**: 27
- **signal_shape**: [27111, 590]
- **fs_bcg**: 10
- **window_s**: 60

## Array Layout (per .npz)
| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `X` | [N, 590] | float64 | BCG X-axis, z-scored @ 10 Hz |
| `Y` | [N, 590] | float64 | BCG Y-axis, z-scored @ 10 Hz |
| `apnea_label` | [N] | int | 0=Normal, 1=Apnea/Hypopnea |
| `stage_label` | [N] | int | 0=Sleep, 1=Wake |
| `patient_id`  | [N] | int | Patient identifier |
| `fs_bcg`      | scalar | int | Sampling rate = 10 Hz |
