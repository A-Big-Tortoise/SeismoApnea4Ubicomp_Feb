# Experiment Log — AASM-grounded Apnea Classification

**Task**: Binary classification of 60s BCG windows (X + Y axis geophone) as normal vs apnea/hypopnea.  
**Data**: Train=37,510 (10.3% apnea) | Test=27,111 (15.4% apnea)  
**Signal**: Z-scored, 1 Hz LP-filtered, 10 Hz, 590 samples per window

---

## Feature Set Evolution

### v1 — 24 AASM-inspired features (per-channel X and Y)
Features per channel (×2):
- `env_mean`, `env_std`, `env_cv`, `env_min_ratio`
- `apnea_frac`, `max_apnea_s`, `has_apnea`  (AASM 90% drop ≥10s)
- `hypopnea_frac`, `max_hypopnea_s`, `has_hypopnea`  (AASM 70% drop ≥10s)
- `regularity` (FFT autocorrelation peak), `resp_power`

### v2 — 30 features (v1 + 6 combined XY features)
Added: `xy_has_apnea`, `xy_has_hypopnea`, `xy_max_apnea_s`, `xy_max_hypopnea_s`, `xy_regularity`, `xy_env_cv`  
(XY = minimum across X and Y — requires both channels to agree)

### v3 — 40 features (v2 + 10 block/recovery features)
Added per channel: `block_min_ratio`, `block_cv`, `recovery_ratio`, `drop_ratio`, `env_iqr`  
Plus: `xy_block_min_ratio`  
(10-second non-overlapping blocks to capture sustained amplitude collapse)

---

## Calibration Snapshot (v3 features, training set)
Key discriminative features (Δ = apnea mean − normal mean):

| Feature               | Normal  | Apnea   | Δ       |
|-----------------------|---------|---------|---------|
| x_has_hypopnea        | 0.4860  | 0.8968  | +0.4108 |
| xy_has_hypopnea       | 0.3643  | 0.7913  | +0.4269 |
| x_env_cv              | 0.6310  | 0.8889  | +0.2579 |
| x_env_std             | 0.4588  | 0.6538  | +0.1949 |
| x_max_hypopnea_s      | 15.01   | 25.32   | +10.31  |
| xy_max_hypopnea_s     | 11.71   | 21.41   | +9.70   |
| x_regularity          | 0.6085  | 0.4664  | −0.1420 |
| x_env_min_ratio       | 0.1706  | 0.0611  | −0.1095 |
| x_block_min_ratio     | ~0.16   | ~0.06   | ~−0.10  |

Note: 48.6% of normal windows have `x_has_hypopnea=True` (within-window baseline is too local — AASM uses a 2-min running baseline). This causes the high FP rate.

---

## Results Table

| Exp | Model                    | Features | bACC   | F1     | TP   | FP   | TN    | FN   |
|-----|--------------------------|----------|--------|--------|------|------|-------|------|
| 1   | RandomForest (balanced)  | v1 (24)  | 0.6793 | 0.4120 | 2471 | 5348 | 17587 | 1705 |
| 2   | RandomForest (balanced)  | v2 (30)  | 0.6774 | 0.4137 | 2391 | 4993 | 17942 | 1785 |
| 3   | HistGradientBoosting     | v2 (30)  | 0.7072 | 0.4274 | 2941 | 6646 | 16289 | 1235 |
| 4   | HistGradientBoosting     | v3 (40)  | 0.7121 | 0.4316 | 2988 | 6683 | 16252 | 1188 |

**Best so far: Exp 4** — HGB (max_iter=800, max_depth=5, lr=0.03, min_samples_leaf=20, l2=0.1) + 40 features

---

## Misclassification Diagnosis

### False Negatives (missed apnea)
Pattern: Apnea windows where the model sees "Regular respiratory oscillations, no sustained dropout". These are likely **subtle hypopnea events** where amplitude reduction is real but below the AASM apnea threshold in the X/Y BCG signal. Without SpO2 data, these are fundamentally ambiguous.

### False Positives (false alarms)
Pattern: Normal windows where "amplitude dropped >30% of baseline for 14–35s". Likely causes:
- Naturally variable breathing amplitude (deep vs shallow breaths)
- Body position changes temporarily damping the BCG signal
- Windows at sleep stage transitions
- The within-window 90th percentile baseline is lower than the true multi-minute baseline

---

## Model Config (Final)

```python
HistGradientBoostingClassifier(
    max_iter=800, max_depth=5, learning_rate=0.03,
    min_samples_leaf=20, l2_regularization=0.1,
    class_weight='balanced', random_state=42,
)
```

Top-10 features by RF importance (v3):
1. x_block_min_ratio      (0.090)
2. x_max_hypopnea_s       (0.071)
3. xy_max_hypopnea_s      (0.068)
4. xy_block_min_ratio     (0.049)
5. x_env_std              (0.049)
6. x_env_iqr              (0.047)
7. x_env_mean             (0.043)
8. x_has_hypopnea         (0.042)
9. y_env_iqr              (0.039)
10. y_max_hypopnea_s      (0.039)

---

## Required Final Output (Exp 4)

```
test_bacc:  0.7121
test_f1:    0.4316
test_cm:    TP=2988 FP=6683 TN=16252 FN=1188
```
