# LLM-based Apnea Detection — X/Y Geophone Signal Classification

You are an expert in sleep apnea diagnosis, IoT sensing, and geophone-based respiratory monitoring. Your file system access is strictly limited to `LLM_Apnea/signals_ml/`.

## Objective

Classify each 60-second respiratory segment as **normal** or **abnormal** (apnea/hypopnea event), using only the X and Y axis geophone signals as your primary evidence. X and Y geophone axes capture thoracic and abdominal respiratory effort respectively, analogous to THO and ABD in polysomnography — but treat them as independent signals.

**Core strategy**: Apply AASM scoring criteria directly to X/Y geophone signals. Learn what normal vs. apnea respiratory oscillation looks like in X/Y from the labeled training data, then classify test data.

---

## 0. Scope and File Access

You may **only** read, create, or modify files inside:

```
LLM_Apnea/signals_ml/
```

Everything outside this folder is off-limits.

---

## 1. Data Pipeline

### 1.1 Raw Data Layout

Source: `fold_data_p109_MTL_60s` — relevant columns extracted by `generate_data.py`:

| Column Range   | Content                                          | Rate   |
| -------------- | ------------------------------------------------ | ------ |
| `[0:6000]`     | X-axis BCG (geophone)                            | 100 Hz |
| `[6000:12000]` | Y-axis BCG (geophone)                            | 100 Hz |
| `[-4]`         | Patient ID                                       | —      |
| `[-2]`         | Binary stage label (0=Sleep, 1=Wake)             | —      |
| `[-1]`         | Binary apnea label (0=Normal, 1=Apnea/Hypopnea)  | —      |

### 1.2 Signal Processing Pipeline

Both BCG channels (X, Y) undergo identical processing:

1. **Low-pass filter**: 4th-order Butterworth, cutoff 1.0 Hz, reflect-padded edges
2. **Downsample**: 100 Hz → 10 Hz (polyphase resampling, factor 10)
3. **Crop**: samples [5:595] → 590 usable samples (59 s), removing filter edge artifacts
4. **Z-score normalize**: per-window mean subtraction and standard deviation scaling

After processing, respiratory oscillation (~0.1–0.5 Hz) becomes clearly visible. During apnea, both amplitude and regularity of oscillations diminish.

### 1.3 Label Encodings

**Apnea label** (binary, per window):

| Code | Meaning            |
| ---- | ------------------ |
| 0    | Normal             |
| 1    | Apnea / Hypopnea   |

**Stage label** (binary, per window):

| Code | Meaning |
| ---- | ------- |
| 0    | Sleep   |
| 1    | Wake    |

### 1.4 Dataset Splits

- **Train set**: fold1 — all windows with both labels available.
- **Test set**: fold2 — all windows with both labels available (labels held out during prediction).

### 1.5 Dataset Preparation

```bash
python generate_data.py
```

**`generate_data.py` is read-only — never modify it.**

Output structure:

```
LLM_Apnea/signals_ml/
├── train.npz
├── test.npz
└── dataset_description.md
```

**Array layout per `.npz`**:

| Key           | Shape     | dtype   | Description                          |
| ------------- | --------- | ------- | ------------------------------------ |
| `X`           | [N, 590]  | float64 | BCG X-axis, z-scored @ 10 Hz        |
| `Y`           | [N, 590]  | float64 | BCG Y-axis, z-scored @ 10 Hz        |
| `apnea_label` | [N]       | int     | 0=Normal, 1=Apnea/Hypopnea           |
| `stage_label` | [N]       | int     | 0=Sleep, 1=Wake                      |
| `patient_id`  | [N]       | int     | Patient identifier                   |
| `fs_bcg`      | scalar    | int     | Sampling rate = 10 Hz                |

---

## 2. AASM Clinical Reference

Source: AASM Manual for the Scoring of Sleep and Associated Events v2.6.

### Apnea/Hypopnea Definitions

| Event                 | Criterion                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Obstructive Apnea** | Peak signal excursion drops ≥90% from baseline for ≥10 s, with continued or increased inspiratory effort  |
| **Central Apnea**     | Peak signal excursion drops ≥90% from baseline for ≥10 s, with absent inspiratory effort                  |
| **Mixed Apnea**       | Peak signal excursion drops ≥90% from baseline for ≥10 s, absent effort initially → resumed effort later |
| **Hypopnea**          | Peak signal excursion drops ≥30% from baseline for ≥10 s, with ≥3% desaturation or arousal               |

### Severity (AHI)

| Severity     | AHI             |
| ------------ | --------------- |
| Normal       | < 5 events/hour |
| Mild OSA     | 5 ≤ AHI < 15   |
| Moderate OSA | 15 ≤ AHI < 30  |
| Severe OSA   | AHI ≥ 30       |

### Sensor Characteristics

Bed-mounted seismic sensor (BCG). X and Y axes capture respiratory body movements at 100 Hz. After 1 Hz low-pass filtering and downsampling to 10 Hz, the respiratory oscillation (~0.1–0.5 Hz) becomes clearly visible. During apnea, both amplitude and regularity of oscillations diminish.

---

## 3. Codebase

| File                | Role                                                                |
| ------------------- | ------------------------------------------------------------------- |
| `generate_data.py`  | Dataset preparation — **read-only, never modify**                  |
| `tool.py`           | Your DSP toolkit — create and modify freely                        |
| `main.py`           | Figure generator and measurement runner — create and modify freely |

---

## 4. Setup (Run Once)

### 4.1 Read the Codebase

Read every in-scope file carefully before proceeding.

### 4.2 Prepare the Dataset

```bash
python generate_data.py
```

### 4.3 Scaffold Files

**`tool.py`** — DSP helpers: bandpass filters, envelope extractors, peak detectors, slope calculators, segment annotators, sliding window statistics.

**`main.py`** — figure generator and measurement runner. Default output is saved plots in `plots/`.

### 4.4 Confirm Readiness

Print: dataset splits loaded, references initialized, tool.py and main.py scaffolded, GPU availability.

---

## 5. Permissions

### You CAN

* Generate and examine plots freely
* Write DSP code to sharpen visual observations
* Read and visualize training examples
* Use GPU 0 or CPU

### You CANNOT

* Modify `generate_data.py`
* Access any file outside `LLM_Apnea/signals_ml/`
* Look at test labels before completing training phase
* Treat a metric change as a conclusion without visual evidence

---

## 6. Few Shot Learning Phase

This is the critical phase. You must learn what normal vs. apnea respiratory oscillation looks like in X/Y geophone signals, then build AASM-grounded decision rules.

### Step 1 — Internalize AASM Standards

Parse the AASM criteria above. Extract and retain: apnea/hypopnea definitions, minimum event duration (≥10 s), amplitude reduction thresholds (≥90% for apnea, ≥30% for hypopnea), and event type distinctions. These are your ground truth scoring rules.

### Step 2 — Analyze Labeled Training Examples

For **every** training sample:

```
1. PLOT      — Plot X and Y channels for the window.

2. OBSERVE   — Describe the respiratory oscillation pattern:
               amplitude, regularity, frequency, any dropout periods.

3. CALIBRATE — Apply AASM amplitude/duration criteria to the X/Y signals.
               Note which features distinguish apnea (label=1) from
               normal (label=0) windows.

4. Build Classification Logic — Base all final decisions on X and Y
               signals only. Build rules from observed patterns.
```

---

## 7. Test Phase — Classify Using X/Y Only

Apply your calibrated criteria to **every** test sample:

```
1. PLOT     — Plot each test sample (X and Y only).

2. PREDICT  — For each segment, produce:

   Segment: [filename or ID]
   Label: normal | abnormal
   Confidence: high | medium | low
   Reasoning: [1–2 sentences citing specific signal features,
               e.g., "X-axis amplitude dropped >80% for 14s at t=22s,
               meeting AASM apnea criteria."]

3. EVALUATE — Compare predictions against ground truth (apnea_label).
              Compute bACC, F1, and full confusion matrix (TP/FP/TN/FN).

4. DIAGNOSE — For every misclassified sample:
              · Re-examine the plot — what did you miss?
              · Was it a failure of theory, observation, or sensor limitation?
              · Log the error pattern.
```

---

## 8. Metrics

Three lenses, equal weight:

### Required stdout format

```
test_bacc:  0.XX
test_f1:    0.XX
test_cm:    TP=X FP=X TN=X FN=X
```

### Definitions

**Balanced Accuracy (bACC)**: `(Sensitivity + Specificity) / 2 = (TP/(TP+FN) + TN/(TN+FP)) / 2`

**F1 Score (apnea-positive)**: `2TP / (2TP + FP + FN)`

**Confusion Matrix**: TP / FP / TN / FN

---

## 9. Final Output

```
=== Final Results ===

             bACC    F1     TP    FP    TN    FN
Test         0.XX   0.XX   X     X     X     X

=== Test Set Predictions ===
Segment   | Label    | Confidence | Duration (s) | Reasoning
--------- | -------- | ---------- | ------------ | ---------
...
```

For each test sample:

* **Label**: normal / abnormal
* **Confidence**: high / medium / low
* **Duration**: predicted apnea duration in seconds (0 for normal)
* **Reasoning**: must reference specific X/Y signal features — describe what you observe in the signal, not just feature names
