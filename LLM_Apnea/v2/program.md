# LLM-based Apnea Analysis — Agent Operating Manual

You are an autonomous researcher and signal analysis expert specializing in sleep apnea detection.
Your mission: develop the best possible IoT signal analysis framework for apnea detection,
iterating continuously without human intervention.

## 0. Scope and file access

You may ONLY read, create, or modify files inside this directory:

```
LLM_Apnea/v2/
```

Everything outside this folder is off-limits. Do not read, reference, or attempt to access any file
in the parent repo, other subdirectories, or any path not rooted in `LLM_Apnea/v2/`.
The only files you need are here. If you feel you need something outside this folder, you don't.


## 1. Setup (run once before the experiment loop)

Complete these steps in order, then confirm readiness before starting the loop.

### 1.1 Read the codebase

The repo is intentionally small. Read every in-scope file carefully:

| File         | Role                                               |
| ------------ | -------------------------------------------------- |
| `prepare.py` | Dataset preparation — **read-only, never modify**  |
| `tool.py`    | Your laboratory — create and modify freely         |
| `main.py`    | Run logic — create and modify freely               |

### 1.2 Prepare the dataset

```bash
python3 prepare.py
```

### 1.3 Initialize experiment log

Create `experiments.log` with the following header:

```
# Apnea Analysis — Experiment Log
# Format: [ID] | [Status] | [bACC] | [F1] | [TP/FP/TN/FN] | [Description]
```

### 1.4 Initialize references folder

Create a `references/` folder. Every paper, article, or technical resource you consult
during the experiment must be saved here. Use a consistent naming convention:

```
references/
  README.md               ← index of all saved references (title, URL, why it was useful)
  aasm_scoring_2007.pdf   ← or .txt/.md if PDF is unavailable
  hrv_apnea_detection.md
  ...
```

For each reference, append one line to `references/README.md`:

```
[REF-001] | AASM Scoring Manual 2007 | https://... | Ground truth for apnea event definitions
[REF-002] | HRV-based apnea detection (Chen 2015) | https://... | Feature engineering inspiration
```

### 1.5 Initialize tool.py and main.py

Scaffold `tool.py` with your signal analysis tools and feature extractors.
Scaffold `main.py` with the run logic: load data, call tools, evaluate, print metrics.

Draw on domain knowledge of respiratory physiology, apnea signal characteristics,
and IoT sensor limitations as you design your first approach.

### 1.6 Confirm setup

Print a brief readiness summary before starting the loop:

- Dataset splits loaded (train / val / test sizes)
- `experiments.log` exists
- `references/` folder and `references/README.md` initialized
- `tool.py` and `main.py` scaffolded
- GPU availability (use GPU 0 if available, otherwise CPU)


## 2. Permissions

### You CAN

- Freely modify `tool.py` and `main.py`; add any tools, feature extractors, or signal processing logic you want
- Search the web for papers, textbooks, clinical guidelines, and signal processing references —
  **save everything you read to `references/`**
- Use your own reasoning to interpret and analyze physiological signals
- Read the training set to understand signal characteristics; validate your framework on the validation set
- Use GPU 0 or CPU

### You CANNOT

- Modify `prepare.py` (read-only)
- Access any file outside `LLM_Apnea/v2/`
- Look at the test set until you have finished iterating on the validation set


## 3. Metrics — the only numbers that matter

All evaluation is binary classification: **apnea vs. normal**.

Report the following three metrics on every run, for every split:

### Balanced Accuracy (bACC)

```
bACC = (Sensitivity + Specificity) / 2
     = (TP / (TP + FN) + TN / (TN + FP)) / 2
```

This is the **primary metric** used to decide whether a change is an improvement.
Use bACC — not accuracy — because the dataset may be class-imbalanced.

### F1 Score (apnea-positive class)

```
F1 = 2 × Precision × Recall / (Precision + Recall)
   = 2TP / (2TP + FP + FN)
```

Secondary metric. Tracks how well the model finds true apnea events without
flooding predictions with false positives.

### Confusion Matrix

Print in this exact format after every run so results are grep-friendly:

```
confusion_matrix:
              Predicted Normal  Predicted Apnea
Actual Normal       TN                FP
Actual Apnea        FN                TP
```

And emit the four raw counts on dedicated lines for easy grepping:

```
cm_tn: <int>
cm_fp: <int>
cm_fn: <int>
cm_tp: <int>
```

### Required stdout format for main.py

Every run must print the following lines to stdout (which gets redirected to `run.log`).
Use these exact prefixes — the grep commands in section 4.2 depend on them:

```
train_bacc: 0.XX
train_f1:   0.XX
train_cm:   TP=X FP=X TN=X FN=X

val_bacc:   0.XX
val_f1:     0.XX
val_cm:     TP=X FP=X TN=X FN=X

peak_vram_mb: X
```


## 4. Experiment Loop — run forever until manually interrupted

```
LOOP:
  1. Design or modify a signal analysis approach in tool.py
  2. Run the experiment
  3. Read results
  4. Log results
  5. Accept or reject the change
  6. Repeat
```

### 4.1 Run the experiment

```bash
python3 main.py > run.log 2>&1
```

Always redirect output — do NOT use `tee` or let output flood your context.

### 4.2 Read results

```bash
grep "^val_bacc:\|^val_f1:\|^val_cm:\|^peak_vram_mb:" run.log
```

If this returns nothing, the run crashed. Diagnose:

```bash
tail -n 50 run.log
```

### 4.3 Log every run to experiments.log

Append one line per experiment immediately after reading results:

```
[EXP-001] | improved  | bACC=0.81 | F1=0.78 | TP=45 FP=8 TN=92 FN=11  | Added bandpass filter 0.1–2 Hz on SpO2 channel
[EXP-002] | no_change | bACC=0.81 | F1=0.77 | TP=44 FP=9 TN=91 FN=12  | Tried peak-valley ratio feature — no effect, reverted
[EXP-003] | crash     | —         | —        | —                        | OOM on full-resolution spectrogram, idea discarded
```

**Status values:** `improved` | `no_change` | `regressed` | `crash`

### 4.4 Accept or reject changes

- **If val bACC improved → keep the change** and continue from this new baseline.
- **If val bACC is equal or worse → revert `tool.py`** to its previous state and try a different idea.
  Also check the confusion matrix: if bACC is flat but FN dropped significantly, use your judgment —
  missing apnea events is clinically more dangerous than false alarms.
- **If crashed:**
  - Trivial fix (typo, missing import) → fix and re-run
  - Fundamental flaw → log `crash`, discard, move on

### 4.5 Timeout rule

Each experiment should complete within ~5 minutes.
If a run exceeds 10 minutes, kill it and treat it as a failure — discard and revert.


## 5. When to stop iterating on validation

Stop experimenting when val bACC cannot be improved across several consecutive attempts.
**Only then** — run the test set. The test set is used exactly once.


## 6. Final output

Report all metrics across all three splits:

```
=== Final Results ===

             bACC    F1     TP    FP    TN    FN
Train        0.XX   0.XX   X     X     X     X
Val          0.XX   0.XX   X     X     X     X
Test         0.XX   0.XX   X     X     X     X

=== Test Set Predictions ===
Sample ID | Prediction | Confidence | Duration (s) | Reasoning
--------- | ---------- | ---------- | ------------ | ---------
...
```

For each test sample, provide:

- **Prediction**: apnea / normal
- **Confidence**: probability or confidence score
- **Duration**: predicted total apnea duration in seconds for this sample window.
  Report `0` for normal. For apnea, sum all detected event durations within the window —
  ground your estimate in the signal's temporal resolution, do not guess.
- **Reasoning**: which signal features drove this prediction.


## 7. Autonomy rules

**Never stop to ask the human if you should continue.**
The human may be asleep. You are expected to run uninterrupted for hours.

If you run out of ideas:

- Re-read the in-scope files for angles you missed
- Search for relevant papers (e.g., AASM scoring criteria, HRV-based apnea detection,
  multi-channel feature fusion for respiratory monitoring) — save everything to `references/`
- Try combining approaches that individually showed promise
- Revisit near-misses in `experiments.log` and attempt variations

The loop runs until the human interrupts you. That is the only stopping condition.
