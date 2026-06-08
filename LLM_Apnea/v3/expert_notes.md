# Expert Notes — Calibrated Diagnostic Criteria

## Training Calibration Summary

### Theory-Confirmed Features
1. **Amplitude ratio is the most discriminative feature** — min(X_ar10, Y_ar10) achieves separation ~1.4 between classes. Apnea windows show lower 10th-percentile-to-median amplitude ratio due to sustained amplitude drops during events.
2. **Concordance between X and Y axes** — both axes typically show similar patterns during apnea events. Discordance (one axis affected, the other not) is more suggestive of artifact.
3. **Hypopnea dominates** — 23/25 apnea training samples are hypopnea, only 2 OSA. The AASM ≥90% amplitude drop criterion applies mainly to full apnea; hypopnea shows ≥30% partial reduction.

### Theory-Adjusted Features
1. **Expected**: Clear flat segments during apnea. **Observed**: Most training apnea (hypopnea) shows partial amplitude reduction, not complete cessation. Only OSA sample #005 shows textbook flat segments.
2. **Expected**: Post-event hyperventilation as a clear marker. **Observed**: Recovery breaths are present but variable. In long events (>40s), recovery may occur at window edges and be truncated.
3. **Expected**: Z-normalization would preserve relative patterns. **Observed**: Z-normalization creates a critical confound — movement artifacts inflate the scale, making normal breathing appear flat. Artifacts at #034, #027, #049 all show this pattern.

### Sensor-Specific Insights (BCG)
1. **BCG amplitude reflects chest wall displacement** — the 1 Hz LP filter + 10 Hz downsample isolates the respiratory band. What we see IS respiratory effort.
2. **X-axis (anteroposterior) is generally noisier** than Y-axis for respiratory detection in these signals.
3. **Y-axis amplitude ratio may be slightly more discriminative** than X-axis (separation 1.43 vs 1.39).

### Calibrated Decision Rules

**Primary criterion**: `min(X_amp_ratio_10_50, Y_amp_ratio_10_50)` — the minimum amplitude ratio across both axes.

| min_ar10 Range | Prediction | Confidence | Notes |
|----------------|------------|------------|-------|
| < 0.45         | APNEA      | High       | Clear sustained amplitude reduction in at least one axis |
| 0.45 – 0.55    | APNEA      | Medium     | Likely apnea but check for movement artifacts |
| 0.55 – 0.65    | Borderline | Low        | Visual inspection required; artifact vs subtle apnea |
| 0.65 – 0.75    | NORMAL     | Low        | Likely normal but could be subtle hypopnea |
| > 0.75         | NORMAL     | High       | Stable amplitude envelope throughout |

**Threshold**: min_ar10 < 0.63 → APNEA (training bACC = 0.86)

**Artifact correction heuristics** (for borderline cases):
- **Movement artifact signature**: Isolated sharp spike (max/p90 ratio > 15) + low_dur < 5s + one axis much less affected → reclassify to NORMAL
- **Subtle apnea signature**: Both axes show moderate reduction + breathing_regularity > 0.2 + envelope CV > 0.3 → lean toward APNEA

### Edge Cases from Training
- **#023 (OSA, 10s)**: Minimum-duration event, min_ar10=0.740. MISSED by feature-based approach — event too short relative to window.
- **#019 (Hypopnea, 31.5s)**: min_ar10=0.641, subtle amplitude changes. Borderline miss.
- **#034 (Normal, artifact)**: min_ar10=0.283. Movement artifact mimics apnea in features but NOT in morphology.
