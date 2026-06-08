# Expert Prior Knowledge — BCG-Based Sleep Apnea Detection

## 1. Pathophysiology of Sleep Apnea

### Obstructive Sleep Apnea (OSA)
- Upper airway collapses despite continued respiratory effort
- Thoracic and abdominal effort persists but becomes paradoxical (out-of-phase)
- Airflow ceases or drops ≥90% for ≥10 seconds (AASM definition)
- Terminated by arousal → post-apnea compensatory hyperventilation (large-amplitude breaths)

### Central Sleep Apnea (CSA)
- Absence of both airflow AND respiratory effort
- No paradoxical breathing — all channels go flat simultaneously
- BCG signal shows near-complete cessation of respiratory oscillation

### Hypopnea
- Partial airflow reduction ≥30% for ≥10s with ≥3% desaturation or arousal
- More subtle amplitude reduction than full apnea
- BCG shows diminished but not absent respiratory oscillation

### Mixed Apnea
- Begins as central (no effort), transitions to obstructive (effort against collapsed airway)
- BCG may show flat segment followed by paradoxical effort pattern

## 2. Signal Manifestations in BCG/Accelerometer

### Normal Breathing
- Regular quasi-sinusoidal oscillation in the respiratory band (0.15–0.4 Hz, ~9–24 breaths/min)
- Consistent amplitude envelope across the window
- X-axis (anteroposterior) captures chest wall displacement from breathing
- Y-axis (lateral) captures secondary respiratory motion

### Apnea Signature in BCG
- **Amplitude reduction/cessation**: Respiratory oscillation drops dramatically (≥90% for apnea, ≥30% for hypopnea)
- **Flat segments**: During events, BCG respiratory signal approaches baseline noise
- **Post-event surge**: After event termination, compensatory hyperventilation produces transient high-amplitude oscillations
- **Duration**: ≥10 seconds of reduced amplitude (by AASM definition)
- **Cardiac artifact**: During respiratory cessation, cardiac BCG component (higher frequency) may become more visible as respiratory masking disappears

### Key Signal Features
- **Envelope amplitude**: Moving average of |signal| — drops during events
- **Local variance**: Sliding window variance — low during apnea, high during normal breathing
- **Respiratory Peak Variance (RPV)**: Variance of successive peak amplitudes — elevated in apnea windows due to alternating flat/hyperventilation segments
- **Amplitude ratio**: Ratio of segment amplitude to window baseline — directly implements AASM ≥90% drop criterion
- **Breathing rate regularity**: Normal breathing is periodic; apnea windows show irregular or absent periodicity

## 3. Temporal Characteristics

- **Minimum event duration**: 10 seconds (AASM)
- **Typical event duration**: 15–30 seconds for individual events
- **Our window length**: ~59 seconds (590 samples at 10 Hz)
- **Event occupancy**: An apnea-labeled 60s window may contain 1–3 events, or one long event
- **Expected apnea duration in window**: 10–50 seconds based on metadata (mean ~40s from training data)

## 4. Known Confounders

- **Movement artifacts**: Large transient spikes from body position changes — distinguish by duration (seconds) and morphology (non-rhythmic)
- **Positional changes**: Sustained amplitude shift after body movement — look for abrupt baseline change
- **Periodic limb movements**: Rhythmic but non-respiratory oscillation — different frequency from breathing
- **REM respiratory variability**: Normal REM breathing is more irregular than NREM — don't over-call mild variability as hypopnea
- **Signal processing artifacts**: Filter edge effects (mitigated by cropping in prepare.py), z-score normalization means absolute amplitude is lost — must use relative amplitude patterns within the window

## 5. Sensor-Specific Considerations

### BCG at 10 Hz, LP filtered at 1 Hz
- Our signals are already low-pass filtered to 1 Hz and downsampled to 10 Hz
- This means we see ONLY the respiratory component (and below)
- Cardiac BCG (~1–2 Hz) has been filtered out
- The respiratory band (0.15–0.4 Hz) is fully captured
- Z-score normalization means: amplitude = standard deviations from mean, not physical units

### X vs Y Axis
- X (anteroposterior): Primary respiratory axis — strongest breathing signal
- Y (lateral): Secondary respiratory axis — useful for confirming bilateral patterns
- Both should show concordant changes during apnea events

### THO/ABD (Train Only)
- Thoracic and abdominal respiratory effort belts — gold standard for respiratory effort
- Available only in training to calibrate understanding; test set has only X, Y
- THO/ABD help distinguish OSA (effort present, airflow absent) from CSA (effort absent)

---

## 6. Diagnostic Hypotheses (Pre-Data)

### What I Expect to See in Apnea Windows
1. **Clear amplitude reduction** in both X and Y channels lasting ≥10s
2. **Post-event hyperventilation** — 2–4 large-amplitude breaths after the flat segment
3. **Higher RPV** — alternating low (event) and high (recovery) amplitude creates high peak variance
4. **Concordance between X and Y** — both axes should show similar patterns
5. **THO/ABD confirmation** (train only) — effort channels should clarify event type

### What I Expect to See in Normal Windows
1. **Stable amplitude envelope** throughout the 59s window
2. **Regular respiratory periodicity** — consistent inter-breath interval
3. **Low RPV** — successive breaths have similar amplitude
4. **X and Y concordance** in maintaining stable oscillation

### Most Discriminative Features (Predicted)
1. **Minimum amplitude ratio** (min envelope / median envelope) — most direct proxy for AASM criterion
2. **Duration of low-amplitude segments** — apnea requires ≥10s
3. **Amplitude variance across the window** — high in apnea (events + recovery), low in normal
4. **Breathing rate regularity** — disrupted in apnea windows

### Hardest Edge Cases (Anticipated)
1. **Short or mild hypopnea** — amplitude reduction may be subtle, especially after z-normalization
2. **Normal REM variability** — irregular but NOT pathological breathing
3. **Movement-contaminated windows** — artifact may mask or mimic apnea patterns
4. **Apnea at window edges** — partial events may not show complete apnea-recovery cycle
5. **Post-normalization ambiguity** — z-scoring removes absolute scale; a window with one brief event may have most of its variance from normal breathing, making the event hard to detect relative to window statistics
