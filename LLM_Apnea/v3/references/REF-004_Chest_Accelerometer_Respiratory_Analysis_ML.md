# REF-004: Respiratory Analysis During Sleep Using a Chest-Worn Accelerometer: A Machine Learning Approach

**Source:** Stubbe L, Houel N, Cottin F. "Respiratory analysis during sleep using a chest-worn accelerometer: A machine learning approach." *Biomedical Signal Processing and Control*, 2023;79:104888.

**URL:** https://www.sciencedirect.com/science/article/pii/S1746809422004888

**Type:** Research article

---

## Overview

Introduces an accelerometer-based screening tool for estimating respiratory rate and detecting breathing cessations (apnea/hypopnea) during sleep, combining signal processing with machine learning.

## Methods

### Signal Processing
- Chest-worn triaxial accelerometer captures respiratory motion
- **Adaptive bandpass filtering** combined with principal component analysis (PCA) to extract respiratory signal
- Feature-based time-domain techniques found **superior to filter-based and frequency-domain techniques**
- Time-domain features more useful for studying patients with irregular breathing due to apneas

### Novel Feature: Respiratory Peak Variance (RPV)
- A custom feature designed to detect irregular respiratory activity
- RPV captures the variability in peak amplitudes of the respiratory signal over a window
- During normal breathing: low RPV (consistent peak amplitudes)
- During apnea/hypopnea: high RPV (amplitude drops followed by recovery, or absence of peaks)
- Serves as part of a home screening tool for sleep-related breathing disorders

### Detection Performance
- Respiratory rate estimation: **mean difference of 1.8 breaths per minute** vs. reference
- Respiratory cessation detection:
  - **Sensitivity: 76.05%**
  - **Specificity: 70.05%**
  - **Overall accuracy: 70.95%**

### Key Features Used
- Respiratory peak variance (RPV)
- Signal amplitude envelope statistics
- Breathing rate variability
- Time-domain features (mean, std, energy, zero-crossing rate of filtered signal)

---

## Relevance to BCG X/Y Axis Apnea Detection

### Direct Applicability
This work is highly relevant because it uses an **accelerometer** (same modality as our BCG sensor) placed on the chest to detect apnea events.

### Transferable Methods
1. **Respiratory Peak Variance (RPV):** Compute variance of peak amplitudes in a sliding window over the BCG respiratory signal -- a direct indicator of breathing irregularity
2. **Adaptive filtering + PCA:** When using multi-axis (X, Y) accelerometer data, PCA can combine axes to extract the strongest respiratory component
3. **Time-domain features are preferred** over frequency-domain for apnea detection
4. **Feature set for our implementation:**
   - Amplitude envelope statistics (mean, std, min, max over 30-60s windows)
   - Peak-to-peak interval variability
   - Signal energy in respiratory band
   - Ratio of current amplitude to baseline amplitude (directly implements AASM amplitude criterion)

### Limitations to Consider
- Sensitivity/specificity of ~76%/70% leaves room for improvement
- Performance varies with sleep position and sensor placement
- Multi-axis fusion (X + Y) may improve over single-axis analysis
