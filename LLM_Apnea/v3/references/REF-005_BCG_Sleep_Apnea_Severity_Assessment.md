# REF-005: Assessing the Severity of Sleep Apnea Syndrome Based on Ballistocardiogram

**Source:** Hwang SH, Lee HJ, Yoon HN, et al. "Assessing the severity of sleep apnea syndrome based on ballistocardiogram." *PLOS ONE*, 2017;12(4):e0175351.

**URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0175351

**Type:** Research article (validation study)

---

## Overview

Developed a BCG-based system for assessing sleep apnea severity, validated against polysomnography (PSG) on 136 subjects. Used a three-stage framework achieving 94.12% accuracy.

## Methods

### Data Collection
- 136 subjects with concurrent PSG and BCG recordings
- BCG signal captured via load cells embedded in the bed
- Full-night recordings in a sleep lab

### Three-Stage Framework
1. **Stage 1: Signal preprocessing and respiratory signal extraction**
   - Bandpass filtering to isolate respiratory component from BCG
   - Artifact detection and removal (movement epochs)

2. **Stage 2: Feature extraction per epoch**
   - Epoch-level features from the respiratory BCG signal
   - Features capturing amplitude changes, breathing regularity, and signal quality

3. **Stage 3: Classification**
   - Machine learning classifiers to categorize AHI severity
   - Binary classification: AHI < 15 vs. AHI >= 15 (moderate-to-severe)
   - Multi-class severity assessment

### Results
- **94.12% accuracy** for severity classification
- **87.5% accuracy** for binary classification (AHI < 15 vs. >= 15) across 32 subjects in a simplified approach
- Demonstrated that direct BCG feature extraction (without separately deriving respiratory and heart rate signals) can be effective

### Key Features
- Amplitude-based features from BCG respiratory signal
- Variability measures capturing breathing irregularity
- Signal energy and power spectral density features
- Epoch-level statistics aggregated over the full night

---

## Relevance to BCG X/Y Axis Apnea Detection

### Validation of BCG for Apnea Assessment
- Confirms that BCG signals contain sufficient information to detect and classify sleep apnea severity
- 94% accuracy validates the modality for clinical screening

### Feature Design Insights
1. **Amplitude reduction features** are the most direct analog to AASM scoring criteria
2. **Epoch-based analysis** (e.g., 30-second or 60-second windows) with aggregation is a proven approach
3. **Direct feature extraction** from the raw BCG can outperform the traditional pipeline of first extracting respiratory rate, then analyzing it
4. **Severity estimation** can be achieved by estimating AHI from event counts per hour

### Architecture Considerations
- A two-stage approach (event detection then severity classification) vs. direct severity estimation are both viable
- For our task (event-level detection from BCG X/Y), the event detection approach aligns better with AASM criteria
