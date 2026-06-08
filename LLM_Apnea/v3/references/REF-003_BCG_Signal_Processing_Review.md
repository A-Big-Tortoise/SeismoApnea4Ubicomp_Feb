# REF-003: Ballistocardiogram Signal Processing: A Review

**Source:** Sadek I, Biswas J, Abdulrazak B. "Ballistocardiogram signal processing: a review." *Health Information Science and Systems*, 2019;7:10.

**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6522616/

**Type:** Review article

---

## Overview

Comprehensive review of BCG signal processing methods covering sensor technologies, signal extraction, and clinical applications including sleep monitoring.

## Key Findings

### BCG Signal Characteristics
- BCG captures the ballistic forces of the heart caused by sudden ejection of blood into great vessels with each heartbeat
- The signal also contains **respiratory components** and **body movement artifacts**
- BCG modalities include: bed-embedded pads, accelerometers, piezo-force sensitive resistors (FSR), fiber-optic sensors, and RFID-based systems

### Respiratory Signal Extraction from BCG
- **Frequency bands:**
  - BCG/cardiac component: **3-15 Hz** (extracted via 6th order Butterworth bandpass filter)
  - Respiratory component: **0.15-0.4 Hz** (extracted via 4th order Butterworth bandpass filter)
  - Body movement: typically < 0.1 Hz or broadband high-amplitude
- Respiratory rate typically ranges 12-20 breaths/min (0.2-0.33 Hz) in adults

### Apnea Detection from BCG
- Apneas are detected based on the **variance of the respiratory signal** using a moving window technique
- During apnea events:
  - Respiratory amplitude decreases dramatically (near zero for obstructive apnea)
  - Heart rate variability changes (bradycardia during event, tachycardia at termination)
  - Signal variance within the respiratory band drops significantly
- Moving window variance analysis is effective for event-level detection

### Multi-axis Analysis
- Accelerometer-based BCG captures signals in X, Y, and Z axes
- Different axes capture different components of cardiorespiratory motion
- Signal quality varies by axis depending on sensor placement and body position

---

## Relevance to BCG X/Y Axis Apnea Detection

### Signal Processing Pipeline
1. **Bandpass filter** the raw accelerometer signal:
   - Respiratory band: 0.15-0.4 Hz (4th order Butterworth)
   - Cardiac band: 3-15 Hz (6th order Butterworth) -- useful for heart rate extraction
2. **Compute signal envelope** (e.g., Hilbert transform or peak detection)
3. **Compute moving variance** or amplitude in the respiratory band
4. **Threshold** against baseline to detect amplitude reductions

### Feature Extraction for Apnea Events
- **Signal variance** in respiratory band (primary feature)
- **Amplitude envelope** of filtered respiratory signal
- **Respiratory rate** changes (irregular or absent during events)
- **Heart rate variability** markers from cardiac BCG component
- **Cross-axis correlation** between X and Y respiratory signals
