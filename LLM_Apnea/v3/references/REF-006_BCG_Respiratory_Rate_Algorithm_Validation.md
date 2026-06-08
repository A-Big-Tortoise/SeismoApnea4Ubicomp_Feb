# REF-006: Evaluating a BCG-Derived Respiratory Rate Algorithm Through Comprehensive Clinical Validation

**Source:** "Evaluating a ballistocardiography derived respiratory rate algorithm through comprehensive clinical validation across multiple settings." *Scientific Reports*, 2025;15:33325.

**URL:** https://www.nature.com/articles/s41598-025-33325-8

**Type:** Clinical validation study

---

## Overview

Large-scale validation of a BCG-derived respiratory rate algorithm across multiple clinical settings including sleep labs, hospitals, and home environments. Demonstrates robust performance even in the presence of sleep-disordered breathing.

## Key Results

### Overall Performance
- **92.68% detection rate** across 68,342 reference datapoints
- **Mean absolute error: 1.29 BrPM** (breaths per minute)
- Validated across multiple clinical settings

### Performance in Sleep Apnea Patients
- In sleep apnea studies (hospital settings):
  - **Detection rate: ~93-96%**
  - **Mean absolute error: ~1.1 BrPM**
- Even with irregular breathing patterns (apnea, COPD):
  - Algorithm usually obtains correct readings
  - **~93% detection rate** for alerting clinicians to true respiratory changes

### Apnea Event Handling
- The algorithm is designed to handle periods of absent or irregular breathing
- Uses variance-based detection in the respiratory signal
- Moving window technique applied to identify low-variance (apneic) epochs

---

## Relevance to BCG X/Y Axis Apnea Detection

### Validated Signal Processing Approach
1. BCG respiratory signal extraction via bandpass filtering is clinically validated
2. The respiratory rate algorithm's high performance in apnea patients confirms that BCG reliably captures breathing even during disordered sleep
3. Moving window variance analysis is an effective and validated method for identifying apneic epochs

### Key Takeaways for Implementation
- **Respiratory rate estimation** can serve as a secondary feature (abnormally low or absent rate indicates apnea)
- **High detection rates (93-96%)** in apnea patients suggest the BCG respiratory signal preserves enough information for event detection
- **Multi-setting validation** supports generalizability of BCG-based respiratory monitoring
- **Error of ~1.1-1.3 BrPM** indicates the respiratory signal is clean enough for amplitude-based analysis
