# REF-007: Validating a Smart Bed Against Polysomnography for Sleep Apnea Detection

**Source:** "Validating a smart bed against polysomnography for sleep apnea detection." *Scientific Reports*, 2025.

**URL:** https://www.nature.com/articles/s41598-025-07336-4

**Type:** Clinical validation study

---

## Overview

Validation of a smart bed platform using BCG data for sleep apnea detection against gold-standard polysomnography. Demonstrates clinically useful accuracy for moderate-to-severe apnea screening.

## Key Results

### Detection Performance
- **83.3% accuracy** for identifying moderate-to-severe apnea (AHI >= 15)
- **Sensitivity: 76%** (correctly identifies individuals with apnea)
- **Specificity: 85%** (correctly identifies individuals without apnea)

### Algorithm Approach
- BCG signals captured from pressure sensors in the smart bed
- Extracts respiratory and cardiac features from the BCG signal
- Uses machine learning to estimate AHI and classify severity
- Event-level detection followed by AHI estimation

---

## Relevance to BCG X/Y Axis Apnea Detection

### Clinical Benchmarks
- 83% accuracy / 76% sensitivity / 85% specificity represents a realistic performance target for BCG-based apnea detection
- The sensitivity gap (76%) highlights that some apnea events are difficult to detect from BCG alone (particularly hypopneas scored by arousal or mild desaturation)

### Design Lessons
1. **AHI >= 15 threshold** (moderate-to-severe) is a practical clinical cutoff for screening
2. **Specificity > sensitivity** pattern is common in BCG-based detection -- the method is better at confirming absence of apnea than detecting all events
3. **Smart bed BCG** is analogous to our chest-worn accelerometer BCG in terms of the signal content (respiratory + cardiac mechanical signals)
