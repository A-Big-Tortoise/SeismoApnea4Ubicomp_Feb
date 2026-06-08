# References Index

Literature references for BCG-based sleep apnea detection from accelerometer X/Y axis signals.

## Reference List

```
[REF-001] | Rules for Scoring Respiratory Events in Sleep (AASM 2012 Update) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3459210/ | Authoritative AASM definitions for apnea (>=90% amplitude drop, >=10s) and hypopnea (>=30% drop, >=10s, >=3% desat or arousal) scoring criteria
[REF-002] | AASM Scoring Manual Version 3 (2023) | https://aasm.org/wp-content/uploads/2024/02/SummaryofChanges_Document_3-1.pdf | Latest AASM update: 4% desaturation rule downgraded to optional; 3% desat/arousal criterion is now the standard
[REF-003] | BCG Signal Processing: A Review (Sadek et al. 2019) | https://pmc.ncbi.nlm.nih.gov/articles/PMC6522616/ | Comprehensive review of BCG signal processing; respiratory band 0.15-0.4 Hz; apnea detected via moving window variance of respiratory signal
[REF-004] | Chest-Worn Accelerometer Respiratory Analysis (Stubbe et al. 2023) | https://www.sciencedirect.com/science/article/pii/S1746809422004888 | ML approach for apnea detection from chest accelerometer; introduces Respiratory Peak Variance (RPV) feature; 76% sensitivity, 70% specificity
[REF-005] | BCG Sleep Apnea Severity Assessment (Hwang et al. 2017) | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0175351 | Three-stage BCG framework for apnea severity; 94% accuracy on 136 subjects; validates direct BCG feature extraction approach
[REF-006] | BCG Respiratory Rate Algorithm Validation (2025) | https://www.nature.com/articles/s41598-025-33325-8 | Large-scale validation (68K datapoints); 93-96% detection rate in apnea patients; confirms BCG respiratory signal reliability
[REF-007] | Smart Bed BCG vs. PSG for Apnea Detection (2025) | https://www.nature.com/articles/s41598-025-07336-4 | Smart bed BCG validation; 83% accuracy, 76% sensitivity, 85% specificity for moderate-to-severe apnea (AHI>=15)
```

## Summary of Key Findings for BCG Apnea Detection

### AASM Scoring Criteria (applicable to BCG signals)
- **Apnea:** >=90% amplitude drop for >=10 seconds
- **Hypopnea:** >=30% amplitude drop for >=10 seconds (with >=3% desaturation or arousal)
- **90% rule:** At least 90% of event duration must meet amplitude criterion

### BCG Signal Processing
- **Respiratory band:** 0.15-0.4 Hz (4th order Butterworth bandpass)
- **Cardiac/BCG band:** 3-15 Hz (6th order Butterworth bandpass)
- **Primary detection method:** Moving window variance/amplitude analysis of respiratory signal

### Validated Features for Detection
- Signal amplitude envelope (mean, std, min, max per window)
- Respiratory Peak Variance (RPV)
- Signal energy in respiratory band
- Amplitude ratio (current vs. baseline) -- directly implements AASM criteria
- Breathing rate variability

### Expected Performance Range
- Severity classification: 83-94% accuracy
- Event-level detection: 76% sensitivity, 70-85% specificity
- Respiratory rate estimation: 1.1-1.8 BrPM error
