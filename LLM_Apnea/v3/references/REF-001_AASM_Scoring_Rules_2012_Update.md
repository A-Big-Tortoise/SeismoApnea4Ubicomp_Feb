# REF-001: Rules for Scoring Respiratory Events in Sleep (AASM 2012 Update)

**Source:** Berry RB, Budhiraja R, Gottlieb DJ, et al. "Rules for Scoring Respiratory Events in Sleep: Update of the 2007 AASM Manual for the Scoring of Sleep and Associated Events." *Journal of Clinical Sleep Medicine*, 2012;8(5):597-619.

**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3459210/

**Type:** Clinical guideline / Authoritative scoring standard

---

## Key Definitions

### Apnea Scoring Rule
Score a respiratory event as an **apnea** when ALL of the following are met:
1. There is a drop in the peak thermal sensor excursion by **>=90% of baseline**
2. The duration of the >=90% drop in sensor signal is **>=10 seconds**
3. At least **90% of the event's duration** meets the amplitude reduction criteria

### Apnea Sub-types
- **Obstructive apnea:** Continued or increased inspiratory effort throughout the entire period of absent airflow
- **Central apnea:** Absent inspiratory effort throughout the entire period of absent airflow
- **Mixed apnea:** Absent inspiratory effort in the initial portion, followed by resumption of effort in the second portion

### Hypopnea Scoring Rules

**Recommended Rule (1A):**
1. Peak signal excursion drops by **>=30%** of pre-event baseline (nasal pressure or PAP device flow)
2. Duration of the >=30% drop is **>=10 seconds**
3. There is a **>=3% oxygen desaturation** from pre-event baseline **and/or** the event is associated with an **arousal**
4. At least **90% of the event's duration** meets the amplitude reduction criteria

**Acceptable/Optional Rule (1B):**
1. Peak signal excursion drops by **>=30%** of pre-event baseline
2. Duration of the >=30% drop is **>=10 seconds**
3. There is a **>=4% oxygen desaturation** from pre-event baseline
4. At least **90% of the event's duration** meets the amplitude reduction criteria

### Apnea-Hypopnea Index (AHI)
- AHI = (total apneas + hypopneas) / total sleep time in hours
- Severity classification:
  - Normal: AHI < 5
  - Mild: AHI 5-14
  - Moderate: AHI 15-29
  - Severe: AHI >= 30

---

## Relevance to BCG-based Apnea Detection

For accelerometer-derived respiratory signals (BCG X/Y axes), the AASM rules translate to:

1. **Apnea detection:** Look for periods where the respiratory signal amplitude drops to <=10% of baseline for >=10 seconds
2. **Hypopnea detection:** Look for periods where the respiratory signal amplitude drops by >=30% (to <=70% of baseline) for >=10 seconds
3. **Duration rule:** All events must last at least 10 seconds
4. **90% rule:** At least 90% of the event duration must meet the amplitude criterion
5. **Without SpO2 data**, we cannot apply the desaturation criterion -- but the amplitude-based rules alone are applicable to BCG signals

### Practical Thresholds for BCG Implementation
- Compute a rolling baseline of respiratory signal amplitude (e.g., median over previous 2 minutes)
- Apnea: signal amplitude < 10% of baseline for >= 10 seconds
- Hypopnea: signal amplitude < 70% of baseline for >= 10 seconds
- Minimum event gap: events separated by < 10 seconds of normal breathing may be merged
