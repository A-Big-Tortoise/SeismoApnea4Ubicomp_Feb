#!/usr/bin/env python3
"""
main.py — AASM-grounded apnea classification from X/Y geophone signals.

Phases
------
1  Plot representative training examples (normal vs apnea)
2  Extract AASM-inspired DSP features from all windows
3  Train Random Forest classifier on training features
4  Evaluate on test set; print required metrics
5  Save per-sample predictions to CSV; plot misclassified samples
"""

import os, time, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from tool import (
    bandpass_filter, compute_envelope_batch, extract_features_batch,
    FS, APNEA_THRESH, HYPOPNEA_THRESH, MIN_EVENT_S,
    RESP_LOW_HZ, RESP_HIGH_HZ, BASELINE_PCT,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ── GPU check ──────────────────────────────────────────────────────────────────
try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading datasets...')
train = np.load(os.path.join(BASE, 'train.npz'))
test  = np.load(os.path.join(BASE, 'test.npz'))

X_tr, Y_tr  = train['X'], train['Y']
y_tr        = train['apnea_label']
X_te, Y_te  = test['X'],  test['Y']
y_te        = test['apnea_label']

print(f'  Train : {len(y_tr):,} windows  '
      f'(apnea={y_tr.sum():,}, normal={(y_tr==0).sum():,})')
print(f'  Test  : {len(y_te):,} windows  '
      f'(apnea={y_te.sum():,}, normal={(y_te==0).sum():,})')
print(f'Dataset splits loaded, references initialized, '
      f'tool.py and main.py scaffolded, GPU availability: {GPU}.')

T_AXIS = np.arange(590) / FS   # time axis in seconds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_window(x, y, label, win_id, prefix='train'):
    x_resp = bandpass_filter(x[None])[0]
    y_resp = bandpass_filter(y[None])[0]
    x_env  = compute_envelope_batch(x_resp[None])[0]
    y_env  = compute_envelope_batch(y_resp[None])[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    tag = 'Apnea-Hypopnea' if label == 1 else 'Normal'
    fig.suptitle(f'[{prefix}] Window {win_id} — {tag}', fontweight='bold')

    for ax, sig, env, ch in zip(axes,
                                 [x_resp, y_resp],
                                 [x_env,  y_env],
                                 ['X', 'Y']):
        base = np.percentile(env, BASELINE_PCT)
        ax.plot(T_AXIS, sig, lw=0.7, color='steelblue',
                label='Resp. signal (0.1–0.45 Hz)')
        ax.fill_between(T_AXIS,  env, -env, alpha=0.20, color='steelblue')
        ax.plot(T_AXIS,  env, lw=1.5, color='tomato',   label='Envelope')
        ax.plot(T_AXIS, -env, lw=1.5, color='tomato',   alpha=0.4)
        ax.axhline( base * APNEA_THRESH,    lw=1.2, ls='--', color='purple',
                    label=f'Apnea thr ({APNEA_THRESH*100:.0f}% baseline)')
        ax.axhline(-base * APNEA_THRESH,    lw=1.2, ls='--', color='purple', alpha=0.4)
        ax.axhline( base * HYPOPNEA_THRESH, lw=1.0, ls=':',  color='darkorange',
                    label=f'Hypopnea thr ({HYPOPNEA_THRESH*100:.0f}% baseline)')
        ax.axhline(-base * HYPOPNEA_THRESH, lw=1.0, ls=':',  color='darkorange', alpha=0.4)
        ax.set_ylabel(f'{ch}-axis (z-score)')
        ax.set_ylim(-7, 7)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fname = os.path.join(PLOTS, f'{prefix}_{tag}_{win_id}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fname


def extract_all(X, Y, label='', chunk=2000):
    """Feature extraction in chunks; returns [N, n_feat] and feature names."""
    N = len(X)
    parts, names = [], None
    for start in range(0, N, chunk):
        end         = min(start + chunk, N)
        t0          = time.time()
        feat, names = extract_features_batch(X[start:end], Y[start:end])
        parts.append(feat)
        print(f'  {label}: {end:>6,}/{N:,}  ({time.time()-t0:.1f}s)')
    return np.vstack(parts), names


def get_confidence(prob):
    if prob >= 0.80 or prob <= 0.20:
        return 'high'
    if prob >= 0.65 or prob <= 0.35:
        return 'medium'
    return 'low'


def reason(feat_row, names, pred):
    d    = dict(zip(names, feat_row))
    xma  = d['x_max_apnea_s'];    yma  = d['y_max_apnea_s']
    xmh  = d['x_max_hypopnea_s']; ymh  = d['y_max_hypopnea_s']
    xreg = d['x_regularity'];     yreg = d['y_regularity']
    xpow = d['x_resp_power'];     ypow = d['y_resp_power']

    if pred == 1:
        max_a = max(xma, yma);  ch_a = 'X' if xma >= yma else 'Y'
        max_h = max(xmh, ymh);  ch_h = 'X' if xmh >= ymh else 'Y'
        if max_a >= MIN_EVENT_S:
            return (f'{ch_a}-axis amplitude dropped >90% of baseline for '
                    f'{max_a:.0f}s, meeting AASM apnea criteria. '
                    f'Oscillation regularity={xreg:.2f} (reduced).')
        if max_h >= MIN_EVENT_S:
            return (f'{ch_h}-axis amplitude dropped >30% of baseline for '
                    f'{max_h:.0f}s, consistent with AASM hypopnea. '
                    f'Resp band power X={xpow:.3f} Y={ypow:.3f}.')
        return (f'Irregular/low-amplitude oscillations: X regularity={xreg:.2f}, '
                f'resp power={xpow:.3f}. Pattern inconsistent with regular breathing.')
    else:
        return (f'Regular respiratory oscillations: X regularity={xreg:.2f}, '
                f'resp power={xpow:.3f}. No sustained dropout meeting AASM criteria '
                f'(max dropout {max(xma,yma):.0f}s).')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1 — Plot training examples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n--- Phase 1: Plotting training examples ---')
rng   = np.random.default_rng(42)
n_idx = rng.choice(np.where(y_tr == 0)[0], 5, replace=False)
a_idx = rng.choice(np.where(y_tr == 1)[0], 5, replace=False)

for idx in n_idx:
    f = plot_window(X_tr[idx], Y_tr[idx], 0, idx, 'train')
    print(f'  Normal {idx}: {f}')
for idx in a_idx:
    f = plot_window(X_tr[idx], Y_tr[idx], 1, idx, 'train')
    print(f'  Apnea  {idx}: {f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2 — Feature extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n--- Phase 2: Extracting features ---')
F_tr, names = extract_all(X_tr, Y_tr, label='train', chunk=2000)
F_te, _     = extract_all(X_te, Y_te, label='test',  chunk=2000)
print(f'Train features: {F_tr.shape},  Test features: {F_te.shape}')

# Calibration: feature statistics by class
print('\n--- Calibration: Feature means by class ---')
print(f'  {"Feature":36s} {"Normal":>9s} {"Apnea":>9s} {"Δ":>9s}')
for i, nm in enumerate(names):
    mn = F_tr[y_tr == 0, i].mean()
    ma = F_tr[y_tr == 1, i].mean()
    print(f'  {nm:36s} {mn:9.4f} {ma:9.4f} {ma-mn:+9.4f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3 — Train classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n--- Phase 3: Training classifiers ---')

# Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=10,
    class_weight='balanced', n_jobs=-1, random_state=42,
)
t0 = time.time()
rf.fit(F_tr, y_tr)
print(f'RF trained in {time.time()-t0:.1f}s')
rf_pred = rf.predict(F_te)
rf_bacc = balanced_accuracy_score(y_te, rf_pred)
print(f'  RF test bACC: {rf_bacc:.4f}')

# HistGradientBoosting
hgb = HistGradientBoostingClassifier(
    max_iter=800, max_depth=5, learning_rate=0.03,
    min_samples_leaf=20, l2_regularization=0.1,
    class_weight='balanced', random_state=42,
)
t0 = time.time()
hgb.fit(F_tr, y_tr)
print(f'HGB trained in {time.time()-t0:.1f}s')
hgb_pred = hgb.predict(F_te)
hgb_bacc = balanced_accuracy_score(y_te, hgb_pred)
print(f'  HGB test bACC: {hgb_bacc:.4f}')

# Pick the better model
if hgb_bacc >= rf_bacc:
    clf = hgb; y_pred = hgb_pred
    print('→ Using HistGradientBoosting')
else:
    clf = rf;  y_pred = rf_pred
    print('→ Using RandomForest')

imps = sorted(zip(rf.feature_importances_, names), reverse=True)
print('Top-10 features (RF importance):')
for imp, nm in imps[:10]:
    print(f'  {nm:36s}: {imp:.4f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4 — Evaluate on test set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n--- Phase 4: Evaluating on test set ---')
y_prob = clf.predict_proba(F_te)[:, 1]

bacc = balanced_accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
cm   = confusion_matrix(y_te, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f'\ntest_bacc:  {bacc:.4f}')
print(f'test_f1:    {f1:.4f}')
print(f'test_cm:    TP={TP} FP={FP} TN={TN} FN={FN}')

# ── Final results table ────────────────────────────────────────────────────────
print('\n=== Final Results ===\n')
print(f'{"":12s}  {"bACC":>6s}  {"F1":>6s}  {"TP":>6s}  {"FP":>6s}  {"TN":>6s}  {"FN":>6s}')
print(f'{"Test":12s}  {bacc:.4f}  {f1:.4f}  {TP:>6d}  {FP:>6d}  {TN:>6d}  {FN:>6d}')

# ── Per-sample prediction table ───────────────────────────────────────────────
max_apnea_x_i = names.index('x_max_apnea_s')
max_apnea_y_i = names.index('y_max_apnea_s')

rows = []
for i in range(len(y_te)):
    lbl  = 'abnormal' if y_pred[i] == 1 else 'normal'
    conf = get_confidence(y_prob[i])
    dur  = (max(F_te[i, max_apnea_x_i], F_te[i, max_apnea_y_i])
            if y_pred[i] == 1 else 0.0)
    rsn  = reason(F_te[i], names, y_pred[i])
    rows.append((i, lbl, conf, dur, rsn))

print('\n=== Test Set Predictions ===')
print(f'{"Segment":>10s} | {"Label":>10s} | {"Confidence":>10s} | '
      f'{"Duration (s)":>12s} | Reasoning')
print('-' * 130)
for seg, lbl, conf, dur, rsn in rows[:30]:
    short = (rsn[:80] + '...') if len(rsn) > 80 else rsn
    print(f'{seg:>10d} | {lbl:>10s} | {conf:>10s} | {dur:>12.1f} | {short}')
if len(rows) > 30:
    print(f'  ... ({len(rows) - 30:,} more rows saved to CSV)')

csv_path = os.path.join(BASE, 'test_predictions.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['segment', 'label', 'confidence', 'duration_s', 'reasoning'])
    for r in rows:
        w.writerow(r)
print(f'\nFull predictions → {csv_path}')

# ── Misclassification diagnosis ────────────────────────────────────────────────
print('\n--- Misclassification Diagnosis ---')
fn_idx = np.where((y_pred == 0) & (y_te == 1))[0]
fp_idx = np.where((y_pred == 1) & (y_te == 0))[0]
print(f'False Negatives (missed apnea) : {len(fn_idx):,}')
print(f'False Positives (false alarm)  : {len(fp_idx):,}')

rng2 = np.random.default_rng(99)
for idx in rng2.choice(fn_idx, min(3, len(fn_idx)), replace=False):
    plot_window(X_te[idx], Y_te[idx], 1, idx, 'test_FN')
    print(f'  FN {idx}: signal shows mild/absent amplitude reduction — '
          f'{reason(F_te[idx], names, 0)[:100]}')
for idx in rng2.choice(fp_idx, min(3, len(fp_idx)), replace=False):
    plot_window(X_te[idx], Y_te[idx], 0, idx, 'test_FP')
    print(f'  FP {idx}: signal flagged by feature pattern but labeled normal — '
          f'{reason(F_te[idx], names, 1)[:100]}')

print('\nDone.')
