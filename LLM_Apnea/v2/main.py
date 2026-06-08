#!/usr/bin/env python3
"""
main.py — Run apnea detection experiment.

Handcrafted features + PCA + Logistic Regression.
"""

import os
import json
import time
import numpy as np
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from tool import extract_features, features_to_array

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET  = os.path.join(BASE_DIR, 'dataset')


def load_split(split):
    """Load all examples from a split."""
    data_dir = os.path.join(DATASET, split, 'data')
    files = sorted(glob(os.path.join(data_dir, 'example_*.npz')))

    feat_list, labels, eids = [], [], []
    for fpath in files:
        d = np.load(fpath, allow_pickle=True)
        x_sig, y_sig = d['X'], d['Y']

        feat_dict = extract_features(x_sig, y_sig)
        feat_arr, _ = features_to_array(feat_dict)
        feat_list.append(feat_arr)

        eid = os.path.basename(fpath).replace('.npz', '')
        eids.append(eid)

        if 'apnea_label' in d:
            labels.append(int(d['apnea_label']))
        else:
            labels.append(None)

    return np.array(feat_list), labels, eids


def load_ground_truth(split):
    gt_path = os.path.join(DATASET, split, 'ground_truth.json')
    with open(gt_path) as f:
        return json.load(f)


def compute_metrics(y_true, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return bacc, f1, int(tp), int(fp), int(tn), int(fn)


def print_metrics(prefix, bacc, f1, tp, fp, tn, fn):
    print(f'{prefix}_bacc:   {bacc:.4f}')
    print(f'{prefix}_f1:     {f1:.4f}')
    print(f'{prefix}_cm:     TP={tp} FP={fp} TN={tn} FN={fn}')


def main():
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading train split...')
    X_train, y_train_raw, _ = load_split('train')
    y_train = np.array(y_train_raw)

    print('Loading val split...')
    X_val, _, val_eids = load_split('val')
    gt_val = load_ground_truth('val')
    y_val = np.array([gt_val[eid]['apnea_label'] for eid in val_eids])

    print(f'Features: {X_train.shape[1]}')

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train))
    X_val_s = np.nan_to_num(scaler.transform(X_val))

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca = PCA(n_components=15, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    print(f'PCA: {X_train_s.shape[1]} → {X_train_pca.shape[1]}, '
          f'explained var: {pca.explained_variance_ratio_.sum():.2%}')

    # ── Train ─────────────────────────────────────────────────────────────────
    print('Training LogReg...')
    clf = LogisticRegression(C=0.2, penalty='l2', solver='lbfgs',
                             max_iter=2000, random_state=42)
    clf.fit(X_train_pca, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_train_pred = clf.predict(X_train_pca)
    y_val_pred = clf.predict(X_val_pca)

    print()
    print_metrics('train', *compute_metrics(y_train, y_train_pred))
    print()
    print_metrics('val', *compute_metrics(y_val, y_val_pred))

    # ── VRAM ──────────────────────────────────────────────────────────────────
    print(f'\npeak_vram_mb: 0')
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
