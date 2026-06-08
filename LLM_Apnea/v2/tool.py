#!/usr/bin/env python3
"""
tool.py — Signal analysis tools for BCG-based apnea detection.

Signals available:
  - Train: X, Y, THO, ABD (590 samples @ 10 Hz, z-scored), event_ts, stage_ts
  - Val/Test: X, Y only (590 samples @ 10 Hz, z-scored)

All feature extractors must work with X, Y only (the common denominator).
"""

import numpy as np
from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis
import pywt


FS = 10       # sampling rate after preprocessing (Hz)
SEQ_LEN = 590 # samples per channel


# ── Spectral features ────────────────────────────────────────────────────────

def spectral_features(sig, fs=FS):
    """PSD-based features from a 1D signal."""
    f, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)), noverlap=128)

    total_power = np.trapz(psd, f) + 1e-12

    # Respiratory band: 0.1–0.5 Hz (6–30 breaths/min)
    resp_mask = (f >= 0.1) & (f <= 0.5)
    resp_power = np.trapz(psd[resp_mask], f[resp_mask])
    resp_ratio = resp_power / total_power

    # Low-freq band: 0.01-0.1 Hz (body movement / apnea cessation)
    lf_mask = (f >= 0.01) & (f < 0.1)
    lf_power = np.trapz(psd[lf_mask], f[lf_mask])
    lf_ratio = lf_power / total_power

    # High-freq band: 0.5-2 Hz (cardiac, snoring artifacts)
    hf_mask = (f >= 0.5) & (f <= 2.0)
    hf_power = np.trapz(psd[hf_mask], f[hf_mask])
    hf_ratio = hf_power / total_power

    # Dominant frequency
    peak_freq = f[np.argmax(psd)]

    # Spectral entropy
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # Spectral edge frequency (95% of power)
    cum_power = np.cumsum(psd)
    sef95_idx = np.searchsorted(cum_power, 0.95 * cum_power[-1])
    sef95 = f[min(sef95_idx, len(f) - 1)]

    return {
        'resp_ratio': resp_ratio,
        'lf_ratio': lf_ratio,
        'hf_ratio': hf_ratio,
        'peak_freq': peak_freq,
        'spectral_entropy': spectral_entropy,
        'sef95': sef95,
    }


# ── Envelope / modulation features ───────────────────────────────────────────

def envelope_features(sig, fs=FS):
    """Hilbert envelope features — apnea causes amplitude modulation
    (periodic drops during cessation, surges during arousal)."""
    analytic = hilbert(sig)
    env = np.abs(analytic)

    # Envelope statistics
    env_mean = np.mean(env)
    env_cv = np.std(env) / (env_mean + 1e-8)

    # Split into segments and check for amplitude drops
    n_seg = 6  # ~10s segments
    seg_len = len(env) // n_seg
    seg_means = [np.mean(env[i * seg_len:(i + 1) * seg_len]) for i in range(n_seg)]
    seg_means = np.array(seg_means)

    # Modulation depth: max/min ratio of segment envelopes
    mod_depth = (seg_means.max() - seg_means.min()) / (env_mean + 1e-8)

    # Envelope trend: is there a crescendo-decrescendo pattern?
    # (common in post-apnea recovery)
    env_diff = np.diff(seg_means)
    n_sign_changes = np.sum(np.diff(np.sign(env_diff)) != 0)

    return {
        'env_cv': env_cv,
        'mod_depth': mod_depth,
        'env_sign_changes': float(n_sign_changes),
        'env_min_seg': float(seg_means.min()),
        'env_max_seg': float(seg_means.max()),
    }


# ── Amplitude / statistical features ─────────────────────────────────────────

def amplitude_features(sig):
    """Statistical features of signal amplitude."""
    return {
        'range': np.ptp(sig),
        'iqr': np.percentile(sig, 75) - np.percentile(sig, 25),
        'skewness': float(skew(sig)),
        'kurtosis': float(kurtosis(sig)),
        'zero_crossing_rate': np.sum(np.diff(np.sign(sig)) != 0) / len(sig),
        'p10': float(np.percentile(np.abs(sig), 10)),
        'p90': float(np.percentile(np.abs(sig), 90)),
        'p90_p10_ratio': float(np.percentile(np.abs(sig), 90) / (np.percentile(np.abs(sig), 10) + 1e-8)),
    }


# ── Temporal variability features ─────────────────────────────────────────────

def temporal_variability(sig, fs=FS):
    """Sliding-window energy variability at multiple scales."""
    features = {}

    for window_s, label in [(5, '5s'), (10, '10s'), (15, '15s')]:
        win = window_s * fs
        n_windows = len(sig) // win
        if n_windows < 2:
            features[f'energy_cv_{label}'] = 0.0
            continue

        energies = np.array([np.mean(sig[i * win:(i + 1) * win] ** 2) for i in range(n_windows)])
        mean_e = np.mean(energies) + 1e-12
        features[f'energy_cv_{label}'] = np.std(energies) / mean_e

    return features


# ── Autocorrelation features ─────────────────────────────────────────────────

def autocorrelation_features(sig, fs=FS):
    """Autocorrelation-based features — regular breathing produces strong
    periodic autocorrelation; apnea disrupts this."""
    sig_centered = sig - np.mean(sig)
    acf = np.correlate(sig_centered, sig_centered, mode='full')
    acf = acf[len(acf) // 2:]  # positive lags only
    acf = acf / (acf[0] + 1e-12)  # normalize

    # Find first peak in ACF after 1.5s lag (respiratory period)
    min_lag = int(1.5 * fs)
    max_lag = int(8.0 * fs)  # max ~8s breathing period
    acf_segment = acf[min_lag:min(max_lag, len(acf))]

    if len(acf_segment) < 5:
        return {'acf_peak': 0.0, 'acf_peak_lag_s': 0.0}

    peaks, _ = find_peaks(acf_segment, height=0.0)

    if len(peaks) == 0:
        return {'acf_peak': 0.0, 'acf_peak_lag_s': 0.0}

    best_peak = peaks[np.argmax(acf_segment[peaks])]
    acf_peak_val = acf_segment[best_peak]
    acf_peak_lag = (best_peak + min_lag) / fs

    return {
        'acf_peak': float(acf_peak_val),
        'acf_peak_lag_s': float(acf_peak_lag),
    }


# ── Respiratory rate estimation ───────────────────────────────────────────────

def respiratory_rate_features(sig, fs=FS):
    """Estimate breathing rate from peak-to-peak intervals."""
    peaks, _ = find_peaks(sig, distance=int(1.5 * fs), prominence=0.1)

    if len(peaks) < 3:
        return {
            'resp_rate_bpm': 0.0,
            'resp_rate_std': 0.0,
            'n_breaths': float(len(peaks)),
            'breath_regularity': 0.0,
        }

    intervals = np.diff(peaks) / fs
    resp_rate = 60.0 / np.mean(intervals)

    mean_int = np.mean(intervals)
    regularity = 1.0 - (np.std(intervals) / (mean_int + 1e-8))

    return {
        'resp_rate_bpm': resp_rate,
        'resp_rate_std': np.std(intervals) * 60.0 / (mean_int ** 2 + 1e-8),
        'n_breaths': float(len(peaks)),
        'breath_regularity': max(0, regularity),
    }


# ── Cross-channel features ───────────────────────────────────────────────────

def cross_channel_features(x_sig, y_sig):
    """Features capturing X-Y channel relationship."""
    corr = np.corrcoef(x_sig, y_sig)[0, 1]
    mag = np.sqrt(x_sig ** 2 + y_sig ** 2)

    # Phase coherence: do X and Y have similar envelope patterns?
    env_x = np.abs(hilbert(x_sig))
    env_y = np.abs(hilbert(y_sig))
    env_corr = np.corrcoef(env_x, env_y)[0, 1]

    return {
        'xy_correlation': corr if np.isfinite(corr) else 0.0,
        'xy_env_correlation': env_corr if np.isfinite(env_corr) else 0.0,
        'mag_std': np.std(mag),
    }


# ── Wavelet features ─────────────────────────────────────────────────────────

def wavelet_features(sig, wavelet='db4', level=4):
    """Multi-resolution wavelet decomposition features.
    At 10 Hz with db4 level 4:
      - cA4: 0-0.3125 Hz (very slow, body movement)
      - cD4: 0.3125-0.625 Hz (respiratory)
      - cD3: 0.625-1.25 Hz (cardiac fundamental)
      - cD2: 1.25-2.5 Hz (cardiac harmonics)
      - cD1: 2.5-5 Hz (high-freq noise/artifacts)
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    features = {}

    for i, c in enumerate(coeffs):
        band = f'wl{i}'  # 0=approx, 1-4=detail
        energy = np.mean(c ** 2)
        features[f'{band}_energy'] = energy
        features[f'{band}_entropy'] = float(-np.sum(
            (c ** 2 / (np.sum(c ** 2) + 1e-12)) *
            np.log2(c ** 2 / (np.sum(c ** 2) + 1e-12) + 1e-12)))

    # Ratio of respiratory band energy to total
    total_e = sum(np.mean(c ** 2) for c in coeffs) + 1e-12
    features['wl_resp_ratio'] = np.mean(coeffs[1] ** 2) / total_e  # cD4 = respiratory
    features['wl_cardiac_ratio'] = np.mean(coeffs[2] ** 2) / total_e  # cD3 = cardiac

    return features


# ── Spectrogram temporal features ─────────────────────────────────────────────

def spectrogram_features(sig, fs=FS):
    """Time-frequency features from short-time spectrograms.
    Captures how frequency content changes over time — apnea events
    show temporal disruption of respiratory spectral power."""
    from scipy.signal import spectrogram as stft

    f, t, Sxx = stft(sig, fs=fs, nperseg=64, noverlap=48, nfft=128)

    # Respiratory band power over time
    resp_mask = (f >= 0.1) & (f <= 0.5)
    resp_power_t = Sxx[resp_mask, :].sum(axis=0)

    # How variable is respiratory power across time?
    rp_mean = resp_power_t.mean() + 1e-12
    rp_cv = resp_power_t.std() / rp_mean

    # Does respiratory power drop then recover? (apnea signature)
    if len(resp_power_t) >= 3:
        rp_min_idx = np.argmin(resp_power_t)
        rp_dip_position = rp_min_idx / (len(resp_power_t) - 1)  # 0=start, 1=end
        rp_dip_depth = 1.0 - (resp_power_t.min() / rp_mean)
    else:
        rp_dip_position = 0.5
        rp_dip_depth = 0.0

    # Total spectral flux (change in spectral content between frames)
    flux = np.sqrt(np.mean(np.diff(Sxx, axis=1) ** 2))

    return {
        'stft_resp_cv': rp_cv,
        'stft_resp_dip_depth': rp_dip_depth,
        'stft_resp_dip_pos': rp_dip_position,
        'stft_spectral_flux': flux,
    }


# ── Master feature extractor ─────────────────────────────────────────────────

def extract_features(x_sig, y_sig):
    """Extract all features from X and Y signals. Returns a flat dict."""
    features = {}

    for prefix, sig in [('x', x_sig), ('y', y_sig)]:
        for name, val in spectral_features(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in envelope_features(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in amplitude_features(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in temporal_variability(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in autocorrelation_features(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in respiratory_rate_features(sig).items():
            features[f'{prefix}_{name}'] = val
        for name, val in wavelet_features(sig).items():
            features[f'{prefix}_{name}'] = val

    for name, val in cross_channel_features(x_sig, y_sig).items():
        features[name] = val

    return features


def features_to_array(feature_dict):
    """Convert feature dict to sorted numpy array (consistent ordering)."""
    keys = sorted(feature_dict.keys())
    return np.array([feature_dict[k] for k in keys]), keys


# ── Neural network approach ──────────────────────────────────────────────────

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Tiny 1D CNN for apnea detection from 2-channel BCG signals."""

    def __init__(self, in_channels=2, seq_len=590):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),         # 590 -> 147

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),         # 147 -> 36

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # 36 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x).squeeze(-1)


def train_cnn(X_train, y_train, X_val, y_val,
              epochs=200, lr=1e-3, weight_decay=1e-2, device='cuda:0'):
    """Train the small CNN with heavy regularisation."""
    model = SmallCNN().to(device)

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_bacc = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)

        # Data augmentation: add noise during training
        if epoch % 2 == 0:
            noise = torch.randn_like(X_tr) * 0.05
            logits_aug = model(X_tr + noise)
            loss = (loss + loss_fn(logits_aug, y_tr)) / 2

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Eval
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_preds = (val_logits > 0).cpu().numpy().astype(int)
                from sklearn.metrics import balanced_accuracy_score
                val_bacc = balanced_accuracy_score(y_val, val_preds)
                if val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_bacc
