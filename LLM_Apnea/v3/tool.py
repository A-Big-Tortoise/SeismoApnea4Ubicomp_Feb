#!/usr/bin/env python3
"""
tool.py
-------
DSP toolkit for BCG-based apnea detection.
Signal processing helpers: filtering, envelope extraction, peak detection,
sliding window statistics, and segment annotation.

All functions assume signals at 10 Hz (590 samples ≈ 59 s) after prepare.py processing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert


# ── Constants ─────────────────────────────────────────────────────────────────
FS = 10          # sampling rate (Hz)
WIN_S = 59       # window duration in seconds (590 samples)
N_SAMPLES = 590  # samples per channel


# ── Filtering ─────────────────────────────────────────────────────────────────

def bandpass_filter(sig, lo_hz, hi_hz, fs=FS, order=4):
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo_hz / nyq, hi_hz / nyq], btype='band')
    padlen = min(3 * max(len(b), len(a)), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=padlen)


def lowpass_filter(sig, cutoff_hz, fs=FS, order=4):
    """Butterworth low-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    padlen = min(3 * max(len(b), len(a)), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=padlen)


def respiratory_bandpass(sig, fs=FS):
    """Extract respiratory band (0.15–0.5 Hz) from BCG signal."""
    return bandpass_filter(sig, 0.15, 0.5, fs=fs)


# ── Envelope Extraction ──────────────────────────────────────────────────────

def amplitude_envelope(sig, window_s=3.0, fs=FS):
    """Compute smoothed amplitude envelope using sliding RMS."""
    win = int(window_s * fs)
    if win < 1:
        win = 1
    squared = sig ** 2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(squared, kernel, mode='same'))
    return rms


def hilbert_envelope(sig):
    """Amplitude envelope via Hilbert transform."""
    analytic = hilbert(sig)
    return np.abs(analytic)


# ── Peak Detection ────────────────────────────────────────────────────────────

def detect_respiratory_peaks(sig, fs=FS, min_breath_s=2.0, max_breath_s=8.0):
    """Detect respiratory peaks with physiological constraints.
    Normal breathing: 8–30 breaths/min → period 2–7.5 s."""
    min_dist = int(min_breath_s * fs)
    peaks, props = find_peaks(sig, distance=min_dist, prominence=0.1)
    return peaks, props


def peak_amplitudes(sig, peaks):
    """Get amplitudes at detected peaks."""
    if len(peaks) == 0:
        return np.array([])
    return sig[peaks]


# ── Sliding Window Statistics ─────────────────────────────────────────────────

def sliding_variance(sig, window_s=5.0, fs=FS):
    """Compute sliding window variance."""
    win = int(window_s * fs)
    if win < 1:
        win = 1
    out = np.zeros_like(sig)
    for i in range(len(sig)):
        start = max(0, i - win // 2)
        end = min(len(sig), i + win // 2 + 1)
        out[i] = np.var(sig[start:end])
    return out


def sliding_mean(sig, window_s=5.0, fs=FS):
    """Compute sliding window mean of absolute values."""
    win = int(window_s * fs)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    return np.convolve(np.abs(sig), kernel, mode='same')


def sliding_energy(sig, window_s=5.0, fs=FS):
    """Compute sliding window energy (sum of squares)."""
    win = int(window_s * fs)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    return np.convolve(sig ** 2, kernel, mode='same')


# ── Segment Analysis ─────────────────────────────────────────────────────────

def find_low_amplitude_segments(envelope, threshold_ratio=0.3, min_duration_s=10.0, fs=FS):
    """Find segments where envelope drops below threshold_ratio * median.
    Returns list of (start_sample, end_sample) tuples."""
    median_amp = np.median(envelope)
    threshold = threshold_ratio * median_amp
    below = envelope < threshold

    segments = []
    in_segment = False
    start = 0
    for i in range(len(below)):
        if below[i] and not in_segment:
            start = i
            in_segment = True
        elif not below[i] and in_segment:
            duration_s = (i - start) / fs
            if duration_s >= min_duration_s:
                segments.append((start, i))
            in_segment = False
    if in_segment:
        duration_s = (len(below) - start) / fs
        if duration_s >= min_duration_s:
            segments.append((start, len(below)))

    return segments


def amplitude_ratio(envelope, segment_percentile=10, baseline_percentile=50):
    """Compute ratio of low-amplitude segments to baseline.
    Lower ratio = more severe amplitude drop (apnea-like)."""
    low = np.percentile(envelope, segment_percentile)
    baseline = np.percentile(envelope, baseline_percentile)
    if baseline < 1e-8:
        return 1.0
    return low / baseline


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(sig, fs=FS):
    """Extract a dictionary of features from a single-channel signal."""
    env = amplitude_envelope(sig, window_s=3.0, fs=fs)
    resp = respiratory_bandpass(sig, fs=fs)
    peaks, _ = detect_respiratory_peaks(resp, fs=fs)
    peak_amps = peak_amplitudes(resp, peaks)

    features = {}

    # Envelope statistics
    features['env_mean'] = np.mean(env)
    features['env_std'] = np.std(env)
    features['env_min'] = np.min(env)
    features['env_max'] = np.max(env)
    features['env_range'] = features['env_max'] - features['env_min']
    features['env_cv'] = features['env_std'] / (features['env_mean'] + 1e-8)

    # Amplitude ratio (AASM-inspired)
    features['amp_ratio_10_50'] = amplitude_ratio(env, 10, 50)
    features['amp_ratio_5_50'] = amplitude_ratio(env, 5, 50)

    # Low-amplitude segment detection
    segments = find_low_amplitude_segments(env, threshold_ratio=0.3, min_duration_s=10.0, fs=fs)
    features['n_low_segments'] = len(segments)
    features['total_low_duration_s'] = sum((e - s) / fs for s, e in segments)

    # Peak statistics (respiratory regularity)
    if len(peak_amps) >= 2:
        features['rpv'] = np.var(peak_amps)  # Respiratory Peak Variance
        features['peak_amp_cv'] = np.std(peak_amps) / (np.mean(peak_amps) + 1e-8)
        ipi = np.diff(peaks) / fs  # inter-peak intervals
        features['breathing_rate'] = 60.0 / np.mean(ipi) if np.mean(ipi) > 0 else 0
        features['breathing_regularity'] = np.std(ipi) / (np.mean(ipi) + 1e-8)
    else:
        features['rpv'] = 0.0
        features['peak_amp_cv'] = 0.0
        features['breathing_rate'] = 0.0
        features['breathing_regularity'] = 0.0

    # Signal energy
    features['signal_energy'] = np.mean(sig ** 2)
    features['resp_energy'] = np.mean(resp ** 2)

    # Sliding variance statistics
    sv = sliding_variance(sig, window_s=5.0, fs=fs)
    features['var_mean'] = np.mean(sv)
    features['var_std'] = np.std(sv)
    features['var_min'] = np.min(sv)
    features['var_range'] = np.max(sv) - np.min(sv)

    return features


# ── Time Axis Helper ──────────────────────────────────────────────────────────

def time_axis(n_samples=N_SAMPLES, fs=FS):
    """Return time axis in seconds."""
    return np.arange(n_samples) / fs


# ── Data Loading Helper ──────────────────────────────────────────────────────

def load_example(path):
    """Load a single .npz example and return as dict."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
