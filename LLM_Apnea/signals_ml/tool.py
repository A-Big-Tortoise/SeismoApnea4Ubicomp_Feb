"""tool.py — DSP toolkit for AASM-grounded apnea classification."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

# ── Processing constants ───────────────────────────────────────────────────────
FS              = 10      # Hz (sample rate after generate_data.py pipeline)
RESP_LOW_HZ     = 0.10   # lower respiratory band edge
RESP_HIGH_HZ    = 0.45   # upper respiratory band edge
BASELINE_PCT    = 90     # envelope percentile used as amplitude baseline
APNEA_THRESH    = 0.10   # AASM: amplitude < 10% baseline → apnea territory
HYPOPNEA_THRESH = 0.70   # AASM: amplitude < 70% baseline → hypopnea territory
MIN_EVENT_S     = 10     # AASM: minimum duration (seconds) for a scored event


def bandpass_filter(signals, low_hz=RESP_LOW_HZ, high_hz=RESP_HIGH_HZ,
                    fs=FS, order=3):
    """Butterworth bandpass (SOS) with reflect padding; 1-D or 2-D, last axis."""
    nyq = 0.5 * fs
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    ndim = signals.ndim
    T    = signals.shape[-1]
    pad  = [(0, 0)] * (ndim - 1) + [(T, T)]
    out  = sosfiltfilt(sos, np.pad(signals, pad, mode='reflect'), axis=-1)
    return out[..., T:-T]


def compute_envelope_batch(signals):
    """Hilbert amplitude envelope [N, T] → [N, T]."""
    return np.abs(hilbert(signals, axis=1))


def compute_regularity_batch(signals):
    """FFT-based normalised autocorrelation peak in the respiratory lag range.

    Searches lags 10–200 samples (1–20 s @ 10 Hz).
    Higher → more periodic (healthy breathing).
    """
    N, T  = signals.shape
    sig   = signals - signals.mean(axis=1, keepdims=True)
    nfft  = 1 << int(np.ceil(np.log2(2 * T)))
    F     = np.fft.rfft(sig, n=nfft, axis=1)
    ac    = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :T]
    ac   /= (ac[:, 0:1] + 1e-8)
    hi    = min(201, T)
    return ac[:, 10:hi].max(axis=1)


def compute_resp_power_batch(signals, fs=FS,
                              low=RESP_LOW_HZ, high=RESP_HIGH_HZ):
    """Fraction of spectral power inside the respiratory band [N,]."""
    T     = signals.shape[1]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signals, axis=1)) ** 2
    mask  = (freqs >= low) & (freqs <= high)
    return power[:, mask].sum(axis=1) / (power.sum(axis=1) + 1e-8)


def longest_run_batch(mask_2d):
    """Vectorised longest consecutive True run per row [N, T] → [N,].

    Uses cumulative-sum differencing; no Python loops.
    """
    m      = mask_2d.astype(np.int64)
    cs     = np.cumsum(m, axis=1)
    cs_f   = np.where(mask_2d, 0, cs)                    # cs at False positions
    cmax   = np.maximum.accumulate(cs_f, axis=1)          # running max of cs_f
    streak = np.where(mask_2d, cs - cmax, 0)
    return streak.max(axis=1)


def extract_features_batch(X_batch, Y_batch, fs=FS):
    """Extract AASM-inspired features for a batch of windows.

    Parameters
    ----------
    X_batch : [N, 590] float64 — BCG X-axis, z-scored @ 10 Hz
    Y_batch : [N, 590] float64 — BCG Y-axis, z-scored @ 10 Hz

    Returns
    -------
    arr   : [N, n_feat] float64
    names : list[str] — sorted feature names, same order as arr columns
    """
    min_s  = int(MIN_EVENT_S * fs)
    X_resp = bandpass_filter(X_batch)
    Y_resp = bandpass_filter(Y_batch)

    res = {}
    for p, resp in [('x', X_resp), ('y', Y_resp)]:
        env  = compute_envelope_batch(resp)                        # [N, T]
        base = np.percentile(env, BASELINE_PCT, axis=1)[:, None]  # [N, 1]
        safe = np.maximum(base, 1e-8)

        res[f'{p}_env_mean']      = env.mean(axis=1)
        res[f'{p}_env_std']       = env.std(axis=1)
        res[f'{p}_env_cv']        = env.std(axis=1) / (env.mean(axis=1) + 1e-8)
        res[f'{p}_env_min_ratio'] = env.min(axis=1) / safe[:, 0]

        a_mask = env < (APNEA_THRESH    * safe)
        res[f'{p}_apnea_frac']    = a_mask.mean(axis=1)
        a_runs = longest_run_batch(a_mask)
        res[f'{p}_max_apnea_s']   = a_runs / fs
        res[f'{p}_has_apnea']     = (a_runs >= min_s).astype(float)

        h_mask = env < (HYPOPNEA_THRESH * safe)
        res[f'{p}_hypopnea_frac'] = h_mask.mean(axis=1)
        h_runs = longest_run_batch(h_mask)
        res[f'{p}_max_hypopnea_s']= h_runs / fs
        res[f'{p}_has_hypopnea']  = (h_runs >= min_s).astype(float)

        res[f'{p}_regularity']    = compute_regularity_batch(resp)
        res[f'{p}_resp_power']    = compute_resp_power_batch(resp, fs)

    # ── Combined X+Y features (require both channels to agree) ────────────────
    res['xy_has_apnea']      = np.minimum(res['x_has_apnea'],      res['y_has_apnea'])
    res['xy_has_hypopnea']   = np.minimum(res['x_has_hypopnea'],   res['y_has_hypopnea'])
    res['xy_max_apnea_s']    = np.minimum(res['x_max_apnea_s'],    res['y_max_apnea_s'])
    res['xy_max_hypopnea_s'] = np.minimum(res['x_max_hypopnea_s'], res['y_max_hypopnea_s'])
    res['xy_regularity']     = (res['x_regularity'] + res['y_regularity']) / 2
    res['xy_env_cv']         = (res['x_env_cv']      + res['y_env_cv'])      / 2

    # ── Sliding-window amplitude collapse: worst 10-s block ───────────────────
    # Mean envelope in each non-overlapping 10-s (100-sample) block → min block
    # Normalized to baseline: low ratio = sustained collapse (apnea indicator)
    X_env_full = compute_envelope_batch(X_resp)
    Y_env_full = compute_envelope_batch(Y_resp)
    T = X_env_full.shape[1]
    block = int(MIN_EVENT_S * fs)   # 100 samples
    n_blocks = T // block
    X_blocks = X_env_full[:, :n_blocks*block].reshape(-1, n_blocks, block).mean(axis=2)
    Y_blocks = Y_env_full[:, :n_blocks*block].reshape(-1, n_blocks, block).mean(axis=2)
    x_base = np.percentile(X_env_full, BASELINE_PCT, axis=1, keepdims=True)
    y_base = np.percentile(Y_env_full, BASELINE_PCT, axis=1, keepdims=True)
    # Block-minimum ratio to baseline
    res['x_block_min_ratio'] = X_blocks.min(axis=1) / (x_base[:, 0] + 1e-8)
    res['y_block_min_ratio'] = Y_blocks.min(axis=1) / (y_base[:, 0] + 1e-8)
    # Block coefficient of variation (measures within-window amplitude swings)
    res['x_block_cv']  = X_blocks.std(axis=1) / (X_blocks.mean(axis=1) + 1e-8)
    res['y_block_cv']  = Y_blocks.std(axis=1) / (Y_blocks.mean(axis=1) + 1e-8)
    res['xy_block_min_ratio'] = np.minimum(res['x_block_min_ratio'],
                                            res['y_block_min_ratio'])

    # ── Amplitude recovery: block after the nadir vs the nadir itself ─────────
    # High recovery ratio = amplitude rebounds after the drop (apnea-like pattern)
    for prefix_b, blk, bse in [('x', X_blocks, x_base[:, 0]),
                                 ('y', Y_blocks, y_base[:, 0])]:
        nadir_idx = blk.argmin(axis=1)            # index of quietest block [N,]
        nadir_val = blk.min(axis=1)               # amplitude at nadir [N,]
        # Block after nadir (clamp to last block)
        after_idx = np.minimum(nadir_idx + 1, n_blocks - 1)
        after_val = blk[np.arange(len(blk)), after_idx]
        res[f'{prefix_b}_recovery_ratio'] = after_val / (nadir_val + 1e-8)
        # Block before nadir (clamp to first)
        before_idx = np.maximum(nadir_idx - 1, 0)
        before_val = blk[np.arange(len(blk)), before_idx]
        res[f'{prefix_b}_drop_ratio'] = nadir_val / (before_val + 1e-8)

    # ── IQR of envelope: robustly captures amplitude spread ───────────────────
    res['x_env_iqr'] = (np.percentile(X_env_full, 75, axis=1) -
                         np.percentile(X_env_full, 25, axis=1))
    res['y_env_iqr'] = (np.percentile(Y_env_full, 75, axis=1) -
                         np.percentile(Y_env_full, 25, axis=1))

    names = sorted(res.keys())
    arr   = np.column_stack([res[k] for k in names])
    arr   = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, names
