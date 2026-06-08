"""
Batch BBI-based lag estimation for multiple patients.

Reads patient URLs from Mapping/SleepLab_Lag.xlsx (Logs sheet, New_URL column).
For each patient: randomly samples 15 segments (3-5 min), filters for stable ones
(BBI correlation >= 0.8), computes lag (4 shifts, drop max/min, average middle 2).

Usage:
    python Mapping/batch_estimate_lag.py

Output:
    Mapping/lag_results.json
"""

import numpy as np
import pandas as pd
import time
import json
import random
import warnings
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from statsmodels.tsa.stattools import acf
from influxdb import InfluxDBClient
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")

PLOT_DIR = 'Mapping/lag_plots'

# ============================================================
# InfluxDB configs
# ============================================================
INFLUX_BSG = {
    'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake',
    'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True
}
INFLUX_SLEEP = {
    'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab',
    'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True
}

EXCEL_PATH = 'Mapping/SleepLab_Lag.xlsx'

# ============================================================
# Signal processing functions (from estimate_lag.ipynb)
# ============================================================

def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
        client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'], ssl=influx['ssl'])
    else:
        client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'], ssl=influx['ssl'])
    query = ('SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \'' + unit +
             '\' AND time >= ' + str(int(start_timestamp * 10e8)) + ' AND time < ' + str(int(end_timestamp * 10e8)))
    result = client.query(query)
    points = list(result.get_points())
    values = list(map(operator.itemgetter(data_name), points))
    data = np.array(values)
    return data


def low_pass_filter(data, Fs, low, order):
    b, a = butter(order, low / (Fs * 0.5), 'low')
    if data.ndim == 1:
        N = len(data)
        padded = np.pad(data, (N, N), mode='reflect')
    elif data.ndim == 2:
        N = data.shape[1]
        padded = np.pad(data, ((0, 0), (N, N)), mode='reflect')
    filtered_data = filtfilt(b, a, padded)
    if data.ndim == 1:
        filtered_data = filtered_data[N:-N]
    elif data.ndim == 2:
        filtered_data = filtered_data[:, N:-N]
    return filtered_data


def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        return (signal - mean) / std
    return signal - mean


def get_bbi(sig, Fs=100, low=0.8):
    s = low_pass_filter(sig, Fs, low, order=3)
    acf_vals = acf(s, nlags=Fs * 30, fft=True)
    acf_peaks, _ = find_peaks(acf_vals, distance=Fs * 2)
    if len(acf_peaks) > 0:
        period_samples = acf_peaks[0]
    else:
        period_samples = Fs * 4
    min_dist = int(period_samples * 0.6)
    peaks, _ = find_peaks(s, distance=min_dist,
                          height=np.mean(s) + 0.3 * np.std(s))
    bbi = np.diff(peaks) / Fs
    bbi_times = (peaks[:-1] + np.diff(peaks) / 2) / Fs
    return peaks, bbi, bbi_times, s


def align_bbi_by_shift(bbi_times_bcg, bbi_bcg, bbi_times_ref, bbi_ref,
                       search_range=(-5.0, 5.0)):
    f_bcg = interp1d(bbi_times_bcg, bbi_bcg, kind='linear',
                     bounds_error=False, fill_value=np.nan)
    f_ref = interp1d(bbi_times_ref, bbi_ref, kind='linear',
                     bounds_error=False, fill_value=np.nan)
    t_lo = max(bbi_times_bcg[0], bbi_times_ref[0]) + abs(search_range[0])
    t_hi = min(bbi_times_bcg[-1], bbi_times_ref[-1]) - abs(search_range[1])
    if t_lo >= t_hi:
        return 0.0, 0.0, np.array([]), np.array([])
    t_common = np.linspace(t_lo, t_hi, 300)
    ref_interp = f_ref(t_common)

    def neg_corr(shift):
        bcg_shifted = f_bcg(t_common + shift)
        valid = ~(np.isnan(bcg_shifted) | np.isnan(ref_interp))
        if valid.sum() < 5:
            return 0.0
        return -np.corrcoef(bcg_shifted[valid], ref_interp[valid])[0, 1]

    shifts = np.linspace(search_range[0], search_range[1], 500)
    corrs = np.array([-neg_corr(s) for s in shifts])
    best_idx = np.argmax(corrs)
    res = minimize_scalar(neg_corr,
                          bounds=(shifts[max(0, best_idx - 5)],
                                  shifts[min(len(shifts) - 1, best_idx + 5)]),
                          method='bounded')
    return res.x, -res.fun, shifts, corrs


# ============================================================
# Helper: parse URL
# ============================================================

def parse_url(url):
    mac = url.split('var-mac=')[1].split('&')[0]
    start_ms = int(url.split('from=')[1].split('&')[0])
    end_ms = int(url.split('to=')[1].split('&')[0])
    start_time = start_ms // 1000
    end_time = end_ms // 1000
    room = url.split('/d/')[1].split('/')[0][-1]
    if room == '2':
        x_axis, y_axis = 'E', 'N'
    else:
        x_axis, y_axis = 'X', 'Y'
    return mac, start_time, end_time, x_axis, y_axis


# ============================================================
# Load patients from Excel
# ============================================================

def load_patients_from_excel(path=EXCEL_PATH):
    df = pd.read_excel(path, sheet_name='Logs')
    # Filter rows with valid New_URL
    df = df.dropna(subset=['New_URL'])
    df = df[df['New_URL'].str.contains('from=', na=False)]
    patients = []
    for _, row in df.iterrows():
        patients.append({
            'id': int(row['ID']),
            'name': str(row['Identifier']),
            'url': str(row['New_URL']),
        })
    return patients


# ============================================================
# Helper: fetch one segment (4 signals) with rate limiting
# ============================================================

query_counter = 0

def fetch_segment(mac, seg_start, seg_end, x_axis, y_axis):
    """Fetch X, Y, THO, ABD for one time segment. Returns (raw_X, raw_Y, tho, abd) or None."""
    global query_counter
    try:
        raw_X = read_influx(INFLUX_BSG, unit=mac, table_name=x_axis,
                            data_name="value", start_timestamp=seg_start, end_timestamp=seg_end)
        query_counter += 1
        raw_Y = read_influx(INFLUX_BSG, unit=mac, table_name=y_axis,
                            data_name="value", start_timestamp=seg_start, end_timestamp=seg_end)
        query_counter += 1
        tho_sig = read_influx(INFLUX_SLEEP, unit=mac, table_name='SleepLab_test',
                              data_name="Effort THO", start_timestamp=seg_start, end_timestamp=seg_end)
        query_counter += 1
        abd_sig = read_influx(INFLUX_SLEEP, unit=mac, table_name='SleepLab_test',
                              data_name="Effort ABD", start_timestamp=seg_start, end_timestamp=seg_end)
        query_counter += 1

        # Rate limit: sleep 5s every 5 queries
        if query_counter % 5 == 0:
            time.sleep(5)

        # Validate lengths
        min_len = min(len(raw_X), len(raw_Y), len(tho_sig), len(abd_sig))
        if min_len < 100 * 60:  # less than 1 minute usable
            return None
        raw_X = raw_X[:min_len]
        raw_Y = raw_Y[:min_len]
        tho_sig = tho_sig[:min_len]
        abd_sig = abd_sig[:min_len]
        return raw_X, raw_Y, tho_sig, abd_sig
    except Exception as e:
        print(f"    [WARN] fetch failed: {e}")
        return None


# ============================================================
# Helper: check segment quality
# ============================================================

def is_stable_segment(raw_X, raw_Y, tho_sig, abd_sig, Fs=100, corr_threshold=0.8):
    """
    Check if a segment is stable.
    Returns (is_stable, best_corr, reason).
    reason is '' if stable, otherwise a string describing why it was rejected.
    """
    try:
        # Kurtosis check
        for sig, lbl in [(raw_X, 'X'), (raw_Y, 'Y')]:
            k = kurtosis(sig)
            if k > 5:
                return False, 0.0, f'kurtosis_{lbl}={k:.1f}'

        # X-Y low-freq alignment check: they should be in phase (from same source)
        lf_x = low_pass_filter(raw_X, Fs, 0.8, order=3)
        lf_y = low_pass_filter(raw_Y, Fs, 0.8, order=3)
        xy_corr = np.corrcoef(lf_x, lf_y)[0, 1]
        if abs(xy_corr) < 0.5:
            return False, 0.0, f'XY_LF_shift_r={xy_corr:.3f}'

        # Spike check
        for sig, lbl in [(raw_X, 'X'), (raw_Y, 'Y'), (tho_sig, 'THO'), (abd_sig, 'ABD')]:
            z = np.abs((sig - np.mean(sig)) / (np.std(sig) + 1e-10))
            if np.max(z) > 8:
                return False, 0.0, f'spike_{lbl}_z={np.max(z):.1f}'

        _, bbi_x, t_x, _ = get_bbi(raw_X, Fs=Fs)
        _, bbi_y, t_y, _ = get_bbi(raw_Y, Fs=Fs)
        _, bbi_tho, t_tho, _ = get_bbi(tho_sig, Fs=Fs, low=0.8)
        _, bbi_abd, t_abd, _ = get_bbi(abd_sig, Fs=Fs, low=0.8)

        if len(bbi_x) < 5 or len(bbi_y) < 5 or len(bbi_tho) < 5 or len(bbi_abd) < 5:
            return False, 0.0, 'too_few_BBI_peaks'

        # BBI outlier check
        for bbi, lbl in [(bbi_x, 'X'), (bbi_y, 'Y'), (bbi_tho, 'THO'), (bbi_abd, 'ABD')]:
            med = np.median(bbi)
            if np.any(bbi > 3 * med) or np.any(bbi < med / 3):
                return False, 0.0, f'BBI_outlier_{lbl}'

        t_lo = max(t_x[0], t_y[0], t_tho[0], t_abd[0])
        t_hi = min(t_x[-1], t_y[-1], t_tho[-1], t_abd[-1])
        if t_hi - t_lo < 30:
            return False, 0.0, 'BBI_overlap<30s'

        t_common = np.linspace(t_lo, t_hi, 100)

        f_x = interp1d(t_x, bbi_x, kind='linear', bounds_error=False, fill_value=np.nan)
        f_y = interp1d(t_y, bbi_y, kind='linear', bounds_error=False, fill_value=np.nan)
        f_tho = interp1d(t_tho, bbi_tho, kind='linear', bounds_error=False, fill_value=np.nan)
        f_abd = interp1d(t_abd, bbi_abd, kind='linear', bounds_error=False, fill_value=np.nan)

        corrs = []
        for f_bcg, f_ref in [(f_x, f_tho), (f_x, f_abd), (f_y, f_tho), (f_y, f_abd)]:
            b = f_bcg(t_common)
            r = f_ref(t_common)
            valid = ~(np.isnan(b) | np.isnan(r))
            if valid.sum() < 10:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(b[valid], r[valid])[0, 1])

        best_corr = max(corrs)
        if best_corr >= corr_threshold:
            return True, best_corr, ''
        return False, best_corr, f'low_corr={best_corr:.3f}'

    except Exception as e:
        return False, 0.0, f'exception: {e}'


# ============================================================
# Helper: compute lag for one stable segment
# ============================================================

def compute_segment_lag(raw_X, raw_Y, tho_sig, abd_sig, Fs=100):
    """
    Compute 4 BBI shifts, remove max and min, average the middle 2.
    Returns (lag_seconds, detail_dict) or (None, None) on failure.
    """
    try:
        _, bbi_x, t_x, _ = get_bbi(raw_X, Fs=Fs)
        _, bbi_y, t_y, _ = get_bbi(raw_Y, Fs=Fs)
        _, bbi_tho, t_tho, _ = get_bbi(tho_sig, Fs=Fs, low=0.8)
        _, bbi_abd, t_abd, _ = get_bbi(abd_sig, Fs=Fs, low=0.8)

        shift_x_tho, corr_x_tho, _, _ = align_bbi_by_shift(t_x, bbi_x, t_tho, bbi_tho)
        shift_x_abd, corr_x_abd, _, _ = align_bbi_by_shift(t_x, bbi_x, t_abd, bbi_abd)
        shift_y_tho, corr_y_tho, _, _ = align_bbi_by_shift(t_y, bbi_y, t_tho, bbi_tho)
        shift_y_abd, corr_y_abd, _, _ = align_bbi_by_shift(t_y, bbi_y, t_abd, bbi_abd)

        shifts = [shift_x_tho, shift_x_abd, shift_y_tho, shift_y_abd]
        corrs = [corr_x_tho, corr_x_abd, corr_y_tho, corr_y_abd]

        # Remove max and min, average middle 2
        sorted_shifts = sorted(shifts)
        lag = (sorted_shifts[1] + sorted_shifts[2]) / 2.0

        detail = {
            'shifts': {'x_tho': shift_x_tho, 'x_abd': shift_x_abd,
                       'y_tho': shift_y_tho, 'y_abd': shift_y_abd},
            'corrs': {'x_tho': corr_x_tho, 'x_abd': corr_x_abd,
                      'y_tho': corr_y_tho, 'y_abd': corr_y_abd},
            'lag_trimmed_mean': lag
        }
        return lag, detail
    except Exception as e:
        print(f"    [WARN] lag computation failed: {e}")
        return None, None


# ============================================================
# Plotting: save diagnostic figures for each stable segment
# ============================================================

def plot_segment(raw_X, raw_Y, tho_sig, abd_sig, pid, name, seg_idx,
                 lag, detail, Fs=100):
    """Save a 4-panel diagnostic figure for one stable segment."""
    low_cut = 0.8
    shift_samples = int(round(lag * Fs))

    # BBI for all 4 signals
    peaks_x, bbi_x, t_x, filt_x = get_bbi(raw_X, Fs=Fs)
    peaks_y, bbi_y, t_y, filt_y = get_bbi(raw_Y, Fs=Fs)
    peaks_tho, bbi_tho, t_tho, filt_tho = get_bbi(tho_sig, Fs=Fs, low=0.8)
    peaks_abd, bbi_abd, t_abd, filt_abd = get_bbi(abd_sig, Fs=Fs, low=0.8)

    # Correlation vs shift curves
    _, _, shifts_x_tho, corrs_x_tho = align_bbi_by_shift(t_x, bbi_x, t_tho, bbi_tho)
    _, _, shifts_x_abd, corrs_x_abd = align_bbi_by_shift(t_x, bbi_x, t_abd, bbi_abd)
    _, _, shifts_y_tho, corrs_y_tho = align_bbi_by_shift(t_y, bbi_y, t_tho, bbi_tho)
    _, _, shifts_y_abd, corrs_y_abd = align_bbi_by_shift(t_y, bbi_y, t_abd, bbi_abd)

    s = detail['shifts']
    c = detail['corrs']

    fig, axes = plt.subplots(4, 1, figsize=(14, 18))

    # --- Panel 1: Filtered signals with detected peaks ---
    ax = axes[0]
    for sig, pk, lbl, color in [
        (filt_x, peaks_x, 'BCG X', 'tab:blue'),
        (filt_y, peaks_y, 'BCG Y', 'tab:orange'),
        (filt_tho, peaks_tho, 'THO', 'tab:green'),
        (filt_abd, peaks_abd, 'ABD', 'tab:red'),
    ]:
        t = np.arange(len(sig)) / Fs
        ax.plot(t, normalize_signal(sig), color=color, alpha=0.6, label=lbl)
        ax.plot(pk / Fs, normalize_signal(sig)[pk], 'kx', markersize=4)
    ax.set_ylabel('Normalized')
    ax.set_title(f'[{pid}] {name} - Seg {seg_idx}: Detected Breathing Peaks')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: BBI curves overlaid ---
    ax = axes[1]
    ax.plot(t_x, bbi_x, 'o-', label='BCG X', alpha=0.7, markersize=3)
    ax.plot(t_y, bbi_y, 's-', label='BCG Y', alpha=0.7, markersize=3)
    ax.plot(t_tho, bbi_tho, '^-', label='THO', alpha=0.7, markersize=3)
    ax.plot(t_abd, bbi_abd, 'v-', label='ABD', alpha=0.7, markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BBI (s)')
    ax.set_title('Respiration Interval Variation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Correlation vs shift ---
    ax = axes[2]
    if len(shifts_x_tho) > 0:
        ax.plot(shifts_x_tho, corrs_x_tho, label=f'X-THO ({s["x_tho"]:+.2f}s, r={c["x_tho"]:.3f})')
        ax.plot(shifts_x_abd, corrs_x_abd, label=f'X-ABD ({s["x_abd"]:+.2f}s, r={c["x_abd"]:.3f})')
        ax.plot(shifts_y_tho, corrs_y_tho, label=f'Y-THO ({s["y_tho"]:+.2f}s, r={c["y_tho"]:.3f})')
        ax.plot(shifts_y_abd, corrs_y_abd, label=f'Y-ABD ({s["y_abd"]:+.2f}s, r={c["y_abd"]:.3f})')
    ax.axvline(lag, color='k', ls='--', alpha=0.5, label=f'lag={lag:+.3f}s')
    ax.set_xlabel('Time shift (s)')
    ax.set_ylabel('Pearson r')
    ax.set_title('BBI Correlation vs Time Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Low-freq waveforms before/after alignment ---
    ax = axes[3]
    lf_x = normalize_signal(low_pass_filter(raw_X, Fs, low_cut, order=3))
    lf_y = normalize_signal(low_pass_filter(raw_Y, Fs, low_cut, order=3))
    lf_tho = normalize_signal(low_pass_filter(tho_sig, Fs, low_cut, order=3))
    lf_abd = normalize_signal(low_pass_filter(abd_sig, Fs, low_cut, order=3))

    N = len(lf_x)
    if shift_samples > 0:
        bx_a, by_a = lf_x[shift_samples:], lf_y[shift_samples:]
        tho_a, abd_a = lf_tho[:N - shift_samples], lf_abd[:N - shift_samples]
    elif shift_samples < 0:
        ss = abs(shift_samples)
        bx_a, by_a = lf_x[:N - ss], lf_y[:N - ss]
        tho_a, abd_a = lf_tho[ss:], lf_abd[ss:]
    else:
        bx_a, by_a, tho_a, abd_a = lf_x, lf_y, lf_tho, lf_abd

    t_a = np.arange(len(bx_a)) / Fs
    ax.plot(t_a, bx_a, label='BCG X (shifted)', alpha=0.7)
    ax.plot(t_a, by_a, label='BCG Y (shifted)', alpha=0.7)
    ax.plot(t_a, tho_a, label='THO', alpha=0.7, linewidth=1.5)
    ax.plot(t_a, abd_a, label='ABD', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title(f'Low-Freq (<{low_cut} Hz) After Alignment (shift={lag:+.3f}s = {shift_samples:+d} samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    patient_dir = os.path.join(PLOT_DIR, f'{pid:03d}_{name}')
    os.makedirs(patient_dir, exist_ok=True)
    fig_path = os.path.join(patient_dir, f'seg_{seg_idx:02d}_ok.png')
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"    -> saved {fig_path}")


def plot_rejected_segment(raw_X, raw_Y, tho_sig, abd_sig, pid, name, seg_idx,
                          reason, Fs=100):
    """Save a simple low-freq plot for a rejected segment."""
    low_cut = 0.8
    lf_x = normalize_signal(low_pass_filter(raw_X, Fs, low_cut, order=3))
    lf_y = normalize_signal(low_pass_filter(raw_Y, Fs, low_cut, order=3))
    lf_tho = normalize_signal(low_pass_filter(tho_sig, Fs, low_cut, order=3))
    lf_abd = normalize_signal(low_pass_filter(abd_sig, Fs, low_cut, order=3))

    t = np.arange(len(lf_x)) / Fs

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(t, lf_x, label='BCG X (LF)', alpha=0.7)
    ax.plot(t, lf_y, label='BCG Y (LF)', alpha=0.7)
    ax.plot(t, lf_tho, label='THO (LF)', alpha=0.7, linewidth=1.5)
    ax.plot(t, lf_abd, label='ABD (LF)', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title(f'[{pid}] {name} - Seg {seg_idx}: REJECTED ({reason})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    patient_dir = os.path.join(PLOT_DIR, f'{pid:03d}_{name}')
    os.makedirs(patient_dir, exist_ok=True)
    fig_path = os.path.join(patient_dir, f'seg_{seg_idx:02d}_rejected.png')
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


# ============================================================
# Main: process all patients
# ============================================================

def find_clean_windows(mac, x_axis, start_time, end_time, Fs=100,
                       win_sec=300, stride_sec=60, kurt_thresh=5.0):
    """
    Pre-scan: fetch entire night X signal, compute rolling kurtosis,
    return list of (win_start, win_end) timestamps for clean windows.
    """
    global query_counter
    print(f"  Pre-scanning X signal for clean windows ...")
    raw_X = read_influx(INFLUX_BSG, unit=mac, table_name=x_axis,
                        data_name="value", start_timestamp=start_time, end_timestamp=end_time)
    query_counter += 1
    if query_counter % 5 == 0:
        time.sleep(5)

    N = len(raw_X)
    win_samples = win_sec * Fs
    stride_samples = stride_sec * Fs

    clean_windows = []
    pos = 0
    while pos + win_samples <= N:
        chunk = raw_X[pos:pos + win_samples]
        k = kurtosis(chunk)
        if k <= kurt_thresh:
            t_start = start_time + pos // Fs
            t_end = t_start + win_sec
            clean_windows.append((t_start, t_end, k))
        pos += stride_samples

    print(f"  Found {len(clean_windows)} clean windows (kurtosis <= {kurt_thresh}) "
          f"out of {(N - win_samples) // stride_samples + 1} total")
    return clean_windows


def process_patient(patient, n_samples=30, seg_min_sec=180, seg_max_sec=360,
                    Fs=100, corr_threshold=0.8, seed=42):
    pid = patient['id']
    name = patient['name']
    url = patient['url']
    mac, start_time, end_time, x_axis, y_axis = parse_url(url)
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"[{pid}] {name}")
    print(f"  MAC: {mac}, duration: {total_duration}s ({total_duration/3600:.1f}h)")

    # Pre-scan to find clean windows
    clean_windows = find_clean_windows(mac, x_axis, start_time, end_time,
                                       Fs=Fs, win_sec=300, stride_sec=60)

    if len(clean_windows) == 0:
        print(f"  WARNING: no clean windows found for {name}")
        return {
            'id': pid, 'name': name, 'url': url, 'mac': mac,
            'n_stable': 0, 'n_sampled': 0, 'stable_lags': [],
            'median_lag_sec': None, 'segment_details': []
        }

    # Sort by kurtosis (cleanest first), pick up to n_samples
    clean_windows.sort(key=lambda x: x[2])
    # Remove overlapping windows: keep cleanest, skip if overlaps with already selected
    selected = []
    for w_start, w_end, k in clean_windows:
        overlap = False
        for s_start, s_end, _ in selected:
            if w_start < s_end and w_end > s_start:
                overlap = True
                break
        if not overlap:
            selected.append((w_start, w_end, k))
        if len(selected) >= n_samples:
            break

    # Randomize segment length within each clean window
    rng = random.Random(seed + pid)
    segments = []
    for w_start, w_end, k in selected:
        seg_len = rng.randint(seg_min_sec, seg_max_sec)
        win_dur = w_end - w_start
        if seg_len > win_dur:
            seg_len = win_dur
        max_offset = win_dur - seg_len
        offset = rng.randint(0, max(0, max_offset))
        seg_start = w_start + offset
        seg_end = seg_start + seg_len
        segments.append((seg_start, seg_end, k))

    print(f"  Selected {len(segments)} clean segments (kurtosis range: "
          f"{segments[0][2]:.1f} - {segments[-1][2]:.1f})")

    stable_lags = []
    segment_details = []

    for i, (seg_start, seg_end, pre_kurt) in enumerate(segments):
        seg_dur = seg_end - seg_start
        print(f"  Segment {i+1}/{len(segments)}: {seg_dur}s (pre-kurt={pre_kurt:.1f}) ...", end=' ')

        data = fetch_segment(mac, seg_start, seg_end, x_axis, y_axis)
        if data is None:
            print("SKIP (fetch failed)")
            continue

        raw_X, raw_Y, tho_sig, abd_sig = data
        stable, best_corr, reason = is_stable_segment(raw_X, raw_Y, tho_sig, abd_sig,
                                                       Fs=Fs, corr_threshold=corr_threshold)
        if not stable:
            print(f"SKIP ({reason})")
            plot_rejected_segment(raw_X, raw_Y, tho_sig, abd_sig, pid, name,
                                  seg_idx=i+1, reason=reason, Fs=Fs)
            continue

        lag, detail = compute_segment_lag(raw_X, raw_Y, tho_sig, abd_sig, Fs=Fs)
        if lag is None:
            print("SKIP (lag computation failed)")
            plot_rejected_segment(raw_X, raw_Y, tho_sig, abd_sig, pid, name,
                                  seg_idx=i+1, reason='lag_failed', Fs=Fs)
            continue

        print(f"OK  lag={lag:+.3f}s, corr={best_corr:.3f}")
        stable_lags.append(lag)
        detail['seg_start'] = seg_start
        detail['seg_end'] = seg_end
        detail['best_bbi_corr'] = best_corr
        segment_details.append(detail)

        # Save diagnostic plot
        plot_segment(raw_X, raw_Y, tho_sig, abd_sig, pid, name,
                     seg_idx=i+1, lag=lag, detail=detail, Fs=Fs)

    if len(stable_lags) == 0:
        print(f"  WARNING: no stable segments found for {name}")
        patient_lag = None
    else:
        patient_lag = float(np.median(stable_lags))
        print(f"  --- {name}: {len(stable_lags)} stable segments, "
              f"median lag = {patient_lag:+.3f}s ({patient_lag*100:+.1f} samples)")

    return {
        'id': pid,
        'name': name,
        'url': url,
        'mac': mac,
        'n_stable': len(stable_lags),
        'n_sampled': len(segments),
        'stable_lags': stable_lags,
        'median_lag_sec': patient_lag,
        'segment_details': segment_details
    }


def main(target_ids=None):
    patients = load_patients_from_excel()
    print(f"Loaded {len(patients)} patients from {EXCEL_PATH}")

    if target_ids is not None:
        patients = [p for p in patients if p['id'] in target_ids]
        print(f"Filtered to {len(patients)} patients: {target_ids}")

    if len(patients) == 0:
        print("No valid patients found.")
        return

    all_results = []
    for idx, patient in enumerate(patients):
        result = process_patient(patient)
        all_results.append(result)
        # Sleep between patients to avoid overloading the server
        if idx < len(patients) - 1:
            print(f"  Sleeping 20s before next patient ...")
            time.sleep(20)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if r['median_lag_sec'] is not None:
            print(f"  [{r['id']:3d}] {r['name']:15s}  lag = {r['median_lag_sec']:+.3f}s  "
                  f"({r['n_stable']}/{r['n_sampled']} stable)")
        else:
            print(f"  [{r['id']:3d}] {r['name']:15s}  NO STABLE SEGMENTS  "
                  f"(0/{r['n_sampled']})")

    # Save JSON
    out_path = 'Mapping/lag_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save summary table as CSV
    rows = []
    for r in all_results:
        rows.append({
            'ID': r['id'],
            'Name': r['name'],
            'MAC': r['mac'],
            'N_Stable': r['n_stable'],
            'N_Sampled': r['n_sampled'],
            'Median_Lag_Sec': r['median_lag_sec'],
            'Median_Lag_Samples': round(r['median_lag_sec'] * 100, 1) if r['median_lag_sec'] is not None else None,
            'Stable_Lags': str(r['stable_lags']),
        })
    df_out = pd.DataFrame(rows)
    csv_path = 'Mapping/lag_results.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
