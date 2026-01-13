import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import trange
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt


def normalize_1d(data):
	data = (data - np.mean(data)) / np.std(data)
	return data


def low_pass_filter(data, Fs, low, order):
	b, a = butter(order, low/(Fs * 0.5), 'low')
	N = len(data)
	padded = np.pad(data, (N, N), mode='reflect')
	filtered_data = filtfilt(b, a, padded)
	return filtered_data[N:-N]


def generate_biosignal_v2(
	fs=100,
	hours_range=(4, 8),
	resp_freq_range=(0.2, 0.5),
	hr_freq_range=(1.0, 2.0),
	amp_range=(5e4, 1e5),
	baseline_range=(2e4, 4e4),
	movement_amp_ratio=(0.3, 0.8),
	num_resp_harmonics=4,
	seed=None
):
	if seed is not None:
		np.random.seed(seed)

	# ---- duration ----
	hours = np.random.uniform(*hours_range)
	duration_sec = int(hours * 3600)
	N = duration_sec * fs
	t = np.arange(N) / fs

	# ---- frequencies ----
	f_resp = np.random.uniform(*resp_freq_range)
	f_hr = np.random.uniform(*hr_freq_range)

	# ---- amplitude & baseline ----
	A = np.random.uniform(*amp_range)
	baseline = np.random.uniform(*baseline_range)

	# =========================
	# Respiration (low-freq)
	# =========================
	resp = np.zeros_like(t)
	for k in range(1, num_resp_harmonics + 1):
		phase = np.random.uniform(0, 2 * np.pi)
		resp += (1.0 / k) * np.sin(2 * np.pi * k * f_resp * t + phase)
	resp /= np.max(np.abs(resp))

	# =========================
	# Movement: low-frequency large-amplitude noise
	# =========================
	movement_raw = np.random.randn(N)

	# low-pass filter (<0.1 Hz)
	b, a = butter(2, 0.3 / (fs / 2), btype='low')
	movement = filtfilt(b, a, movement_raw)
	movement /= np.max(np.abs(movement))

	movement_amp = A * np.random.uniform(*movement_amp_ratio) *3

	# =========================
	# Axis-specific mixing
	# =========================
	signal = np.zeros((2, N))
	axis_gain = np.random.uniform(0.8, 1.2, size=2)

	for i in range(2):
		signal[i] = (
			baseline
			+ axis_gain[i] * (
				A * resp
				+ movement_amp * movement
			)
		)

	meta = {
		"fs": fs,
		"hours": hours,
		"resp_freq": f_resp,
		"hr_freq": f_hr,
		"resp_amp": A,
		"movement_amp": movement_amp,
		"baseline": baseline
	}

	return signal, fs, meta


def signal_process(signal):
	sig = low_pass_filter(signal, Fs=100, low=0.6, order=3)
	sig = resample_poly(sig,1,10)
	sig = sig[5:595]
	sig = normalize_1d(sig)
	return sig

if __name__ == "__main__":
	data, fs, meta = generate_biosignal_v2(seed=42)
	print(data.shape)  # (2, N)
	onnx_model = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Inference/onnx_models/model_1.onnx'
	ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

	for i in trange(0,  data.shape[1], 100):

		# preprecessing
		sig_x = signal_process(data[0, i: i+6000])
		sig_y = signal_process(data[1, i: i+6000])
		
		signals = np.stack([sig_x, sig_y], axis=-1) 
		signals = np.expand_dims(signals, axis=0)  # (1, 590, 2)
		signals = signals.transpose(0, 2, 1)
		signals = signals.astype(np.float32)

		outs = ort_session.run(None, {"input": signals})
		# print(outs)