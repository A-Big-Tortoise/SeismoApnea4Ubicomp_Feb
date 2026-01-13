import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import trange
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt




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

def normalize_1d(data):
	data = (data - np.mean(data)) / np.std(data)
	return data


def low_pass_filter(data, Fs, low, order):
	b, a = butter(order, low/(Fs * 0.5), 'low')
	N = len(data)
	padded = np.pad(data, (N, N), mode='reflect')
	filtered_data = filtfilt(b, a, padded)
	return filtered_data[N:-N]



def signal_process(sig):
	sig = resample_poly(sig,1,10)
	sig = low_pass_filter(sig, Fs=10, low=0.6, order=3)
	sig = sig[5:595]
	sig = normalize_1d(sig)
	return sig

# if __name__ == "__main__":
# 	data, fs, meta = generate_biosignal_v2(seed=42)
# 	print(data.shape)  # (2, N)

# 	for onnx_id in range(1, 5):
# 		onnx_model = f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Inference/onnx_models/model_{onnx_id}.onnx'
# 		ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

# 		for i in trange(0,  data.shape[1], 100):
# 			# preprecessing
# 			sig_x = signal_process(data[0, i: i+6000])
# 			sig_y = signal_process(data[1, i: i+6000])
			
# 			signals = np.stack([sig_x, sig_y], axis=-1) 
# 			signals = np.expand_dims(signals, axis=0)  # (1, 590, 2)
# 			signals = signals.transpose(0, 2, 1)
# 			signals = signals.astype(np.float32)

# 			outs = ort_session.run(None, {"input": signals})
			
import os, time
import numpy as np
import psutil
from tqdm import trange
import onnxruntime as ort

proc = psutil.Process(os.getpid())

def rss_mb():
	return proc.memory_info().rss / (1024 ** 2)

class StageTimer:
	def __init__(self):
		self.t = {}
		self.m = {}
		self._t0 = {}
		self._m0 = {}

	def start(self, name):
		self._t0[name] = time.perf_counter()
		self._m0[name] = rss_mb()

	def stop(self, name):
		dt = (time.perf_counter() - self._t0[name]) * 1000  # ms
		dm = rss_mb() - self._m0[name]
		self.t.setdefault(name, []).append(dt)
		self.m.setdefault(name, []).append(dm)

	def summary(self):
		def stat(x):
			x = np.array(x, dtype=float)
			return float(x.mean()), float(np.percentile(x, 95)), float(x.max())
		print("\n==== Time (ms): mean / p95 / max ====")
		for k, v in self.t.items():
			mean, p95, mx = stat(v)
			print(f"{k:15s}: {mean:8.3f} / {p95:8.3f} / {mx:8.3f}")
		print("\n==== RSS Δ (MB): mean / p95 / max ====")
		for k, v in self.m.items():
			mean, p95, mx = stat(v)
			print(f"{k:15s}: {mean:8.3f} / {p95:8.3f} / {mx:8.3f}")

# if __name__ == "__main__":
#     data, fs, meta = generate_biosignal_v2(seed=42)
#     print(data.shape) 

#     onnx_model = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Inference/onnx_models/model_1.onnx'

#     so = ort.SessionOptions()
#     # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     ort_session = ort.InferenceSession(onnx_model, sess_options=so, providers=['CPUExecutionProvider'])

#     timer = StageTimer()

#     win = 6000
#     hop = 100

#     i0 = 0
#     sig_x = signal_process(data[0, i0:i0+win])
#     sig_y = signal_process(data[1, i0:i0+win])
#     signals = np.stack([sig_x, sig_y], axis=-1)
#     signals = np.expand_dims(signals, axis=0).transpose(0, 2, 1).astype(np.float32)
#     _ = ort_session.run(None, {"input": signals})

#     for i in trange(0, data.shape[1] - win + 1, hop):
#         timer.start("preprocess")
#         x = data[0, i:i+win]
#         y = data[1, i:i+win]

#         sig_x = signal_process(x)
#         sig_y = signal_process(y)

#         signals = np.stack([sig_x, sig_y], axis=-1)          # (L, 2)
#         signals = np.expand_dims(signals, axis=0)            # (1, L, 2)
#         signals = signals.transpose(0, 2, 1).astype(np.float32)  # (1, 2, L)
#         timer.stop("preprocess")


#         timer.start("inference")
#         outs = ort_session.run(None, {"input": signals})
#         timer.stop("inference")

#         # timer.start("post")
#         # ...
#         # timer.stop("post")

#     timer.summary()
#     print(f"\nFinal RSS (MB): {rss_mb():.2f}")
	
import os, gc, time
import psutil, resource
import onnxruntime as ort
import numpy as np
from tqdm import trange

proc = psutil.Process(os.getpid())


def rss_mb():
	return proc.memory_info().rss / (1024**2)

def peak_rss_mb():
	x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	# Linux: KB, macOS: bytes（经验判断）
	if x > 10**8:
		return x / (1024**2)
	else:
		return x / 1024.0

def mem_barrier():
	gc.collect()
	time.sleep(0.05)

if __name__ == "__main__":
	data, fs, meta = generate_biosignal_v2(seed=42)
	print(data.shape)

	onnx_model = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Inference/onnx_models/model_1.onnx'
	print(f"ONNX file size (MB): {os.path.getsize(onnx_model)/(1024**2):.2f}")

	# ---- measure before load ----
	mem_barrier()
	rss0, peak0 = rss_mb(), peak_rss_mb()
	print(f"RSS before load (MB): {rss0:.2f} | Peak so far (MB): {peak0:.2f}")

	so = ort.SessionOptions()
	ort_session = ort.InferenceSession(onnx_model, sess_options=so, providers=['CPUExecutionProvider'])


	model = onnx.load(onnx_model)
	from onnx import numpy_helper

	total_params = 0

	for init in model.graph.initializer:
		arr = numpy_helper.to_array(init)
		total_params += arr.size

	print(f"Total parameters: {total_params:,}")
	
	
	# ---- measure after load ----
	mem_barrier()
	rss1, peak1 = rss_mb(), peak_rss_mb()
	print(f"RSS after load  (MB): {rss1:.2f} | Peak so far (MB): {peak1:.2f}")
	print(f"LOAD delta      (MB): {rss1 - rss0:.2f}")

	# ---- warmup (first run) ----
	win = 6000
	i0 = 0
	sig_x = signal_process(data[0, i0:i0+win])
	sig_y = signal_process(data[1, i0:i0+win])
	signals = np.stack([sig_x, sig_y], axis=-1)
	signals = np.expand_dims(signals, axis=0).transpose(0, 2, 1).astype(np.float32)

	_ = ort_session.run(None, {"input": signals})

	# ---- measure after first run ----
	mem_barrier()
	rss2, peak2 = rss_mb(), peak_rss_mb()
	print(f"RSS after 1st run(MB): {rss2:.2f} | Peak so far (MB): {peak2:.2f}")
	print(f"1st-run delta    (MB): {rss2 - rss1:.2f}")

	# ---- your profiling loop (unchanged) ----
	timer = StageTimer()
	hop = 100

	for i in trange(0, data.shape[1] - win + 1, hop):
		timer.start("preprocess")
		x = data[0, i:i+win]
		y = data[1, i:i+win]
		sig_x = signal_process(x)
		sig_y = signal_process(y)
		signals = np.stack([sig_x, sig_y], axis=-1)
		signals = np.expand_dims(signals, axis=0).transpose(0, 2, 1).astype(np.float32)
		timer.stop("preprocess")

		timer.start("inference")
		outs = ort_session.run(None, {"input": signals})
		timer.stop("inference")

	timer.summary()
	print(f"\nFinal RSS (MB): {rss_mb():.2f}")
	print(f"Peak RSS  (MB): {peak_rss_mb():.2f}")
