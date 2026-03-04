import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
# from Code.utils_dsp import kurtosis
from statsmodels.tsa.stattools import acf

def low_pass_filter(data, Fs, low, order):
	b, a = signal.butter(order, low/(Fs * 0.5), 'low')

	if data.ndim == 1:
		N = len(data)
		padded = np.pad(data, (N, N), mode='reflect')
	elif data.ndim == 2:
		N = data.shape[1]
		padded = np.pad(data, ((0, 0), (N, N)), mode='reflect')
	
	filtered_data = signal.filtfilt(b, a, padded)

	if data.ndim == 1:
		filtered_data = filtered_data[N:-N]
	elif data.ndim == 2:
		filtered_data = filtered_data[:, N:-N]
	
	return filtered_data

def kurtosis(signal):
	signal = np.array(signal)
	mean = np.mean(signal)
	std = np.std(signal, ddof=0)
	kurt = np.mean((signal - mean)**4) / std**4
	return kurt

def rr_estimate(signal, Fs):
	acf_values = acf(signal, nlags=Fs*30, fft=True)
	peaks, _ = find_peaks(acf_values, distance=Fs*3, height=np.mean(acf_values) + 0.5 * np.std(acf_values))
	if len(peaks) > 1:
		peak = peaks[0]
		rr_interval = peak / Fs
		rr = 60 / rr_interval
		return rr
	else: 
		return -1


def cal_ratio(signal, rr):
	peaks, _ = find_peaks(signal, distance=6000 / rr * 0.65, height=np.mean(signal) + 0.35 * np.std(signal))
	valleys, _ = find_peaks(-signal, distance=6000 / rr * 0.65, height=-np.mean(signal) + 0.35 * np.std(signal))
	ratios = []
	for peak in peaks:
		valley_left = valleys[valleys < peak][-1] if np.any(valleys < peak) else None
		valley_right = valleys[valleys > peak][0] if np.any(valleys > peak) else None
		if valley_left is not None and valley_right is not None:
			peak_left = peak - valley_left
			peak_right = valley_right - peak
			ratio = peak_right / peak_left
			ratios.append(ratio)
			
	if len(ratios) <= 8:
		return signal, peaks, valleys, -1
	else: 
		ratio = np.mean(sorted(ratios)[4:-4])
		if ratio < 1: 
			ratio = 1 / ratio
			signal = signal * -1
			return signal, valleys, peaks, ratio
		else:
			return signal, peaks, valleys, ratio



folder = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Data/fold_data_p109_MTL_60s/'
import shutil
shutil.rmtree('/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/ie/')
os.mkdir(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/ie/')

ratio_estimates = []
ratio_labels = []


for file in os.listdir(folder):
	data = np.load(os.path.join(folder, file))
	print(f'{file}: {data.shape}')

	# random_idxes = np.random.sample(data.shape[0], size=15, replace=False)
	random_idxes = np.random.choice(data.shape[0], size=200, replace=False)

	for idx in random_idxes:
		random_idx = [idx]
		x, y, tho, abd = data[random_idx, :6000], data[random_idx, 6000:12000], data[random_idx, 18000: 24000], data[random_idx, 24000: 30000]

		x = low_pass_filter(x, Fs=100, low=0.8, order=3)
		y = low_pass_filter(y, Fs=100, low=0.8, order=3)
		tho = low_pass_filter(tho, Fs=100, low=0.8, order=3)
		abd = low_pass_filter(abd, Fs=100, low=0.8, order=3)


		rr_x = rr_estimate(x[0], Fs=100)
		rr_y = rr_estimate(y[0], Fs=100)
		rr_tho = rr_estimate(tho[0], Fs=100)
		rr_abd = rr_estimate(abd[0], Fs=100)
		rr = np.mean([rr_x, rr_y, rr_tho, rr_abd])

		if kurtosis(abd) > 3: continue  
		if kurtosis(y) > 3: continue
		if kurtosis(x) > 3: continue
		if kurtosis(tho) > 3: continue
	
		x, y, tho, abd = x[0], y[0], tho[0], abd[0]
		x, peaks_x, valleys_x, ratio_x = cal_ratio(x, rr)
		y, peaks_y, valleys_y, ratio_y = cal_ratio(y, rr)
		tho, peaks_tho, valleys_tho, ratio_tho = cal_ratio(tho, rr)
		abd, peaks_abd, valleys_abd, ratio_abd = cal_ratio(abd, rr)

		if ratio_x == -1 or ratio_y == -1 or ratio_tho == -1 or ratio_abd == -1:
			continue

		ratio_estimate = np.mean([ratio_x, ratio_y])
		# ratio_estimate = ratio_y
		# ratio_estimate = min(ratio_x, ratio_y)
		ratio_label = np.mean([ratio_tho, ratio_abd])
		
	
		ratio_estimates.append(ratio_estimate)	
		ratio_labels.append(ratio_label)
		# print(f'File: {file}, Index: {idx}, Ratios - X: {ratio_x}, Y: {ratio_y}, THO: {ratio_tho}, ABD: {ratio_abd}')

		if np.abs(ratio_estimate - ratio_label) > 0.5:
			print(f'File: {file}, Index: {idx}, Ratios - X: {ratio_x}, Y: {ratio_y}, THO: {ratio_tho}, ABD: {ratio_abd}')
			fig, axes = plt.subplots(4, 1, figsize=(15, 10))
			axes[0].plot(x, color='blue', alpha=0.5)
			axes[0].plot(peaks_x, x[peaks_x], 'x', color='black')
			axes[0].plot(valleys_x, x[valleys_x], 'o', color='cyan')
			axes[1].plot(y, color='orange', alpha=0.5)
			axes[1].plot(peaks_y, y[peaks_y], 'x', color='black')
			axes[1].plot(valleys_y, y[valleys_y], 'o', color='cyan')
			axes[2].plot(tho, color='green', alpha=0.5)
			axes[2].plot(peaks_tho, tho[peaks_tho], 'x', color='black')
			axes[2].plot(valleys_tho, tho[valleys_tho], 'o', color='cyan')
			axes[3].plot(abd, color='red', alpha=0.5)
			axes[3].plot(peaks_abd, abd[peaks_abd], 'x', color='black')
			axes[3].plot(valleys_abd, abd[valleys_abd], 'o', color='cyan')
			axes[0].set_title('X-axis, Kurtosis: {:.2f}, Ratio: {:.2f}, RR: {:.2f}'.format(kurtosis(x), ratio_x, rr_x))
			axes[1].set_title('Y-axis, Kurtosis: {:.2f}, Ratio: {:.2f}, RR: {:.2f}'.format(kurtosis(y), ratio_y, rr_y))
			axes[2].set_title('THO, Kurtosis: {:.2f}, Ratio: {:.2f}, RR: {:.2f}'.format(kurtosis(tho), ratio_tho, rr_tho))
			axes[3].set_title('ABD, Kurtosis: {:.2f}, Ratio: {:.2f}, RR: {:.2f}'.format(kurtosis(abd), ratio_abd, rr_abd))	
			plt.tight_layout()
			plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/ie/{file[:-4]}_{random_idx[0]}.png')
			plt.close()
	# break
# plt.plot(ratio_labels, ratio_estimates, 'o', alpha=0.05)
# plt.xlabel('Ratio Labels')
# plt.ylabel('Ratio Estimates')
# plt.title(f'Ratio Labels vs Ratio Estimates, corr: {np.corrcoef(ratio_labels, ratio_estimates)[0, 1]:.2f}')
# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/ratio_005.png', dpi=300)
# plt.close()