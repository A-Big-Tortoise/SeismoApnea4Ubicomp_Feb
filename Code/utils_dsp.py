import numpy as np
from scipy import signal
import torch
from tqdm import tqdm


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


def band_pass_filter(data, Fs, lowcut, highcut, order):
	b, a = signal.butter(order, [lowcut / (Fs * 0.5), highcut / (Fs * 0.5)], btype='band')

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



def denoise(data, low=0.8, Fs=100):
	return low_pass_filter(data, Fs=Fs, low=low, order=3)


def denoise_iter(data, low=0.8, Fs=100):
	filtered_data = np.zeros_like(data)
	for cnt, signal in tqdm(enumerate(data)):
		signal = low_pass_filter(signal, Fs=Fs, low=low, order=3)
		filtered_data[cnt] = signal
	return filtered_data


def denoise_band(data, Fs=100):
	return band_pass_filter(data, Fs=Fs, lowcut=1.25, highcut=8.5, order=3)


def normalize(data):
	data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
	return data



def g_rational(x, k):
	return x / (1.0 + np.abs(x) / k)



def normalize2(data):
	data = data - np.mean(data, axis=1, keepdims=True)
	values = data.flatten()
	data_P_low, data_P_high = np.percentile(values, [2, 98])
	print(f'Data_P_High: {data_P_high}')
	data = g_rational(data, k=data_P_high)
	data = data / np.std(data, axis=1, keepdims=True)
	return data



def normalize_1d(data):
	data = (data - np.mean(data)) / np.std(data)
	return data

# def modify_magnitude_with_gaussian_noise(x_in, fs=10, band=(0, 1), noise_std=5, clip_negative=True, return_spectrum=False):
#     x = x_in.astype(np.float32)
#     N = len(x)

#     # Use real FFT for efficiency (since x is real-valued)
#     X = np.fft.rfft(x)
#     freqs = np.fft.rfftfreq(N, d=1/fs)  # Only positive freqs

#     mag = np.abs(X)
#     phase = np.angle(X)
#     mag_noisy = mag.copy()

#     f_low, f_high = band
#     mask = (freqs >= f_low) & (freqs <= f_high)
#     k = np.sum(mask)

#     if k > 0:
#         noise = np.random.normal(0, noise_std, size=k)
#         if clip_negative:
#             mag_noisy[mask] = np.clip(mag_noisy[mask] + noise, a_min=0.0, a_max=None)
#         else:
#             mag_noisy[mask] = mag_noisy[mask] + noise

#     # Reconstruct complex spectrum and perform inverse rFFT
#     X_noisy = mag_noisy * np.exp(1j * phase)
#     x_noisy = np.fft.irfft(X_noisy, n=N)

#     if return_spectrum:
#         return x_noisy, mag, mag_noisy
#     else:
#         return x_noisy


# def modify_magnitude_with_gaussian_noise_batch(x_in, fs=10, band=(0,1), noise_std=5, clip_negative=True):
#     x = x_in.to(torch.float32)
#     B, C, N = x.shape
#     X = torch.fft.rfft(x, dim=-1)                          # [B,C,Nr]
#     freqs = torch.fft.rfftfreq(N, d=1/fs).to(x.device)     # [Nr]
#     mag, phase = torch.abs(X), torch.angle(X)
#     f_low, f_high = band
#     mask = (freqs >= f_low) & (freqs <= f_high)            # [Nr]
#     k = int(mask.sum().item())
#     if k > 0:
#         noise = torch.randn(B, C, k, device=x.device) * noise_std
#         mag_noisy = mag.clone()
#         if clip_negative:
#             mag_noisy[..., mask] = torch.clamp(mag_noisy[..., mask] + noise, min=0.0)
#         else:
#             mag_noisy[..., mask] = mag_noisy[..., mask] + noise
#         X_noisy = mag_noisy * torch.exp(1j * phase)
#         x_noisy = torch.fft.irfft(X_noisy, n=N, dim=-1)
#         return x_noisy
#     return x






def modify_magnitude_with_gaussian_noise_batch(x_in, fs=10, band=(0, 1), noise_std=5, clip_negative=True, return_spectrum=False):
	"""
	Apply Gaussian noise to magnitude spectrum of each channel in a batch of 2-channel signals.

	Args:
		x_in (torch.Tensor): Input tensor of shape [B, 2, N].
		fs (float): Sampling frequency in Hz.
		band (tuple): Frequency range (f_low, f_high) in Hz to apply noise.
		noise_std (float): Standard deviation of Gaussian noise.
		clip_negative (bool): Whether to clip magnitude values below 0.
		return_spectrum (bool): If True, also return original and modified magnitude spectra.

	Returns:
		x_noisy (torch.Tensor): Time-domain signals after adding noise, shape [B, 2, N].
		(Optional) mag, mag_noisy
	"""
	x = x_in.clone()
	B, C, N = x.shape
	f_low, f_high = band

	# FFT: [B, C, N] -> complex spectrum
	X = torch.fft.fft(x, dim=-1)
	freqs = torch.fft.fftfreq(N, d=1/fs).to(x.device)  # [N]

	mag = torch.abs(X)         # [B, C, N]
	phase = torch.angle(X)     # [B, C, N]
	mag_noisy = mag.clone()    # [B, C, N]

	# Mask for target frequency band
	mask = (freqs.abs() >= f_low) & (freqs.abs() <= f_high)  # [N]

	# Broadcast mask to shape [1, 1, N]
	mask = mask.view(1, 1, -1)  # for broadcasting
	num_freqs = mask.sum()

	# Add Gaussian noise to selected frequencies
	noise = torch.randn(B, C, num_freqs, device=x.device) * noise_std  # [B, 2, #freq]
	mag_noisy[mask.expand_as(mag_noisy)] += noise.flatten()

	if clip_negative:
		mag_noisy = torch.clamp(mag_noisy, min=0.0)

	# Reconstruct complex spectrum and IFFT
	X_noisy = mag_noisy * torch.exp(1j * phase)
	x_noisy = torch.fft.ifft(X_noisy, dim=-1).real  # [B, 2, N]

	if return_spectrum:
		return x_noisy, mag, mag_noisy
	else:
		return x_noisy



def modify_magnitude_with_gaussian_noise(x_in, fs=10, band=(0, 1), noise_std=5, clip_negative=True, return_spectrum=False):
	"""
	Add Gaussian white noise to the magnitude spectrum within a specified frequency band.

	Args:
		x (np.ndarray): 1D time-domain signal.
		fs (float): Sampling frequency in Hz.
		band (tuple): (f_low, f_high) frequency range in Hz to apply noise.
		noise_std (float): Standard deviation of Gaussian noise to add to magnitude.
		clip_negative (bool): Clip negative magnitudes after noise addition.
		return_spectrum (bool): If True, also return original & modified magnitude spectra.

	Returns:
		x_modified (np.ndarray): Time-domain signal reconstructed with noisy magnitude.
		(Optional) original_mag, modified_mag
	"""
	x = x_in.copy()
	N = len(x)
	X = np.fft.fft(x)
	freqs = np.fft.fftfreq(N, d=1/fs)

	mag = np.abs(X)
	phase = np.angle(X)
	mag_noisy = mag.copy()

	f_low, f_high = band
	indices = np.where((np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high))[0]
	
	# Add real-valued Gaussian white noise to magnitude
	mag_noisy[indices] += np.random.normal(loc=0, scale=noise_std, size=len(indices))

	if clip_negative:
		mag_noisy = np.clip(mag_noisy, 0, None)

	X_noisy = mag_noisy * np.exp(1j * phase)
	x_noisy = np.fft.ifft(X_noisy).real

	if return_spectrum:
		return x_noisy, mag, mag_noisy
	else:
		return x_noisy
	
