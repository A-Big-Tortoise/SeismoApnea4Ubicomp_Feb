import numpy as np
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import trange
import onnx
import onnxruntime as ort
import os, gc, time
import psutil, resource
proc = psutil.Process(os.getpid())
import paho.mqtt.client as mqtt
import struct
import sys
import queue
import netifaces



data_queue = queue.Queue(2000)


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



def pack_beddot_data(timestamp, data_interval, data):
	# First, convert the MAC address into byte sequence
	packed_data = struct.pack("H", len(data))  # data length
	packed_data += struct.pack("Q", timestamp)  # timestamp
	packed_data += struct.pack("I", data_interval)  # data interval 
	for item in data:
		packed_data += struct.pack("i", item)
		
	return packed_data


# Function to Publish data
def upload_data(client):
	"""Retrieves data from the queue and uploads it via MQTT."""
	# Try connecting to the broker
	try:
		client.connect(MQTT_BROKER, MQTT_PORT, 30)
		client.loop_start()
	except Exception as e:
		print(f"Error connecting to MQTT: {e}")
		sys.exit(1)

	while True:
		if not data_queue.empty():
			data = data_queue.get()
			data = data.rstrip(b'}').split(b',')
			channel = data.pop(0).decode('utf-8')[-2]
			start_timestamp = int(float(data.pop(0)) * 1000000)
			int_data_list = [int(data_point.decode('utf-8').strip()) for data_point in data]
			# Pack and send the sine wave data series
			packed_data = pack_beddot_data(start_timestamp, DATA_INTERVAL, int_data_list)
			if channel == 'Z':
				channel = 'geophone'
			elif channel == 'E':
				channel = 'X'
			elif channel == 'N':
				channel = 'Y'
			MQTT_TOPIC = MQTT_TOPIC_PRE + channel
			client.publish(MQTT_TOPIC, packed_data, qos=1)
			

def mac_address():
	macEth = None
	data = netifaces.interfaces()
	for i in data:
		if i == 'wlan0': #'en0': # 'eth0':
			interface = netifaces.ifaddresses(i)
			info = interface[netifaces.AF_LINK]
			if info:
				macEth = interface[netifaces.AF_LINK][0]["addr"]
				
	return macEth



MQTT_BROKER = "172.21.85.54"
MQTT_PORT = 1883
DATA_INTERVAL = 10000


WINDOW = 25  # window length

def publish_windowed_data(client, data, fs):
    """
    data: ndarray, shape [C, T]
    """
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 30)
        client.loop_start()
    except Exception as e:
        print(f"Error connecting to MQTT: {e}")
        sys.exit(1)

    C, T = data.shape
    start_ts = int(time.time() * 1e6)  # microsecond timestamp

    for start in range(0, T - WINDOW + 1, WINDOW):
        for ch in range(C):
            sig_window = data[ch, start:start + WINDOW]
            int_data_list = sig_window.astype(int).tolist()

            # channel mapping
            if ch == 2:
                channel = "geophone"   # Z
            elif ch == 0:
                channel = "X"          # E
            elif ch == 1:
                channel = "Y"          # N
            else:
                continue

            ts = start_ts + int(start * DATA_INTERVAL)

            packed_data = pack_beddot_data(
                ts,
                DATA_INTERVAL,
                int_data_list
            )

            topic = MQTT_TOPIC_PRE + channel
            client.publish(topic, packed_data, qos=1)

        # 可选：模拟实时（否则是 burst 发送）
        time.sleep(WINDOW * DATA_INTERVAL / 1e6)



if __name__ == "__main__":
	data, fs, meta = generate_biosignal_v2(seed=42)
	print(data.shape)

	# unit = mac_address()
	MQTT_TOPIC_PRE = f"/UGA/SeismoApnea/"


	# window = 25
	# for i in trange(0, data.shape[1] - window + 1, window):
	# 	sig_window = data[:, i:i+window]
	client = mqtt.Client()
	publish_windowed_data(client, data, fs)