import numpy as np
import os, sys
import warnings
warnings.filterwarnings("ignore")
from influxdb import InfluxDBClient
import operator
import onnxruntime as ort
from time import sleep
from datetime import datetime, timezone
import pytz
from scipy.signal import butter, lfilter, filtfilt, resample_poly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.plotly_xz_mtl import load_model_MTL
import yaml
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def low_pass_filter(data, Fs, low, order):
	b, a = butter(order, low/(Fs * 0.5), 'low')

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


def normalize(data):
	data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
	return data


def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
	if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
		client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
	else:
		client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
	query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
	result = client.query(query)
	points = list(result.get_points())
	values =  list(map(operator.itemgetter(data_name), points))
	times  =  list(map(operator.itemgetter('time'),  points))
	data = np.array(values)
	return data, times


def connect_influxdb(influxdb_conf):
	url_str = influxdb_conf["ip"].split("://")
	if len(url_str) >= 2:
		ssl_val = True if url_str[0] == "https" else False
		url_ip = url_str[1]
	else:
		ssl_val = True
		url_ip = url_str[0]
	# print(f'url_ip: {url_ip}')
	verify = False

	influxdb_client = InfluxDBClient(url_ip, influxdb_conf["port"], influxdb_conf["user"], influxdb_conf["password"],
									 influxdb_conf["db"], ssl=ssl_val, verify_ssl=verify)

	try:
		if influxdb_client.ping():
			print(f"Successfully connected to the Influxdb! {url_ip}")
		else:
			print(f"Failed to connect to the Influxdb {url_ip} , exit")
			exit(1)
	except Exception as e:
		print(f"Ping failure, Connection to the Influxdb {url_ip} failed!, exit")
		exit(1)
	return influxdb_client


if __name__ == "__main__":
	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
	influx_vitals_psg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	url = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1743833962796&to=1743834079231&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
	url = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1743834661398&to=1743834777833&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
	# url = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1743836957025&to=1743837106519&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
	# url = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1743840067651&to=1743840400855&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
	# url = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1743840254199&to=1743840700624&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
	mac = url.split('var-mac=')[1].split('&')[0]
	start_time = int(url.split('from=')[1].split('&')[0]) / 1000
	end_time = int(url.split('to=')[1].split('&')[0]) / 1000
	
	time_fix = 228.5 / 100
	# time_fix = 0.0

	start_time_bsg = start_time - time_fix
	end_time_bsg = end_time - time_fix

	start_time_psg = start_time
	end_time_psg = end_time


	raw_Z, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='Z', 
						data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
	raw_Y, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='N', 
						data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
	raw_X, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='E', 
						data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
	raw_THO, _ = read_influx(influx_vitals_psg, unit=mac, table_name='SleepLab_test', 
						data_name="Effort THO", start_timestamp=start_time_psg, end_timestamp=end_time)
	raw_ABD, _ = read_influx(influx_vitals_psg, unit=mac, table_name='SleepLab_test', 
						data_name="Effort ABD", start_timestamp=start_time_psg, end_timestamp=end_time)
	
	X = low_pass_filter(raw_X, 100, 1, 3)
	Y = low_pass_filter(raw_Y, 100, 1, 3)
	X = (X - np.mean(X)) / np.std(X) * -1
	Y = (Y - np.mean(Y)) / np.std(Y)
	THO = (raw_THO - np.mean(raw_THO)) / np.std(raw_THO)
	ABD = (raw_ABD - np.mean(raw_ABD)) / np.std(raw_ABD)
	print(f'raw_Z shape: {raw_Z.shape}, raw_Y shape: {raw_Y.shape}, raw_X shape: {raw_X.shape}, raw_THO shape: {raw_THO.shape}, raw_ABD shape: {raw_ABD.shape}')
	
	peaks_x, _ = find_peaks(X, distance=100, height=np.quantile(X, 0.99))
	peaks_abd, _ = find_peaks(ABD, distance=100, height=np.quantile(ABD, 0.99))


	# print(f'peaks_x: {peaks_x}, peaks_abd: {peaks_abd}')
	# print(f'peaks_abd-peaks_x: {peaks_abd - peaks_x}')
	fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
	axes[0].plot(Y, label='Y')
	axes[0].plot(ABD, label='ABD')
	# axes[0].plot(THO, label='THO')
	axes[0].set_title('Y-axis')
	axes[0].legend()
	axes[1].plot(X, label='X')
	# axes[1].scatter(peaks_x, X[peaks_x], color='red', label='Peaks')
	# axes[1].plot(ABD, label='ABD')
	axes[1].plot(THO, label='THO')
	# axes[1].scatter(peaks_abd, ABD[peaks_abd], color='green', label='ABD Peaks')
	axes[1].set_title('X-axis')
	axes[1].legend()
	axes[2].plot(THO, label='THO')
	axes[2].set_title('THO')
	axes[2].legend()
	axes[3].plot(ABD, label='ABD')
	axes[3].set_title('ABD')
	axes[3].legend()
	plt.tight_layout()
	plt.savefig('raw_signals_test.png')