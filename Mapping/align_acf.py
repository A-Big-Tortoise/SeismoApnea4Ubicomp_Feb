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
from statsmodels.tsa.stattools import acf


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


def best_lag_cal(L, xcorr):
	lags = np.arange(-(L-1), L)
	idx = np.argmax(xcorr)
	best_lag = lags[idx]
	return best_lag

if __name__ == "__main__":
	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
	influx_vitals_psg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	urls = [
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741409805854&to=1741409962173&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741408979689&to=1741409147405&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741406798063&to=1741406960772&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741406585333&to=1741406702706&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741406895639&to=1741406952837&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741408984073&to=1741409109409&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1741409798757&to=1741409924093&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns'
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1741500353231&to=1741500464698',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1741500966294&to=1741501077761',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1741501686490&to=1741501751797',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1741502202921&to=1741502281726',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1741502399930&to=1741502478735',
		
	]
	for cnt, url in enumerate(urls):
		mac = url.split('var-mac=')[1].split('&')[0]
		start_time = int(url.split('from=')[1].split('&')[0]) / 1000
		end_time = int(url.split('to=')[1].split('&')[0]) / 1000
		
		# time_fix = 228.5 / 100
		time_fix = 0.0

		start_time_bsg = start_time - time_fix
		end_time_bsg = end_time - time_fix

		start_time_psg = start_time
		end_time_psg = end_time


		# raw_Z, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='Z', 
							# data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
		raw_Y, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='Y', 
		# raw_Y, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='N', 
							data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
		raw_X, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='X', 
		# raw_X, _ = read_influx(influx_vitals_bsg, unit=mac, table_name='E', 
							data_name="value", start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
		raw_THO, _ = read_influx(influx_vitals_psg, unit=mac, table_name='SleepLab_test', 
							data_name="Effort THO", start_timestamp=start_time_psg, end_timestamp=end_time)
		raw_ABD, _ = read_influx(influx_vitals_psg, unit=mac, table_name='SleepLab_test', 
							data_name="Effort ABD", start_timestamp=start_time_psg, end_timestamp=end_time)
		
		X = low_pass_filter(raw_X, 100, 0.35, 3) * -1
		Y = low_pass_filter(raw_Y, 100, 0.35, 3)
		X = (X - np.mean(X)) / np.std(X) 
		Y = (Y - np.mean(Y)) / np.std(Y)

		THO = low_pass_filter(raw_THO, 100, 1, 3)
		ABD = low_pass_filter(raw_ABD, 100, 1, 3)
		THO = (THO - np.mean(THO)) / np.std(THO)
		ABD = (ABD - np.mean(ABD)) / np.std(ABD)
		
		# peaks_x, _ = find_peaks(X, distance=100, height=np.quantile(X, 0.99))
		# peaks_abd, _ = find_peaks(ABD, distance=100, height=np.quantile(ABD, 0.99))


		# print(f'peaks_x: {peaks_x}, peaks_abd: {peaks_abd}')
		# print(f'peaks_abd-peaks_x: {peaks_abd - peaks_x}')
		fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
		axes[0].plot(Y, label='Y')
		axes[0].set_title('Y-axis')
		axes[0].legend()
		axes[1].plot(X, label='X')
		axes[1].set_title('X-axis')
		axes[1].legend()
		axes[2].plot(THO, label='THO')
		axes[2].set_title('THO')
		axes[2].legend()
		axes[3].plot(ABD, label='ABD')
		axes[3].set_title('ABD')
		axes[3].legend()
		plt.tight_layout()
		plt.savefig('/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/figs/{}.png'.format(cnt))


		print(f'raw_Y shape: {raw_Y.shape}, raw_X shape: {raw_X.shape}, raw_THO shape: {raw_THO.shape}, raw_ABD shape: {raw_ABD.shape}')




		fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
		Y_THO = np.correlate(Y, THO, mode='full')
		denom = (np.linalg.norm(Y) * np.linalg.norm(THO)) + 1e-12
		Y_THO = Y_THO / denom * -1
		best_lag_Y_THO = best_lag_cal(len(Y), Y_THO)
		axes[0].plot(Y_THO)
		axes[0].set_title(f'Y-THO (lag: {best_lag_Y_THO})')
		X_THO = np.correlate(X, THO, mode='full')
		denom = (np.linalg.norm(X) * np.linalg.norm(THO)) + 1e-12
		X_THO = X_THO / denom * -1
		best_lag_X_THO = best_lag_cal(len(X), X_THO)
		axes[1].plot(X_THO, label='X-THO')
		axes[1].set_title(f'X-THO (lag: {best_lag_X_THO})')
		Y_ABD = np.correlate(Y, ABD, mode='full')
		denom = (np.linalg.norm(Y) * np.linalg.norm(ABD)) + 1e-12
		Y_ABD = Y_ABD / denom * -1
		best_lag_Y_ABD = best_lag_cal(len(Y), Y_ABD)
		axes[2].plot(Y_ABD, label='Y-ABD')
		axes[2].set_title(f'Y-ABD (lag: {best_lag_Y_ABD})')
		X_ABD = np.correlate(X, ABD, mode='full')
		denom = (np.linalg.norm(X) * np.linalg.norm(ABD)) + 1e-12
		X_ABD = X_ABD / denom * -1
		best_lag_X_ABD = best_lag_cal(len(X), X_ABD)
		axes[3].plot(X_ABD, label='X-ABD')
		axes[3].set_title(f'X-ABD (lag: {best_lag_X_ABD})')
		axes[4].plot(Y_THO, label='Y-THO')
		axes[4].plot(X_THO, label='X-THO')
		axes[4].plot(Y_ABD, label='Y-ABD')
		axes[4].plot(X_ABD, label='X-ABD')
		axes[4].set_title('All Correlations')
		axes[4].legend()
		plt.tight_layout()
		plt.savefig('/home/jiayu/SeismoApnea4Ubicomp_Feb/Mapping/figs/{}_correlation.png'.format(cnt))

