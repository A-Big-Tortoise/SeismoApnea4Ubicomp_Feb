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
import torch
import yaml


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


def convert_ms_timestamp_to_ny_datetime(timestamp_ms):
	dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
	ny_tz = pytz.timezone('America/New_York')
	dt_ny = dt_utc.astimezone(ny_tz)
	return dt_ny


def inference(XY, model, device, threshold_stage, threshold_apnea):
	XY = torch.tensor(XY, dtype=torch.float32).to(device)
	pred_res_stage = []
	pred_res_apnea = []

	model.eval()
	with torch.no_grad():
		outputs1_stage, outputs1_apnea, _, _, _ = model(XY)
		outputs2_stage, outputs2_apnea, _, _, _ = model(torch.flip(XY, dims=[2]))
		outputs3_stage, outputs3_apnea, _, _, _ = model(-XY)

		output_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
		probs_stage = torch.softmax(output_stage, dim=1)  # shape: [batch, num_classes]
		prob_class1_stage = probs_stage[:, 1].cpu().numpy()
		pred_stage = (prob_class1_stage >= threshold_stage).astype(int)
		pred_res_stage.extend(pred_stage)

		output_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
		probs_apnea = torch.softmax(output_apnea, dim=1)  # shape: [batch, num_classes]
		prob_class1_apnea = probs_apnea[:, 1].cpu().numpy()
		pred_apnea = (prob_class1_apnea >= threshold_apnea).astype(int)
		pred_res_apnea.extend(pred_apnea)

	return np.array(pred_res_stage), np.array(pred_res_apnea)


def load_configs(config_path):	
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
		fold2id = config["fold_id_mapping"]
		fold2threshold_stage = config["fold_to_threshold_stage"]
		fold2threshold_apnea = config["fold_to_threshold_apnea"]
	return fold2id, fold2threshold_stage, fold2threshold_apnea



if __name__ == "__main__":

	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	seg_duration_rr = 60
	step = 1

	mac = 'b8:27:eb:ec:64:08'

	dest = {'ip': 'sensorserver.engr.uga.edu', 'port': 8086, 'db': 'healthresult', 'user': 'algtest', 'password': 'sensorweb711'}
	dest_client = connect_influxdb(dest)



	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	models = []
	for i in range(1, 5):
		model_folder = f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Losses/Models/MTL_ws1_wa1/fold{i}/PatchTST_patchlen24_nlayer4_dmodel64_nhead4_dff256/'
		model = load_model_MTL(model_folder, seg_duration_rr, device, axis=2)
		models.append(model)
	print('len(models): ', len(models))
	config_path = f"/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Losses/configs/MTL_ws1_wa1.yaml"
	fold2id, threshold_stages, threshold_apneas = load_configs(config_path)




	data_dict = {
			"Jiayu": (1770321733633, 1770321878664),
			"Jiahui": (1770322049249, 1770322309666),
			"Yida": (1770322576738, 1770322874030)
			# "Zixuan": ()     
		}
	# Zixuan

	# # Sahas
	# start_time = 1770322977722 / 1000
	# end_time = 1770323204088 / 1000



	pred_stage_lst = [1] * 60
	pred_apnea_lst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	print('Initial pred_stage_lst: ', pred_stage_lst)
	filtered_apnea_lst = []
	filtered_stage_lst = []

	for subject, (start_time, end_time) in data_dict.items():
		start_time, end_time = start_time / 1000, end_time / 1000

		while start_time - 60 < end_time:
			json_body = []

			rr_start_time = start_time - seg_duration_rr*1.01
			raw_Z_rr, _ = read_influx(influx_vitals_bsg, unit=mac, 
								table_name='Z', data_name="value",
								start_timestamp=rr_start_time, 
								end_timestamp=start_time)
			raw_Y_rr, _ = read_influx(influx_vitals_bsg, unit=mac, 
								table_name='Y', data_name="value",
								start_timestamp=rr_start_time, 
								end_timestamp=start_time)
			
			raw_X_rr, _ = read_influx(influx_vitals_bsg, unit=mac, 
								table_name='X', data_name="value",
								start_timestamp=rr_start_time, 
								end_timestamp=start_time)

			if len(raw_Z_rr) < 100*seg_duration_rr: 
				start_time += step
				continue
			

			def preprocess(raw_signal, seg_duration):
				signal = np.array(raw_signal[-100*seg_duration:])
				signal = low_pass_filter(signal, Fs=100, low=0.8, order=3)
				signal = resample_poly(signal,1,10)
				signal = signal[5:595]
				signal = (signal - np.mean(signal)) / np.std(signal)
				return signal
			X = preprocess(raw_X_rr, seg_duration_rr)
			Y = preprocess(raw_Y_rr, seg_duration_rr)		
			XY = np.stack([X, Y], axis=0)[np.newaxis, :, :]
			
			
			pred_stages = []
			pred_apneas = []
			for fold_idx in range(1, 5):
				model = models[fold_idx-1]
				fold_idx = str(fold_idx)
				pred_stage, pred_apnea = inference(XY, model, device, threshold_stage=threshold_stages[fold_idx], threshold_apnea=threshold_apneas[fold_idx])	
				print(f'Fold {fold_idx}, Predicted Stage: {pred_stage}, Predicted Apnea: {pred_apnea}')
				pred_stages.append(pred_stage)
				pred_apneas.append(pred_apnea)

			
			pred_stage_final = int(0) if np.sum(np.array(pred_stages)) < 2 else int(1)
			pred_apnea_final = int(1) if np.sum(np.array(pred_apneas)) >= 3 else int(0)
			pred_apnea_final = int(0) if pred_stage_final == 1 else pred_apnea_final
			print(f'Final Predicted Stage: {pred_stage_final}, Final Predicted Apnea: {pred_apnea_final}')
			
			pred_stage_lst.append(pred_stage_final)
			pred_apnea_lst.append(pred_apnea_final)
			filtered_stage_lst.append(np.median(np.array(pred_stage_lst[-60:])).astype(int))
			filtered_apnea_lst.append(np.median(np.array(pred_apnea_lst[-10:])).astype(int))

			start_time += step
