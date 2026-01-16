import pandas as pd
import re
from influxdb import InfluxDBClient
import operator
from datetime import datetime
import pytz
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm import trange
from download_data_apnea import resample_poly, low_pass_filter, normalize_1d
	
plt.rcParams.update({
		# 'font.family': 'Times New Roman',   # 字体
		'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],

		'font.size': 14,                    # 默认字体大小
		'axes.titlesize': 16,               # 子图标题
		'axes.labelsize': 16,               # x/y 轴标签
		'xtick.labelsize': 14,              # x 轴刻度
		'ytick.labelsize': 14,              # y 轴刻度
		'legend.fontsize': 14,              # 图例
	})




def signal_process(sig):

	sig = low_pass_filter(sig, Fs=100, low=0.75, order=3)
	sig = normalize_1d(sig)

	return sig



def local_time_epoch(time, zone="America/New_York"):
	local_tz = pytz.timezone(zone)
	localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
	local_dt = local_tz.localize(localTime, is_dst=None)
	epoch = local_dt.timestamp()
	return epoch


def mac_to_int(mac):
	res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
	if res is None:
		raise ValueError('invalid mac address')
	return int(res.group(0).replace(':', ''), 16)

def int_to_mac(n):
	if not (0 <= n < (1 << 48)):
		raise ValueError("Integer out of range for MAC address (0 to 2^48-1)")
	hex_str = f'{n:012x}'  
	return ':'.join(hex_str[i:i+2] for i in range(0, 12, 2))


def mac_to_int(mac):
	res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
	if res is None:
		raise ValueError('invalid mac address')
	return int(res.group(0).replace(':', ''), 16)


def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
	if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
		client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
	else:
		client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])

	query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
	print(query)
	result = client.query(query)

	points = list(result.get_points())
	values =  list(map(operator.itemgetter(data_name), points))
	times  =  list(map(operator.itemgetter('time'),  points))
	data = np.array(values)
	return data, times


def plot_example(X, Y, Z, Effort_THO, Effort_ABD, Flow, Events, TYPE):
	time = np.arange(0, len(Y)) / 100
	row = 5

	fig, axes = plt.subplots(row, 1, figsize=(12, 2.35 *  row), sharex=True)
	axes[0].plot(time, normalize_1d(X), label='X (norm)')
	axes[0].plot(time, signal_process(X), label='X (filt)')
	axes[1].plot(time, normalize_1d(Y), label='Y (norm)')
	axes[1].plot(time, signal_process(Y), label='Y (filt)')
	axes[2].plot(time, normalize_1d(Z), label='Z (norm)')
	axes[2].plot(time, signal_process(Z), label='Z (filt)')

	axes[3].plot(time, normalize_1d(Effort_THO), label='Thoracic')
	axes[3].plot(time, normalize_1d(Effort_ABD), label='Abdominal')

	axes[4].plot(time, normalize_1d(Flow), label='Airflow')

	time_event = np.arange(0, len(Events)) / 100 * 10
	Events = np.asarray(Events).squeeze()
	idx = np.where(Events > 0)[0] 
	if len(idx) > 0:
		s, e = int(idx[0]), int(idx[-1])

		time_event = np.arange(0, len(Events)) / 100 * 10
		t0, t1 = float(time_event[s]), float(time_event[e])

		for ax in axes:
			ax.axvspan(t0, t1, alpha=0.1)

	for i in range(row):
		axes[i].legend(loc='upper right')
	titles = [
		'(a) X-axis vibration',
		'(b) Y-axis vibration',
		'(c) Z-axis vibration',	
		'(d) Respiratory effort (THO & ABD)',
		'(e) Airflow'
	]

	for ax, title in zip(axes, titles):
		ax.set_title(title, fontsize=16)
	plt.xlabel('Time [sec]')
	plt.tight_layout()
	plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_true/{TYPE}_exp.png', bbox_inches='tight', dpi=300)
	plt.close()



# def plot_example(X, Y, Z, Effort_THO, Effort_ABD, Flow, Events, TYPE):
# 	time = np.arange(0, len(Y)) / 100
# 	row = 4

# 	fig, axes = plt.subplots(row, 1, figsize=(12, 2.35 *  row), sharex=True)
# 	axes[0].plot(time, normalize_1d(X), label='X (norm)')
# 	axes[0].plot(time, signal_process(X), label='X (filt)')
# 	axes[1].plot(time, normalize_1d(Y), label='Y (norm)')
# 	axes[1].plot(time, signal_process(Y), label='Y (filt)')
	

# 	axes[2].plot(time, normalize_1d(Effort_THO), label='Thoracic')
# 	axes[2].plot(time, normalize_1d(Effort_ABD), label='Abdominal')

# 	axes[3].plot(time, normalize_1d(Flow), label='TFlow')

# 	time_event = np.arange(0, len(Events)) / 100 * 10
# 	Events = np.asarray(Events).squeeze()
# 	idx = np.where(Events > 0)[0] 
# 	if len(idx) > 0:
# 		s, e = int(idx[0]), int(idx[-1])

# 		time_event = np.arange(0, len(Events)) / 100 * 10
# 		t0, t1 = float(time_event[s]), float(time_event[e])

# 		for ax in axes:
# 			ax.axvspan(t0, t1, alpha=0.1)

# 	for i in range(row):
# 		axes[i].legend(loc='upper right')
# 	titles = [
# 		'(a) X-axis vibration',
# 		'(b) Y-axis vibration',
# 		'(c) Respiratory effort (THO & ABD)',
# 		'(d) Airflow'
# 	]

# 	for ax, title in zip(axes, titles):
# 		ax.set_title(title, fontsize=16)
# 	plt.xlabel('Time [sec]')
# 	plt.tight_layout()
# 	plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_true/{TYPE}_exp_template.png', bbox_inches='tight', dpi=300)
# 	plt.close()




if __name__ == "__main__":
	# normal = 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom3/sleep-lab-tri-axis-monitoring-room-3?orgId=1&var-mac=b8:27:eb:c2:a0:f9&var-name=vitalsigns&from=1747632328697&to=1747632427921'
	csa_urls = [
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1745649066937&to=1745649143679',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1745979761042&to=1745979824296&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom2/sleep-lab-tri-axis-monitoring-room-2?orgId=1&from=1745984748177&to=1745984805873&var-mac=b8:27:eb:ab:e3:b7&var-name=vitalsigns',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom3/sleep-lab-tri-axis-monitoring-room-3?orgId=1&var-mac=b8:27:eb:c2:a0:f9&var-name=vitalsigns&from=1747631485293&to=1747631584517',
	]

	hyp_urls = [
			'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1741587588366&to=1741587700773',	]


	osa_urls = [
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1755147769209&to=1755147866929',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1755148532690&to=1755148599829',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1755148806565&to=1755148861901',
		'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1754791789929&to=1754791856396',
	]




	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
	influx_vitals_sleep = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	fs = 100

	TYPE = 'Hyp'  # 'CSA' or 'OSA'

	if TYPE == 'CSA':
		selected_urls = csa_urls
	elif TYPE == 'Hyp':
		selected_urls = hyp_urls
	elif TYPE == 'OSA':
		selected_urls = osa_urls


	for cnt, url in enumerate(selected_urls):
		MAC = url.split('var-mac=')[1].split('&')[0]
		unix_start_time = int(url.split('from=')[1].split('&')[0]) / 1000.0
		unix_end_time = int(url.split('to=')[1].split('&')[0]) / 1000.0
		# unix_start_time = 1742012136.270
		# unix_end_time = unix_start_time + 60
		Room = url.split('Room')[1].split('/')[0][-1]

		print(f'Processing MAC: {MAC}, Room: {Room}, Start time: {unix_start_time}, End time: {unix_end_time}')
		
		start_latency = 4

		start_time = unix_start_time + start_latency
		end_time = start_time + 60

		start_time_bsg = unix_start_time + start_latency + 0.3
		end_time_bsg = start_time_bsg + 60

		if Room == 2: table_names = {'X':'E', 'Y':'N', 'Z':'Z'}
		else: table_names = {'X':'X', 'Y':'Y', 'Z':'Z'}


		X, _ = read_influx(influx_vitals_bsg, unit=MAC, 
						table_name=table_names['X'], data_name='value', 
						start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)


		Y, _ = read_influx(influx_vitals_bsg, unit=MAC, 
						table_name=table_names['Y'], data_name='value', 
						start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)

		Z, _ =  read_influx(influx_vitals_bsg, unit=MAC, 
						table_name=table_names['Z'], data_name='value', 
						start_timestamp=start_time_bsg, end_timestamp=end_time_bsg)
		
		Effort_THO, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
						table_name='SleepLab_test', data_name='Effort THO', 
						start_timestamp=start_time, end_timestamp=end_time)
		
		
		Effort_ABD, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
						table_name='SleepLab_test', data_name='Effort ABD', 
						start_timestamp=start_time, end_timestamp=end_time)

		
		TFlow, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
						table_name='SleepLab_test', data_name='TFlow', 
						start_timestamp=start_time, end_timestamp=end_time)

		PFlow_1, _ =  read_influx(influx_vitals_sleep, unit=MAC,
						table_name='SleepLab_test', data_name='PFlow_1', 
						start_timestamp=start_time, end_timestamp=end_time)

		Events, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
						table_name='SleepLab_test_5', data_name='Events', 
						start_timestamp=start_time, end_timestamp=end_time)


		RERAs, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
						table_name='SleepLab_test_5', data_name='RERAs', 
						start_timestamp=start_time, end_timestamp=end_time)

		final_event = Events+RERAs
		final_event[360:435] = 1
		final_event[160:290] = 0
		final_event[:10] = 0
		plot_example(X, Y, Z, Effort_THO, Effort_ABD, TFlow, final_event, TYPE)	

		# time = np.arange(0, len(Y)) / fs
		# row = 4

		# fig, axes = plt.subplots(row, 1, figsize=(15, 2.35 *  row))
		# axes[0].plot(time, normalize_1d(X), label='X (norm)')
		# axes[0].plot(time, signal_process(X), label='X (filt)')
		# axes[1].plot(time, normalize_1d(Y), label='Y (norm)')
		# axes[1].plot(time, signal_process(Y), label='Y (filt)')

		# axes[2].plot(time, normalize_1d(Effort_THO), label='Thoracic')
		# axes[2].plot(time, normalize_1d(Effort_ABD), label='Abdominal')

		# axes[3].plot(time, normalize_1d(TFlow), label='TFlow')
		# # axes[3].plot(time, normalize_1d(PFlow_1), label='PFlow')
		
		# final_event = Events+RERAs
		# final_event[360:435] = 1
		# final_event[160:225] = 0
		# time_event = np.arange(0, len(final_event)) / fs *10
		# axes[3].plot(time_event, final_event, label='Events', color='black')	
		# for i in range(row):
		# 	axes[i].legend(loc='upper right')
		# titles = [
		# 	'(a) X-axis vibration',
		# 	'(b) Y-axis vibration',
		# 	'(c) Respiratory effort (THO & ABD)'
		# ]

		# for ax, title in zip(axes, titles):
		# 	ax.set_title(title, fontsize=16)
		# plt.xlabel('Time [sec]')
		# plt.tight_layout()
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_true/{TYPE}_exp.png', bbox_inches='tight', dpi=300)
		# plt.close()

				


			