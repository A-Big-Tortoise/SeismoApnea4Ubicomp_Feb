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
def signal_process(sig):

	sig = low_pass_filter(sig, Fs=100, low=1, order=3)
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


# def read_excel(file_path):
#     df = pd.read_excel(file_path, sheet_name='Logs')
#     print(df.keys())
#     df_waiting = df[df['Status'] == 'Uploaded']
#     outputs = []
#     for index, row in df_waiting.iterrows():
#         Patient = row['Patient'].split(',')[0]
#         Mac = row['Mac'] 
#         Room = row['Room']
#         ID = row['ID']
#         Date = str(row['Start Time'].month) + '.' + str(row['Start Time'].day)
#         Start_Time, End_Time = row['Start Time'], row['End Time']
#         Sleep_Status = row['SStatus']
#         print(f"Patient: {Patient}, ID: {ID}, Mac: {Mac}, date: {Date}, StartTime: {Start_Time}, EndTime: {End_Time}")
		
#         print()
#         outputs.append((Patient, Mac, Date, Room, ID, Sleep_Status, Start_Time, End_Time))

#     return outputs

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


duration = 90
overlap = 30


if __name__ == "__main__":
	# infos = read_excel('/home/test/SleepLab/Code/SleepLab.xlsx')
	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
	influx_vitals_sleep = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	fs = 100


	MAC = 'b8:27:eb:1f:c1:84'
	ID = '5'
	unix_start_time = 1742012136.270
	# unix_end_time = 1742012199.726
	unix_end_time = unix_start_time + 60
	Room = 1
	


	start_time = unix_start_time
	end_time = unix_end_time

	start_time_bsg = start_time + 0.3
	end_time_bsg = end_time + 0.3

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

	
	Events, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
					table_name='SleepLab_test_5', data_name='Events', 
					start_timestamp=start_time, end_timestamp=end_time)
	
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


	
	time = np.arange(0, len(Y)) / fs
	row = 3
	from scipy.signal import medfilt
	X = medfilt(X, kernel_size=11)

	fig, axes = plt.subplots(row, 1, figsize=(15, 2.35 *  row), sharex=True)
	axes[0].plot(time, normalize_1d(X), label='X (norm)', alpha=0.5)
	axes[0].plot(time, signal_process(X), label='X (filt)', linewidth=2)
	axes[1].plot(time, normalize_1d(Y), label='Y (norm)', alpha=0.5)
	axes[1].plot(time, signal_process(Y), label='Y (filt)', linewidth=2)

	axes[2].plot(time, normalize_1d(Effort_THO), label='Thoracic', linewidth=2)
	axes[2].plot(time, normalize_1d(Effort_ABD), label='Abdominal', linewidth=2)
	
	for i in range(row):
		axes[i].legend(loc='upper right')
	titles = [
		'(a) X-axis vibration',
		'(b) Y-axis vibration',
		'(c) Respiratory effort (THO & ABD)'
	]

	for ax, title in zip(axes, titles):
		ax.set_title(title, fontsize=16)
	plt.xlabel('Time [sec]')
	plt.tight_layout()
	plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_new/rec_exp_mildfilter.png', bbox_inches='tight', dpi=300)
	plt.close()

	# RERAs, _ = read_influx(influx_vitals_sleep, unit=MAC,
	#                         table_name='SleepLab_test_5', data_name='RERAs',
	#                         start_timestamp=start_time, end_timestamp=end_time)
	# if len(RERAs) != 10 * duration:
	#     print(f'RERAs.shape: {RERAs.shape}')
	#     continue

	# Leg_1, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
	#                 table_name='SleepLab_test', data_name='Leg 1', 
	#                 start_timestamp=start_time, end_timestamp=end_time)
	# # Leg_1_Max = max(abs(Leg_1))
	# if len(Leg_1) != 200 * duration:
	#     print(f'Leg_1.shape: {Leg_1.shape}')
	#     continue

	# Leg_2, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
	#                 table_name='SleepLab_test', data_name='Leg 2', 
	#                 start_timestamp=start_time, end_timestamp=end_time)
	# if len(Leg_2) != 200 * duration:
	#     print(f'Leg_2.shape: {Leg_2.shape}')
	#     continue


	# Body, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
	#                 table_name='SleepLab_test', data_name='Body', 
	#                 start_timestamp=start_time, end_timestamp=end_time)
	# if len(Body) != duration:
	#     print(f'Body.shape: {Body.shape}')
	#     continue
	
	# Sleep_Stage, _ = read_influx(influx_vitals_sleep, unit=MAC,
	#                         table_name='SleepLab_Stage', data_name='staginglabel_v1',
	#                         start_timestamp=start_time, end_timestamp=end_time)

	# if len(Sleep_Stage) != duration:
	#     print(f'Sleep_Stage.shape: {Sleep_Stage.shape}')
	#     continue



			


		