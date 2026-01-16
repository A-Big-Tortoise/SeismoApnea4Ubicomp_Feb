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
from scipy.signal import butter, filtfilt, resample_poly


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


def read_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='Logs')
    print(df.keys())
    df_waiting = df[df['Status'] == 'Uploaded']
    outputs = []
    for index, row in df_waiting.iterrows():
        Patient = row['Patient'].split(',')[0]
        Mac = row['Mac'] 
        Room = row['Room']
        ID = row['ID']
        Date = str(row['Start Time'].month) + '.' + str(row['Start Time'].day)
        Start_Time, End_Time = row['Start Time'], row['End Time']
        Sleep_Status = row['SStatus']
        print(f"Patient: {Patient}, ID: {ID}, Mac: {Mac}, date: {Date}, StartTime: {Start_Time}, EndTime: {End_Time}")
        
        print()
        outputs.append((Patient, Mac, Date, Room, ID, Sleep_Status, Start_Time, End_Time))

    return outputs

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

    result = client.query(query)

    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    data = np.array(values)
    return data, times


def low_pass_filter(data, Fs, low, order):
	b, a = butter(order, low/(Fs * 0.5), 'low')
	N = len(data)
	padded = np.pad(data, (N, N), mode='reflect')
	filtered_data = filtfilt(b, a, padded)
	return filtered_data[N:-N]


def signal_process(sig):
	sig = resample_poly(sig,1,10)
	# sig = low_pass_filter(sig, Fs=10, low=0.75, order=3)
	sig = low_pass_filter(sig, Fs=10, low=1, order=3)
	# sig = sig[5:595]
	sig = normalize_1d(sig)
	return sig

def normalize_1d(data):
	data = (data - np.mean(data)) / np.std(data)
	return data


if __name__ == "__main__":
    infos = read_excel('/home/test/SleepLab/Code/SleepLab.xlsx')
    influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
    influx_vitals_sleep = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}



    MAC = ''
    ID = ''
    unix_start_time = 0
    unix_end_time = 0
    Room = 1
    


    start_time = unix_start_time
    end_time = unix_end_time

    if Room == 2: table_names = {'X':'E', 'Y':'N', 'Z':'Z'}
    else: table_names = {'X':'X', 'Y':'Y', 'Z':'Z'}


    X, _ = read_influx(influx_vitals_bsg, unit=MAC, 
                    table_name=table_names['X'], data_name='value', 
                    start_timestamp=start_time, end_timestamp=end_time)


    Y, _ = read_influx(influx_vitals_bsg, unit=MAC, 
                    table_name=table_names['Y'], data_name='value', 
                    start_timestamp=start_time, end_timestamp=end_time)

    Z, _ =  read_influx(influx_vitals_bsg, unit=MAC, 
                    table_name=table_names['Z'], data_name='value', 
                    start_timestamp=start_time, end_timestamp=end_time)

    
    Effort_THO, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
                    table_name='SleepLab_test', data_name='Effort THO', 
                    start_timestamp=start_time, end_timestamp=end_time)
    
    
    Effort_ABD, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
                    table_name='SleepLab_test', data_name='Effort ABD', 
                    start_timestamp=start_time, end_timestamp=end_time)

    
    Events, _ =  read_influx(influx_vitals_sleep, unit=MAC, 
                    table_name='SleepLab_test_5', data_name='Events', 
                    start_timestamp=start_time, end_timestamp=end_time)
    
    
    
    row = 3
    fig, axes = plt.subplots()

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



            


        