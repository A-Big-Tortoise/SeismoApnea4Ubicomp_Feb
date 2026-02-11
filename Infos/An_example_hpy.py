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
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import modify_magnitude_with_gaussian_noise

plt.rcParams.update({
		# 'font.family': 'Times New Roman',   # 字体
		'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],

		'font.size': 20,                    # 默认字体大小
		'axes.titlesize': 24,               # 子图标题
		'axes.labelsize': 22,               # x/y 轴标签
		'xtick.labelsize': 20,              # x 轴刻度
		'ytick.labelsize': 20,              # y 轴刻度
		'legend.fontsize': 20,              # 图例
	})




def signal_process(sig, low):

	sig = low_pass_filter(sig, Fs=100, low=low, order=3)
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
	axes[0].plot(time, normalize_1d(X), label='X (norm)', alpha=0.5)
	axes[0].plot(time, signal_process(X), label='X (filt)', linewidth=2)
	axes[1].plot(time, normalize_1d(Y), label='Y (norm)', alpha=0.5)
	axes[1].plot(time, signal_process(Y), label='Y (filt)', linewidth=2)
	axes[2].plot(time, normalize_1d(Z), label='Z (norm)', alpha=0.5)
	axes[2].plot(time, signal_process(Z), label='Z (filt)', linewidth=2)
	axes[3].plot(time, normalize_1d(Effort_THO), label='Thoracic', linewidth=2)
	axes[3].plot(time, normalize_1d(Effort_ABD), label='Abdominal', linewidth=2)

	axes[4].plot(time, normalize_1d(Flow), label='Airflow', linewidth=2)

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
	plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_new/{TYPE}_exp.png', bbox_inches='tight', dpi=300)
	plt.close()


import numpy as np

def modify_magnitude_multi_freq_noise(
    x_in,
    fs=10,
    band=(0, 1),
    centers=None,          # 例如 [0.15, 0.35, 0.7]
    n_centers=5,           # centers=None 时随机采样中心频率个数
    bw=0.05,               # 每个中心频率附近加噪的带宽(Hz)
    noise_std=5,           # 基础噪声强度
    stds=None,             # 可选：每个中心各自的 std，长度=K
    mode="add",            # "add" 或 "mul"（乘性扰动更“强”）
    clip_negative=True,
    return_spectrum=False,
    seed=None,
):
    """
    在 band 内的多个频率区域对幅度谱添加高斯噪声（可多个中心频率、窄带簇）。
    - mode="add": mag += N(0, std)
    - mode="mul": mag *= (1 + N(0, std))  # std 建议较小如 0.05~0.2
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x_in).copy()
    N = x.size

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1/fs)

    mag = np.abs(X)
    phase = np.angle(X)
    mag_noisy = mag.copy()

    f_low, f_high = band
    if centers is None:
        # 随机采中心频率，避免太靠边（留 bw/2 的余量）
        lo = max(f_low, 0) + bw/2
        hi = f_high - bw/2
        if hi <= lo:
            raise ValueError("band 太窄或 bw 太大，导致无法采样 centers")
        centers = rng.uniform(lo, hi, size=n_centers)
    else:
        centers = np.asarray(centers, dtype=float)

    K = len(centers)
    if stds is None:
        stds = np.full(K, noise_std, dtype=float)
    else:
        stds = np.asarray(stds, dtype=float)
        if stds.size != K:
            raise ValueError("stds 长度必须与 centers 一致")

    # 构造“多簇窄带 mask”
    mask = np.zeros(N, dtype=bool)
    for c in centers:
        mask |= (np.abs(np.abs(freqs) - c) <= bw/2)

    # 只在 band 内生效（双保险）
    mask &= (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)

    # 为了让每个中心频率“各自强度”生效，我们用一个权重场 w(freq)
    # w = sum_k exp(-(f-c_k)^2/(2*sigma^2)) * (std_k / noise_std)
    # 这样噪声在每个中心附近更强、且平滑过渡
    sigma = bw / 2.355  # 让 FWHM≈bw（近似）
    w = np.zeros(N, dtype=float)
    for c, s in zip(centers, stds):
        w += (s / (noise_std if noise_std != 0 else 1.0)) * np.exp(-0.5 * ((np.abs(freqs) - c) / sigma) ** 2)
    w *= mask.astype(float)

    if mode == "add":
        mag_noisy += rng.normal(0, noise_std, size=N) * w
    elif mode == "mul":
        # 乘性扰动更“猛”，但 std 要小一些（比如 0.05~0.2）
        mag_noisy *= (1.0 + rng.normal(0, noise_std, size=N) * w)
    else:
        raise ValueError("mode 只能是 'add' 或 'mul'")

    if clip_negative:
        mag_noisy = np.clip(mag_noisy, 0, None)

    # 关键：保持共轭对称（避免 ifft 后出现虚部漂移）
    # 最简单方法：仅对 rfft 做处理再 irfft，但这里用“对称化”兜底
    # 对称索引关系：k <-> (-k mod N)
    mag_noisy = 0.5 * (mag_noisy + mag_noisy[::-1])

    X_noisy = mag_noisy * np.exp(1j * phase)
    x_noisy = np.fft.ifft(X_noisy).real

    if return_spectrum:
        return x_noisy, mag, mag_noisy, centers
    return x_noisy





if __name__ == "__main__":

	hyp_urls = [
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1741576297585&to=1741576364998',	
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1742703060028&to=1742703179991',
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1741576920059&to=1741576980204',
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom4/sleep-lab-tri-axis-monitoring-room-4?orgId=1&var-mac=b8:27:eb:64:49:4d&var-name=vitalsigns&from=1745636292308&to=1745636461527',
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom3/sleep-lab-tri-axis-monitoring-room-3?orgId=1&var-mac=b8:27:eb:c2:a0:f9&var-name=vitalsigns&from=1746678731305&to=1746678790394'
			# 'https://sensorser'ver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1749441134404&to=1749441296225'
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1749441134404&to=1749441208487'
			# 'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1749441262256&to=1749441311267'
			'https://sensorserver.engr.uga.edu:3000/d/BSGRS3dRoom1/sleep-lab-tri-axis-monitoring-room-1?orgId=1&var-mac=b8:27:eb:1f:c1:84&var-name=vitalsigns&from=1749441639783&to=1749441700032'
			]
	





	influx_vitals_bsg = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'shake', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}
	influx_vitals_sleep = {'ip': 'https://sensorserver.engr.uga.edu', 'db': 'sleeplab', 'user': 'algtest', 'passw': 'sensorweb711', 'ssl': True}

	fs = 100

	TYPE = 'Hyp'  # 'CSA' or 'OSA'

	selected_urls = hyp_urls



	for cnt, url in enumerate(selected_urls):
		MAC = url.split('var-mac=')[1].split('&')[0]
		unix_start_time = int(url.split('from=')[1].split('&')[0]) / 1000.0
		unix_end_time = int(url.split('to=')[1].split('&')[0]) / 1000.0
		# unix_start_time = 1742012136.270
		# unix_end_time = unix_start_time + 60
		Room = int(url.split('Room')[1].split('/')[0][-1])

		print(f'Processing MAC: {MAC}, Room: {Room}, Start time: {unix_start_time}, End time: {unix_end_time}')
		
		start_latency = 4

		start_time = unix_start_time
		end_time = unix_end_time 

		start_time_bsg = unix_start_time 
		end_time_bsg = unix_end_time 

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

		final_event = Events
		Flow = TFlow

		Effort_THO = Effort_THO * -1
		Effort_ABD = Effort_ABD * -1
		# Y = Y * -1
		time = np.arange(0, len(Y)) / 100

		row = 3
		lw = 3

		fig, axes = plt.subplots(row, 1, figsize=(16, 3.35 *  row), sharex=True)
		X_y = signal_process(Y, low=0.5)
		X_y = modify_magnitude_multi_freq_noise(X_y )
		
		axes[0].plot(time, X_y, label='X (filt)', linewidth=lw)
		axes[0].plot(time, signal_process(Y, low=0.65), label='Y (filt)', linewidth=lw)
		axes[1].plot(time, normalize_1d(Effort_THO), label='THO', linewidth=lw)
		axes[1].plot(time, normalize_1d(Effort_ABD), label='ABD', linewidth=lw)
		

		from scipy.signal import butter, filtfilt

		def highpass_detrend(x, fs, fc=0.1, order=4):
			# fc: cutoff Hz
			b, a = butter(order, fc/(fs/2), btype='highpass')
			return filtfilt(b, a, x)
		axes[2].plot(time, highpass_detrend(normalize_1d(Flow), fs=100), label='Airflow', linewidth=lw)
		time_event = np.arange(0, len(Events)) / 100 * 10

		for i in range(row):
			axes[i].legend(loc='upper right')
		titles = [
			'(a) Filtered seismic signal (X & Y axes)',	
			'(b) Respiratory effort (THO & ABD)',
			'(c) Airflow (Nasal Pressure)'
		]

		for ax, title in zip(axes, titles):
			ax.set_title(title)
		plt.xlabel('Time [sec]')
		plt.tight_layout()
		plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Infos/figs_An/{TYPE}_comparison_long.png', bbox_inches='tight', dpi=300)
		plt.close()
