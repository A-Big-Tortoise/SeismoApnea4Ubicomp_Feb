import pandas as pd
import pynvml
import random
import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dl import ApneaDataset, ApneaDataset_MTL, BalancedBatchSampler, ApneaDataset_MTL_REC, ApneaDataset_TriClass
from Code.utils_dsp import denoise, denoise_band, normalize, modify_magnitude_with_gaussian_noise, denoise_iter
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.signal import resample_poly
from sklearn.metrics import (
	balanced_accuracy_score,
	f1_score
)


def calculate_icc_standard(labels, preds):
	data = np.column_stack([labels, preds])
	
	n = data.shape[0]  # 受试者数量
	k = data.shape[1]  # 评分者数量 (2)
	
	# 计算各种均值
	grand_mean = np.mean(data)                    # 总均值
	subject_means = np.mean(data, axis=1)         # 每个受试者的均值
	rater_means = np.mean(data, axis=0)           # 每个评分者的均值
	
	# 计算平方和
	# SST: Total sum of squares
	SST = np.sum((data - grand_mean) ** 2)
	
	# SSR: Sum of squares for rows (subjects/between subjects)
	SSR = k * np.sum((subject_means - grand_mean) ** 2)
	
	# SSC: Sum of squares for columns (raters/between raters)
	SSC = n * np.sum((rater_means - grand_mean) ** 2)
	
	# SSE: Sum of squares error (residual)
	SSE = SST - SSR - SSC
	
	# 计算自由度
	df_r = n - 1              # 受试者自由度
	df_c = k - 1              # 评分者自由度
	df_e = (n - 1) * (k - 1)  # 残差自由度
	
	# 计算均方
	MSR = SSR / df_r  # Mean square for subjects
	MSC = SSC / df_c  # Mean square for raters
	MSE = SSE / df_e  # Mean square error
	
	# 计算ICC(2,1) - Two-way random effects, absolute agreement, single measures
	# ICC = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)
	numerator = MSR - MSE
	denominator = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
	
	icc = numerator / denominator
	return icc


def ahi_to_severity(ahi):
	if ahi <= 5:
		return 0    # Normal
	elif ahi <= 15:
		return 1    # Mild
	elif ahi <= 30:
		return 2    # Moderate
	else:
		return 3    # Severe
  

  
def calculate_cm(y_true_list, y_pred_list, legend):
	y_true = np.array([ahi_to_severity(x) for x in y_true_list])
	y_pred = np.array([ahi_to_severity(x) for x in y_pred_list])
	bacc = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='macro')
	return f'{legend}, BACC, F1: {bacc*100:.2f}\%, {f1*100:.2f}\%'


def idx2filename(idx):
	df = pd.read_excel('/home/jiayu/SleepApnea4Ubicomp/Code/SleepLab.xlsx', sheet_name='Logs')
	df = df[df['ID'] == idx]
	patient_name = df['Patient'].values[0].split(',')[0]
	day = df['Identifier'].values[0].split('_')[-1]
	print(patient_name, day)
	filename = f'{patient_name}_{day}'
	return filename


def read_excel(file_path='/home/jiayu/SleepApnea4Ubicomp/Code/SleepLab.xlsx'):
	df = pd.read_excel(file_path, sheet_name='Logs')
	print(df.keys())
	df_waiting = df[df['Status'] == 'Uploaded']
	outputs = []
	for index, row in df_waiting.iterrows():
		Patient = row['Patient'].split(',')[0]
		Start_time = row['Start Time']
		Lights_out = row['Lights Out']
		Sleep_latency = row['SLatency']
		Lights_on = row['Lights On']
		date_roughly = row['Identifier'].split('_')[1]
		print(f"Patient: {Patient}")
		print()
		outputs.append((Patient, Start_time, Lights_out, Sleep_latency, Lights_on, date_roughly))

	return outputs



def time2timestamp(time_str):
	t = pd.Timestamp(time_str)
	t_ny = t.tz_localize('America/New_York')
	unix_time = t_ny.timestamp()
	return unix_time



def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False



def choose_gpu_by_model_process_count():

	def is_deep_learning_process(pid):
		try:
			cmdline_path = f"/proc/{pid}/cmdline"
			if not os.path.exists(cmdline_path):
				return False
			with open(cmdline_path, "r") as f:
				cmdline = f.read().lower()
				return "python" in cmdline or "torch" in cmdline or "tensorflow" in cmdline
		except:
			return False
		
	pynvml.nvmlInit()
	device_count = pynvml.nvmlDeviceGetCount()
	gpu_model_process_counts = []
	for i in range(device_count):
		handle = pynvml.nvmlDeviceGetHandleByIndex(i)
		try: procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
		except pynvml.NVMLError: procs = []
		model_procs = [p for p in procs if is_deep_learning_process(p.pid)]
		print(f"GPU {i}: Total processes = {len(procs)}, DL processes = {len(model_procs)}")
		gpu_model_process_counts.append(len(model_procs))
	best_gpu = gpu_model_process_counts.index(min(gpu_model_process_counts))
	pynvml.nvmlShutdown()
	return best_gpu



def load_val_test_data(data_path, fold_idx):
	if fold_idx == 1:
		train_fold = [3, 4]
		valid_fold = [2]
	if fold_idx == 2:
		train_fold = [1, 4]
		valid_fold = [3]
	if fold_idx == 3:
		train_fold = [1, 2]
		valid_fold = [4]
	if fold_idx == 4:
		train_fold = [2, 3]
		valid_fold = [1]
	
	val_data, test_data = [], []
	for i in range(1, 5):
		print(f'Loading data from fold {i} ...')
		data_file_path = data_path + f'fold{i}.npy'
		all_data = np.load(data_file_path)
		if i in train_fold: pass
		elif i in valid_fold: val_data.append(all_data)
		else: test_data.append(all_data)

	print(f'Total val files: {len(val_data)}, Total test files: {len(test_data)}')
	val_data = np.concatenate(val_data)
	test_data = np.concatenate(test_data)
	print(f'Loaded Val Data: {val_data.shape}, Loaded Test Data: {test_data.shape}')
	return val_data, test_data



def load_train_val_test_data(data_path, fold_idx):
	if fold_idx == 1:
		train_fold = [3, 4]
		valid_fold = [2]
	if fold_idx == 2:
		train_fold = [1, 4]
		valid_fold = [3]
	if fold_idx == 3:
		train_fold = [1, 2]
		valid_fold = [4]
	if fold_idx == 4:
		train_fold = [2, 3]
		valid_fold = [1]
	
	train_data, val_data, test_data = [], [], []
	for i in range(1, 5):
		print(f'Loading data from fold {i} ...')
		data_file_path = data_path + f'fold{i}.npy'
		all_data = np.load(data_file_path)
		if i in train_fold: train_data.append(all_data)
		elif i in valid_fold: val_data.append(all_data)
		else: test_data.append(all_data)

	print(f'Total train files: {len(train_data)}, Total val files: {len(val_data)}, Total test files: {len(test_data)}')
	train_data = np.concatenate(train_data)
	print(f'After concatenation, train_data.shape: {train_data.shape}')
	val_data = np.concatenate(val_data)
	test_data = np.concatenate(test_data)
	print(f'Loaded Train Data: {train_data.shape}, Loaded Val Data: {val_data.shape}, Loaded Test Data: {test_data.shape}')
	return train_data, val_data, test_data



def load_train_val_data(data_path, fold_idx):
	data_files = np.sort(os.listdir(data_path))   

	train_data, val_data = [], []
	for i in range(len(data_files)):
		print('Loading data: ', data_files[i])
		all_data = np.load(data_path + data_files[i])
		if fold_idx == i + 1: val_data.append(all_data)
		else: train_data.append(all_data)
	print(f'Total train files: {len(train_data)}, Total val files: {len(val_data)}')
	train_data = np.concatenate(train_data)
	print(f'After concatenation, train_data.shape: {train_data.shape}')
	val_data = np.concatenate(val_data)
	print(f'Loaded Train Data: {train_data.shape}, Loaded Val Data: {val_data.shape}')
	
	return train_data, val_data


def load_val_data(data_path, fold_idx):
	data_files = np.sort(os.listdir(data_path))   

	val_data = []
	for i in range(len(data_files)):
		print('Loading data: ', data_files[i])
		if fold_idx == i + 1: 
			all_data = np.load(data_path + data_files[i])
			val_data.append(all_data)
	val_data = np.concatenate(val_data)
	print(f'Loaded Val Data: {val_data.shape}')
	
	return val_data



def augmentation(X, Y, Z, Events, others):
	augmented_X, augmented_Y, augmented_Z = [], [], []
	augmented_Events, augmented_Others = [], []

	for i in range(len(Y)):
		cnt = 0

		if Events[i] != 0: 
			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i])]
			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i])]
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i])]
			augmented_X.extend(x_aug)
			augmented_Y.extend(y_aug)
			augmented_Z.extend(z_aug)
			cnt += 4


			# x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
			# y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
			# z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
			# augmented_X.extend(x_aug)
			# augmented_Y.extend(y_aug)
			# augmented_Z.extend(z_aug)
			# cnt += 3


		else:
			if np.random.rand() < 0.1:
				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
				augmented_X.extend(x_aug)
				augmented_Y.extend(y_aug)
				augmented_Z.extend(z_aug)
				cnt += 3
			if np.random.rand() < 0.1:
				x_aug = modify_magnitude_with_gaussian_noise(X[i])
				y_aug = modify_magnitude_with_gaussian_noise(Y[i])
				z_aug = modify_magnitude_with_gaussian_noise(Z[i])
				augmented_X.append(x_aug)
				augmented_Y.append(y_aug)
				augmented_Z.append(z_aug)
				cnt += 1
			
		augmented_Events.extend([Events[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	X = np.concatenate((X, np.array(augmented_X)), axis=0)
	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return X, Y, Z, Events, others


def augmentation_TriClass(X, Y, Z, Labels, others):
	augmented_X, augmented_Y, augmented_Z = [], [], []
	augmented_Labels, augmented_Others = [], []

	for i in range(len(Y)):
		cnt = 0

		if Labels[i] != 0: 
			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i])]
			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i])]
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i])]
			augmented_X.extend(x_aug)
			augmented_Y.extend(y_aug)
			augmented_Z.extend(z_aug)
			cnt += 4
		else:
			if np.random.rand() < 0.1:
				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
				augmented_X.extend(x_aug)
				augmented_Y.extend(y_aug)
				augmented_Z.extend(z_aug)
				cnt += 3
			if np.random.rand() < 0.1:
				x_aug = modify_magnitude_with_gaussian_noise(X[i])
				y_aug = modify_magnitude_with_gaussian_noise(Y[i])
				z_aug = modify_magnitude_with_gaussian_noise(Z[i])
				augmented_X.append(x_aug)
				augmented_Y.append(y_aug)
				augmented_Z.append(z_aug)
				cnt += 1

		augmented_Labels.extend([Labels[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	X = np.concatenate((X, np.array(augmented_X)), axis=0)
	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
	Labels = np.concatenate((Labels, np.array(augmented_Labels)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return X, Y, Z, Labels, others

def augmentation_MTL_Z(Z, Events, Stages, others, std):
	augmented_Z = []
	augmented_Events, augmented_Stages, augmented_Others = [], [], []

	for i in range(len(Z)):
		cnt = 0

		if Events[i] != 0 or Stages[i] != 0: 
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i], noise_std=std)]
			augmented_Z.extend(z_aug)
			cnt += 4
		else:
			# if np.random.rand() < 0.01:
			# 	x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
			# 	y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
		
			if np.random.rand() < 0.1:
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
				augmented_Z.extend(z_aug)
				cnt += 3
			if np.random.rand() < 0.1:
				z_aug = modify_magnitude_with_gaussian_noise(Z[i], noise_std=std)
				augmented_Z.append(z_aug)
				cnt += 1

		augmented_Events.extend([Events[i]] * cnt)
		augmented_Stages.extend([Stages[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
	Stages = np.concatenate((Stages, np.array(augmented_Stages)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return Z, Events, Stages, others


def augmentation_MTL(X, Y, Z, Events, Stages, others, std):
	augmented_X, augmented_Y, augmented_Z = [], [], []
	augmented_Events, augmented_Stages, augmented_Others = [], [], []

	for i in range(len(Y)):
		cnt = 0

		if Events[i] != 0 or Stages[i] != 0: 
			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i], noise_std=std)]
			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i], noise_std=std)]
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i], noise_std=std)]
			augmented_X.extend(x_aug)
			augmented_Y.extend(y_aug)
			augmented_Z.extend(z_aug)
			cnt += 4
		else:
			# if np.random.rand() < 0.01:
			# 	x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
			# 	y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
			# 	z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
			# 	augmented_X.extend(x_aug)
			# 	augmented_Y.extend(y_aug)
			# 	augmented_Z.extend(z_aug)
			# 	cnt += 3
			# if np.random.rand() < 0.01:
			# 	x_aug = modify_magnitude_with_gaussian_noise(X[i])
			# 	y_aug = modify_magnitude_with_gaussian_noise(Y[i])
			# 	z_aug = modify_magnitude_with_gaussian_noise(Z[i])
			# 	augmented_X.append(x_aug)
			# 	augmented_Y.append(y_aug)
			# 	augmented_Z.append(z_aug)
			# 	cnt += 1
			if np.random.rand() < 0.1:
				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
				augmented_X.extend(x_aug)
				augmented_Y.extend(y_aug)
				augmented_Z.extend(z_aug)
				cnt += 3
			if np.random.rand() < 0.1:
				x_aug = modify_magnitude_with_gaussian_noise(X[i], noise_std=std)
				y_aug = modify_magnitude_with_gaussian_noise(Y[i], noise_std=std)
				z_aug = modify_magnitude_with_gaussian_noise(Z[i], noise_std=std)
				augmented_X.append(x_aug)
				augmented_Y.append(y_aug)
				augmented_Z.append(z_aug)
				cnt += 1

		augmented_Events.extend([Events[i]] * cnt)
		augmented_Stages.extend([Stages[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	X = np.concatenate((X, np.array(augmented_X)), axis=0)
	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
	Stages = np.concatenate((Stages, np.array(augmented_Stages)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return X, Y, Z, Events, Stages, others


def augmentation_MTL_REC(X, Y, Z, THO, ABD, Events, Stages, others):
	augmented_X, augmented_Y, augmented_Z = [], [], []
	augmented_THO, augmented_ABD = [], []
	augmented_Events, augmented_Stages, augmented_Others = [], [], []

	for i in range(len(Y)):
		cnt = 0

		if Events[i] != 0 or Stages[i] != 0: 
			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i])]
			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i])]
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i])]
			tho_aug = [-THO[i], THO[i, ::-1], -THO[i, ::-1], THO[i]]
			abd_aug = [-ABD[i], ABD[i, ::-1], -ABD[i, ::-1], ABD[i]]
			augmented_X.extend(x_aug)
			augmented_Y.extend(y_aug)
			augmented_Z.extend(z_aug)
			augmented_THO.extend(tho_aug)
			augmented_ABD.extend(abd_aug)
			cnt += 4
		else:
			if np.random.rand() < 0.1:
				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
				tho_aug = [-THO[i], THO[i, ::-1], -THO[i, ::-1]]
				abd_aug = [-ABD[i], ABD[i, ::-1], -ABD[i, ::-1]]
				augmented_X.extend(x_aug)
				augmented_Y.extend(y_aug)
				augmented_Z.extend(z_aug)
				augmented_THO.extend(tho_aug)
				augmented_ABD.extend(abd_aug)
				cnt += 3
			if np.random.rand() < 0.1:
				x_aug = modify_magnitude_with_gaussian_noise(X[i])
				y_aug = modify_magnitude_with_gaussian_noise(Y[i])
				z_aug = modify_magnitude_with_gaussian_noise(Z[i])
				augmented_X.append(x_aug)
				augmented_Y.append(y_aug)
				augmented_Z.append(z_aug)
				augmented_THO.append(THO[i])
				augmented_ABD.append(ABD[i])
				cnt += 1

		augmented_Events.extend([Events[i]] * cnt)
		augmented_Stages.extend([Stages[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	X = np.concatenate((X, np.array(augmented_X)), axis=0)
	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
	THO = np.concatenate((THO, np.array(augmented_THO)), axis=0)
	ABD = np.concatenate((ABD, np.array(augmented_ABD)), axis=0)

	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
	Stages = np.concatenate((Stages, np.array(augmented_Stages)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return X, Y, Z, THO, ABD, Events, Stages, others




def augmentation2(X, Y, Z, X_hr, Y_hr, Z_hr, Events, others):
	augmented_X, augmented_Y, augmented_Z = [], [], []
	augmented_X_hr, augmented_Y_hr, augmented_Z_hr = [], [], []
	augmented_Events, augmented_Others = [], []

	for i in range(len(Y)):
		cnt = 0

		if Events[i] != 0: 
			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i])]
			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i])]
			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i])]

			x_hr_aug = [-X_hr[i], X_hr[i, ::-1], -X_hr[i, ::-1], modify_magnitude_with_gaussian_noise(X_hr[i])]
			y_hr_aug = [-Y_hr[i], Y_hr[i, ::-1], -Y_hr[i, ::-1], modify_magnitude_with_gaussian_noise(Y_hr[i])]
			z_hr_aug = [-Z_hr[i], Z_hr[i, ::-1], -Z_hr[i, ::-1], modify_magnitude_with_gaussian_noise(Z_hr[i])]

			augmented_X.extend(x_aug)
			augmented_Y.extend(y_aug)
			augmented_Z.extend(z_aug)

			augmented_X_hr.extend(x_hr_aug)
			augmented_Y_hr.extend(y_hr_aug)
			augmented_Z_hr.extend(z_hr_aug)

			cnt += 4
		else:
			if np.random.rand() < 0.1:
				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]

				x_hr_aug = [-X_hr[i], X_hr[i, ::-1], -X_hr[i, ::-1]]
				y_hr_aug = [-Y_hr[i], Y_hr[i, ::-1], -Y_hr[i, ::-1]]
				z_hr_aug = [-Z_hr[i], Z_hr[i, ::-1], -Z_hr[i, ::-1]]

				augmented_X.extend(x_aug)
				augmented_Y.extend(y_aug)
				augmented_Z.extend(z_aug)

				augmented_X_hr.extend(x_hr_aug)
				augmented_Y_hr.extend(y_hr_aug)
				augmented_Z_hr.extend(z_hr_aug)

				cnt += 3
			if np.random.rand() < 0.1:
				x_aug = modify_magnitude_with_gaussian_noise(X[i])
				y_aug = modify_magnitude_with_gaussian_noise(Y[i])
				z_aug = modify_magnitude_with_gaussian_noise(Z[i])

				x_hr_aug = modify_magnitude_with_gaussian_noise(X_hr[i])
				y_hr_aug = modify_magnitude_with_gaussian_noise(Y_hr[i])
				z_hr_aug = modify_magnitude_with_gaussian_noise(Z_hr[i])

				augmented_X.append(x_aug)
				augmented_Y.append(y_aug)
				augmented_Z.append(z_aug)

				augmented_X_hr.append(x_hr_aug)
				augmented_Y_hr.append(y_hr_aug)
				augmented_Z_hr.append(z_hr_aug)

				cnt += 1

		augmented_Events.extend([Events[i]] * cnt)
		augmented_Others.extend([others[i]] * cnt)

	X = np.concatenate((X, np.array(augmented_X)), axis=0)
	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)

	X_hr = np.concatenate((X_hr, np.array(augmented_X_hr)), axis=0)
	Y_hr = np.concatenate((Y_hr, np.array(augmented_Y_hr)), axis=0)
	Z_hr = np.concatenate((Z_hr, np.array(augmented_Z_hr)), axis=0)

	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

	return X, Y, Z, X_hr, Y_hr, Z_hr, Events, others




def remove_sleep_stage(data, sleep_stage_idx, nseg):
	sleepstage = data[:, sleep_stage_idx:sleep_stage_idx+nseg]
	masks = []
	for i in range(len(sleepstage)):
		stage = sleepstage[i]
		if 4 in stage or -1 in stage:
			masks.append(False)
		else:
			masks.append(True)
	masks = np.array(masks)
	print(f'Removing {np.sum(~masks)} samples due to Sleep Stage')
	data = data[masks]
	return data



def data_preprocess(data, Type, raw, sleep_index, nseg, index):
	print('-'*50)
	print(f'In {Type} ...')

	# Remove Mixed Apnea
	# valid_mask_MixedApnea = data[:, -1] < 4
	# data = data[valid_mask_MixedApnea] 
	# print(f'Removing {np.sum(~valid_mask_MixedApnea)} samples due to Mixed Apnea')


	# Remove Sleep Stage 
	if index == 'AHI':
		data = remove_sleep_stage(data, sleep_index, nseg)
	


	# data = remove_sleep_stage(data, sleep_index, nseg)

	X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
	# X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]
	# if Type == 'train':
	#     print(f'train with belt')
	#     X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]
	# elif Type == 'val':
	#     print(f'val with bsg')
	#     # X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
	#     X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]

	# others = data[:, :-1]
	others = data[:, -5:-1]
	Events = data[:, -1]

	if raw:
		# X, Y, Z = denoise(X, low=10), denoise(Y, low=10), denoise(Z, low=10)
		X, Y, Z = denoise(X), denoise(Y), denoise(Z)
		X, Y, Z = resample_poly(X,1,5,axis=1), resample_poly(Y,1,5,axis=1), resample_poly(Z,1,5,axis=1)
		X, Y, Z = X[:, 5:-5], Y[:, 5:-5], Z[:, 5:-5]
	else:
		# X, Y, Z = denoise(X), denoise(Y), denoise(Z)
		X = denoise_iter(X)
		Y = denoise_iter(Y)
		Z = denoise_iter(Z)
		print('Denoising')
		X, Y, Z = resample_poly(X,1,10,axis=1), resample_poly(Y,1,10,axis=1), resample_poly(Z,1,10,axis=1)
		X, Y, Z = X[:, 5:595], Y[:, 5:595], Z[:, 5:595]
		print(f'Downsampled')

	# X, Y, Z = normalize2(X), normalize2(Y), normalize2(Z)
	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	print("Normalization")
	if Type == 'train':
		X, Y, Z, Events, others = augmentation(X, Y, Z, Events, others)
		print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	unique_events = np.unique(Events)
	for i in range(len(unique_events)):
		print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))
	
	return X, Y, Z, Events, others


def data_preprocess_TriClass(data, Type):
	print('-'*50)
	print(f'In {Type} ...')

	X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]

	others = data[:, -6:-2]
	Labels = data[:, -1]
	


	# X, Y, Z = denoise(X), denoise(Y), denoise(Z)
	X = denoise_iter(X)
	Y = denoise_iter(Y)
	Z = denoise_iter(Z)
	print('Denoising')
	X, Y, Z = resample_poly(X,1,10,axis=1), resample_poly(Y,1,10,axis=1), resample_poly(Z,1,10,axis=1)
	X, Y, Z = X[:, 5:595], Y[:, 5:595], Z[:, 5:595]
	print(f'Downsampled')

	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	print("Normalization")
	if Type == 'train':
		X, Y, Z, Labels, others = augmentation_TriClass(X, Y, Z, Labels, others)
		print(f'After Augmentation, Labels_{Type}.shape: ', Labels.shape)

	unique_labels = np.unique(Labels)
	for i in range(len(unique_labels)):
		print(f'Labels_{Type} {unique_labels[i]}: ', np.sum(Labels == unique_labels[i]))
	
	return X, Y, Z, Labels, others


def data_preprocess_MTL_Seqlen(data, Type, duration):
	print('-'*50)
	print(f'In {Type} ...')

	seqlen = duration * 100
	downsampled_seqlen = duration * 10
	X, Y, Z = data[:, :seqlen], data[:, seqlen:seqlen*2], data[:, seqlen*2:seqlen*3]

	others = data[:, -6:-2]
	Stages = data[:, -2]
	Events = data[:, -1]


	# X, Y, Z = denoise(X), denoise(Y), denoise(Z)
	X = denoise_iter(X)
	Y = denoise_iter(Y)
	Z = denoise_iter(Z)
	print('Denoising')
	X, Y, Z = resample_poly(X,1,10,axis=1), resample_poly(Y,1,10,axis=1), resample_poly(Z,1,10,axis=1)
	X, Y, Z = X[:, 5:downsampled_seqlen-5], Y[:, 5:downsampled_seqlen-5], Z[:, 5:downsampled_seqlen-5]
	print(f'Downsampled')

	# X, Y, Z = normalize2(X), normalize2(Y), normalize2(Z)
	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	print("Normalization")
	if Type == 'train':
		X, Y, Z, Events, Stages, others = augmentation_MTL(X, Y, Z, Events, Stages, others)
		print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	unique_events = np.unique(Events)
	for i in range(len(unique_events)):
		print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))
	
	return X, Y, Z, Stages, Events, others


def data_preprocess_MTL_Z(data, Type, std=5):
	print('-'*50)
	print(f'In {Type} ...')

	Z = data[:, 12000:18000]
	# others = data[:, :-1]
	others = data[:, -6:-2]
	Stages = data[:, -2]
	Events = data[:, -1]

	# X, Y, Z = denoise(X), denoise(Y), denoise(Z)
	Z = denoise_iter(Z)
	print('Denoising')
	Z = resample_poly(Z,1,4,axis=1)
	Z = Z[:, 5:int(60*100/4)-5]
	print(f'Downsampled')

	# X, Y, Z = normalize2(X), normalize2(Y), normalize2(Z)
	Z = normalize(Z)
	print("Normalization")
	if Type == 'train':
		Z, Events, Stages, others = augmentation_MTL_Z(Z, Events, Stages, others, std)
		print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	unique_events = np.unique(Events)
	for i in range(len(unique_events)):
		print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))
	
	return Z, Stages, Events, others





def data_preprocess_MTL(data, Type, std=5):
	print('-'*50)
	print(f'In {Type} ...')

	# Remove Mixed Apnea
	# valid_mask_MixedApnea = data[:, -1] < 4
	# data = data[valid_mask_MixedApnea] 
	# print(f'Removing {np.sum(~valid_mask_MixedApnea)} samples due to Mixed Apnea')


	# Remove Sleep Stage 
	# if index == 'AHI':
	# 	data = remove_sleep_stage(data, sleep_index, nseg)
	


	# data = remove_sleep_stage(data, sleep_index, nseg)

	X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
	# X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]
	# if Type == 'train':
	#     print(f'train with belt')
	#     X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]
	# elif Type == 'val':
	#     print(f'val with bsg')
	#     # X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
	#     X, Y, Z = data[:, 18000:24000], data[:, 24000:30000], data[:, 12000:18000]

	# others = data[:, :-1]
	others = data[:, -6:-2]
	Stages = data[:, -2]
	Events = data[:, -1]
	


	# X, Y, Z = denoise(X), denoise(Y), denoise(Z)
	X = denoise_iter(X)
	Y = denoise_iter(Y)
	Z = denoise_iter(Z)
	print('Denoising')
	X, Y, Z = resample_poly(X,1,10,axis=1), resample_poly(Y,1,10,axis=1), resample_poly(Z,1,10,axis=1)
	X, Y, Z = X[:, 5:595], Y[:, 5:595], Z[:, 5:595]
	print(f'Downsampled')

	# X, Y, Z = normalize2(X), normalize2(Y), normalize2(Z)
	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	print("Normalization")
	if Type == 'train':
		X, Y, Z, Events, Stages, others = augmentation_MTL(X, Y, Z, Events, Stages, others, std)
		print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	unique_events = np.unique(Events)
	for i in range(len(unique_events)):
		print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))
	
	return X, Y, Z, Stages, Events, others


def data_preprocess_MTL_REC(data, Type):
	print('-'*50)
	print(f'In {Type} ...')

	X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
	THO, ABD = data[:, 18000: 24000], data[:, 24000: 30000]
	others = data[:, -6:-2]
	Stages = data[:, -2]
	Events = data[:, -1]


	X = denoise_iter(X)
	Y = denoise_iter(Y)
	Z = denoise_iter(Z)
	THO = denoise_iter(THO)
	ABD = denoise_iter(ABD)
	print('Denoising')

	X, Y, Z = resample_poly(X,1,10,axis=1), resample_poly(Y,1,10,axis=1), resample_poly(Z,1,10,axis=1)
	THO, ABD = resample_poly(THO,1,10,axis=1), resample_poly(ABD,1,10,axis=1)
	X, Y, Z = X[:, 5:595], Y[:, 5:595], Z[:, 5:595]
	THO, ABD = THO[:, 5:595], ABD[:, 5:595]
	print(f'Downsampled')

	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	THO, ABD = normalize(THO), normalize(ABD)
	print("Normalization")
	if Type == 'train':
		# X, Y, Z, Events, Stages, others = augmentation_MTL_REC(X, Y, Z, Events, Stages, others)
		X, Y, Z, THO, ABD, Events, Stages, others = augmentation_MTL_REC(X, Y, Z, THO, ABD, Events, Stages, others)
		print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	unique_events = np.unique(Events)
	for i in range(len(unique_events)):
		print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))
	
	return X, Y, Z, THO, ABD, Stages, Events, others



def npy2dataset_sleep(data_path, fold_idx, args, inference=False):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_train, Y_train, Z_train, Events_train, others_train = data_preprocess(train_data, 'train', args.raw, args.sleep_index, args.nseg, args.Index)
	X_val, Y_val, Z_val, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)
	X_test, Y_test, Z_test, Events_test, others_test = data_preprocess(test_data, 'test', args.raw, args.sleep_index, args.nseg, args.Index)


	print(X_train.shape, Y_train.shape, Z_train.shape)
	print(X_val.shape, Y_val.shape, Z_val.shape)
	print(X_test.shape, Y_test.shape, Z_test.shape)

	
	if args.XYZ == 'XY':
		Signals_train = np.stack([X_train, Y_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test], axis=-1)
	elif args.XYZ == 'XYZ':
		print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, Z_train: {Z_train.shape}')
		Signals_train = np.stack([X_train, Y_train, Z_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val, Z_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test, Z_test], axis=-1)
	
	print(f'{args.XYZ} shape: Train: {Signals_train.shape}, Val: {Signals_val.shape}, Test: {Signals_test.shape}')
	

	if inference:   
		train_dataset = ApneaDataset(Signals_train, Events_train, others_train)
		train_loader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=False, drop_last=False)
		val_dataset = ApneaDataset(Signals_val, Events_val, others_val)
		# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
		test_dataset = ApneaDataset(Signals_test, Events_test, others_test)
		test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
		return train_loader, val_loader, test_loader
	else:
		train_dataset = ApneaDataset(Signals_train, Events_train)
		train_sampler = BalancedBatchSampler(train_dataset.labels_np, batch_size=args.batch_size)
		train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
		# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
		val_dataset = ApneaDataset(Signals_val, Events_val)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
		test_dataset = ApneaDataset(Signals_test, Events_test)
		test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
		return train_loader, val_loader, test_loader
 


def npy2dataset_inference_sleep(data_path, fold_idx, args, inference=False):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_val, Y_val, Z_val, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)
	X_test, Y_test, Z_test, Events_test, others_test = data_preprocess(test_data, 'test', args.raw, args.sleep_index, args.nseg, args.Index)


	print(X_val.shape, Y_val.shape, Z_val.shape)
	print(X_test.shape, Y_test.shape, Z_test.shape)

	
	if args.XYZ == 'XY':
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test], axis=-1)

	print(f'{args.XYZ} shape: Val: {Signals_val.shape}, Test: {Signals_test.shape}')
	


	val_dataset = ApneaDataset(Signals_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	test_dataset = ApneaDataset(Signals_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	return val_loader, test_loader




def npy2dataset(data_path, fold_idx, args, inference=False):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data = load_train_val_data(data_path, fold_idx)


	X_train, Y_train, Z_train, Events_train, others_train = data_preprocess(train_data, 'train', args.raw, args.sleep_index, args.nseg, args.Index)
	X_val, Y_val, Z_val, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)


	print(X_train.shape, Y_train.shape, Z_train.shape)
	print(X_val.shape, Y_val.shape, Z_val.shape)

	
	if args.XYZ == 'XY':
		Signals_train = np.stack([X_train, Y_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
	elif args.XYZ == 'XYZ':
		print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, Z_train: {Z_train.shape}')
		Signals_train = np.stack([X_train, Y_train, Z_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val, Z_val], axis=-1)
	elif args.XYZ == 'Y':
		Signals_train = np.expand_dims(Y_train, axis=-1)
		Signals_val = np.expand_dims(Y_val, axis=-1)

	print(f'{args.XYZ} shape: Train: {Signals_train.shape}, Val: {Signals_val.shape}')
	

	if inference:   
# loader = DataLoader(dataset, batch_sampler=sampler)
		train_dataset = ApneaDataset(Signals_train, Events_train, others_train)
		train_loader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=False, drop_last=False)
		val_dataset = ApneaDataset(Signals_val, Events_val, others_val)
		# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
		return train_loader, val_loader
	else:

		train_dataset = ApneaDataset(Signals_train, Events_train)
		train_sampler = BalancedBatchSampler(train_dataset.labels_np, batch_size=args.batch_size)
		train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
		# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
		val_dataset = ApneaDataset(Signals_val, Events_val)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

		return train_loader, val_loader



def npy2dataset_true(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)


	X_train, Y_train, _, Events_train, others_train = data_preprocess(train_data, 'train', args.raw, args.sleep_index, args.nseg, args.Index)
	X_val, Y_val, _, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)
	X_test, Y_test, _, Events_test, others_test = data_preprocess(test_data, 'test', args.raw, args.sleep_index, args.nseg, args.Index)

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_train = np.stack([X_train, Y_train], axis=-1) 
	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	train_dataset = ApneaDataset(Signals_train, Events_train)
	train_sampler = BalancedBatchSampler(train_dataset.labels_np, batch_size=args.batch_size)
	train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
	# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset(Signals_val, Events_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset(Signals_test, Events_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader




def npy2dataset_inference_true(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_val, Y_val, _, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)
	X_test, Y_test, _, Events_test, others_test = data_preprocess(test_data, 'test', args.raw, args.sleep_index, args.nseg, args.Index)

	print(f' X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_val: {X_val.shape}, Y_val: {Y_val.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset(Signals_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset(Signals_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return val_loader, test_loader




from torch.utils.data import Sampler, DataLoader

class ThreeGroupRatioBatchSampler(Sampler):
    """
    每个 batch 固定比例：
      Wake (stage=1)                 : 1/2
      Sleep & Non-Apnea (stage=0,e=0): 1/4
      Sleep & Apnea (stage=0,e=1)    : 1/4

    等价于：
      Sleep:Wake = 1:1
      Sleep内部 Apnea:Non-Apnea = 1:1
    """
    def __init__(self, stages, events, batch_size, drop_last=True, seed=42):
        self.stages = np.asarray(stages)
        self.events = np.asarray(events)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        if self.batch_size % 4 != 0:
            raise ValueError(f"batch_size 必须能被4整除（你现在是 {self.batch_size}），才能严格做到 1/2,1/4,1/4。")

        self.idx_wake = np.where(self.stages == 1)[0]
        self.idx_s0   = np.where((self.stages == 0) & (self.events == 0))[0]
        self.idx_s1   = np.where((self.stages == 0) & (self.events == 1))[0]

        if len(self.idx_wake) == 0 or len(self.idx_s0) == 0 or len(self.idx_s1) == 0:
            raise ValueError(f"某一组样本为0：wake={len(self.idx_wake)}, sleep0={len(self.idx_s0)}, sleep1={len(self.idx_s1)}")

        self.n_wake = self.batch_size // 2
        self.n_s0   = self.batch_size // 4
        self.n_s1   = self.batch_size // 4

        # 一个 epoch 能出多少个 batch：由“最稀缺组”决定（不 replacement 的情况下）
        # 这里我们仍按这个长度定义 epoch，但采样时允许 replacement 补齐，训练更稳
        limiting = min(len(self.idx_wake) // self.n_wake,
                       len(self.idx_s0)   // self.n_s0,
                       len(self.idx_s1)   // self.n_s1)
        self.num_batches = max(limiting, 1)

    def __len__(self):
        return self.num_batches

    def _sample(self, idx, k):
        # 优先不放回；不够则放回补齐
        if len(idx) >= k:
            return self.rng.choice(idx, size=k, replace=False)
        else:
            return self.rng.choice(idx, size=k, replace=True)

    def __iter__(self):
        for _ in range(self.num_batches):
            w  = self._sample(self.idx_wake, self.n_wake)
            s0 = self._sample(self.idx_s0,   self.n_s0)
            s1 = self._sample(self.idx_s1,   self.n_s1)

            batch = np.concatenate([w, s0, s1])
            self.rng.shuffle(batch)
            yield batch.tolist()


def npy2dataset_MTL(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data = load_train_val_data(data_path, fold_idx)


	X_train, Y_train, _, Stages_train, Events_train, others_train = data_preprocess_MTL(train_data, 'train')
	X_val, Y_val, _, Stages_val, Events_val, others_val = data_preprocess_MTL(val_data, 'val')

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}')

	Signals_train = np.stack([X_train, Y_train], axis=-1) 
	Signals_val = np.stack([X_val, Y_val], axis=-1)
	print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_val: {X_val.shape}, Y_val: {Y_val.shape}')

	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	train_dataset = ApneaDataset_MTL(Signals_train, Stages_train, Events_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader


def npy2dataset_true_TriClass(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_train, Y_train, _, Labels_train, others_train = data_preprocess_TriClass(train_data, 'train')
	X_val, Y_val, _, Labels_val, others_val = data_preprocess_TriClass(val_data, 'val')
	X_test, Y_test, _, Labels_test, others_test = data_preprocess_TriClass(test_data, 'test')

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	if args.XYZ == 'XY':
		Signals_train = np.stack([X_train, Y_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test], axis=-1)
	elif args.XYZ == 'X':
		Signals_train = np.expand_dims(X_train, axis=-1)
		Signals_val = np.expand_dims(X_val, axis=-1)
		Signals_test = np.expand_dims(X_test, axis=-1)
	elif args.XYZ == 'Y':
		Signals_train = np.expand_dims(Y_train, axis=-1)
		Signals_val = np.expand_dims(Y_val, axis=-1)
		Signals_test = np.expand_dims(Y_test, axis=-1)


	train_dataset = ApneaDataset_TriClass(Signals_train, Labels_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_TriClass(Signals_val, Labels_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_TriClass(Signals_test, Labels_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader



def npy2dataset_true_MTL_Seqlen(data_path, fold_idx, duration, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_train, Y_train, _, Stages_train, Events_train, others_train = data_preprocess_MTL_Seqlen(train_data, 'train', duration)
	X_val, Y_val, _, Stages_val, Events_val, others_val = data_preprocess_MTL_Seqlen(val_data, 'val', duration)
	X_test, Y_test, _, Stages_test, Events_test, others_test = data_preprocess_MTL_Seqlen(test_data, 'test', duration)

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	if args.XYZ == 'XY':
		Signals_train = np.stack([X_train, Y_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test], axis=-1)
	elif args.XYZ == 'X':
		Signals_train = np.expand_dims(X_train, axis=-1)
		Signals_val = np.expand_dims(X_val, axis=-1)
		Signals_test = np.expand_dims(X_test, axis=-1)
	elif args.XYZ == 'Y':
		Signals_train = np.expand_dims(Y_train, axis=-1)
		Signals_val = np.expand_dims(Y_val, axis=-1)
		Signals_test = np.expand_dims(Y_test, axis=-1)


	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	train_dataset = ApneaDataset_MTL(Signals_train, Stages_train, Events_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL(Signals_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader



def npy2dataset_true_MTL_Z(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	Z_train, Stages_train, Events_train, others_train = data_preprocess_MTL_Z(train_data, 'train', args.std)
	Z_val, Stages_val, Events_val, others_val = data_preprocess_MTL_Z(val_data, 'val', args.std)
	Z_test, Stages_test, Events_test, others_test = data_preprocess_MTL_Z(test_data, 'test', args.std)

	print(f'Z_train: {Z_train.shape}, Z_val: {Z_val.shape}, Z_test: {Z_test.shape}')

	if args.XYZ == 'Z':
		Signals_train = np.expand_dims(Z_train, axis=-1)
		Signals_val = np.expand_dims(Z_val, axis=-1)
		Signals_test = np.expand_dims(Z_test, axis=-1)



	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	train_dataset = ApneaDataset_MTL(Signals_train, Stages_train, Events_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL(Signals_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader





def npy2dataset_true_MTL(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	X_train, Y_train, Z_train, Stages_train, Events_train, others_train = data_preprocess_MTL(train_data, 'train', args.std)
	X_val, Y_val, Z_val, Stages_val, Events_val, others_val = data_preprocess_MTL(val_data, 'val', args.std)
	X_test, Y_test, Z_test, Stages_test, Events_test, others_test = data_preprocess_MTL(test_data, 'test', args.std)

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	if args.XYZ == 'XY':
		Signals_train = np.stack([X_train, Y_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test], axis=-1)
	elif args.XYZ == 'X':
		Signals_train = np.expand_dims(X_train, axis=-1)
		Signals_val = np.expand_dims(X_val, axis=-1)
		Signals_test = np.expand_dims(X_test, axis=-1)
	elif args.XYZ == 'Y':
		Signals_train = np.expand_dims(Y_train, axis=-1)
		Signals_val = np.expand_dims(Y_val, axis=-1)
		Signals_test = np.expand_dims(Y_test, axis=-1)
	elif args.XYZ == 'Z':
		Signals_train = np.expand_dims(Z_train, axis=-1)
		Signals_val = np.expand_dims(Z_val, axis=-1)
		Signals_test = np.expand_dims(Z_test, axis=-1)
	elif args.XYZ == 'XYZ':
		Signals_train = np.stack([X_train, Y_train, Z_train], axis=-1) 
		Signals_val = np.stack([X_val, Y_val, Z_val], axis=-1)
		Signals_test = np.stack([X_test, Y_test, Z_test], axis=-1)
	elif args.XYZ == 'XZ':
		Signals_train = np.stack([X_train, Z_train], axis=-1) 
		Signals_val = np.stack([X_val, Z_val], axis=-1)
		Signals_test = np.stack([X_test, Z_test], axis=-1)
	elif args.XYZ == 'YZ':
		Signals_train = np.stack([Y_train, Z_train], axis=-1) 
		Signals_val = np.stack([Y_val, Z_val], axis=-1)
		Signals_test = np.stack([Y_test, Z_test], axis=-1)

	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	train_dataset = ApneaDataset_MTL(Signals_train, Stages_train, Events_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL(Signals_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader




def npy2dataset_true_MTL_REC(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)


	X_train, Y_train, _, Stages_train, Events_train, others_train = data_preprocess_MTL(train_data, 'train')
	X_val, Y_val, _, Stages_val, Events_val, others_val = data_preprocess_MTL(val_data, 'val')
	X_test, Y_test, _, Stages_test, Events_test, others_test = data_preprocess_MTL(test_data, 'test')

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_train = np.stack([X_train, Y_train], axis=-1) 
	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	train_dataset = ApneaDataset_MTL(Signals_train, Stages_train, Events_train)

	# batch_sampler = ThreeGroupRatioBatchSampler(
	# 	stages=Stages_train,
	# 	events=Events_train,
	# 	batch_size=args.batch_size,
	# 	drop_last=True)
	# train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL(Signals_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader




def npy2dataset_true_MTL_REC(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)

	
	X_train, Y_train, _, THO_train, ABD_train, Stages_train, Events_train, others_train = data_preprocess_MTL_REC(train_data, 'train')
	X_val, Y_val, _, THO_val, ABD_val, Stages_val, Events_val, others_val = data_preprocess_MTL_REC(val_data, 'val')
	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_MTL_REC(test_data, 'test')

	print(f'X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_train = np.stack([X_train, Y_train], axis=-1) 
	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	print('...Data Distribution...')
	print('In Training')
	print(f'Wake/Sleep: {np.sum(Stages_train==1)}/{np.sum(Stages_train==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_train==1)}/{np.sum(Events_train==0)-np.sum(Stages_train==1)}')

	print('In Validation')
	print(f'Wake/Sleep: {np.sum(Stages_val==1)}/{np.sum(Stages_val==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_val==1)}/{np.sum(Events_val==0)-np.sum(Stages_val==1)}')

	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	train_dataset = ApneaDataset_MTL_REC(Signals_train, THO_train, ABD_train, Stages_train, Events_train)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	val_dataset = ApneaDataset_MTL_REC(Signals_val, THO_val, ABD_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader



def npy2dataset_inference_true_MTL(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	# _, val_data, test_data = load_train_val_test_data(data_path, fold_idx)
	val_data, test_data = load_val_test_data(data_path, fold_idx)


	X_val, Y_val, _, Stages_val, Events_val, others_val = data_preprocess_MTL(val_data, 'val')
	X_test, Y_test, _, Stages_test, Events_test, others_test = data_preprocess_MTL(test_data, 'test')

	print(f' X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL(Signals_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return val_loader, test_loader

def npy2dataset_inference_MTL(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, val_data = load_train_val_data(data_path, fold_idx)


	X_val, Y_val, _, Stages_val, Events_val, others_val = data_preprocess_MTL(val_data, 'val')

	print(f' X_val: {X_val.shape}')

	Signals_val = np.stack([X_val, Y_val], axis=-1)

	val_dataset = ApneaDataset_MTL(Signals_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return val_loader


def npy2dataset_inference_true_MTL_REC(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, val_data, test_data = load_train_val_test_data(data_path, fold_idx)


	X_val, Y_val, _, THO_val, ABD_val, Stages_val, Events_val, others_val = data_preprocess_MTL_REC(val_data, 'val')
	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_MTL_REC(test_data, 'test')

	print(f' X_val: {X_val.shape}, X_test: {X_test.shape}')

	Signals_val = np.stack([X_val, Y_val], axis=-1)
	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')

	val_dataset = ApneaDataset_MTL_REC(Signals_val, THO_val, ABD_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_MTL_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return val_loader, test_loader



def npy2dataset_inference(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	val_data = load_val_data(data_path, fold_idx)
	X_val, Y_val, Z_val, Events_val, others_val = data_preprocess(val_data, 'val', args.raw, args.sleep_index, args.nseg, args.Index)
	print(X_val.shape, Y_val.shape, Z_val.shape)
	if args.XYZ == 'XY':
		Signals_val = np.stack([X_val, Y_val], axis=-1)
		print(f'Y_val: {Y_val.shape}')
	elif args.XYZ == 'XYZ':
		print(f'X_val: {X_val.shape}, Y_val: {Y_val.shape}, Z_val: {Z_val.shape}')
		Signals_val = np.stack([X_val, Y_val, Z_val], axis=-1)
	elif args.XYZ == 'Y':
		Signals_val = np.expand_dims(Y_val, axis=-1)
		Signals_val = np.expand_dims(Y_val, axis=-1)

	print(f'{args.XYZ} shape: Val: {Signals_val.shape}')
	val_dataset = ApneaDataset(Signals_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	return val_loader
