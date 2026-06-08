import pandas as pd
import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from utils_dl_mapping import ApneaDataset_REC
from Code.utils_dsp import normalize, modify_magnitude_with_gaussian_noise, denoise_iter
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import tqdm


# def augmentation_MTL_REC(X, Y, Z, THO, ABD, Events, Stages, others):
# 	augmented_X, augmented_Y, augmented_Z = [], [], []
# 	augmented_THO, augmented_ABD = [], []
# 	augmented_Events, augmented_Stages, augmented_Others = [], [], []

# 	for i in range(len(Y)):
# 		cnt = 0

# 		if Events[i] != 0 or Stages[i] != 0: 
# 			x_aug = [-X[i], X[i, ::-1], -X[i, ::-1], modify_magnitude_with_gaussian_noise(X[i])]
# 			y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1], modify_magnitude_with_gaussian_noise(Y[i])]
# 			z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1], modify_magnitude_with_gaussian_noise(Z[i])]
# 			tho_aug = [-THO[i], THO[i, ::-1], -THO[i, ::-1], THO[i]]
# 			abd_aug = [-ABD[i], ABD[i, ::-1], -ABD[i, ::-1], ABD[i]]
# 			augmented_X.extend(x_aug)
# 			augmented_Y.extend(y_aug)
# 			augmented_Z.extend(z_aug)
# 			augmented_THO.extend(tho_aug)
# 			augmented_ABD.extend(abd_aug)
# 			cnt += 4
# 		else:
# 			if np.random.rand() < 0.1:
# 				x_aug = [-X[i], X[i, ::-1], -X[i, ::-1]]
# 				y_aug = [-Y[i], Y[i, ::-1], -Y[i, ::-1]]
# 				z_aug = [-Z[i], Z[i, ::-1], -Z[i, ::-1]]
# 				tho_aug = [-THO[i], THO[i, ::-1], -THO[i, ::-1]]
# 				abd_aug = [-ABD[i], ABD[i, ::-1], -ABD[i, ::-1]]
# 				augmented_X.extend(x_aug)
# 				augmented_Y.extend(y_aug)
# 				augmented_Z.extend(z_aug)
# 				augmented_THO.extend(tho_aug)
# 				augmented_ABD.extend(abd_aug)
# 				cnt += 3
# 			if np.random.rand() < 0.1:
# 				x_aug = modify_magnitude_with_gaussian_noise(X[i])
# 				y_aug = modify_magnitude_with_gaussian_noise(Y[i])
# 				z_aug = modify_magnitude_with_gaussian_noise(Z[i])
# 				augmented_X.append(x_aug)
# 				augmented_Y.append(y_aug)
# 				augmented_Z.append(z_aug)
# 				augmented_THO.append(THO[i])
# 				augmented_ABD.append(ABD[i])
# 				cnt += 1

# 		augmented_Events.extend([Events[i]] * cnt)
# 		augmented_Stages.extend([Stages[i]] * cnt)
# 		augmented_Others.extend([others[i]] * cnt)

# 	X = np.concatenate((X, np.array(augmented_X)), axis=0)
# 	Y = np.concatenate((Y, np.array(augmented_Y)), axis=0)
# 	Z = np.concatenate((Z, np.array(augmented_Z)), axis=0)
# 	THO = np.concatenate((THO, np.array(augmented_THO)), axis=0)
# 	ABD = np.concatenate((ABD, np.array(augmented_ABD)), axis=0)

# 	Events = np.concatenate((Events, np.array(augmented_Events)), axis=0)
# 	Stages = np.concatenate((Stages, np.array(augmented_Stages)), axis=0)
# 	others = np.concatenate((others, np.array(augmented_Others)), axis=0)

# 	return X, Y, Z, THO, ABD, Events, Stages, others



def data_preprocess_REC(data, Type):
	print('-'*50)
	print(f'In {Type} ...')


	# data = data[:5000]

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
	# if Type == 'train':
	# 	X, Y, Z, THO, ABD, Events, Stages, others = augmentation_MTL_REC(X, Y, Z, THO, ABD, Events, Stages, others)
	# 	print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	# unique_events = np.unique(Events)
	# for i in range(len(unique_events)):
	# 	print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))

	return X, Y, Z, THO, ABD, Stages, Events, others


def data_preprocess_REC_aligned(data, Type):
	print('-'*50)
	print(f'In {Type} ...')


	# data = data[:5000]

	WIN = (data.shape[1] - 6) // 5  # per-channel samples (5600 after lag crop)
	X, Y, Z = data[:, :WIN], data[:, WIN:2*WIN], data[:, 2*WIN:3*WIN]
	THO, ABD = data[:, 3*WIN:4*WIN], data[:, 4*WIN:5*WIN]
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
	crop_end = WIN // 10 - 5  # e.g. 555 when WIN=5600
	X, Y, Z = X[:, 5:crop_end], Y[:, 5:crop_end], Z[:, 5:crop_end]
	THO, ABD = THO[:, 5:crop_end], ABD[:, 5:crop_end]
	print(f'Downsampled')

	print(f'Before Normalization, X_{Type}.shape: {X.shape}, Y_{Type}.shape: {Y.shape}, Z_{Type}.shape: {Z.shape}')

	X, Y, Z = normalize(X), normalize(Y), normalize(Z)
	THO, ABD = normalize(THO), normalize(ABD)
	print("Normalization")
	# if Type == 'train':
	# 	X, Y, Z, THO, ABD, Events, Stages, others = augmentation_MTL_REC(X, Y, Z, THO, ABD, Events, Stages, others)
	# 	print(f'After Augmentation, Events_{Type}.shape: ', Events.shape)

	# unique_events = np.unique(Events)
	# for i in range(len(unique_events)):
	# 	print(f'Events_{Type} {unique_events[i]}: ', np.sum(Events == unique_events[i]))

	return X, Y, Z, THO, ABD, Stages, Events, others




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


def load_train_val_test_data_aligned(data_path, fold_idx, lag_csv_path=None):
	"""
	Load train/val/test data with lag correction from lag_results.csv.
	Subjects with NaN Median_Lag_Sec are excluded.

	Lag correction (no circular shift):
	  - MAX_LAG = 400 samples (4s at 100Hz); all epochs trimmed to WIN = 6000 - MAX_LAG = 5600 samples/channel
	  - lag > 0: PSG lags BSG → BSG[0:WIN], PSG[lag:lag+WIN]
	  - lag < 0: BSG lags PSG → BSG[|lag|:|lag|+WIN], PSG[0:WIN]
	Output column layout: [X(WIN) | Y(WIN) | Z(WIN) | THO(WIN) | ABD(WIN) | others(6)]
	Subject ID is stored in data[:, -4].
	"""
	MAX_LAG = 400  # 4 seconds at 100 Hz
	WIN = 6000 - MAX_LAG  # 5600 samples per channel

	if lag_csv_path is None:
		lag_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lag_results.csv')

	lag_df = pd.read_csv(lag_csv_path)
	removed_df = lag_df[lag_df['Median_Lag_Sec'].isna()]
	if len(removed_df) > 0:
		print(f'Subjects removed (no lag estimate): {len(removed_df)}')
		for _, row in removed_df.iterrows():
			print(f'  ID={int(row["ID"])}, Name={row["Name"]}')
	lag_df = lag_df.dropna(subset=['Median_Lag_Sec'])
	lag_map = dict(zip(lag_df['ID'].astype(int), lag_df['Median_Lag_Samples'].round().astype(int)))
	valid_ids = set(lag_map.keys())
	print(f'Valid subjects with lag: {len(valid_ids)}')

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

	def load_fold_with_lag(fold_num):
		data_file_path = data_path + f'fold{fold_num}.npy'
		data = np.load(data_file_path)
		subject_ids = data[:, -4].astype(int)

		# Remove subjects with no lag estimate
		mask = np.isin(subject_ids, list(valid_ids))
		n_removed = np.sum(~mask)
		if n_removed > 0:
			removed = np.unique(subject_ids[~mask])
			print(f'  Fold {fold_num}: removing {n_removed} epochs from subjects {removed} (no lag estimate)')
		data = data[mask]
		subject_ids = subject_ids[mask]

		# Build aligned array: 5 channels × WIN + 6 metadata cols
		N = len(data)
		data_aligned = np.empty((N, WIN * 5 + 6), dtype=data.dtype)

		for uid in np.unique(subject_ids):
			lag = int(np.clip(lag_map[uid], -MAX_LAG, MAX_LAG))
			bsg_start = max(0, -lag)  # where to start reading BSG
			psg_start = max(0,  lag)  # where to start reading PSG
			rows = subject_ids == uid

			src = data[rows]
			data_aligned[rows, 0      : WIN    ] = src[:, bsg_start         : bsg_start + WIN        ]  # X
			data_aligned[rows, WIN    : 2*WIN  ] = src[:, 6000 + bsg_start  : 6000 + bsg_start + WIN ]  # Y
			data_aligned[rows, 2*WIN  : 3*WIN  ] = src[:, 12000 + bsg_start : 12000 + bsg_start + WIN]  # Z
			data_aligned[rows, 3*WIN  : 4*WIN  ] = src[:, 18000 + psg_start : 18000 + psg_start + WIN]  # THO
			data_aligned[rows, 4*WIN  : 5*WIN  ] = src[:, 24000 + psg_start : 24000 + psg_start + WIN]  # ABD
			data_aligned[rows, 5*WIN  :        ] = src[:, -6:]                                           # metadata

		return data_aligned

	train_data, val_data, test_data = [], [], []
	for i in range(1, 5):
		print(f'Loading data from fold {i} ...')
		fold_data = load_fold_with_lag(i)
		if i in train_fold: train_data.append(fold_data)
		elif i in valid_fold: val_data.append(fold_data)
		else: test_data.append(fold_data)

	print(f'Total train files: {len(train_data)}, Total val files: {len(val_data)}, Total test files: {len(test_data)}')
	train_data = np.concatenate(train_data)
	print(f'After concatenation, train_data.shape: {train_data.shape}')
	val_data = np.concatenate(val_data)
	test_data = np.concatenate(test_data)
	print(f'Loaded Train Data: {train_data.shape}, Loaded Val Data: {val_data.shape}, Loaded Test Data: {test_data.shape}')
	return train_data, val_data, test_data




def npy2dataset_REC(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data(data_path, fold_idx)
	
	X_train, Y_train, _, THO_train, ABD_train, Stages_train, Events_train, others_train = data_preprocess_REC(train_data, 'train')
	X_val, Y_val, _, THO_val, ABD_val, Stages_val, Events_val, others_val = data_preprocess_REC(val_data, 'val')
	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_REC(test_data, 'test')

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

	train_dataset = ApneaDataset_REC(Signals_train, THO_train, ABD_train, Stages_train, Events_train, others_train)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
	
	val_dataset = ApneaDataset_REC(Signals_val, THO_val, ABD_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
	
	test_dataset = ApneaDataset_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader




def npy2dataset_REC_aligned(data_path, fold_idx, args, lag_csv_path=None):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	train_data, val_data, test_data = load_train_val_test_data_aligned(data_path, fold_idx, lag_csv_path)

	X_train, Y_train, _, THO_train, ABD_train, Stages_train, Events_train, others_train = data_preprocess_REC_aligned(train_data, 'train')
	X_val, Y_val, _, THO_val, ABD_val, Stages_val, Events_val, others_val = data_preprocess_REC_aligned(val_data, 'val')
	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_REC_aligned(test_data, 'test')

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

	train_dataset = ApneaDataset_REC(Signals_train, THO_train, ABD_train, Stages_train, Events_train, others_train)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	val_dataset = ApneaDataset_REC(Signals_val, THO_val, ABD_val, Stages_val, Events_val, others_val)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	test_dataset = ApneaDataset_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return train_loader, val_loader, test_loader




def npy2dataset_REC_inference(data_path, fold_idx, args):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, _, test_data = load_train_val_test_data(data_path, fold_idx)

	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_REC(test_data, 'test')

	print(f'X_test: {X_test.shape}')

	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'Signals_test: {Signals_test.shape}')

	print('...Data Distribution...')
	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	test_dataset = ApneaDataset_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return test_loader


def npy2dataset_REC_inference_aligned(data_path, fold_idx, args, lag_csv_path=None):
	"""Shape of each batch: [batch_size, channels, seq_len]"""
	_, _, test_data = load_train_val_test_data_aligned(data_path, fold_idx, lag_csv_path)

	X_test, Y_test, _, THO_test, ABD_test, Stages_test, Events_test, others_test = data_preprocess_REC_aligned(test_data, 'test')

	print(f'X_test: {X_test.shape}')

	Signals_test = np.stack([X_test, Y_test], axis=-1)
	print(f'Signals_test: {Signals_test.shape}')

	print('...Data Distribution...')
	print('In Testing')
	print(f'Wake/Sleep: {np.sum(Stages_test==1)}/{np.sum(Stages_test==0)}')
	print(f'Apnea/Non-Apnea: {np.sum(Events_test==1)}/{np.sum(Events_test==0)-np.sum(Stages_test==1)}')

	test_dataset = ApneaDataset_REC(Signals_test, THO_test, ABD_test, Stages_test, Events_test, others_test)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

	return test_loader

