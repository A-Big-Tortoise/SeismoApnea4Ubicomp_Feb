import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import denoise, normalize_1d
from Code.models.clf import ApneaClassifier_PatchTST_MTL_REC
from Code.utils import choose_gpu_by_model_process_count, calculate_icc_standard, ahi_to_severity, calculate_cm
from scipy.signal import resample_poly 
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from Code.plotly_xz_mtl_rec import concatenate_segments, inference_REC \
	 , get_wake_masks_pred, get_wake_masks_29 \
	 , count_continuous_ones, compute_segmented_mae \
     , load_configs, load_model_MTL_REC, compute_tst_mae
from Code.plotly_xz_mtl import ratio_check_lst, process_allnight_data
from matplotlib import pyplot as plt

if __name__ == "__main__":
	data_folder = 'Data/data_60s_30s_yingjian2/'
	duration = 60
	overlap = 30
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	df = pd.read_excel('Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')

	id2fold = {}
	step_sig_apn = 150
	step_sig_sleep = 10

	Experiment = 'Main'	

	model_folder_name = f'MTL_ws1_wa1_REC0.001'
	config_path = f"Experiments/{Experiment}/configs/{model_folder_name}.yaml"


	fold2id, fold_to_threshold_stage, fold_to_threshold_apn = load_configs(config_path)
	
	for fold_name, id_list in fold2id.items():
		for _id in id_list:
			id2fold[_id] = fold_name


	for file in sorted(os.listdir(data_folder)):
		# ============ Load Data ============
		data = np.load(os.path.join(data_folder, file))
		ID_npy = data[0, -2]		
		# print(f'Processing Patient ID: {ID_npy}')

		# ============ Data Check ============
		if ID_npy not in id2fold: continue
		# if ID_npy in [50, 25, 108, 134, 120, 153, 119]: continue
		if ID_npy in [24, 50, 25, 134, 153, 119, 114, 99, 32]: continue
		if not ratio_check_lst(ID_npy): continue
		sleep_time_excel = df.loc[df['ID'] == ID_npy, 'Duration(h)'].values[0]  * df.loc[df['ID'] == ID_npy, 'SEfficiency'].values[0] * 0.01
		if sleep_time_excel <= 2:
			print(f'Skipping patient {ID_npy} due to short sleep time: {sleep_time_excel:.2f} h\n')
			continue

		# ============ Data Loading and Processing ============
		fold_idx = id2fold.get(ID_npy)

		print(f'Patient ID: {ID_npy}, Shape: {data.shape}, Assigned to fold: {fold_idx}')

		X, Y, Z = data[:, :6000], data[:, 6000:12000], data[:, 12000:18000]
		THO, ABD = data[:, 18000:24000], data[:, 24000:30000]
		SleepStage = data[:, 42660:42720]
		Event = data[:, 42720:43320]



		X_concat, time_xyz = process_allnight_data(X, denoising=True)
		Y_concat, time_xyz = process_allnight_data(Y, denoising=True)
		Z_concat, time_xyz = process_allnight_data(Z, denoising=True)
		THO_concat, time_tho_abd = process_allnight_data(THO, denoising=False)
		ABD_concat, time_tho_abd = process_allnight_data(ABD, denoising=False)


		SleepStage_concat = concatenate_segments(SleepStage, int((duration - overlap) * 1))
		Event_concat = concatenate_segments(Event, int((duration - overlap) * 10))
		time_sleep = np.arange(len(SleepStage_concat)) / 1
		time_event = np.arange(len(Event_concat)) / 10


		# ============ Inference ============
		model_folder = f'Experiments/Main/Models/{model_folder_name}/fold{fold_idx}/PatchTST_patchlen24_nlayer4_dmodel64_nhead4_dff256/'
		model = load_model_MTL_REC(model_folder, duration, device)
		
		_, pred_res_apn = inference_REC(X_concat, Y_concat, model, device, step_sig_apn, threshold=fold_to_threshold_apn[fold_idx], duration=duration)	
		pad_length_apn = 600 // step_sig_apn
		pred_res_apn = np.pad(pred_res_apn, (pad_length_apn, 0), mode='constant', constant_values=0)
		pred_time_apn = np.arange(len(pred_res_apn)) * step_sig_apn / 10  


		pred_res_sleep, _ = inference_REC(X_concat, Y_concat, model, device, step_sig_sleep, threshold=fold_to_threshold_stage[fold_idx], duration=duration)
		pad_length_sleep = 60 // step_sig_sleep
		pred_res_sleep = np.pad(pred_res_sleep, (pad_length_sleep, 1), mode='constant', constant_values=1)
		pred_time_sleep = np.arange(len(pred_res_sleep)) * step_sig_sleep / 10

		wake_masks = get_wake_masks_pred(pred_res_sleep, step_sig_apn // 10)	
		if ID_npy == 29: wake_masks = get_wake_masks_29(SleepStage_concat, step_sig_apn // 10)
		pred_res_apn[wake_masks] = 0 

		pred_res_processed = np.zeros_like(pred_res_apn)
		for i in range(1, len(pred_res_apn)):
			if pred_res_apn[i] == 1 and pred_res_processed[i-1] == 1:
				pred_res_processed[i] = 0
			else:
				pred_res_processed[i] = pred_res_apn[i]

		true_apnea_events = df.loc[df['ID'] == ID_npy, ['CA', 'OA', 'MA', 'HYP']].values.flatten()
		n_true_apnea_events = np.sum(true_apnea_events)
		
		_, n_apnea_events = count_continuous_ones(pred_res_processed)

		sleep_time_pred = np.sum(pred_res_sleep==0)  / (3600 / (step_sig_sleep / 10))
		print(f'True Sleep Time (hours): {sleep_time_excel:.2f} h')
		print(f'Pred Sleep Time (hours): {sleep_time_pred:.2f} h')
	

		AHI_label = df.loc[df['ID'] == ID_npy, 'AHI'].values[0]
		if ID_npy == 7: AHI_label = df.loc[df['ID'] == ID_npy, 'RDI'].values[0]
		if ID_npy == 106: AHI_label = 30 / (7.32 * 83.5 / 100)
		if ID_npy == 134: AHI_label = 13.73
		if ID_npy == 108: AHI_label = 9.6


		AHI_preds_processed_label = n_apnea_events / sleep_time_pred

		print(f'# Apnea Events: {n_apnea_events}, AHI (predicted): {AHI_preds_processed_label:.2f}')
		print(f'# True Apnea Events: {n_true_apnea_events}, AHI (labels): {AHI_label:.2f}')

		SleepStage_concat = np.array(SleepStage_concat)
		SleepStage_concat[(SleepStage_concat >= 0) & (SleepStage_concat <= 3)] = 0
		SleepStage_concat[SleepStage_concat == -1] = 1
		SleepStage_concat[SleepStage_concat == 4] = 1

		Event_concat = np.array(Event_concat)
		Event_concat[Event_concat >= 1] = 1
		Event_concat[Event_concat == -1] = 0


		row = 3
		fig, axes = plt.subplots(row, 1, figsize=(20, 2.35 *  row))
		axes[0].plot(normalize_1d(X_concat), label='X (norm)', alpha=0.5)
		axes[0].plot(normalize_1d(Y_concat), label='Y (norm)', alpha=0.5)
		axes[0].plot(normalize_1d(Z_concat), label='Z (norm)', alpha=0.5)
		axes[1].plot(time_sleep, SleepStage_concat, label='Sleep Stage (0: Wake, 1-4: Sleep)', color='green')
		axes[1].plot(time_event, Event_concat, label='Apnea Events (0: No event, 1: Event)', color='orange')
		
		axes[2].plot(pred_time_sleep, pred_res_sleep, label='Predicted Sleep Stage (0: Wake, 1: Sleep)', color='green')
		axes[2].plot(pred_time_apn, pred_res_processed, label='Predicted Apnea Events (0: No event, 1: Event)', color='orange')
		axes[0].set_title(f'Patient {ID_npy}, AHI (pred): {AHI_preds_processed_label:.2f}, AHI (label): {AHI_label:.2f}')
		plt.tight_layout()
		plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_rec_overnight/Patient_{ID_npy}_overnight_plot.png')
