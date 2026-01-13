import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import denoise, normalize_1d
from Code.models.clf import ApneaClassifier_PatchTST_MTL
from Code.utils import choose_gpu_by_model_process_count, calculate_icc_standard, ahi_to_severity, calculate_cm
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from Code.plotly_xz_mtl import plot_person_level_results_sleep \
	 , plot_person_level_results, concatenate_segments, inference \
	 , get_wake_masks_pred, get_wake_masks_29 \
	 , count_continuous_ones \
     , load_configs, load_model_MTL
from Code.plotly_xz_mtl import ratio_check_lst, process_allnight_data
from Code.plotly_xz_mtl_rec import compute_segmented_mae, compute_tst_mae

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

	Experiment = 'Seqlen'	

	duration_model = 30

	model_folder_name = f'XY_{duration_model}s_F1_ws1_wa1'
	config_path = f"Experiments/{Experiment}/configs/{model_folder_name}.yaml"


	fold2id, fold_to_threshold_stage, fold_to_threshold_apn = load_configs(config_path)
	
	for fold_name, id_list in fold2id.items():
		for _id in id_list:
			id2fold[_id] = fold_name

	TST_labels, TST_preds = [], []
	AHI_labels, AHI_preds = [], []
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
		THO_concat, time_tho_abd = process_allnight_data(THO, denoising=False)
		ABD_concat, time_tho_abd = process_allnight_data(ABD, denoising=False)


		SleepStage_concat = concatenate_segments(SleepStage, int((duration - overlap) * 1))
		Event_concat = concatenate_segments(Event, int((duration - overlap) * 10))
		time_sleep = np.arange(len(SleepStage_concat)) / 1
		time_event = np.arange(len(Event_concat)) / 10


		# ============ Inference ============
		model_folder = f'Experiments/{Experiment}/Models/{model_folder_name}/fold{fold_idx}/PatchTST_patchlen24_nlayer4_dmodel64_nhead4_dff256/'
		model = load_model_MTL(model_folder, duration_model, device)
		
		_, pred_res_apn = inference(X_concat, Y_concat, model, device, step_sig_apn, threshold=fold_to_threshold_apn[fold_idx], duration=duration_model)	
		pad_length_apn = duration_model * 10 // step_sig_apn
		pred_res_apn = np.pad(pred_res_apn, (pad_length_apn, 0), mode='constant', constant_values=0)
		pred_time_apn = np.arange(len(pred_res_apn)) * step_sig_apn / 10  


		pred_res_sleep, _ = inference(X_concat, Y_concat,model, device, step_sig_sleep, threshold=fold_to_threshold_stage[fold_idx], duration=duration_model)
		pad_length_sleep = duration_model * 10 // step_sig_sleep
		pred_res_sleep = np.pad(pred_res_sleep, (pad_length_sleep, 1), mode='constant', constant_values=1)
		pred_time_sleep = np.arange(len(pred_res_sleep)) * step_sig_sleep / 10

		wake_masks = get_wake_masks_pred(pred_res_sleep, step_sig_apn // 10, duration=duration_model)	
		if ID_npy == 29:
			wake_masks = get_wake_masks_29(SleepStage_concat, step_sig_apn // 10, duration=duration_model)
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

		TST_labels.append(sleep_time_excel)
		TST_preds.append(sleep_time_pred)
		AHI_labels.append(AHI_label)
		AHI_preds.append(AHI_preds_processed_label)
		
	log_path = f'Experiments/{Experiment}/Models/{model_folder_name}/logs.txt'
	if not os.path.exists(os.path.dirname(log_path)):
		os.makedirs(os.path.dirname(log_path))


	path = f'Experiments/{Experiment}/Models/{model_folder_name}/New_Seismo_AHI_{step_sig_apn//10}s_larger2_change106108134_change29_no153119241149932_with108'
	sleep_path = f'Experiments/{Experiment}/Models/{model_folder_name}/New_Seismo_TST_{step_sig_sleep//10}s_larger2_change106108134_change29_no153119241149932_with108'



	AHI_labels, AHI_preds = np.array(AHI_labels), np.array(AHI_preds)	
	result = compute_segmented_mae(AHI_labels, AHI_preds)
	print(result) 


	lines = []
	lines.append(f'All #: {len(AHI_labels)}\n')

	result_tst = compute_tst_mae(TST_labels, TST_preds, 'All')
	print(result_tst)
	lines.append(result_tst)



	AHI_labels, AHI_preds = np.array(AHI_labels), np.array(AHI_preds)	
	result_mae = compute_segmented_mae(AHI_labels, AHI_preds)
	print(result_mae) 
	lines.append(result_mae)


	icc = calculate_icc_standard(AHI_labels, AHI_preds)
	result_icc = f'ICC: {icc:.3f}\n'
	print(result_icc)
	lines.append(result_icc)

	result_cm = calculate_cm(AHI_labels, AHI_preds, 'All')
	print(result_cm)
	lines.append(result_cm)


	lines.append('\n')
	lines.append('============================\n')


	with open(log_path, "a", encoding="utf-8") as f:
		for line in lines:
			f.write(line + "\n")
	
	
	plot_person_level_results(
		y_true_list=AHI_labels,
		y_pred_list=AHI_preds,
		fig_path=path+'.png'
	)	

	plot_person_level_results_sleep(
		y_true_list=TST_labels,
		y_pred_list=TST_preds,
		ahi_label_list=AHI_labels,
		fig_path=sleep_path+'.png'
	)