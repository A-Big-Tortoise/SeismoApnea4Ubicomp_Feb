import numpy as np
import os, sys
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import denoise, normalize_1d
from Code.models.clf import ApneaClassifier_PatchTST_TriClass
from Code.utils import choose_gpu_by_model_process_count
from Code.plotly_xz_mtl import plot_person_level_results_sleep \
	 , plot_person_level_results, concatenate_segments \
	 , ratio_check_lst, get_wake_masks_pred, get_wake_masks_29 \
	 , process_allnight_data, count_continuous_ones, compute_segmented_mae \
     , load_configs
import torch
import pandas as pd



def load_model_TriClass(model_folder, device, axis=2):
	patch_len = 24
	n_layers = 4
	d_model = 64
	n_heads = 4
	d_ff = 256
	output_class = 3

	# model = ApneaClassifier_PatchTST_MTL(
	model = ApneaClassifier_PatchTST_TriClass(
		input_size=axis, num_classes=output_class,
		seq_len=590, patch_len=patch_len,
		stride=patch_len // 2,
		n_layers=n_layers, d_model=d_model,
		n_heads=n_heads, d_ff=d_ff,
		axis=axis,
		dropout=0.2,
		mask_ratio=0.5
		).to(device)
	sorted_files = sorted(os.listdir(model_folder), key=lambda x: int(x.split('_')[0][5:]), reverse=False)
	model_path = model_folder + sorted_files[-1]
	checkpoint = torch.load(model_path, weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	return model


def inference_TriClass(X_concat, Y_concat, model, device, step_sig, XY=None):
	if XY is None or XY == 'XY':
		model_input = np.stack([X_concat, Y_concat], axis=0)  # shape (2, N)
		model_input = model_input.reshape(1, 2, -1)  # shape (1, 2, N)
	elif XY == 'X':
		model_input = X_concat.reshape(1, 1, -1)  # shape (1, 1, N)
	elif XY == 'Y':
		model_input = Y_concat.reshape(1, 1, -1)  # shape (1, 1, N)

	batch_size = 512

	segments = []

	for i in range(0, model_input.shape[2] - 600 + step_sig, step_sig):
		segment = model_input[:, :, i+5:i+595]
		if segment.shape[2] == 590:
			segments.append(segment)


	segments = np.concatenate(segments, axis=0)  # shape: [N, C, 590]
	segments = (segments - np.mean(segments, axis=2, keepdims=True)) / (np.std(segments, axis=2, keepdims=True) + 1e-6)
	segments_tensor = torch.tensor(segments, dtype=torch.float32).to(device)

	pred_res = []

	model.eval()
	with torch.no_grad():
		for start in range(0, len(segments_tensor), batch_size):
			end = start + batch_size
			batch = segments_tensor[start:end]
			outputs1, _, _ = model(batch)
			outputs2, _, _ = model(torch.flip(batch, dims=[2]))
			outputs3, _, _ = model(-batch)

			output = (outputs1 + outputs2 + outputs3) / 3
			probs = torch.softmax(output, dim=1)  # shape: [batch, num_classes]
			prob_class1 = probs[:, 1].cpu().numpy()
			# pred = (prob_class1 >= threshold).astype(int)
			pred = np.argmax(probs.cpu().numpy(), axis=1)
			pred_res.extend(pred)

	return np.array(pred_res)





if __name__ == "__main__":
	data_folder = 'Data/data_60s_30s_yingjian2/'
	duration = 60
	overlap = 30
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	df = pd.read_excel('Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')

	id2fold = {}

	step_sig = 350
	# step_sig_sleep = 10
	
	XYZ = 'XY'
	Experiment = 'MTL'
	model_folder_name = f'{XYZ}_60s_0.5'
	fold2id, _, _ = load_configs(f"Experiments/{Experiment}/configs/{XYZ}_60s_F1_ws1_wa0.yaml")

	for fold_name, id_list in fold2id.items():
		for _id in id_list:
			id2fold[_id] = fold_name

	TST_labels, TST_preds = [], []
	AHI_labels, AHI_preds = [], []
	for file in sorted(os.listdir(data_folder)):
		# ============ Load Data ============
		data = np.load(os.path.join(data_folder, file))
		ID_npy = data[0, -2]		

		# ============ Data Check ============
		if ID_npy not in id2fold: continue
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

		X_concat, time_xyz = process_allnight_data(X, duration=duration, overlap=overlap, denoising=True)
		Y_concat, time_xyz = process_allnight_data(Y, duration=duration, overlap=overlap, denoising=True)
		THO_concat, time_tho_abd = process_allnight_data(THO, duration=duration, overlap=overlap, denoising=False)
		ABD_concat, time_tho_abd = process_allnight_data(ABD, duration=duration, overlap=overlap, denoising=False)

		SleepStage_concat = concatenate_segments(SleepStage, int((duration - overlap) * 1))
		Event_concat = concatenate_segments(Event, int((duration - overlap) * 10))
		time_sleep = np.arange(len(SleepStage_concat)) / 1
		time_event = np.arange(len(Event_concat)) / 10


		# ============ Inference ============
		model_folder = f'Experiments/{Experiment}/Models/{model_folder_name}/fold{fold_idx}/PatchTST_patchlen24_nlayer4_dmodel64_nhead4_dff256/'
		model = load_model_TriClass(model_folder, device, axis=len(XYZ))
		
		pred_res = inference_TriClass(X_concat, Y_concat, model, device, step_sig, XY=XYZ)
		pad_length = 60 // step_sig
		pred_res = np.pad(pred_res, (pad_length, 1), mode='constant', constant_values=2)
		pred_time = np.arange(len(pred_res)) * step_sig / 10



		pred_res_processed = np.zeros_like(pred_res)
		for i in range(1, len(pred_res)):
			if pred_res[i] == 1 and pred_res_processed[i-1] == 1:
				pred_res_processed[i] = 0
			else:
				pred_res_processed[i] = pred_res[i]

		true_apnea_events = df.loc[df['ID'] == ID_npy, ['CA', 'OA', 'MA', 'HYP']].values.flatten()
		n_true_apnea_events = np.sum(true_apnea_events)
		
		_, n_apnea_events = count_continuous_ones(pred_res_processed)

		sleep_time_pred = (np.sum(pred_res==0)+np.sum(pred_res==1)) / (3600 / (step_sig / 10))
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
		
	path = f'Experiments/{Experiment}/Models/TriClass_AHI_{step_sig//10}s_larger2_change106108134_change29_no153119241149932_with108'
	sleep_path = f'Experiments/{Experiment}/Models/TriClass_TST_{step_sig//10}s_larger2_change106108134_change29_no153119241149932_with108'


	AHI_labels, AHI_preds = np.array(AHI_labels), np.array(AHI_preds)	
	result = compute_segmented_mae(AHI_labels, AHI_preds)
	print(result) 
	
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