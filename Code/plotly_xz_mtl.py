import numpy as np
import plotly.graph_objects as go
import os, sys
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
import subprocess
from Code.utils_dsp import denoise, normalize_1d
from Code.models.clf import ApneaClassifier_PatchTST_MTL
from utils import choose_gpu_by_model_process_count

from Code.utils import calculate_icc_standard, ahi_to_severity
from scipy.signal import resample_poly 
import torch
from tqdm import trange
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn.metrics import (
	balanced_accuracy_score,
	f1_score
)

from scipy.ndimage import median_filter



def plot_person_level_results_sleep(y_true_list, y_pred_list, ahi_label_list,
							  fig_path):
		# plt.figure(figsize=(8, 6))
		plt.rcParams.update({
			'axes.titlesize': 18,     # 图标题
			'axes.labelsize': 14,     # 坐标轴标题
			'xtick.labelsize': 12,    # x轴刻度
			'ytick.labelsize': 12,    # y轴刻度
			'legend.fontsize': 12,    # 图例
		})

		ahi_severity = np.array([ahi_to_severity(x) for x in ahi_label_list])
		severity_colors = {
			0: "green",
			1: "yellow",
			2: "orange",
			3: "red"
		}
		scatter_colors = [severity_colors[s] for s in ahi_severity]   # 根据真实严重程度上色



		ICC = calculate_icc_standard(y_true_list, y_pred_list)
		Corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]
		max_val = max(max(y_true_list), max(y_pred_list)) + 1


		
		fig, axes = plt.subplots(1, 2, figsize=(14, 6))
		# fig, axes = plt.subplots(1, 3, figsize=(21, 6))

		ax = axes[0]

		ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.7)

		ax.scatter(
			y_pred_list,    # x = predicted
			y_true_list,   # y = label
			c=scatter_colors,
			s=50,
			alpha=0.85
		)

		ax.set_xlabel("# Sleep Time [Predicted]")
		ax.set_ylabel("# Sleep Time [Labels]")
		ax.set_title(f"Person-level Sleep Time \nICC={ICC:.3f}, Corr={Corr:.3f}, MAE={np.mean(np.abs(np.array(y_true_list) - np.array(y_pred_list))):.2f}")


		ax3 = axes[1]
		# ba plot
		ba_diff = np.array(y_pred_list) - np.array(y_true_list)
		ba_avg = (np.array(y_pred_list) + np.array(y_true_list)) / 2
		mean_diff = np.mean(ba_diff)
		std_diff = np.std(ba_diff)
		ax3.scatter(
			ba_avg, ba_diff,
			# c='blue',
			c=scatter_colors,
			s=50,
			alpha=0.7
		)
		upper, lower = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
		ax3.axhline(mean_diff, color='blue', linestyle='--')
		ax3.text(max(ba_avg)*0.8, mean_diff + 0.5, f'Mean Diff: {mean_diff:.2f}', color='red')

		ax3.axhline(upper, color='red', linestyle='--')
		ax3.text(max(ba_avg)*0.8, upper + 0.5, f'+1.96 SD: {upper:.2f}', color='red')

		ax3.axhline(lower, color='red', linestyle='--')
		ax3.text(max(ba_avg)*0.8, lower + 0.5, f'-1.96 SD: {lower:.2f}', color='red')
		
		
		
		ax3.set_xlabel("Average Sleep Time")
		ax3.set_ylabel("Difference in Sleep Time (Predicted - True)")
		ax3.set_title("Bland-Altman Plot, MAE: {:.2f}".format(np.mean(np.abs(ba_diff))))
		# ax3.legend()


		plt.tight_layout()

		plt.savefig(fig_path, dpi=300)
		plt.close()


def plot_person_level_results(y_true_list, y_pred_list,
							  fig_path):
		# plt.figure(figsize=(8, 6))
		plt.rcParams.update({
			'axes.titlesize': 18,     # 图标题
			'axes.labelsize': 14,     # 坐标轴标题
			'xtick.labelsize': 12,    # x轴刻度
			'ytick.labelsize': 12,    # y轴刻度
			'legend.fontsize': 12,    # 图例
		})


		y_true = np.array([ahi_to_severity(x) for x in y_true_list])
		y_pred = np.array([ahi_to_severity(x) for x in y_pred_list])
		labels = ["Normal", "Mild", "Moderate", "Severe"]

		severity_colors = {
			0: "green",
			1: "yellow",
			2: "orange",
			3: "red"
		}
		scatter_colors = [severity_colors[s] for s in y_true]   # 根据真实严重程度上色

		cm = confusion_matrix(y_true, y_pred)

		ICC = calculate_icc_standard(y_true_list, y_pred_list)
		Corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]
		max_val = max(max(y_true_list), max(y_pred_list)) + 1

		# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
		fig, axes = plt.subplots(1, 3, figsize=(21, 6))
		ax = axes[0]
		ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.7)

		ax.scatter(
			y_pred_list,    # x = predicted
			y_true_list,   # y = label
			c=scatter_colors,
			s=50,
			alpha=0.85
		)

		ax.set_xlabel("# AHI [Predicted]")
		ax.set_ylabel("# AHI [Labels]")
		ax.set_title(f"Person-level AHI Scatter\nICC={ICC:.3f}, Corr={Corr:.3f}")

		# Add legend for severity colors
		for s, color in severity_colors.items():
			ax.scatter([], [], color=color, label=labels[s])
		ax.legend(title="True Severity", loc="upper left", fontsize=9)

		ax2 = axes[1]
		sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
					xticklabels=labels, yticklabels=labels, ax=ax2)

		bacc = balanced_accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average='macro')

		ax2.set_xlabel("Predicted Severity")
		ax2.set_ylabel("True Severity")
		ax2.set_title(f"AHI Severity, Bacc: {bacc:.3f}, F1: {f1:.3f}")



		ax3 = axes[2]
		# ba plot
		ba_diff = np.array(y_pred_list) - np.array(y_true_list)
		ba_avg = (np.array(y_pred_list) + np.array(y_true_list)) / 2
		mean_diff = np.mean(ba_diff)
		std_diff = np.std(ba_diff)
		ax3.scatter(
			ba_avg, ba_diff,
			# c='blue',
			c=scatter_colors,
			s=50,
			alpha=0.7
		)
		upper, lower = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
		ax3.axhline(mean_diff, color='blue', linestyle='--')
		ax3.text(max(ba_avg)*0.8, mean_diff + 0.5, f'Mean Diff: {mean_diff:.2f}', color='red')

		ax3.axhline(upper, color='red', linestyle='--')
		ax3.text(max(ba_avg)*0.8, upper + 0.5, f'+1.96 SD: {upper:.2f}', color='red')

		ax3.axhline(lower, color='red', linestyle='--')
		ax3.text(max(ba_avg)*0.8, lower + 0.5, f'-1.96 SD: {lower:.2f}', color='red')
		
		
		
		ax3.set_xlabel("Average AHI")
		ax3.set_ylabel("Difference in AHI (Predicted - True)")
		ax3.set_title("Bland-Altman Plot, MAE: {:.2f}".format(np.mean(np.abs(ba_diff))))
		# ax3.legend()


		plt.tight_layout()

		plt.savefig(fig_path, dpi=300)
		plt.close()

		print("Combined scatter + CM saved to:", fig_path)
		print(cm)


def concatenate_segments(signal, step_size):
	n_segments, segment_length = signal.shape
	total_length = step_size * (n_segments - 1) + segment_length
	concatenated = np.zeros(total_length)
	count = np.zeros(total_length)
	for i in range(n_segments):
		start_idx = i * step_size
		end_idx = start_idx + segment_length
		concatenated[start_idx:end_idx] += signal[i, :]
		count[start_idx:end_idx] += 1
	return concatenated / count


def plot_all_concatenated_signals(signals, 
								  num_apnea_events, num_pred_apnea_events, 
								  ahi, pred_ahi,
								  filename):
	fig = go.Figure()
	for time, signal_data, name, color in signals:
		fig.add_trace(
			go.Scatter(
				x=time,
				y=signal_data,
				mode='lines',
				name=name,
				line=dict(width=1, color=color))
		)

	fig.update_layout(
		title=f'All Signals, # Apnea Event: {num_apnea_events}, # Predicted Apnea Event: {num_pred_apnea_events} \n AHI: {ahi:.2f}, Predicted AHI: {pred_ahi:.2f}',
		xaxis_title='Time (seconds)',
		yaxis_title='Amplitude',
		template='plotly_white',
		hovermode='closest', 
		height=700,
		legend=dict(x=0.02, y=0.98, bordercolor='gray', borderwidth=0.5),
	)
	fig.update_yaxes(autorange=True, fixedrange=False)
	fig.update_xaxes(autorange=True, fixedrange=False)


	server_file = f"/home/jiayu/SleepApnea4Ubicomp/HTML_plots/plot_{filename}_tst.html"
	fig.write_html(server_file, include_plotlyjs="cdn")
	print(f"✅ Saved interactive plot to {server_file}")

	mac_ip = "172.21.117.244"
	mac_user = "jc67805"
	mac_dest = f"/Users/{mac_user}/Desktop/HTML_plots_TST"

	try:
		subprocess.run(
			["scp", server_file, f"{mac_user}@{mac_ip}:{mac_dest}"],
			check=True
		)
		print(f"💾 Successfully copied to {mac_dest} on your Mac ({mac_ip})")
	except Exception as e:
		print(f"⚠️ Failed to copy to Mac: {e}")
		print(f"You can manually run:\nscp {server_file} {mac_user}@{mac_ip}:{mac_dest}")

	return server_file


def inference(X_concat, Y_concat, model, device, step_sig, threshold, XY=None):
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

	pred_res_stage = []
	pred_res_apnea = []

	model.eval()
	with torch.no_grad():
		for start in range(0, len(segments_tensor), batch_size):
			end = start + batch_size
			batch = segments_tensor[start:end]
			outputs1_stage, outputs1_apnea, _, _, _ = model(batch)
			outputs2_stage, outputs2_apnea, _, _, _ = model(torch.flip(batch, dims=[2]))
			outputs3_stage, outputs3_apnea, _, _, _ = model(-batch)

			output_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
			probs_stage = torch.softmax(output_stage, dim=1)  # shape: [batch, num_classes]
			prob_class1_stage = probs_stage[:, 1].cpu().numpy()
			pred_stage = (prob_class1_stage >= threshold).astype(int)
			pred_res_stage.extend(pred_stage)

			output_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
			probs_apnea = torch.softmax(output_apnea, dim=1)  # shape: [batch, num_classes]
			prob_class1_apnea = probs_apnea[:, 1].cpu().numpy()
			pred_apnea = (prob_class1_apnea >= threshold).astype(int)
			pred_res_apnea.extend(pred_apnea)

	return np.array(pred_res_stage), np.array(pred_res_apnea)


def get_wake_masks(SleepStage_concat, step_sleep):
	sleep_segments = []
	for i in range(0, len(SleepStage_concat) - 60 + step_sleep, step_sleep):
		sleep_segment = SleepStage_concat[i:i+60]
		if len(sleep_segment) == 60:
			sleep_segments.append(sleep_segment)

	sleep_segments = np.array(sleep_segments)  # shape: [N, 60]

	wake_masks = []
	for i in range(len(sleep_segments)):
		sleep_segment = sleep_segments[i]
		if -1 in sleep_segment or 4 in sleep_segment:
			wake_masks.append(i)
	
	wake_masks = np.array(wake_masks)
	return wake_masks


def get_wake_masks_pred(SleepStage_concat, step_sleep):
	sleep_segments = []
	for i in range(0, len(SleepStage_concat) - 60 + step_sleep, step_sleep):
		sleep_segment = SleepStage_concat[i:i+60]
		if len(sleep_segment) == 60:
			sleep_segments.append(sleep_segment)

	sleep_segments = np.array(sleep_segments)  # shape: [N, 60]

	wake_masks = []
	for i in range(len(sleep_segments)):
		sleep_segment = sleep_segments[i]
		if 1 in sleep_segment:
			wake_masks.append(i)
	
	wake_masks = np.array(wake_masks)
	return wake_masks


def get_wake_masks_29(SleepStage_concat, step_sleep):
	sleep_segments = []
	for i in range(0, len(SleepStage_concat) - 60 + step_sleep, step_sleep):
		sleep_segment = SleepStage_concat[i:i+60]
		if len(sleep_segment) == 60:
			sleep_segments.append(sleep_segment)

	sleep_segments = np.array(sleep_segments)  # shape: [N, 60]

	wake_masks = []
	for i in range(len(sleep_segments)):
		sleep_segment = sleep_segments[i]
		if np.sum(sleep_segment == 4) >= 55 or -1 in sleep_segment:
			wake_masks.append(i)
	
	wake_masks = np.array(wake_masks)
	return wake_masks


def process_allnight_data(data, duration=60, overlap=30, denoising=False):
	step_seconds = duration - overlap
	sampling_rate = data.shape[1] / duration
	step_size = int(step_seconds * sampling_rate)
	concatenated = concatenate_segments(data, step_size)
	if denoising:
		concatenated = denoise(concatenated)
	
	concatenate = resample_poly(concatenated, 1, 10)
	concatenate = normalize_1d(concatenate)
	return concatenate, np.arange(len(concatenate)) / (sampling_rate/10)


def count_continuous_ones(signal):
	signal = np.array(signal)
	segments = []
	
	in_segment = False
	start_idx = 0
	
	for i in range(len(signal)):
		if signal[i] == 1 and not in_segment:
			in_segment = True
			start_idx = i
		elif signal[i] == 0 and in_segment:
			segments.append({
				'start': start_idx,
				'end': i - 1,
				'length': i - start_idx
			})
			in_segment = False
	
	if in_segment:
		segments.append({
			'start': start_idx,
			'end': len(signal) - 1,
			'length': len(signal) - start_idx
		})
	
	return segments, len(segments)


def compute_segmented_mae(AHI_labels, AHI_preds):
	AHI_labels = np.array(AHI_labels)
	AHI_preds = np.array(AHI_preds)

	segments = np.array([ahi_to_severity(x) for x in AHI_labels])

	maes = {}
	for s in range(4):
		idx = (segments == s)
		if np.sum(idx) == 0:
			maes[s] = None
		else:
			maes[s] = np.mean(np.abs(AHI_labels[idx] - AHI_preds[idx]))

	return {
		"Normal (<5)": maes[0],
		"Mild (5–15)": maes[1],
		"Moderate (15–30)": maes[2],
		"Severe (≥30)": maes[3],
	}


def remove_short_ones(x, min_len=2):
	out = []
	for val, group in groupby(x):
		g = list(group)
		if val == 1 and len(g) < min_len:
			out.extend([0] * len(g))
		else:
			out.extend(g)
	return np.array(out)


def load_configs(config_path):	
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
		fold2id = config["fold_id_mapping"]
		fold2threshold_stage = config["fold_to_threshold_stage"]
		fold2threshold_apnea = config["fold_to_threshold_apnea"]
	return fold2id, fold2threshold_stage, fold2threshold_apnea


def ratio_check(file, threshold=0.75):
	try: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p144_rdi/' + file)
	except: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p137_rdi_useless/' + file)
	
	ratio = data_check_quality.shape[0] / data.shape[0]
	print(f'ratio: {data_check_quality.shape[0] / data.shape[0]:.3f}')
	if ratio < 0.75: return False
	return True


def ratio_check_lst(npy_ID):
	lst = [136, 39, 106, 43, 155, 21, 32, 156, 99, 124, 24, 31, 36, 47, 157, 22, 91, 83, 13, 123, 148, 122, 50, 120, 154, 118, 17, 5, 153, 98, 27, 30, 10, 134, 101, 151, 133, 9, 7, 86, 84, 53, 144, 110, 141, 146, 2, 107, 19, 55, 119, 20, 1, 138, 40, 14, 12, 34, 137, 29, 41, 15, 3, 23, 18, 11, 92, 143, 102, 117, 57, 48, 44, 152, 4, 150, 51, 93, 114, 127, 52, 8, 116, 42, 104, 108, 97, 33, 25]
	if npy_ID in lst: return True
	else: return False

def load_model_MTL(model_folder, device, axis=2):
	patch_len = 24
	n_layers = 4
	d_model = 64
	n_heads = 4
	d_ff = 256
	output_class = 2

	model = ApneaClassifier_PatchTST_MTL(
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



if __name__ == "__main__":
	data_folder = '/home/jiayu/SleepApnea4Ubicomp/Data/data_60s_30s_yingjian2/'
	duration = 60
	overlap = 30
	cuda = '0'
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	df = pd.read_excel('/home/jiayu/SleepApnea4Ubicomp/Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')

	id2fold = {}
	ID_num = 109

	step_sig_apn = 150
	step_sig_sleep = 10
	order = 0
	
	end = 'ws1_wa1_gs0.0001_ga0.0001_mask_lr5_mr0.5'
	model_folder_name = f'MTL_XY_p{ID_num}_45s_F1_{end}'
	# config_path = f"/home/jiayu/SleepApnea4Ubicomp/Code/configs_mtl/p{ID_num}_45s_MTL_ws1_wa1_gs0.0001_ga0.0001_mask_lr5.yaml"
	config_path = f"/home/jiayu/SleepApnea4Ubicomp/Code/configs_mtl/p{ID_num}_45s_MTL_{end}.yaml"

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

		# ============ Data Check ============
		if ID_npy not in id2fold: continue
		# if ID_npy in [50, 25, 108, 134, 120, 153, 119]: continue
		if ID_npy in [24, 50, 25, 134, 153, 119, 114]: continue

		if not ratio_check(file): continue
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
		model = load_model_MTL(fold_idx, model_folder_name, device)
		
		_, pred_res_apn = inference(X_concat, Y_concat, model, device, step_sig_apn, threshold=fold_to_threshold_apn[fold_idx])	
		pad_length_apn = 600 // step_sig_apn
		pred_res_apn = np.pad(pred_res_apn, (pad_length_apn, 0), mode='constant', constant_values=0)
		pred_time_apn = np.arange(len(pred_res_apn)) * step_sig_apn / 10  


		pred_res_sleep, _ = inference(X_concat, Y_concat, model, device, step_sig_sleep, threshold=fold_to_threshold_stage[fold_idx])
		pad_length_sleep = 60 // step_sig_sleep
		pred_res_sleep = np.pad(pred_res_sleep, (pad_length_sleep, 1), mode='constant', constant_values=1)
		pred_time_sleep = np.arange(len(pred_res_sleep)) * step_sig_sleep / 10
		

		# ks = 1
		# pred_res_sleep = median_filter(pred_res_sleep, size=ks, mode='constant', cval=1)

		# =========== Post-processing ============
		# pred_sleep_time_raw = np.sum(pred_res_sleep==0)  / (3600 / (step_sig_sleep / 10))
		# if pred_sleep_time_raw <= 4: order = 8
		# else: order = 0
		# order = 0
		# pred_res_sleep = remove_short_ones(pred_res_sleep, min_len=order+1)		

		wake_masks = get_wake_masks_pred(pred_res_sleep, step_sig_apn // 10)	
		if ID_npy == 29:
			wake_masks = get_wake_masks_29(SleepStage_concat, step_sig_apn // 10)
		pred_res_apn[wake_masks] = 0 

		pred_res_processed = np.zeros_like(pred_res_apn)
		for i in range(1, len(pred_res_apn)):
			if pred_res_apn[i] == 1 and pred_res_processed[i-1] == 1:
				pred_res_processed[i] = 0
			else:
				pred_res_processed[i] = pred_res_apn[i]
		# pred_res_processed = pred_res_apn.copy()

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

		# minus 2
		# AHI_preds_processed_label = AHI_preds_processed_label - 2 if AHI_preds_processed_label > 2 else AHI_preds_processed_label

		TST_labels.append(sleep_time_excel)
		TST_preds.append(sleep_time_pred)
		
		AHI_labels.append(AHI_label)
		AHI_preds.append(AHI_preds_processed_label)
		


		# fig, axes = plt.subplots(3, 1, figsize=(16, 9))
		# plt.title(f'Patient ID: {ID_npy}')
		# SleepStage_concat[(SleepStage_concat < 4) & (SleepStage_concat >= 0)] = 0
		# SleepStage_concat[SleepStage_concat == -1] = 1
		# SleepStage_concat[SleepStage_concat >= 4] = 1		
		
		# axes[0].plot(time_sleep, SleepStage_concat, label='Sleep Stage')
		# axes[0].set_title(f'Sleep Stage Label, {sleep_time_excel:.2f} h')
		# axes[1].plot(pred_time_sleep, pred_res_sleep, label='Predicted Sleep Time')
		# axes[1].set_title(f'Predicted Sleep Time, {sleep_time_pred:.2f} h')
		# axes[2].plot(pred_time_sleep, pred_res_sleep_processed, label='Processed Predicted Sleep Time')
		# axes[2].set_title(f'Processed Sleep Time, {sleep_time_pred_processed:.2f} h')

		# plt.tight_layout()
		# plt.savefig(f'/home/jiayu/SleepApnea4Ubicomp/Models/binary_Sleep_Wake_XY_p109_Supcon_F1_balanced_mask0.5_AHI_45s_sw_AVG_true/figs/{order}order/{file[:-4]}.png')
		# plt.close()

		# if ID_npy in [114] or (AHI_label >15 and AHI_label <= 30):
		# if np.abs(AHI_label - AHI_preds_processed_label) >= 8:
		# 	signals = [
		# 		# (time_xyz, X_concat, 'X-axis', '#1f77b4'),
		# 		# (time_xyz, Y_concat, 'Y-axis', '#ff7f0e'),
		# 		# (time_tho_abd, THO_concat, 'Thoracic', '#d62728'),
		# 		# (time_tho_abd, ABD_concat, 'Abdominal', '#9467bd'),
		# 		(time_sleep, SleepStage_concat, 'Sleep Stage', '#8c564b'),
		# 		(time_event, Event_concat, 'Events', '#e377c2'),
		# 		(pred_time_apn, pred_res_processed, 'Processed Predicted Apnea', '#bcbd22'),
		# 		(pred_time_sleep, pred_res_sleep, 'Predicted Sleep', '#7f7f7f')
		# 	]


		# 	output_file = plot_all_concatenated_signals(signals, 
		# 											n_true_apnea_events, n_apnea_events, 
		# 											AHI_label, AHI_preds_processed_label,
		# 												file[:-4])
		# 	print('---------------------------------------')
			

	# path = f'Models/{model_folder_name}/PatchTST_patchlen24_{step_sig_apn//10}s_larger2_change106_change29_no153119'
	# sleep_path = f'Models/{model_folder_name}/PatchTST_patchlen24_{step_sig_sleep//10}s_TST_larger2_change106_change29_no153119'
	
	path = f'Models/{model_folder_name}/PatchTST_patchlen24_{step_sig_apn//10}s_larger2_change106108134_change29_no15311924114_with108'
	sleep_path = f'Models/{model_folder_name}/PatchTST_patchlen24_{step_sig_sleep//10}s_TST_larger2_change106108134_change29_no15311924114_with108'


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