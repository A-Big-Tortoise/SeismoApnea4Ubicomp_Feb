import numpy as np
import os, sys
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import denoise, normalize_1d
from Code.models.clf import ApneaClassifier_PatchTST_MTL
from Code.utils import choose_gpu_by_model_process_count, calculate_icc_standard, ahi_to_severity, calculate_cm
from Code.plotly_xz_mtl import plot_person_level_results_sleep \
	 , plot_person_level_results, concatenate_segments, inference \
	 , ratio_check_lst, get_wake_masks_pred, get_wake_masks_29 \
	 , process_allnight_data, count_continuous_ones  \
     , load_configs, load_model_MTL
from Code.plotly_xz_mtl_rec import compute_segmented_mae, compute_tst_mae
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from pprint import pprint
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


plt.rcParams.update({
	'axes.titlesize': 18,     # 图标题
	'axes.labelsize': 16,     # 坐标轴标题
	'xtick.labelsize': 14,    # x轴刻度
	'ytick.labelsize': 14,    # y轴刻度
	'legend.fontsize': 14,    # 图例
})



def compute_segmented_mae_overall(AHI_labels, AHI_preds, TST_labels, TST_preds):
	AHI_labels = np.asarray(AHI_labels)
	AHI_preds = np.asarray(AHI_preds)

	segments = np.array([ahi_to_severity(x) for x in AHI_labels])

	results = ''

	severity_names = {
		0: "Normal (<5)",
		1: "Mild (5–15)",
		2: "Moderate (15–30)",
		3: "Severe (≥30)",
	}

	idx_sum = []
	for s in range(4):
		idx = (segments == s)
		idx_sum.append(np.sum(idx))
		key = severity_names[s]

		if np.sum(idx) == 0:
			results += f"{key}: N/A\n"
		else:
			print(f"Computing MAE for {key}, N={np.sum(idx)}")
			errors = np.abs(AHI_labels[idx] - AHI_preds[idx])
			mae = np.mean(errors)
			std = np.std(errors)
			icc = calculate_icc_standard(AHI_labels[idx], AHI_preds[idx])

			# results[key] = f"{mae:.2f} ± {std:.2f}"
			results += f"{key}: {mae:.2f} $\pm$ {std:.2f}\n"
			results += f"  ICC: {icc:.3f}\n"
			correlation = np.corrcoef(AHI_labels[idx], AHI_preds[idx])[0, 1]
			results += f"  Correlation: {correlation:.3f}\n"


			tst_mae = np.mean(np.abs(TST_labels[idx] - TST_preds[idx]))
			tst_std = np.std(np.abs(TST_labels[idx] - TST_preds[idx]))
			results += f"  TST MAE: {tst_mae:.2f} $\pm$ {tst_std:.2f} Hours\n"
	print(results)
	
	return results




def plot_person_level_results_reg(y_true_list, y_pred_list,
							  fig_path):
		alpha = 0.3
		size = 150

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


		ICC = calculate_icc_standard(y_true_list, y_pred_list)
		Corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]
		max_val = max(max(y_true_list), max(y_pred_list)) + 1

		fig, axes = plt.subplots(1, 2, figsize=(12, 6))
		ax = axes[0]
		ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, linewidth=2.5)

		ax.scatter(
			y_pred_list,    # x = predicted
			y_true_list,   # y = label
			c=scatter_colors,
			s=size,
			alpha=alpha
		)

		ax.set_xlabel("# AHI [Predictions]")
		ax.set_ylabel("# AHI [Labels]")
		ax.set_title(f"Person-level AHI Scatter\nICC={ICC:.3f}, Corr={Corr:.3f}")


		legend_elements = [
			Line2D(
				[0], [0],
				marker='o',
				color='none',
				label=labels[s],
				markerfacecolor=color,
				markersize=12,
				markeredgecolor=color,    # ⭐ 去掉黑边
				markeredgewidth=1,       # ⭐ 彻底关掉
				alpha=alpha
			)
			for s, color in severity_colors.items()
		]

		ax.legend(
			handles=legend_elements,
			loc="upper left",
			fontsize=14,
			frameon=True
		)



		ax3 = axes[1]
		ba_diff = np.array(y_pred_list) - np.array(y_true_list)
		ba_avg = (np.array(y_pred_list) + np.array(y_true_list)) / 2
		mean_diff = np.mean(ba_diff)
		std_diff = np.std(ba_diff)
		ax3.scatter(
			ba_avg, ba_diff,
			c=scatter_colors,
			s=size,
			alpha=alpha
		)
		ymin, ymax = ax3.get_ylim()

		ax3.set_ylim([ymin, -ymin*0.65])  # make y-axis symmetric
		upper, lower = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
		ax3.axhline(mean_diff, color='blue', linestyle='--', linewidth=2.5)

		ax3.axhline(upper, color='red', linestyle='--', linewidth=2.5)

		ax3.axhline(lower, color='red', linestyle='--', linewidth=2.5)
		trans = mtransforms.blended_transform_factory(ax3.transAxes, ax3.transData)
		ymin, ymax = ax3.get_ylim()
	
		dy = 0.02 * (ymax - ymin)

		ax3.set_xlabel("Average AHI: (Predictions + Labels) / 2")
		ax3.set_ylabel("Difference in AHI: (Predictions - Labels)")
		ax3.set_title(f"Bland-Altman Plot\nMAE: {np.mean(np.abs(ba_diff)):.2f} $\pm$ {np.std(np.abs(ba_diff)):.2f}")

		def annotate_line(y, title, value, color):
			# title: slightly ABOVE the line
			ax3.text(
				0.99, y + dy,
				title,
				transform=trans,
				ha="right", va="bottom",
				fontsize=14, color=color,
			)
			# value: slightly BELOW the line
			ax3.text(
				0.99, y - dy,
				f"{value:.2f}",
				transform=trans,
				ha="right", va="top",
				fontsize=14, color=color,
			)

		annotate_line(mean_diff, "Mean diff", mean_diff, "blue")
		annotate_line(upper, "+1.96 SD", upper, "red")
		annotate_line(lower, "-1.96 SD", lower, "red")

		plt.tight_layout()

		plt.savefig(fig_path, dpi=300)
		plt.close()


def plot_person_level_results_clf(y_true_list, y_pred_list,
							  fig_path):
		# plt.figure(figsize=(8, 6))
		alpha = 0.3
		size = 150

		y_true = np.array([ahi_to_severity(x) for x in y_true_list])
		y_pred = np.array([ahi_to_severity(x) for x in y_pred_list])
		labels = ["Normal", "Mild", "Moderate", "Severe"]


		fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))

		cm = confusion_matrix(y_true, y_pred)
		row_sum = cm.sum(axis=1, keepdims=True)
		cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)  # safe divide

		annot = np.empty_like(cm, dtype=object)
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				annot[i, j] = f"{cm_pct[i, j]*100:.2f}%\n{cm[i, j]:d}"

		sns.heatmap(
			cm_pct,                      # color by percentage (row-normalized)
			annot=annot, fmt="",         # show our custom text
			cmap="Blues",
			vmin=0.0, vmax=1.0,          # keep color scale consistent
			xticklabels=labels, yticklabels=labels,
			square=True,                 # optional: looks nicer, doesn't change labels
			cbar=False,
			linewidths=0.5, linecolor="black",
			annot_kws={"fontsize": 14, "fontweight": "bold"},
			ax=ax2
		)


		bacc = balanced_accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average='macro')

		ax2.set_xlabel("Severity [Predictions]")
		ax2.set_ylabel("Severity [Labels]")
		ax2.set_title(f"Apnea Severity Classification\nBacc: {bacc*100:.2f}%, Macro-F1: {f1*100:.2f}%")

		plt.tight_layout()

		plt.savefig(fig_path, dpi=300)
		plt.close()

		print("Combined scatter + CM saved to:", fig_path)
		print(cm)





def plot_person_level_results_sleep(y_true_list, y_pred_list, ahi_label_list,
							  fig_path):
		
	alpha = 0.3
	size = 150
	ahi_severity = np.array([ahi_to_severity(x) for x in ahi_label_list])
	severity_colors = {
		0: "green",
		1: "yellow",
		2: "orange",
		3: "red"
	}
	scatter_colors = [severity_colors[s] for s in ahi_severity]   # 根据真实严重程度上色
	labels = ["Normal", "Mild", "Moderate", "Severe"]



	ICC = calculate_icc_standard(y_true_list, y_pred_list)
	Corr = np.corrcoef(y_true_list, y_pred_list)[0, 1]
	max_val = max(max(y_true_list), max(y_pred_list)) + 1


	
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	ax = axes[0]
	ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, linewidth=2.5)

	ax.scatter(
		y_pred_list,    # x = predicted
		y_true_list,   # y = label
		c=scatter_colors,
		s=size,
		alpha=alpha
	)

	ax.set_xlabel("Total Sleep Time (TST) [Predictions]")
	ax.set_ylabel("Total Sleep Time (TST) [Labels]")
	ax.set_title(f"Person-level Sleep Time Scatter\nICC={ICC:.3f}")


	legend_elements = [
		Line2D(
			[0], [0],
			marker='o',
			color='none',
			label=labels[s],
			markerfacecolor=color,
			markersize=12,
			markeredgecolor=color,    # ⭐ 去掉黑边
			markeredgewidth=1,       # ⭐ 彻底关掉
			alpha=alpha
		)
		for s, color in severity_colors.items()
	]

	ax.legend(
		handles=legend_elements,
		loc="upper left",
		fontsize=14,
		frameon=True
	)


	ax3 = axes[1]
	ba_diff = np.array(y_pred_list) - np.array(y_true_list)
	ba_avg = (np.array(y_pred_list) + np.array(y_true_list)) / 2
	mean_diff = np.mean(ba_diff)
	std_diff = np.std(ba_diff)
	ax3.scatter(
		ba_avg, ba_diff,
		c=scatter_colors,
		s=size,
		alpha=alpha
	)
	ymin, ymax = ax3.get_ylim()

	ax3.set_ylim([ymin*1.25, ymax*1.25])  # make y-axis symmetric
	# ax3.set_ylim([ymin, -ymin*0.65])  # make y-axis symmetric
	upper, lower = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
	ax3.axhline(mean_diff, color='blue', linestyle='--', linewidth=2.5)

	ax3.axhline(upper, color='red', linestyle='--', linewidth=2.5)

	ax3.axhline(lower, color='red', linestyle='--', linewidth=2.5)
	trans = mtransforms.blended_transform_factory(ax3.transAxes, ax3.transData)
	ymin, ymax = ax3.get_ylim()

	dy = 0.02 * (ymax - ymin)

	ax3.set_xlabel("Average TST: (Predictions + Labels) / 2")
	ax3.set_ylabel("Difference in TST: (Predictions - Labels)")
	ax3.set_title(f"Bland-Altman Plot\nMAE: {np.mean(np.abs(ba_diff)):.2f} $\pm$ {np.std(np.abs(ba_diff)):.2f} Hours")

	def annotate_line(y, title, value, color):
		# title: slightly ABOVE the line
		ax3.text(
			0.99, y + dy,
			title,
			transform=trans,
			ha="right", va="bottom",
			fontsize=14, color=color,
		)
		# value: slightly BELOW the line
		ax3.text(
			0.99, y - dy,
			f"{value:.2f}",
			transform=trans,
			ha="right", va="top",
			fontsize=14, color=color,
		)

	annotate_line(mean_diff, "Mean diff", mean_diff, "blue")
	annotate_line(upper, "+1.96 SD", upper, "red")
	annotate_line(lower, "-1.96 SD", lower, "red")

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.3)   # ⭐ 调大子图之间的横向间距

	plt.savefig(fig_path, dpi=300)
	plt.close()


def compute_cm_metrics(AHI_labels, AHI_preds):
	AHI_labels = np.asarray(AHI_labels)
	AHI_preds = np.asarray(AHI_preds)

	segments_labels = np.array([ahi_to_severity(x) for x in AHI_labels])
	segments_preds = np.array([ahi_to_severity(x) for x in AHI_preds])
	
	severity_names = {
		0: "Normal (<5)",
		1: "Mild (5–15)",
		2: "Moderate (15–30)",
		3: "Severe (≥30)",
	}

	results = {}
	y_true = segments_labels
	y_pred = segments_preds
	for c in range(4):
		tp = np.sum((y_pred == c) & (y_true == c))
		fp = np.sum((y_pred == c) & (y_true != c))
		fn = np.sum((y_pred != c) & (y_true == c))

		acc = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = (2 * acc * recall / (acc + recall)) if (acc + recall) > 0 else 0.0

		results[severity_names[c]] = {
			"acc (precision)": np.round(acc*100, 2),
			"recall": np.round(recall*100, 2),
			"f1": np.round(f1*100, 2)
		}
	print('results:', results)
	return results




if __name__ == "__main__":
	data_folder = 'Data/data_60s_30s_yingjian2/'
	duration = 60
	overlap = 30
	cuda = '0'
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	df = pd.read_excel('Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')

	id2fold = {}

	step_sig_apn = 140
	step_sig_sleep = 10
	
	XYZ = 'XY'
	Experiment = 'Report'
	model_folder_name = f'XY_60s_F1_ws1_wa1_again'
	# model_folder_name = f'XY_F1_ws1_wa1_was0.001'

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

		# ============ Data Check ============
		
		if ID_npy not in id2fold: continue
		# if ID_npy in [24, 50, 25, 134, 153, 119, 114, 99, 32]: continue
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

		Z_concat, time_xyz = process_allnight_data(Z, duration=duration, overlap=overlap, denoising=True)
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
		model = load_model_MTL(model_folder, duration, device, axis=len(XYZ))
		
		_, pred_res_apn = inference(X_concat, Y_concat, Z_concat, model, device, step_sig_apn, threshold=fold_to_threshold_apn[fold_idx], duration=duration, XY=XYZ)	
		
		pad_length_apn = 600 // step_sig_apn
		pred_res_apn = np.pad(pred_res_apn, (pad_length_apn, 0), mode='constant', constant_values=0)
		pred_time_apn = np.arange(len(pred_res_apn)) * step_sig_apn / 10  


		pred_res_sleep, _ = inference(X_concat, Y_concat, Z_concat, model, device, step_sig_sleep, threshold=fold_to_threshold_stage[fold_idx], duration=duration, XY=XYZ)
		pad_length_sleep = 60 // step_sig_sleep
		pred_res_sleep = np.pad(pred_res_sleep, (pad_length_sleep, 1), mode='constant', constant_values=1)
		pred_time_sleep = np.arange(len(pred_res_sleep)) * step_sig_sleep / 10

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


	path = f'Experiments/{Experiment}/Models/{model_folder_name}/AHI_{step_sig_apn//10}s_larger2_change106108134_change29'
	sleep_path = f'Experiments/{Experiment}/Models/{model_folder_name}/TST_{step_sig_sleep//10}s_larger2_change106108134_change29'


	AHI_labels, AHI_preds = np.array(AHI_labels), np.array(AHI_preds)	
	result = compute_segmented_mae(AHI_labels, AHI_preds)
	print(result) 
	
	lines = []
	lines.append(f'All #: {len(AHI_labels)}\n')

	TST_labels, TST_preds = np.array(TST_labels), np.array(TST_preds)
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

	compute_segmented_mae_overall(AHI_labels, AHI_preds, TST_labels, TST_preds)
	compute_cm_metrics(AHI_labels, AHI_preds)
	data_path = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Report/figs/'
	plot_person_level_results_reg(
		y_true_list=AHI_labels,
		y_pred_list=AHI_preds,
		fig_path=data_path+'overall_apnea_reg_all.png'
	)	
	plot_person_level_results_clf(
		y_true_list=AHI_labels,
		y_pred_list=AHI_preds,
		fig_path=data_path+'overall_apnea_clf_all.png'
	)	
	plot_person_level_results_sleep(
		y_true_list=TST_labels,
		y_pred_list=TST_preds,
		ahi_label_list=AHI_labels,
		fig_path=data_path+'overall_sleep_all.png'
	)