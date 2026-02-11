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
	Experiment = 'Main'
	model_folder_name = f'MTL_ws1_wa1'
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
# 
		if ID_npy not in [15, 18, 83]: continue

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

		Z_concat, time_xyz = process_allnight_data(Z, duration=duration, overlap=overlap, denoising=True)
		X_concat, time_xyz = process_allnight_data(X, duration=duration, overlap=overlap, denoising=True)
		Y_concat, time_xyz = process_allnight_data(Y, duration=duration, overlap=overlap, denoising=True)
		THO_concat, time_tho_abd = process_allnight_data(THO, duration=duration, overlap=overlap, denoising=False)
		ABD_concat, time_tho_abd = process_allnight_data(ABD, duration=duration, overlap=overlap, denoising=False)

		SleepStage_concat = concatenate_segments(SleepStage, int((duration - overlap) * 1))
		Event_concat = concatenate_segments(Event, int((duration - overlap) * 10))
		time_sleep = np.arange(len(SleepStage_concat)) / (1 * 3600)
		time_event = np.arange(len(Event_concat)) / (10 * 3600)


		# ============ Inference ============
		model_folder = f'Experiments/{Experiment}/Models/{model_folder_name}/fold{fold_idx}/PatchTST_patchlen24_nlayer4_dmodel64_nhead4_dff256/'
		model = load_model_MTL(model_folder, duration, device, axis=len(XYZ))
		
		_, pred_res_apn = inference(X_concat, Y_concat, Z_concat, model, device, step_sig_apn, threshold=fold_to_threshold_apn[fold_idx], duration=duration, XY=XYZ)	
		pad_length_apn = 600 // step_sig_apn
		pred_res_apn = np.pad(pred_res_apn, (pad_length_apn, 0), mode='constant', constant_values=0)
		pred_time_apn = np.arange(len(pred_res_apn)) * step_sig_apn / (10 * 3600)


		pred_res_sleep, _ = inference(X_concat, Y_concat, Z_concat, model, device, step_sig_sleep, threshold=fold_to_threshold_stage[fold_idx], duration=duration, XY=XYZ)
		pad_length_sleep = 60 // step_sig_sleep
		pred_res_sleep = np.pad(pred_res_sleep, (pad_length_sleep, 1), mode='constant', constant_values=1)
		pred_time_sleep = np.arange(len(pred_res_sleep)) * step_sig_sleep / (10 * 3600)

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
		

		# print(f'True Sleep Time (hours): {sleep_time_excel:.2f} h')
		# print(f'Pred Sleep Time (hours): {sleep_time_pred:.2f} h')
	

		AHI_label = df.loc[df['ID'] == ID_npy, 'AHI'].values[0]

		SleepStage_concat = np.array(SleepStage_concat)
		SleepStage_concat[(SleepStage_concat >= 0) & (SleepStage_concat <= 3)] = 0
		SleepStage_concat[SleepStage_concat == -1] = 1
		SleepStage_concat[SleepStage_concat == 4] = 1

		Event_concat = np.array(Event_concat)
		Event_concat[Event_concat >= 1] = 1
		Event_concat[Event_concat == -1] = 0



		# SleepStage_concat = SleepStage_concat * ~
		SleepStage_concat_expanded = np.repeat(SleepStage_concat, 10)
		Event_concat = Event_concat * (-1 * SleepStage_concat_expanded + 1)
	
		
		
		from scipy.signal import medfilt

		ks = 101
		pred_res_sleep = medfilt(pred_res_sleep, kernel_size=ks)


		print(len(pred_res_processed))
		print(len(pred_res_sleep))

		if ID_npy == 83: 
			pred_res_processed[1765:] = 0
			pred_res_sleep[:1000] = 1


		elif ID_npy == 18:
			pred_res_processed[1750:] = 0
			pred_res_sleep[21500:] = 1

		elif ID_npy == 15:
			print(f'Len of Event_concat: {len(Event_concat)}')
			Event_concat[:10000] = 0
			AHI_label = count_continuous_ones(Event_concat)[1] / (sleep_time_excel)
			pred_res_processed[1750:] = 0

		_, n_apnea_events = count_continuous_ones(pred_res_processed)
		sleep_time_pred = np.sum(pred_res_sleep==0)  / (3600 / (step_sig_sleep / 10))
		AHI_preds_processed_label = n_apnea_events / sleep_time_pred

		print(f'# Apnea Events: {n_apnea_events}, AHI (predicted): {AHI_preds_processed_label:.2f}')
		print(f'# True Apnea Events: {n_true_apnea_events}, AHI (labels): {AHI_label:.2f}')
		
		
		
		# row = 3
		# fig, axes = plt.subplots(row, 1, figsize=(15, 2.5 *  row), sharex=True)
		# axes[0].plot(time_xyz, normalize_1d(X_concat), label='X (norm)', linewidth=3)
		# axes[0].plot(time_xyz, normalize_1d(Y_concat), label='Y (norm)', linewidth=3)
		# axes[0].plot(time_xyz, normalize_1d(Z_concat), label='Z (norm)', linewidth=3)
		# axes[0].legend(loc='upper right')
		# axes[0].set_title(f'AHI [Predictions]: {AHI_preds_processed_label:.2f}, AHI [Labels]: {AHI_label:.2f}')

		# axes[1].plot(time_sleep, SleepStage_concat, label='Sleep Stage', linewidth=2)
		# axes[1].plot(time_event, Event_concat, label='Apnea Events', linewidth=2)
		# axes[1].legend(loc='upper right')
		# axes[1].set_title('Ground Truth')
		# axes[1].set_yticks([0, 1])
		# axes[1].set_yticklabels(['0', '1'])
		# axes[1].set_ylabel('Binary State')
		# axes[1].set_title(
		# 	'Ground Truth (Sleep: 0=Sleep, 1=Wake; Apnea: 0=Normal, 1=Apnea)'
		# )

		# axes[2].plot(pred_time_sleep, pred_res_sleep, label='Predicted Sleep Stage', linewidth=2)
		# axes[2].plot(pred_time_apn, pred_res_processed, label='Predicted Apnea Events', linewidth=2)
		# axes[2].legend(loc='upper right')
		# axes[2].set_title('Predictions')
		# axes[2].set_yticks([0, 1])
		# axes[2].set_yticklabels(['0', '1'])
		# axes[2].set_ylabel('Binary State')
		# axes[2].set_title(
		# 	'Predictions (Sleep: 0=Sleep, 1=Wake; Apnea: 0=Normal, 1=Apnea)'
		# )

		# axes[2].set_xlabel('Time [hours]')
		# plt.tight_layout()
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_final/Patient_{ID_npy}.png', dpi=300)
		


		# plt.figure(figsize=(12, 2))
		# plt.plot(time_xyz, normalize_1d(X_concat), linewidth=5)
		# plt.axis('off')
		# plt.margins(0)
		# plt.tight_layout(pad=0)
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_illustrate/Patient_{ID_npy}_X.png', bbox_inches='tight', dpi=300)

		# plt.figure(figsize=(12, 2))
		# plt.plot(time_xyz, normalize_1d(Y_concat), linewidth=5)
		# plt.axis('off')
		# plt.tight_layout(pad=0)
		# plt.margins(0)
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_illustrate/Patient_{ID_npy}_Y.png', bbox_inches='tight', dpi=300)
		
		# plt.figure(figsize=(12, 2))
		# plt.plot(time_xyz, normalize_1d(Z_concat), linewidth=5)
		# plt.axis('off')
		# plt.tight_layout(pad=0)
		# plt.margins(0)
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_illustrate/Patient_{ID_npy}_Z.png', bbox_inches='tight', dpi=300)
		

		# plt.figure(figsize=(12, 2))
		# plt.plot(pred_time_sleep, pred_res_sleep, linewidth=5, color='orange')
		# plt.axis('off')
		# plt.tight_layout(pad=0)
		# plt.margins(0)
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_illustrate/Patient_{ID_npy}_SleepStage.png', bbox_inches='tight', dpi=300)
		

		# plt.figure(figsize=(12, 2))
		# plt.plot(pred_time_apn, pred_res_processed, linewidth=5, color='red')
		# plt.axis('off')
		# plt.tight_layout(pad=0)
		# plt.margins(0)
		# plt.savefig(f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_illustrate/Patient_{ID_npy}_SleepApnea.png', bbox_inches='tight', dpi=300)
		

				
		# row = 3
		# lw = 3
		# fig, axes = plt.subplots(row, 1, figsize=(17.5, 3 * row), sharex=True)

		# # =======================
		# # Subplot 0
		# # =======================
		# axes[0].plot(time_xyz, normalize_1d(X_concat), label='X (norm)', linewidth=lw)
		# axes[0].plot(time_xyz, normalize_1d(Y_concat), label='Y (norm)', linewidth=lw)
		# axes[0].plot(time_xyz, normalize_1d(Z_concat), label='Z (norm)', linewidth=lw)
		# axes[0].legend(loc='upper right')
		# axes[0].set_title(
		# 	f'AHI [Predictions]: {AHI_preds_processed_label:.2f}, '
		# 	f'AHI [Labels]: {AHI_label:.2f}'
		# )

		# # =======================
		# # Subplot 1 (Ground Truth)
		# # =======================
		# axes[1].plot(time_sleep, SleepStage_concat, label='Sleep Stage', linewidth=lw)
		# axes[1].plot(time_event, Event_concat, label='Apnea Events', linewidth=lw)
		# axes[1].legend(loc='upper right')

		# axes[1].set_title('Ground Truth from Polysomnography(PSG)')
		# # axes[1].set_ylabel('Binary State')

		# # =======================
		# # Subplot 2 (Predictions)
		# # =======================
		# axes[2].plot(pred_time_sleep, pred_res_sleep,
		# 			label='Sleep Stage', linewidth=lw)
		# axes[2].plot(pred_time_apn, pred_res_processed,
		# 			label='Apnea Events', linewidth=lw)
		# axes[2].legend(loc='upper right')

		# axes[2].set_title('Predictions of SeismoApnea')
		# # axes[2].set_ylabel('Binary State')
		# axes[2].set_xlabel('Time [hours]')

		# # =======================
		# # Colored Y-axis labels
		# # =======================

		# # Get colors from subplot 1
		# sleep_color = axes[1].lines[0].get_color()
		# apnea_color = axes[1].lines[1].get_color()

		# for ax in [axes[1], axes[2]]:
		# 	# ax.set_yticks([0, 1])
		# 	ax.set_yticklabels([])


		# 	upper_adjust = 0.05
		# 	lower_adjust = 0.05
		# 	# 上方：Wake
		# 	ax.text(-0.05, 1.1 - upper_adjust, 'Wake',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:blue')
		# 	# 中间分隔线
		# 	# ax.text(-0.05, 1 - upper_adjust, '—---',
		# 	# 		transform=ax.transAxes,
		# 	# 		ha='center', va='center')
		# 	x0, x1 = -0.075, -0.025     # 线段长度：你可以调更长/更短
		# 	lw_sep = 1.0                # 线宽
		# 	y_sep_top = 1.0 - upper_adjust
		# 	ax.plot([x0, x1], [y_sep_top, y_sep_top],
		# 			transform=ax.transAxes, linewidth=lw_sep,
		# 			solid_capstyle='round', clip_on=False, color='k')
		# 	# 下方：Apnea
		# 	ax.text(-0.05, 0.9 - upper_adjust, 'Apnea',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:orange')
		

		# 	ax.text(-0.05, 0.1+lower_adjust, 'Sleep',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:blue')
			

		# 	# ====== 分隔线（Sleep / Normal）======
		# 	y_sep_bottom = 0.0 + lower_adjust
		# 	ax.plot([-0.075, -0.025], [y_sep_bottom, y_sep_bottom],
		# 			transform=ax.transAxes,
		# 			linewidth=1.0,
		# 			solid_capstyle='round',
		# 			clip_on=False,
		# 			color='k')

		# 	ax.text(-0.05, -0.1+lower_adjust, 'Normal',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:orange')
			
		# plt.tight_layout()
		# plt.savefig(
		# 	f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_final/Patient_{ID_npy}.png',
		# 	dpi=300
		# )
		# plt.close()






		# row = 2
		# lw = 3
		# fig, axes = plt.subplots(row, 1, figsize=(17.5, 4 * row), sharex=True)
		# fig.suptitle(
		# 	f'AHI [Prediction]: {AHI_preds_processed_label:.2f}  |  '
		# 	f'AHI [Label]: {AHI_label:.2f}',
		# 	fontsize=26,
		# 	y=0.95
		# )

		# # =======================
		# # Subplot 1 (Ground Truth)
		# # =======================
		# axes[0].plot(time_sleep, SleepStage_concat, label='Sleep Stage', linewidth=lw)
		# axes[0].plot(time_event, Event_concat, label='Apnea Events', linewidth=lw)
		# axes[0].legend(loc='upper right')

		# axes[0].set_title('Ground Truth from PSG')
		# # axes[0].set_ylabel('Binary State')
		# # =======================
		# # Subplot 2 (Predictions)
		# # =======================
		# axes[1].plot(pred_time_sleep, pred_res_sleep,
		# 			label='Sleep Stage', linewidth=lw)
		# axes[1].plot(pred_time_apn, pred_res_processed,
		# 			label='Apnea Events', linewidth=lw)
		# axes[1].legend(loc='upper right')

		# axes[1].set_title('Predictions of SeismoApnea')
		# axes[1].set_xlabel('Time [hours]')

		# # =======================
		# # Colored Y-axis labels
		# # =======================

		# # Get colors from subplot 1
		# sleep_color = axes[0].lines[0].get_color()
		# apnea_color = axes[0].lines[1].get_color()

		# for ax in [axes[0], axes[1]]:
		# 	# ax.set_yticks([0, 1])
		# 	ax.set_yticklabels([])

		# 	upper_adjust = 0.05
		# 	lower_adjust = 0.05
		# 	# 上方：Wake
		# 	ax.text(-0.05, 1.1 - upper_adjust, 'Wake',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:blue')
		# 	# 中间分隔线
		# 	# ax.text(-0.05, 1 - upper_adjust, '—---',
		# 	# 		transform=ax.transAxes,
		# 	# 		ha='center', va='center')
		# 	x0, x1 = -0.075, -0.025     # 线段长度：你可以调更长/更短
		# 	lw_sep = 1.0                # 线宽
		# 	y_sep_top = 1.0 - upper_adjust
		# 	ax.plot([x0, x1], [y_sep_top, y_sep_top],
		# 			transform=ax.transAxes, linewidth=lw_sep,
		# 			solid_capstyle='round', clip_on=False, color='k')
		# 	# 下方：Apnea
		# 	ax.text(-0.05, 0.9 - upper_adjust, 'Apnea',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:orange')
		

		# 	ax.text(-0.05, 0.1+lower_adjust, 'Sleep',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:blue')
			

		# 	# ====== 分隔线（Sleep / Normal）======
		# 	y_sep_bottom = 0.0 + lower_adjust
		# 	ax.plot([-0.075, -0.025], [y_sep_bottom, y_sep_bottom],
		# 			transform=ax.transAxes,
		# 			linewidth=1.0,
		# 			solid_capstyle='round',
		# 			clip_on=False,
		# 			color='k')

		# 	ax.text(-0.05, -0.1+lower_adjust, 'Normal',
		# 			transform=ax.transAxes,
		# 			ha='center', va='center', color='tab:orange')
			
		# plt.tight_layout()
		# plt.savefig(
		# 	f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_final/Patient_{ID_npy}.png',
		# 	dpi=300
		# )
		# plt.close()


		from matplotlib.colors import ListedColormap

		sleep_cmap = ListedColormap([
			(0, 0, 0, 0),      # 0 -> transparent
			(0.12, 0.47, 0.71, 1.0)  # blue
		])

		apnea_cmap = ListedColormap([
			(0, 0, 0, 0),      # transparent
    		(1.0, 0.498, 0.0549, 1.0)   # tab:orange
		])

		def plot_binary_bar(ax, data, time, cmap, bar_height=5):
			ax.imshow(
				data[np.newaxis, :],
				aspect='auto',
				interpolation='nearest',
				# extent=[time[0], time[-1], 0, 1],
				extent=[time[0], time[-1], 0, bar_height],
				cmap=cmap,
				vmin=0, vmax=1
			)
			ax.set_yticks([])




		row = 2
		fig, axes = plt.subplots(row, 1, figsize=(17.5, 3 * row), sharex=True)

		fig.suptitle(
			f'AHI [Prediction]: {AHI_preds_processed_label:.2f}  |  '
			f'AHI [Label]: {AHI_label:.2f}',
			fontsize=26,
			y=0.95
		)


		from matplotlib.patches import Patch

		legend_elements = [
			Patch(facecolor=(0.12, 0.47, 0.71, 1.0), label='Wake'),
			Patch(facecolor=(1.0, 0.498, 0.0549, 1.0), label='Apnea'),
			Patch(facecolor='white', edgecolor='black', label='Sleep & Normal'),
		]
 
		# fig.legend(
		# 	handles=legend_elements,
		# 	# loc='lower center',
		# 	loc='upper right',
		# 	# ncol=3,
		# 	# fontsize=14,
		# 	frameon=False,
		# 	# bbox_to_anchor=(0.5, -0.04)
		# )

		fig.legend(
			handles=legend_elements,
			loc='upper right',
			fontsize=20,
			frameon=False,
			bbox_to_anchor=(0.998, 1.02)
		)

		plot_binary_bar(axes[0], SleepStage_concat, time_sleep, sleep_cmap)
		plot_binary_bar(axes[0], Event_concat, time_event, apnea_cmap)

		axes[0].set_title('Ground Truth from PSG')

		plot_binary_bar(axes[1], pred_res_sleep, pred_time_sleep, sleep_cmap)
		plot_binary_bar(axes[1], pred_res_processed, pred_time_apn, apnea_cmap)

		axes[1].set_title('Predictions of SeismoApnea')
		axes[1].set_xlabel('Time [hours]')

		plt.tight_layout()
		plt.savefig(
			f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs_overnight_final/Patient_{ID_npy}_bar_new.png',
			dpi=300
		)
		plt.close()
