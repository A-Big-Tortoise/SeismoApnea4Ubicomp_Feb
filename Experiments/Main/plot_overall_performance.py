import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')

from sklearn.metrics import confusion_matrix
# from Code.plotly_xz_mtl_rec import plot_person_level_results_sleep \
	#  , plot_person_level_results, concatenate_segments, inference_REC 
from Code.utils import calculate_icc_standard, ahi_to_severity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from pprint import pprint


def compute_segmented_mae(AHI_labels, AHI_preds, TST_labels, TST_preds):
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




plt.rcParams.update({
	'axes.titlesize': 18,     # 图标题
	'axes.labelsize': 16,     # 坐标轴标题
	'xtick.labelsize': 14,    # x轴刻度
	'ytick.labelsize': 14,    # y轴刻度
	'legend.fontsize': 14,    # 图例
})


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


def plot_person_level_results(y_true_list, y_pred_list,
							  fig_path):
		# plt.figure(figsize=(8, 6))
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

		fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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


		ax2 = axes[1]
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

		ax3 = axes[2]
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

		print("Combined scatter + CM saved to:", fig_path)
		print(cm)



if __name__ == "__main__":
	data_path = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Main/figs/'
	
	AHI_labels = np.load(os.path.join(data_path, 'AHI_labels.npy')).tolist()
	AHI_preds = np.load(os.path.join(data_path, 'AHI_preds.npy')).tolist()
	TST_labels = np.load(os.path.join(data_path, 'TST_labels.npy'))
	TST_preds = np.load(os.path.join(data_path, 'TST_preds.npy'))

	compute_segmented_mae(AHI_labels, AHI_preds, TST_labels, TST_preds)
	compute_cm_metrics(AHI_labels, AHI_preds)
	# plot_person_level_results(
	# 	y_true_list=AHI_labels,
	# 	y_pred_list=AHI_preds,
	# 	fig_path=data_path+'overall_apnea.png'
	# )	

	# plot_person_level_results_sleep(
	# 	y_true_list=TST_labels,
	# 	y_pred_list=TST_preds,
	# 	ahi_label_list=AHI_labels,
	# 	fig_path=data_path+'overall_sleep.png'
	# )



