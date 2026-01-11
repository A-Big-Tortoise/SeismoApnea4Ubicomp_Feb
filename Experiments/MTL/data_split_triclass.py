import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from Code.utils import read_excel, time2timestamp

src_data_folder = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Data/fold_data_p109_MTL_45s/'
save_data_folder = '/home/jiayu/SeismoApnea4Ubicomp_Feb/Data/fold_data_p109_TriClass_60s/'


for src_data_path in os.listdir(src_data_folder):
	data_triclass_labels = []
	print(f'Processing {src_data_path}...')
	data = np.load(os.path.join(src_data_folder, src_data_path), allow_pickle=True)
	data_stage_label = data[:, -2]
	data_apnea_label = data[:, -1]
	print(f'Sleep/Wake labels: {np.unique(data_stage_label, return_counts=True)}')
	print(f'Apnea labels: {np.unique(data_apnea_label, return_counts=True)}')
	for i in range(data.shape[0]):
		stage, apnea = data_stage_label[i], data_apnea_label[i]
		if stage == 1: # Wake
			data_triclass_labels.append(2)
		else: # Sleep
			if apnea == 0:
				data_triclass_labels.append(0) # Normal
			else:
				data_triclass_labels.append(1) # Hypopnea + Apnea
	data_triclass_labels = np.array(data_triclass_labels)
	data_new = np.concatenate((data[:, :-2], data_triclass_labels.reshape(-1, 1)), axis=1)
	print(data_new.shape)

	np.save(os.path.join(save_data_folder, src_data_path), data_new)