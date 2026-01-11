import numpy as np
import os, sys
import pandas as pd
from tqdm import tqdm


def ratio_check(file, threshold=0.75):
	try: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p144_rdi/' + file)
	except: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p137_rdi_useless/' + file)
	
	ratio = data_check_quality.shape[0] / data.shape[0]
	if ratio < threshold: return False
	return True



all_patients = [
  4, 9, 23, 26, 30, 35, 39, 48, 50, 53, 56, 59, 63, 84, 90, 93, 99, 102, 108, 114, 118, 136, 138, 143, 144, 150, 151, 155,
  12, 16, 18, 25, 28, 29, 31, 36, 41, 45, 51, 58, 60, 66, 68, 71, 97, 101, 104, 105, 107, 110, 111, 116, 127, 131, 154,
  1, 10, 11, 15, 22, 32, 33, 34, 38, 42, 44, 52, 64, 87, 91, 92, 96, 98, 106, 119, 121, 123, 130, 141, 153, 156, 157,
  2, 3, 5, 7, 8, 14, 17, 19, 24, 27, 40, 46, 55, 57, 69, 81, 83, 86, 95, 117, 120, 122, 134, 137, 146, 148, 152]



# Useful patients 
if __name__ == "__main__":
	data_folder = '/home/jiayu/SleepApnea4Ubicomp_Feb/Data/data_60s_30s_yingjian2/'
	df = pd.read_excel('/home/jiayu/SleepApnea4Ubicomp_Feb/Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')
	ID_num = 109

	useful_ids = []

	for file in tqdm(sorted(os.listdir(data_folder))):
		# ============ Load Data ============
		data = np.load(os.path.join(data_folder, file))
		ID_npy = data[0, -2]		

		# ============ Data Check ============
		if ID_npy not in all_patients: continue
		if ID_npy in [24, 50, 25, 134, 153, 119, 114]: continue
		if not ratio_check(file): continue
		sleep_time_excel = df.loc[df['ID'] == ID_npy, 'Duration(h)'].values[0]  * df.loc[df['ID'] == ID_npy, 'SEfficiency'].values[0] * 0.01
		if sleep_time_excel <= 2:
			# print(f'Skipping patient {ID_npy} due to short sleep time: {sleep_time_excel:.2f} h\n')
			continue

		AHI_label = df.loc[df['ID'] == ID_npy, 'AHI'].values[0]
		if ID_npy == 7: AHI_label = df.loc[df['ID'] == ID_npy, 'RDI'].values[0]
		if ID_npy == 106: AHI_label = 30 / (7.32 * 83.5 / 100)
		if ID_npy == 134: AHI_label = 13.73
		if ID_npy == 108: AHI_label = 9.6

		useful_ids.append(ID_npy)
	print('Useful IDs:', useful_ids)
	print('Total Useful IDs:', len(useful_ids))

		

