import numpy as np
import os, sys
import pandas as pd
from tqdm import tqdm	
from collections import Counter



# def ratio_check(file, threshold=0.75):
# 	try: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p144_rdi/' + file)
# 	except: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p137_rdi_useless/' + file)
	
# 	ratio = data_check_quality.shape[0] / data.shape[0]
# 	if ratio < threshold: return False
# 	return True



all_patients = [
  4, 9, 23, 26, 30, 35, 39, 48, 50, 53, 56, 59, 63, 84, 90, 93, 99, 102, 108, 114, 118, 136, 138, 143, 144, 150, 151, 155,
  12, 16, 18, 25, 28, 29, 31, 36, 41, 45, 51, 58, 60, 66, 68, 71, 97, 101, 104, 105, 107, 110, 111, 116, 127, 131, 154,
  1, 10, 11, 15, 22, 32, 33, 34, 38, 42, 44, 52, 64, 87, 91, 92, 96, 98, 106, 119, 121, 123, 130, 141, 153, 156, 157,
  2, 3, 5, 7, 8, 14, 17, 19, 24, 27, 40, 46, 55, 57, 69, 81, 83, 86, 95, 117, 120, 122, 134, 137, 146, 148, 152]


def BMI_group(BMI):
	if BMI < 18.5:
		return 'Underweight'
	elif 18.5 <= BMI < 24.9:
		return 'Normal weight'
	elif 24.9 <= BMI < 29.9:
		return 'Overweight'
	else:
		return 'Bbesity'
	
def Age_group(Age):
	if Age <= 39:
		return '<40'
	elif 40 <= Age < 60:
		return '40-59'
	else:
		return '60+'

# Useful patients 
if __name__ == "__main__":
	data_folder = 'Data/data_60s_30s_yingjian2/'
	df = pd.read_excel('Code/SleepLab.xlsx', engine='openpyxl', sheet_name='Logs')
	# useful_ids = []

	# for file in tqdm(sorted(os.listdir(data_folder))):
	# 	# ============ Load Data ============
	# 	data = np.load(os.path.join(data_folder, file))
	# 	ID_npy = data[0, -2]		

	# 	# ============ Data Check ============
	# 	if ID_npy not in all_patients: continue
	# 	if ID_npy in [24, 50, 25, 134, 153, 119, 114]: continue
	# 	if not ratio_check(file): continue
	# 	sleep_time_excel = df.loc[df['ID'] == ID_npy, 'Duration(h)'].values[0]  * df.loc[df['ID'] == ID_npy, 'SEfficiency'].values[0] * 0.01
	# 	if sleep_time_excel <= 2:
	# 		# print(f'Skipping patient {ID_npy} due to short sleep time: {sleep_time_excel:.2f} h\n')
	# 		continue

	# 	AHI_label = df.loc[df['ID'] == ID_npy, 'AHI'].values[0]
	# 	if ID_npy == 7: AHI_label = df.loc[df['ID'] == ID_npy, 'RDI'].values[0]
	# 	if ID_npy == 106: AHI_label = 30 / (7.32 * 83.5 / 100)
	# 	if ID_npy == 134: AHI_label = 13.73
	# 	if ID_npy == 108: AHI_label = 9.6

	# 	useful_ids.append(ID_npy)
	# useful_ids = [136.0, 39.0, 106.0, 155.0, 32.0, 156.0, 99.0, 31.0, 157.0, 22.0, 91.0, 83.0, 123.0, 148.0, 120.0, 154.0, 118.0, 17.0, 5.0, 98.0, 27.0, 30.0, 10.0, 101.0, 151.0, 9.0, 7.0, 86.0, 84.0, 53.0, 144.0, 110.0, 141.0, 146.0, 2.0, 107.0, 19.0, 55.0, 1.0, 138.0, 40.0, 14.0, 12.0, 34.0, 137.0, 29.0, 41.0, 15.0, 3.0, 23.0, 18.0, 92.0, 143.0, 102.0, 117.0, 57.0, 48.0, 44.0, 152.0, 4.0, 150.0, 51.0, 93.0, 127.0, 52.0, 8.0, 116.0, 42.0, 104.0, 108.0, 97.0, 33.0]
	useful_ids = [136.0, 39.0, 106.0, 155.0, 156.0, 31.0, 157.0, 22.0, 91.0, 83.0, 123.0, 148.0, 120.0, 154.0, 118.0, 17.0, 5.0, 98.0, 27.0, 30.0, 10.0, 101.0, 151.0, 9.0, 7.0, 86.0, 84.0, 53.0, 144.0, 110.0, 141.0, 146.0, 2.0, 107.0, 19.0, 55.0, 1.0, 138.0, 40.0, 14.0, 12.0, 34.0, 137.0, 29.0, 41.0, 15.0, 3.0, 23.0, 18.0, 92.0, 143.0, 102.0, 117.0, 57.0, 48.0, 44.0, 152.0, 4.0, 150.0, 51.0, 93.0, 127.0, 52.0, 8.0, 116.0, 42.0, 104.0, 108.0, 97.0, 33.0]
	print('Useful IDs:', useful_ids)
	print('Total Useful IDs:', len(useful_ids))

	Ages = []
	Sexs = []
	BMIs = []
	for ID in useful_ids:
		df_patient = df.loc[df['ID'] == ID]
		# print(df_patient)
		Ages.append(df_patient['Age'].values[0])
		Sexs.append(df_patient['Sex'].values[0])
		BMIs.append(df_patient['BMI'].values[0])
	Ages = np.array(Ages)
	print('Average Age:', np.mean(Ages), np.std(Ages[Ages>18]))
	print('Average Age:', np.mean(Ages[Ages>18]), np.std(Ages[Ages>18]))
	Age_groups = [Age_group(age) for age in Ages]
	counter = Counter(Age_groups)
	print('Age Group Counts:', counter)

	print(f'Male Count: {Sexs.count("M")}, Female Count: {Sexs.count("F")}')
	print('Average BMI:', np.mean(BMIs), np.std(BMIs))
	BMI_groups = [BMI_group(bmi) for bmi in BMIs]
	counter = Counter(BMI_groups)
	print('BMI Group Counts:', counter)


