import numpy as np
import os, sys


def ratio_check(file, threshold=0.75):
	try: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p144_rdi/' + file)
	except: data_check_quality = np.load('/home/jiayu/SleepApnea4Ubicomp/Data/data_p137_rdi_useless/' + file)
	ratio = data_check_quality.shape[0] / data.shape[0]
	if ratio < threshold: return False
	return True



if __name__ == "__main__":
	data_folder = '/home/jiayu/SleepApnea4Ubicomp/Data/data_60s_30s_yingjian2/'
	in_lst_ids = []

	for file in sorted(os.listdir(data_folder)):
		data = np.load(os.path.join(data_folder, file))
		ID_npy = data[0, -2]		
		if not ratio_check(file): continue
		in_lst_ids.append(int(ID_npy))
	print(f'in_lst_ids: {in_lst_ids}')
	print(f'len(in_lst_ids): {len(in_lst_ids)}')
