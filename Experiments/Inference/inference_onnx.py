import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
from Code.utils_dsp import denoise, normalize_1d
from Code.utils import choose_gpu_by_model_process_count, data_preprocess_MTL
from scipy.signal import resample_poly 
import torch
from tqdm import trange
import onnx
import onnxruntime as ort

def load_data(data_path):
	data = []
	for i in range(1, 5):
		print(f'Loading data from fold {i} ...')
		data_file_path = data_path + f'fold{i}.npy'
		data.append(np.load(data_file_path))
	data = np.concatenate(data)
	print(f'All data shape: {data.shape}')
	return data



if __name__ == "__main__":
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

	version="CPU"
	version="GPU"

	data = load_data('Data/fold_data_p109_MTL_45s/')
	X, Y, _, _, _, _ = data_preprocess_MTL(data, 'train')
	signals = np.stack([X, Y], axis=-1) 
	signals = signals.transpose(0, 2, 1)
	signals = signals.astype(np.float32)
	print(f'Signals shape before preprocessing: {signals.shape}')
	onnx_model_folder = 'Experiments/Inference/onnx_models/'
	# for model in os.listdir(onnx_model_folder):
	# 	print(f'Loading ONNX model: {model} ...')
	onnx_model_path = onnx_model_folder + 'model_1_b32.onnx'

	if version == "CPU":
		ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
	else:
		ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
	
	print("providers:", ort_session.get_providers())
	print("provider options:", ort_session.get_provider_options())
	for i in trange(0,  signals.shape[0], 32):
		x = signals[i:i+32, :, :]
		outs = ort_session.run(None, {"input": x})