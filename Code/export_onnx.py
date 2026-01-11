# Remote Changes
# !/usr/bin/env python3
import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Code.utils import npy2dataset
from Code.models.clf import ApneaClassifier_PatchTST, ApneaClassifier_PatchTST_MTL_REC
from Code.utils_dl import inference
import argparse
import torch.onnx
import onnx
import onnx
import onnxruntime as ort
from tqdm import trange


if __name__ == '__main__':
	torch.cuda.empty_cache()

	parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--nseg', type=int, default=60)
	parser.add_argument('--seq_len', type=int, default=590)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--in_channels', type=int, default=1)
	parser.add_argument('--dropout', type=float, default=0.2) 
	parser.add_argument('--XYZ', type=str, default='XY')
	parser.add_argument('--raw', type=str, default=False)
	parser.add_argument('--tta_method', type=str, default='avgnew')
	args = parser.parse_args()

	Model = 'PatchTST' 
	# model_folder_name = f'{args.XYZ}_60s_F1_ws1_wa2_REC0.001'
	device = torch.device('cpu')



	for fold_idx in range(1, 5):
		fold_idx = int(fold_idx)
		model_save_fold_name = f'Models/{args.XYZ}_60s_F1_ws1_wa1_REC0.001_export/fold{fold_idx}/'

		patch_len = 24
		n_layers = 4
		d_model = 64
		n_heads = 4
		d_ff = 256           
		mask_ratio = 0.5
		model = ApneaClassifier_PatchTST_MTL_REC(
		# model = ApneaClassifier_Dual_PatchTST(
			input_size=2, num_classes=2,
			seq_len=args.seq_len, patch_len=patch_len,
			stride=patch_len // 2,
			n_layers=n_layers, d_model=d_model,
			n_heads=n_heads, d_ff=d_ff,
			axis=3 if args.XYZ == 'XYZ' else 2,
			dropout=args.dropout,
			mask_ratio=mask_ratio).to(device)

		model_folder = f'{model_save_fold_name}/PatchTST_patchlen{patch_len}_nlayer{n_layers}_dmodel{d_model}_nhead{n_heads}_dff{d_ff}/'



		sorted_files = sorted(os.listdir(model_folder), key=lambda x: int(x.split('_')[0][5:]), reverse=False)
		# print(f'sorted_files: {sorted(os.listdir(model_folder))}')

		model_path = model_folder + sorted_files[-1]
		# print(f'model_path: {model_path}')
		
		checkpoint = torch.load(model_path, weights_only=False)
		# print(checkpoint.keys())
		model.load_state_dict(checkpoint['model_state_dict'])
		# val_balanced_acc = checkpoint['val_balanced_acc']
		# val_f1 = checkpoint['val_f1']
		# print(f'val_balanced_acc: {val_balanced_acc}')
		# print(f'val_f1: {val_f1}')
		# 打印模型结构，检查第一个MatMul层期望的输入
		print(model)

		# 或者检查W_P层的权重形状
		for name, param in model.named_parameters():
			if 'W_P' in name:
				print(f"{name}: {param.shape}")

		onnx_model_folder = 'Code/onnx_models/'

		onnx_model_path = onnx_model_folder + f'model_{fold_idx}.onnx'

		with torch.no_grad():
			model.eval()
			torch.onnx.export(model, 
								torch.randn(1, 2, 590).to(device), 
								onnx_model_path, 
								verbose=False, input_names=['input'], 
								output_names=['output'],
								opset_version=11,
								do_constant_folding=True,
							)
		onnx_model = onnx.load(onnx_model_path)
		
		try:
			onnx.checker.check_model(onnx_model)
		except Exception as e:
			print('The model is invalid: %s' % e)
		else:
			print('The model is valid!')
			print(f'ONNX model saved at: {onnx_model_path}')

		sess = ort.InferenceSession("Code/onnx_models/model_1.onnx", providers=["CPUExecutionProvider"])

		x = np.random.randn(1, 2, 590).astype(np.float32)
		outs = sess.run(None, {"input": x})
		print([o.shape for o in outs])