import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils import seed_everything, npy2dataset_true_TriClass, choose_gpu_by_model_process_count
from Code.utils_dl import train_classifier_TriClass, inference_TriClass
from Code.models.clf import ApneaClassifier_PatchTST_TriClass
from Code.inference_mtl import threshold_adjustment
import argparse
from torch.optim.lr_scheduler import StepLR 
import yaml
import pandas as pd

def load_checkpoint_TriClass(model_folder):
    sorted_files = sorted(os.listdir(model_folder), key=lambda x: int(x.split('_')[0][5:]), reverse=False)
    print(f'sorted_files: {sorted(os.listdir(model_folder))}')

    model_path = model_folder + sorted_files[-1]
    print(f'model_path: {model_path}')
    
    checkpoint = torch.load(model_path, weights_only=False)
    print(checkpoint.keys())

    val_bacc = checkpoint['val_bacc']
    val_f1 = checkpoint['val_f1']
    print(f'val_bacc: {val_bacc}, val_f1: {val_f1}')

    return checkpoint


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--seq_len', type=int, default=590)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--epochs', type=int, default=75)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--dropout', type=float, default=0.2) 
	parser.add_argument('--XYZ', type=str, default='XY')
	parser.add_argument('--save_bacc', type=float, default=0.6)
	parser.add_argument('--tta_method', type=str, default='avgnew')
	parser.add_argument('--threhold', type=str, default='0.5')
	args = parser.parse_args()
	seed_everything(args.seed)

	Model = 'PatchTST'
	Type = 'TriClass'
	Experiment = 'MTL'

	torch.cuda.empty_cache()
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	model_save_folder = f'{args.XYZ}_60s_{args.threhold}'
	fold_id_threshold = {}
	fold_id_mapping = {}


	for fold_idx in range(1, 5):
		model_save_fold_name = f'Experiments/{Experiment}/Models/{model_save_folder}/fold{fold_idx}/'
		if Model == 'PatchTST':
			patch_len = 24
			n_layers = 4
			d_model = 64
			n_heads = 4
			d_ff = 256           
			mask_ratio = 0.5
			model = ApneaClassifier_PatchTST_TriClass(
				input_size=len(args.XYZ), num_classes=3,
				seq_len=args.seq_len, patch_len=patch_len,
				stride=patch_len // 2,
				n_layers=n_layers, d_model=d_model,
				n_heads=n_heads, d_ff=d_ff,
				axis=len(args.XYZ),
				dropout=args.dropout,
				mask_ratio=mask_ratio).to(device)

			model_save_dir = f'{model_save_fold_name}/PatchTST_patchlen{patch_len}_nlayer{n_layers}_dmodel{d_model}_nhead{n_heads}_dff{d_ff}/'

		if not os.path.exists(model_save_dir): os.makedirs(model_save_dir, exist_ok=True)		
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
		data_path = f'Data/fold_data_p109_{Type}_60s/'
		train_loader, val_loader, test_loader = npy2dataset_true_TriClass(data_path, fold_idx, args)


		train_classifier_TriClass(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			optimizer=optimizer,
			device=device,
			epochs=args.epochs,
			save_dir=model_save_dir,
			save_bacc=args.save_bacc,
			scheduler=scheduler,
			tta_method=args.tta_method,
			threhold=args.threhold
		)

		# yaml_path ç=  f'Experiments/{Experiment}/configs/{model_save_folder}.yaml'


		# "Add Inference Code Here"
		# checkpoint = load_checkpoint_TriClass(model_save_dir)
		# model.load_state_dict(checkpoint['model_state_dict'])
		# preds, labels, probs, others = inference_TriClass(model, test_loader, device, tta_method=args.tta_method)
		# best_threshold = threshold_adjustment(np.array(probs)[:, 1], labels, 'Labels')

		# fold_id_threshold[str(fold_idx)] = float(best_threshold)
		# fold_id_mapping[str(fold_idx)] = np.unique(others[:, -2]).tolist()
		# print(f'fold_id_mapping[{str(fold_idx)}]: {fold_id_mapping[str(fold_idx)]}')

		# with open(yaml_path, 'w') as yaml_file:
		# 	yaml.dump({'fold_id_mapping': fold_id_mapping, 'fold_to_threshold': fold_id_threshold}, yaml_file)
