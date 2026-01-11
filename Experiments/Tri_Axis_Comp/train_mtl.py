import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils import seed_everything, npy2dataset_true_MTL, choose_gpu_by_model_process_count
from Code.utils_dl import train_classifier_MTL
from Code.models.clf import ApneaClassifier_PatchTST_MTL
from Code.inference_mtl import load_checkpoint, inference_MTL, threshold_adjustment
import argparse
from torch.optim.lr_scheduler import StepLR 
import yaml
import pandas as pd


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--seq_len', type=int, default=590)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--epochs', type=int, default=75)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--dropout', type=float, default=0.2) 
	parser.add_argument('--XYZ', type=str, default='Y')
	parser.add_argument('--save_bacc', type=float, default=0.75)
	parser.add_argument('--tta_method', type=str, default='avgnew')
	parser.add_argument('--threhold', type=str, default='F1')
	parser.add_argument('--lambda_s', type=float, default=1)
	parser.add_argument('--lambda_a', type=float, default=1)
	parser.add_argument('--lambda_as', type=float, default=0)
	parser.add_argument('--lambda_hinge', type=float, default=0)
	parser.add_argument('--gama_stage', type=float, default=0)
	parser.add_argument('--gama_apnea', type=float, default=0)

	args = parser.parse_args()
	seed_everything(args.seed)

	Model = 'PatchTST'
	Type = 'MTL'
	Experiment = 'Tri_Axis_Comp'

	torch.cuda.empty_cache()
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	model_save_folder = f'{args.XYZ}_60s_{args.threhold}_ws{args.lambda_s}_wa{args.lambda_a}'
	fold_id_threshold_stage = {}
	fold_id_threshold_apnea = {}
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
			model = ApneaClassifier_PatchTST_MTL(
				input_size=len(args.XYZ), num_classes=2,
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
		data_path = f'Data/fold_data_p109_{Type}_45s/'
		train_loader, val_loader, test_loader = npy2dataset_true_MTL(data_path, fold_idx, args)


		train_classifier_MTL(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			weight_stage=args.lambda_s,
			weight_apnea=args.lambda_a,
			weight_as=args.lambda_as,
			ASloss=True if args.lambda_as !=0 else False,
			HingeLoss=True if args.lambda_hinge !=0 else False,
			weight_hinge=args.lambda_hinge,
			supcon_stage=True if args.gama_stage != 0 else False,
			supcon_apnea=True if args.gama_apnea != 0 else False,
			gama_stage=args.gama_stage,
			gama_apnea=args.gama_apnea, 
			optimizer=optimizer,
			device=device,
			epochs=args.epochs,
			save_dir=model_save_dir,
			save_bacc=args.save_bacc,
			scheduler=scheduler,
			tta_method=args.tta_method,
			threhold=args.threhold
		)

		yaml_path =  f'Experiments/{Experiment}/configs/{model_save_folder}.yaml'


		"Add Inference Code Here"
		checkpoint = load_checkpoint(model_save_dir)
		model.load_state_dict(checkpoint['model_state_dict'])

		preds_stage, labels_stage, probs_stage, preds_apnea, labels_apnea, probs_apnea, others, stage_apnea_mask = inference_MTL(model, test_loader, device, tta_method=args.tta_method)
		best_threshold_stage = threshold_adjustment(np.array(probs_stage)[:, 1], labels_stage, 'Stage')
		best_threshold_apnea = threshold_adjustment(np.array(probs_apnea)[:, 1], labels_apnea, 'Apnea')

		fold_id_threshold_stage[str(fold_idx)] = float(best_threshold_stage)
		fold_id_threshold_apnea[str(fold_idx)] = float(best_threshold_apnea)
		fold_id_mapping[str(fold_idx)] = np.unique(others[:, -2]).tolist()
		print(f'fold_id_mapping[{str(fold_idx)}]: {fold_id_mapping[str(fold_idx)]}')

		with open(yaml_path, 'w') as yaml_file:
			yaml.dump({'fold_id_mapping': fold_id_mapping, 'fold_to_threshold_stage': fold_id_threshold_stage, 'fold_to_threshold_apnea': fold_id_threshold_apnea}, yaml_file)
