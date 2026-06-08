import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils import seed_everything, npy2dataset_true, npy2dataset_yingjian, choose_gpu_by_model_process_count
from Code.utils_dl import train_classifier_yingjian
from Code.models.clf import  ApneaClassifier_PatchTST
import argparse
from torch.optim.lr_scheduler import StepLR 


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--seq_len', type=int, default=590)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--epochs', type=int, default=75)
	# parser.add_argument('--lr', type=float, default=2e-4)
	# parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--dropout', type=float, default=0.2) 
	parser.add_argument('--XYZ', type=str, default='Y')
	parser.add_argument('--save_bacc', type=float, default=0.6)
	parser.add_argument('--tta_method', type=str, default=None)

	args = parser.parse_args()

	seed_everything(args.seed)

	Model = 'PatchTST'

	torch.cuda.empty_cache()
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	
	
	for fold_idx in range(1, 6):
		model_save_fold_name = f'Experiments/Yingjian/Models/{args.XYZ}_Yingjian_60s_times15/fold{fold_idx}/'

		if Model == 'PatchTST':
			# patch_len = 16
			patch_len = 24
			n_layers = 5
			d_model = 64
			n_heads = 4
			d_ff = 256           
			mask_ratio = 0
			model = ApneaClassifier_PatchTST(
				input_size=1, num_classes=3,
				seq_len=args.seq_len, patch_len=patch_len,
				stride=patch_len // 2,
				n_layers=n_layers, d_model=d_model,
				n_heads=n_heads, d_ff=d_ff,
				axis=len(args.XYZ),
				dropout=args.dropout,
				mask_ratio=mask_ratio).to(device)

			model_save_dir = f'{model_save_fold_name}/PatchTST_patchlen{patch_len}_nlayer{n_layers}_dmodel{d_model}_nhead{n_heads}_dff{d_ff}/'


		if not os.path.exists(model_save_dir): os.makedirs(model_save_dir, exist_ok=True)		
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 5)
		# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.lr*0.2)

		scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
		# scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
		
		data_path = 'Data/apnea_data/'

		train_loader, val_loader = npy2dataset_yingjian(data_path, fold_idx, args)

		train_classifier_yingjian(
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
		)