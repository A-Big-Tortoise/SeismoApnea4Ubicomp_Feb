import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils import seed_everything, choose_gpu_by_model_process_count
from utils_mapping import npy2dataset_REC
from utils_dl_mapping import train_REC
from Code.models.rec import  PatchTST_REC, VTCN_ED
import argparse
from torch.optim.lr_scheduler import StepLR 




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--seq_len', type=int, default=590)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--epochs', type=int, default=75)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--dropout', type=float, default=0.2) 
	parser.add_argument('--XYZ', type=str, default='XY')
	parser.add_argument('--save_bacc', type=float, default=0.75)
	parser.add_argument('--tta_method', type=str, default='avgnew')

	args = parser.parse_args()

	seed_everything(args.seed)

	Model = 'TCN'
	Type = 'MTL'

	torch.cuda.empty_cache()
	cuda = choose_gpu_by_model_process_count()
	device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
	
	
	for fold_idx in range(1, 5):
		model_save_fold_name = f'Mapping/Models/{args.XYZ}_60s_REC/fold{fold_idx}/'
		if Model == 'PatchTST':
			patch_len = 24
			n_layers = 4
			d_model = 64
			n_heads = 4
			d_ff = 256           
			mask_ratio = 0.5
			model = PatchTST_REC(
				input_size=2, num_classes=2,
				seq_len=args.seq_len, patch_len=patch_len,
				stride=patch_len // 2,
				n_layers=n_layers, d_model=d_model,
				n_heads=n_heads, d_ff=d_ff,
				axis=3 if args.XYZ == 'XYZ' else 2,
				dropout=args.dropout,
				mask_ratio=mask_ratio).to(device)

			model_save_dir = f'{model_save_fold_name}/PatchTST_patchlen{patch_len}_nlayer{n_layers}_dmodel{d_model}_nhead{n_heads}_dff{d_ff}/'
		else:
			out_channels = 1
			n_hid = 32
			n_block = 7
			kernel_size = 7
			model = VTCN_ED(input_size = len(args.XYZ), 
						output_size = out_channels,
						num_channels = [n_hid]*n_block, 
						kernel_size = kernel_size, 
						seq_len = args.seq_len,
						dropout = args.dropout)
			model_save_dir = f'{model_save_fold_name}/VTCN_nhid{n_hid}_nblock{n_block}_kernel{kernel_size}/'

		if not os.path.exists(model_save_dir): os.makedirs(model_save_dir, exist_ok=True)		
		
		model.to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 5)
		scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
		data_path = f'Data/fold_data_p109_{Type}_60s/'
		train_loader, val_loader, test_loader = npy2dataset_REC(data_path, fold_idx, args)

		train_REC(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			optimizer=optimizer,
			device=device,
			epochs=args.epochs,
			save_dir=model_save_dir,
			scheduler=scheduler,
		)
