import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ApneaDataset_REC(Dataset):
    def __init__(self, data, labels_THO, labels_ABD,labels_stage, labels_apnea, others=None):
        """
        data: ndarray, shape [N, C, L] or [N, L, C]
        labels_stage: ndarray/list of 0/1
        labels_apnea: ndarray/list of 0/1
        others: optional extra info
        """
        # 确保 data shape 为 [N, C, L]
        data = torch.FloatTensor(data).transpose(1, 2)

        self.data = data
        self.labels_stage = torch.LongTensor(labels_stage)
        self.labels_apnea = torch.LongTensor(labels_apnea)
        self.labels_tho = torch.FloatTensor(labels_THO)
        self.labels_abd = torch.FloatTensor(labels_ABD)
        self.others = torch.FloatTensor(others) if others is not None else None
        
        # 保留 numpy 版本标签，供 Sampler 使用
        self.labels_stage_np = self.labels_stage.numpy()
        self.labels_apnea_np = self.labels_apnea.numpy()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.others is not None:
            return self.data[idx], self.labels_tho[idx], self.labels_abd[idx], self.labels_stage[idx], self.labels_apnea[idx], self.others[idx]
        else:
            return self.data[idx], self.labels_tho[idx], self.labels_abd[idx], self.labels_stage[idx], self.labels_apnea[idx]




def train_REC(model, train_loader, val_loader, 
					 optimizer, device, epochs, save_dir, 
					 scheduler=None):
	os.makedirs(save_dir, exist_ok=True)

	best_val_loss_rec = float('inf')
	for epoch in range(1, epochs + 1):
		model.train()
		train_loss_rec = 0
		for batch_idx, (data, labels_tho, labels_abd, labels_stage, labels_apnea, others) in enumerate(train_loader):
			data, labels_tho, labels_abd = data.to(device), labels_tho.to(device), labels_abd.to(device)
			
			optimizer.zero_grad()
			rec_tho, rec_abd = model(data)

			# ===== Reconstruction Loss =====
			loss_rec_tho = F.mse_loss(rec_tho, labels_tho, reduction='mean')
			loss_rec_abd = F.mse_loss(rec_abd, labels_abd, reduction='mean')
			loss_rec = loss_rec_tho + loss_rec_abd

			train_loss_rec += loss_rec.item()
			
			loss_rec.backward()
			optimizer.step()

		avg_train_loss_rec = train_loss_rec / len(train_loader)
  
		if scheduler is not None:
			scheduler.step()
			last_lrs = scheduler.get_last_lr()
			# print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")

		model.eval()
		val_loss_rec = 0
		with torch.no_grad():
			for data, labels_tho, labels_abd, labels_stage, labels_apnea, others in val_loader:
				data, labels_tho, labels_abd = data.to(device), labels_tho.to(device), labels_abd.to(device)
				rec_tho, rec_abd = model(data)

				# ===== Reconstruction Loss =====
				loss_rec_tho = F.mse_loss(rec_tho, labels_tho, reduction='mean')
				loss_rec_abd = F.mse_loss(rec_abd, labels_abd, reduction='mean')
				loss_rec = loss_rec_tho + loss_rec_abd
				
				val_loss_rec += loss_rec.item()

		avg_val_loss_rec = val_loss_rec / len(val_loader)

		print(f'\nEpoch {epoch} Results:')
		print(f'Training - Rec Loss: {avg_train_loss_rec*1e4:.6f}')
		print(f'Validation - Rec Loss: {avg_val_loss_rec*1e4:.6f}')

		if avg_val_loss_rec < best_val_loss_rec:
			print(f'✅ Best model found at epoch {epoch}')
			
			model_name = f'epoch{epoch}_{avg_val_loss_rec:.6f}.pth'
			latest_model_path = os.path.join(save_dir, model_name)
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_loss': avg_val_loss_rec
			}, latest_model_path)
			best_val_loss_rec = avg_val_loss_rec




def inference_REC(model, test_loader, device, save_dir):
	os.makedirs(save_dir, exist_ok=True)

	model.eval()
	with torch.no_grad():
		for data, labels_tho, labels_abd, labels_stage, labels_apnea, others in test_loader:
			data, labels_tho, labels_abd = data.to(device), labels_tho.to(device), labels_abd.to(device)
			rec_tho, rec_abd = model(data)
			random_idx = np.random.choice(data.size(0), size=5, replace=False)
			for i in random_idx:
				save_path = os.path.join(save_dir, f'sample_{i}.png')

				fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)
				axes[0].plot(data[i, 0].cpu().numpy(), label='X')
				axes[1].plot(data[i, 1].cpu().numpy(), label='Y')
				axes[2].plot(labels_tho[i].cpu().numpy(), label='THO True')
				axes[3].plot(labels_abd[i].cpu().numpy(), label='ABD True')
				axes[4].plot(rec_tho[i].cpu().numpy(), label='THO Rec')
				axes[5].plot(rec_abd[i].cpu().numpy(), label='ABD Rec')
				for ax in axes:
					ax.legend()
				plt.tight_layout()
				plt.savefig(save_path)
				plt.close()
				print(f'Saved reconstruction plot to {save_path}')
