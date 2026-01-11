import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils_dsp import modify_magnitude_with_gaussian_noise_batch

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)  # 保险归一化
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 数值稳定
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # 屏蔽对角线（自身对比）
        logits_mask = (~torch.eye(batch_size, dtype=torch.bool, device=device)).float()
        mask = mask * logits_mask

        # log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 最终 loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class ApneaDataset(Dataset):
    def __init__(self, data, labels, others=None):
        """
        data: ndarray, shape [N, C, L] or [N, L, C]
        labels: ndarray/list of 0/1
        others: optional extra info
        """
        # 确保 data shape 为 [N, C, L]
        data = torch.FloatTensor(data).transpose(1, 2)

        self.data = data
        self.labels = torch.LongTensor(labels)
        # self.others = torch.LongTensor(others) if others is not None else None
        self.others = torch.FloatTensor(others) if others is not None else None

        # 保留 numpy 版本标签，供 Sampler 使用
        self.labels_np = self.labels.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.others is not None:
            return self.data[idx], self.labels[idx], self.others[idx]
        else:
            return self.data[idx], self.labels[idx]

    def get_class_indices(self):
        """返回正类和负类的索引，用于 BalancedBatchSampler"""
        pos_idx = np.where(self.labels_np == 1)[0]
        neg_idx = np.where(self.labels_np == 0)[0]
        return pos_idx, neg_idx
    

class ApneaDataset_TriClass(Dataset):
    def __init__(self, data, labels, others=None):
        """
        data: ndarray, shape [N, C, L] or [N, L, C]
        others: optional extra info
        """
        # 确保 data shape 为 [N, C, L]
        data = torch.FloatTensor(data).transpose(1, 2)

        self.data = data
        self.labels = torch.LongTensor(labels)
        self.others = torch.FloatTensor(others) if others is not None else None

        # 保留 numpy 版本标签，供 Sampler 使用
        self.labels_np = self.labels.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.others is not None:
            return self.data[idx], self.labels[idx], self.others[idx]
        else:
            return self.data[idx], self.labels[idx]



class ApneaDataset_MTL(Dataset):
    def __init__(self, data, labels_stage, labels_apnea, others=None):
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
        # self.others = torch.LongTensor(others) if others is not None else None
        self.others = torch.FloatTensor(others) if others is not None else None

        # 保留 numpy 版本标签，供 Sampler 使用
        self.labels_stage_np = self.labels_stage.numpy()
        self.labels_apnea_np = self.labels_apnea.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.others is not None:
            return self.data[idx], self.labels_stage[idx], self.labels_apnea[idx], self.others[idx]
        else:
            return self.data[idx], self.labels_stage[idx], self.labels_apnea[idx]




class ApneaDataset_MTL_REC(Dataset):
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


# class ApneaDataset(Dataset):
#     def __init__(self, data, labels, others=None):
#         self.data = torch.FloatTensor(data).transpose(1, 2)
#         self.labels = torch.LongTensor(labels)
#         self.others = torch.LongTensor(others) if others is not None else None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if self.others is not None:
#             return self.data[idx], self.labels[idx], self.others[idx]
#         else:
#             return self.data[idx], self.labels[idx]

class BalancedBatchSampler(Sampler):
    """
    自动补齐少数类样本的平衡采样器：
    每个 batch 保证正负样本数量一致；
    若少数类不足，则随机重复采样。
    """
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert batch_size % 2 == 0, "batch_size 必须是偶数！"
        self.half = batch_size // 2

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

        # 判断谁是少数类
        self.minority_idx = self.pos_idx if len(self.pos_idx) < len(self.neg_idx) else self.neg_idx
        self.majority_idx = self.neg_idx if len(self.pos_idx) < len(self.neg_idx) else self.pos_idx

        # 计算需要重复采样的数量
        self.n_minority = len(self.minority_idx)
        self.n_majority = len(self.majority_idx)
        self.num_batches = self.n_majority // self.half  # 以多数类为基准

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.pos_idx)
            np.random.shuffle(self.neg_idx)

        # 若少数类不足，则随机补齐
        repeat_factor = int(np.ceil(self.n_majority / self.n_minority))
        minority_expanded = np.tile(self.minority_idx, repeat_factor)[:self.n_majority]
        
        if len(self.pos_idx) < len(self.neg_idx):
            pos_full, neg_full = minority_expanded, self.majority_idx
        else:
            pos_full, neg_full = self.majority_idx, minority_expanded

        for i in range(self.num_batches):
            pos_batch = pos_full[i*self.half:(i+1)*self.half]
            neg_batch = neg_full[i*self.half:(i+1)*self.half]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.Tensor([alpha, 1-alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha.to(input.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    

def compute_class_weights(train_loader, weight_strategy):
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    

    if weight_strategy == 'equal':
        class_weights = torch.ones(num_classes)
    
    elif weight_strategy == 'effective_number':
        # Effective number of samples weighting
        beta = 0.999
        class_counts = np.bincount(all_labels)
        print(f'Class counts: {class_counts}')
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.where(effective_num != 0, effective_num, 1)
        class_weights = torch.FloatTensor(weights) * 100

    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}")
    
    print(f"Using class weights: {class_weights.numpy()}")
    return class_weights


def compute_class_weights_new(all_labels, weight_strategy):
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    

    if weight_strategy == 'equal':
        class_weights = torch.ones(num_classes)
    
    elif weight_strategy == 'effective_number':
        beta = 0.999
        class_counts = np.bincount(all_labels)
        print(f'Class counts: {class_counts}')
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.where(effective_num != 0, effective_num, 1)
        class_weights = torch.FloatTensor(weights) * 100

    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}")
    
    print(f"Using class weights: {class_weights.numpy()}")
    return class_weights



def find_best_threshold_F1(y_true, y_prob, num_thresholds=500, average='macro'):
    """
    Find the threshold that maximizes F1 score.

    Args:
        y_true (array-like): True labels
        y_prob (array-like): Predicted probabilities
        num_thresholds (int): Number of thresholds (default=500)
        average (str): 'macro', 'weighted', 'micro', etc.

    Returns:
        dict with threshold, best f1, bacc, confusion matrix, y_pred
    """

    thresholds = np.linspace(0, 1, num_thresholds)
    f1_vals = [
        f1_score(y_true, (y_prob >= t).astype(int), average=average)
        for t in thresholds
    ]
    # f1_vals = np.array([
    #     f1_score(y_true, (y_prob >= t).astype(int), average=average)
    #     for t in thresholds
    # ])

    # mx = f1_vals.max()
    # idx = np.where(np.isclose(f1_vals, mx, atol=1e-12))[0]

    # print("Plateau size:", len(idx))
    # print("Threshold range:", thresholds[idx[0]], "to", thresholds[idx[-1]])
    
    
    idx_best = np.argmax(f1_vals)
    # idx_best = idx_best - int(0.016 * num_thresholds)

    best_th = thresholds[idx_best]
    best_f1 = f1_vals[idx_best]

    y_pred_best = (y_prob >= best_th).astype(int)
    best_bacc = balanced_accuracy_score(y_true, y_pred_best)
    conf = confusion_matrix(y_true, y_pred_best)

    return {
        "best_threshold": best_th,
        "best_f1": best_f1,
        "best_bacc": best_bacc,
        "confusion_matrix": conf,
        "y_pred": y_pred_best
    }






def find_best_threshold_F1_and_Bacc(y_true, y_prob, num_thresholds=500, average='macro'):
    """
    Find the threshold that maximizes (F1 + Bacc) score.

    Args:
        y_true (array-like): True labels
        y_prob (array-like): Predicted probabilities
        num_thresholds (int): Number of thresholds (default=500)
        average (str): 'macro', 'weighted', 'micro', etc.

    Returns:
        dict with threshold, best f1, bacc, confusion matrix, y_pred
    """

    thresholds = np.linspace(0, 1, num_thresholds)
    f1_vals = [
        f1_score(y_true, (y_prob >= t).astype(int), average=average)
        for t in thresholds
    ]
    bacc_vals = [
        balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        for t in thresholds
    ]

    idx_best = np.argmax((np.array(f1_vals) * 2 + np.array(bacc_vals)))
    best_th = thresholds[idx_best]
    best_f1 = f1_vals[idx_best]

    y_pred_best = (y_prob >= best_th).astype(int)
    best_bacc = balanced_accuracy_score(y_true, y_pred_best)
    conf = confusion_matrix(y_true, y_pred_best)

    return {
        "best_threshold": best_th,
        "best_f1": best_f1,
        "best_bacc": best_bacc,
        "confusion_matrix": conf,
        "y_pred": y_pred_best
    }




# def find_best_threshold_weightedF1(y_true, y_prob, num_thresholds=500):
#     """
#     Find the threshold that maximizes weighted F1 score.

#     Args:
#         y_true (array-like): True binary labels (0 or 1)
#         y_prob (array-like): Predicted probabilities for positive class (float [0,1])
#         num_thresholds (int): Number of thresholds to sweep (default: 500)

#     Returns:
#         dict with:
#             - best_threshold (float)
#             - best_f1 (float)
#             - best_bacc (float)
#             - confusion_matrix (np.ndarray)
#     """
#     # Sweep thresholds
#     thresholds = np.linspace(0, 1, num_thresholds)
#     f1_vals = [f1_score(y_true, (y_prob >= t).astype(int), average='weighted') for t in thresholds]

#     # Find the best threshold
#     idx_best = np.argmax(f1_vals)
#     best_th = thresholds[idx_best]
#     best_f1 = f1_vals[idx_best]

#     # Evaluate metrics at best threshold
#     y_pred_best = (y_prob >= best_th).astype(int)
#     best_bacc = balanced_accuracy_score(y_true, y_pred_best)
#     conf = confusion_matrix(y_true, y_pred_best)

#     return {
#         "best_threshold": best_th,
#         "best_f1": best_f1,
#         "best_bacc": best_bacc,
#         "confusion_matrix": conf,
#         "y_pred": y_pred_best  # 新增：最优阈值下的预测结果
#     }




def train_classifier(model, train_loader, val_loader, 
                     optimizer,  device, epochs, save_dir, 
                     save_bacc, gama, supcon, weight_strategy='effective_number', 
                     scheduler=None, tta_method=None, threhold='0.5'):
    """
    Train the classifier with validation and model saving using Focal Loss
    
    Args:
        weight_strategy (str): Strategy for class weights:
            - 'balanced': inverse frequency weighting
            - 'equal': equal weights for all classes
            - 'effective_number': weights based on effective number of samples
    """
    os.makedirs(save_dir, exist_ok=True)

    class_weights = compute_class_weights(train_loader, weight_strategy)
    classification_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    # classification_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # contrastive_loss_fn = InfoNCELosss(device=device)

    if supcon:
        SupConLoss_fn = SupConLoss(temperature=0.07, base_temperature=0.07)

    best_val_bacc = -1
    for epoch in range(1, epochs + 1):
        model.train()

        train_preds, train_labels = [], []
        train_loss, train_loss_cls, train_loss_con = 0, 0, 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            # print(f'O in labels, 1 in labels: {(labels==0).sum().item()}, {(labels==1).sum().item()}')
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, proj_F, _ = model(data)

            loss_cls = classification_loss_fn(outputs, labels)

            if supcon:
                loss_supcon = SupConLoss_fn(proj_F, labels) * gama
                loss = loss_cls + loss_supcon
                train_loss_con += loss_supcon.item()
                train_loss_cls += loss_cls.item()
            else:
                loss = loss_cls

            # print(f'Losses - Classification: {loss_cls.item():.6f}, Contrastive: {loss_supcon.item():.6f}')
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        if scheduler is not None:
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            # print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")


        avg_train_loss = train_loss / len(train_loader)
        if supcon:
            avg_train_loss_cls = train_loss_cls / len(train_loader)
            avg_train_loss_con = train_loss_con / len(train_loader)
            print(f'Epoch {epoch} Training Losses - Total: {avg_train_loss:.6f}, Classification: {avg_train_loss_cls:.6f}, Contrastive: {avg_train_loss_con:.6f}')
            
        train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        

        model.eval()
        val_preds, val_labels = [], []
        val_probs = []
        val_loss, val_loss_cls, val_loss_supcon = 0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                if tta_method is not None:
                    # B, C, T
                    outputs1, proj_F, _ = model(data)
                    outputs2, proj_F, _ = model(torch.flip(data, dims=[2]))
                    outputs3, proj_F, _ = model(-data)
                    


                    loss = classification_loss_fn(outputs1, labels)
                    if tta_method == 'avg':
                        outputs = (outputs1 + outputs2 + outputs3) / 3
                        _, predicted = torch.max(outputs.data, 1)   
                    if tta_method == 'avgnew':
                        outputs4, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                        outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
                        _, predicted = torch.max(outputs.data, 1)

                    if tta_method == 'voting':
                        _, predicted1 = torch.max(outputs1.data, 1)
                        _, predicted2 = torch.max(outputs2.data, 1)
                        _, predicted3 = torch.max(outputs3.data, 1)

                        predicted = (predicted1 + predicted2 + predicted3) // 2
                else:
                    outputs, proj_F, _ = model(data)
                    # loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)

                    loss_cls = classification_loss_fn(outputs, labels)
                    if supcon:
                        loss_supcon = SupConLoss_fn(proj_F, labels) * gama
                        loss = loss_cls + loss_supcon
                        val_loss_supcon += loss_supcon.item()
                        val_loss_cls += loss_cls.item()
                    else:
                        loss = loss_cls
                    
                    val_loss += loss.item()
                    
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(F.softmax(outputs.data, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        if supcon:
            avg_val_loss_cls = val_loss_cls / len(val_loader)
            avg_val_loss_supcon = val_loss_supcon / len(val_loader)
            print(f'Epoch {epoch} Validation Losses - Total: {avg_val_loss:.6f}, Classification: {avg_val_loss_cls:.6f}, Contrastive: {avg_val_loss_supcon:.6f}')

        if threhold == '0.5':
            val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_conf_mat = confusion_matrix(val_labels, val_preds)
        elif threhold == 'F1':
            val_probs_pos = np.array(val_probs)[:, 1]
            threshold_results = find_best_threshold_F1(
                y_true=np.array(val_labels),
                y_prob=val_probs_pos,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold = threshold_results['best_threshold']
            val_balanced_acc = threshold_results['best_bacc']
            val_f1 = threshold_results['best_f1']
            val_conf_mat = threshold_results['confusion_matrix']
        elif threhold == 'macroF1':
            val_probs_pos = np.array(val_probs)[:, 1]
            threshold_results = find_best_threshold_F1(
                y_true=np.array(val_labels),
                y_prob=val_probs_pos,
                num_thresholds=500,
                average='macro'
            )
            val_best_threshold = threshold_results['best_threshold']
            val_balanced_acc = threshold_results['best_bacc']
            val_f1 = threshold_results['best_f1']
            val_conf_mat = threshold_results['confusion_matrix']

            

        print(f'\nEpoch {epoch} Results:')
        print(f'Training - Loss: {avg_train_loss*10000:.4f}, Balanced Acc: {train_balanced_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Validation - Loss: {avg_val_loss*10000:.4f}, Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}')
        print('\nValidation Confusion Matrix:')
        print(val_conf_mat)


        if train_balanced_acc < save_bacc: continue

        if val_balanced_acc > best_val_bacc:
            if threhold == 'F1':
                print(f'✅ Best model found at epoch {epoch} with Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}, Threshold: {val_best_threshold}')
            else:
                print(f'✅ Best model found at epoch {epoch} with Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}')
            best_val_bacc = val_balanced_acc    
            best_val_f1 = val_f1

            model_name = f'epoch{epoch}_{int(val_balanced_acc*100)}_{int(val_f1*100)}.pth'
            latest_model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_balanced_acc': val_balanced_acc,
                'val_f1': val_f1
            }, latest_model_path)

            
def train_classifier_TriClass(model, train_loader, val_loader, 
                     optimizer,  device, epochs, save_dir, 
                     save_bacc, 
                     weight_strategy='effective_number', 
                     scheduler=None, tta_method=None, threhold='0.5'):
    """
    Train the classifier with validation and model saving using Focal Loss
    
    Args:
        weight_strategy (str): Strategy for class weights:
            - 'balanced': inverse frequency weighting
            - 'equal': equal weights for all classes
            - 'effective_number': weights based on effective number of samples
    """
    os.makedirs(save_dir, exist_ok=True)
    labels = train_loader.dataset.labels_np
    class_weights = compute_class_weights_new(labels, weight_strategy)
    classification_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)


    best_val_bacc = -1
    for epoch in range(1, epochs + 1):
        model.train()

        train_preds, train_labels = [], []
        
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, proj_F, _ = model(data)

            loss_cls = classification_loss_fn(outputs, labels)
            
            # print(f'Losses - Classification Stage: {loss_cls_stage.item():.6f}, Classification Apnea: {loss_cls_apnea.item():.6f}')
            train_loss += loss_cls.item()
            
            loss_cls.backward()
            optimizer.step()
            
            _, predicted_labels = torch.max(outputs.data, 1)
            train_preds.extend(predicted_labels.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)

        train_bacc = balanced_accuracy_score(train_labels, train_preds)
        # train_f1 = f1_score(train_labels, train_preds, average='weighted')

        if scheduler is not None:
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            # print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")


        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0    
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(device), labels.to(device)
                if tta_method is not None:
                    # B, C, T
                    outputs1, proj_F, _ = model(data)
                    outputs2, _, _ = model(torch.flip(data, dims=[2]))
                    outputs3, _, _ = model(-data)
                    
                    if tta_method == 'avg':
                        outputs = (outputs1 + outputs2 + outputs3) / 3
                        _, predicted_labels = torch.max(outputs.data, 1) 
                    if tta_method == 'avgnew':
                        outputs4, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                        outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
                        _, predicted_labels = torch.max(outputs.data, 1)
                else:
                    outputs, proj_F, _ = model(data)
                    _, predicted_labels = torch.max(outputs.data, 1)

                loss_cls = classification_loss_fn(outputs, labels)
 

                val_loss += loss_cls.item()


                val_preds.extend(predicted_labels.cpu().numpy())
                val_probs.extend(F.softmax(outputs.data, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())


        avg_val_loss = val_loss / len(val_loader)

        if threhold == 'F1':
            val_probs_pos = np.array(val_probs)[:, 1]
            threshold_results = find_best_threshold_F1(
                y_true=np.array(val_labels),
                y_prob=val_probs_pos,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold = threshold_results['best_threshold']
            val_bacc = threshold_results['best_bacc']
            val_f1 = threshold_results['best_f1']
            val_conf_mat = threshold_results['confusion_matrix']
        elif threhold == '0.5':
            val_best_threshold = 0.5
            val_bacc = balanced_accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_conf_mat = confusion_matrix(val_labels, val_preds)
            

        print(f'\nEpoch {epoch} Results:')
        print(f'Training - Loss: {avg_train_loss*1e4:.4f}')
        print(f'Training - Bacc : {train_bacc:.4f}')
        print(f'Validation - Loss: {avg_val_loss*1e4:.4f}')
        print(f'Validation - Bacc: {val_bacc:.4f}')

        print('\nValidation CM:')
        print(val_conf_mat)

        if train_bacc < save_bacc: continue

        if val_bacc > best_val_bacc:
            print(f'✅ Best model found at epoch {epoch} with Threshold: {val_best_threshold:.4f}')
            
            best_val_bacc = val_bacc
            best_val_f1 = val_f1

            model_name = f'epoch{epoch}_{int(val_bacc*100)}_{int(val_f1*100)}.pth'
            latest_model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bacc': val_bacc,
                'val_f1': val_f1,
            }, latest_model_path)





def train_classifier_MTL(model, train_loader, val_loader, 
                     optimizer,  device, epochs, save_dir, 
                     save_bacc, 
                     gama_stage, supcon_stage, 
                     gama_apnea, supcon_apnea,
                     weight_stage, weight_apnea, weight_as, weight_hinge,
                     weight_strategy='effective_number', 
                     scheduler=None, tta_method=None, threhold='0.5', ASloss=False, HingeLoss=False):
    """
    Train the classifier with validation and model saving using Focal Loss
    
    Args:
        weight_strategy (str): Strategy for class weights:
            - 'balanced': inverse frequency weighting
            - 'equal': equal weights for all classes
            - 'effective_number': weights based on effective number of samples
    """
    os.makedirs(save_dir, exist_ok=True)
    labels_stage, labels_apnea = train_loader.dataset.labels_stage_np, train_loader.dataset.labels_apnea_np
    class_weights_stage = compute_class_weights_new(labels_stage, weight_strategy)
    classification_loss_fn_stage = FocalLoss(alpha=class_weights_stage, gamma=2.0)
    mask = (labels_stage == 0)  & (labels_apnea != -1)
    class_weights_apnea = compute_class_weights_new(labels_apnea[mask], weight_strategy)
    classification_loss_fn_apnea = FocalLoss(alpha=class_weights_apnea, gamma=2.0)

    supconloss_stage = SupConLoss()
    supconloss_apnea = SupConLoss()

    best_val_bacc = -1
    for epoch in range(1, epochs + 1):
        model.train()

        train_preds_stage, train_labels_stage = [], []
        train_preds_apnea, train_labels_apnea = [], []
        
        train_loss, train_loss_cls_stage, train_loss_cls_apnea = 0, 0, 0
        train_loss_AS = 0
        train_loss_Hinge = 0
        train_loss_SupCon_stage = 0
        train_loss_SupCon_apnea = 0

        for batch_idx, (data, labels_stage, labels_apnea) in enumerate(train_loader):
            # print(f'O in labels, 1 in labels: {(labels==0).sum().item()}, {(labels==1).sum().item()}')
            # data, labels = data.to(device), labels.to(device)
            data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
            
            optimizer.zero_grad()
            outputs_stage, outputs_apnea, proj_F, _, _ = model(data)

            loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
            
            stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)      # Sleep 的样本位置
            if stage_apnea_mask.any():                      # 这一 batch 至少有一个 sleep
                loss_apnea_batch = classification_loss_fn_apnea(
                    outputs_apnea[stage_apnea_mask],        # 只取 sleep 的样本
                    labels_apnea[stage_apnea_mask]
                )                                     # FocalLoss 默认 reduction='mean' 就行
                loss_cls_apnea = weight_apnea * loss_apnea_batch
            else:
                # 没有 sleep 样本，就让 apnea loss=0，但图还在
                loss_cls_apnea = 0.0 * loss_cls_stage

            loss = loss_cls_stage + loss_cls_apnea


            if ASloss:
                if stage_apnea_mask.any():
                    p_wake = F.softmax(outputs_stage, dim=1)[stage_apnea_mask, 1]
                    p_apnea = F.softmax(outputs_apnea, dim=1)[stage_apnea_mask, 1]
                    loss_as = (p_wake * p_apnea).mean() * weight_as
                else:
                    loss_as = 0.0 * loss_cls_stage
                train_loss_AS += loss_as.item()
                loss = loss + loss_as
                

            if supcon_stage:
                loss_supcon_stage = supconloss_stage(proj_F, labels_stage)
                train_loss_SupCon_stage += loss_supcon_stage.item()
                loss = loss + gama_stage * loss_supcon_stage

            if supcon_apnea:
                if stage_apnea_mask.any():
                    loss_supcon_apnea = supconloss_apnea(proj_F[stage_apnea_mask], labels_apnea[stage_apnea_mask])
                else:
                    loss_supcon_apnea = 0.0 * loss_cls_stage
                train_loss_SupCon_apnea += loss_supcon_apnea.item()
                loss = loss + gama_apnea * loss_supcon_apnea

            if HingeLoss:
                p_wake = F.softmax(outputs_stage, dim=1)[:, 1].detach()
                p_apnea = F.softmax(outputs_apnea, dim=1)[:, 1]
                base_mask = (labels_stage == 0) & (labels_apnea != -1)     # 只在 GT sleep 上约束（推荐）
                m0 = base_mask.float()
                tau_w, tau_a = 0.6, 0.6
                gate = ((p_wake.detach() > tau_w) & (p_apnea.detach() > tau_a)).float()  # detach 很关键

                margin = 0
                viol = F.relu(p_wake + p_apnea - (1.0 - margin))   # hinge

                den = (m0 * gate).sum() + 1e-6
                loss_hinge = (m0 * gate * viol).sum() / den
                loss_hinge = loss_hinge * weight_hinge

                train_loss_Hinge += loss_hinge.item()
                loss = loss + loss_hinge

                

            # print(f'Losses - Classification Stage: {loss_cls_stage.item():.6f}, Classification Apnea: {loss_cls_apnea.item():.6f}')
            train_loss += loss.item()
            train_loss_cls_stage += loss_cls_stage.item()
            train_loss_cls_apnea += loss_cls_apnea.item()
            
            loss.backward()
            optimizer.step()
            
            _, predicted_stage = torch.max(outputs_stage.data, 1)
            train_preds_stage.extend(predicted_stage.cpu().numpy())
            train_labels_stage.extend(labels_stage.cpu().numpy())

            _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            train_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
            train_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())



        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_cls_stage = train_loss_cls_stage / len(train_loader)
        avg_train_loss_cls_apnea = train_loss_cls_apnea / len(train_loader)
        if ASloss: avg_train_loss_AS = train_loss_AS / len(train_loader)
        if HingeLoss: avg_train_loss_Hinge = train_loss_Hinge / len(train_loader)
        if supcon_stage: avg_train_loss_SupCon_stage = train_loss_SupCon_stage / len(train_loader)
        if supcon_apnea: avg_train_loss_SupCon_apnea = train_loss_SupCon_apnea / len(train_loader)

        train_bacc_stage = balanced_accuracy_score(train_labels_stage, train_preds_stage)
        # train_f1_stage = f1_score(train_labels_stage, train_preds_stage, average='weighted')
        train_bacc_apnea = balanced_accuracy_score(train_labels_apnea, train_preds_apnea)
        # train_f1_apnea = f1_score(train_labels_apnea, train_preds_apnea, average='weighted')

        if scheduler is not None:
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            # print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")


        model.eval()
        val_preds_stage, val_labels_stage, val_probs_stage = [], [], []
        val_preds_apnea, val_labels_apnea, val_probs_apnea = [], [], []
        val_loss, val_loss_cls_stage, val_loss_cls_apnea = 0, 0, 0
        val_loss_AS = 0
        val_loss_Hinge = 0
        val_loss_SupCon_stage = 0
        val_loss_SupCon_apnea = 0
        with torch.no_grad():
            for data, labels_stage, labels_apnea, _ in val_loader:
                data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
                if tta_method is not None:
                    # B, C, T
                    outputs1_stage, outputs1_apnea, proj_F, _, _ = model(data)
                    outputs2_stage, outputs2_apnea, _, _, _ = model(torch.flip(data, dims=[2]))
                    outputs3_stage, outputs3_apnea, _, _, _ = model(-data)
                    
                    if tta_method == 'avg':
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
                        _, predicted_stage = torch.max(outputs_stage.data, 1) 
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)  
                    if tta_method == 'avgnew':
                        outputs4_stage, outputs4_apnea, _, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage + outputs4_stage) / 4
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea + outputs4_apnea) / 4
                        _, predicted_stage = torch.max(outputs_stage.data, 1)
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)
                else:
                    outputs_stage, outputs_apnea, proj_F, _, _ = model(data)
                    _, predicted_stage = torch.max(outputs_stage.data, 1)
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)


                loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
 
                # ===== Apnea: sample-level mask =====
                stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)
                if stage_apnea_mask.any():
                    loss_apnea_batch = classification_loss_fn_apnea(
                        outputs_apnea[stage_apnea_mask],
                        labels_apnea[stage_apnea_mask]
                    )
                    loss_cls_apnea = weight_apnea * loss_apnea_batch
                else:
                    loss_cls_apnea = 0.0 * loss_cls_stage

                loss = loss_cls_stage + loss_cls_apnea  

                if ASloss:
                    if stage_apnea_mask.any():
                        p_wake = F.softmax(outputs_stage, dim=1)[stage_apnea_mask, 1]
                        p_apnea = F.softmax(outputs_apnea, dim=1)[stage_apnea_mask, 1]
                        loss_as = (p_wake * p_apnea).mean() * weight_as
                    else:
                        loss_as = 0.0 * loss_cls_stage
                    val_loss_AS += loss_as.item()
                    loss = loss + loss_as
                
                if HingeLoss:
                    p_wake = F.softmax(outputs_stage, dim=1)[:, 1].detach()
                    p_apnea = F.softmax(outputs_apnea, dim=1)[:, 1]
                    base_mask = (labels_stage == 0) & (labels_apnea != -1)     # 只在 GT sleep 上约束（推荐）
                    m0 = base_mask.float()
                    tau_w, tau_a = 0.6, 0.6
                    gate = ((p_wake.detach() > tau_w) & (p_apnea.detach() > tau_a)).float()  # detach 很关键

                    margin = 0
                    viol = F.relu(p_wake + p_apnea - (1.0 - margin))   # hinge

                    den = (m0 * gate).sum() + 1e-6
                    loss_hinge = (m0 * gate * viol).sum() / den
                    loss_hinge = loss_hinge * weight_hinge

                    val_loss_Hinge += loss_hinge.item()
                    loss = loss + loss_hinge
                
                if supcon_stage:
                    loss_supcon_stage = supconloss_stage(proj_F, labels_stage)
                    val_loss_SupCon_stage += loss_supcon_stage.item()
                    loss = loss + gama_stage * loss_supcon_stage
                
                if supcon_apnea:
                    if stage_apnea_mask.any():
                        loss_supcon_apnea = supconloss_apnea(proj_F[stage_apnea_mask], labels_apnea[stage_apnea_mask])
                    else:
                        loss_supcon_apnea = 0.0 * loss_cls_stage
                    val_loss_SupCon_apnea += loss_supcon_apnea.item()
                    loss = loss + gama_apnea * loss_supcon_apnea


                val_loss += loss.item()
                val_loss_cls_stage += loss_cls_stage.item()
                val_loss_cls_apnea += loss_cls_apnea.item()

                

                val_preds_stage.extend(predicted_stage.cpu().numpy())
                val_probs_stage.extend(F.softmax(outputs_stage.data, dim=1).cpu().numpy())
                val_labels_stage.extend(labels_stage.cpu().numpy())
                val_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
                val_probs_apnea.extend(F.softmax(outputs_apnea.data, dim=1)[stage_apnea_mask].cpu().numpy())
                val_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())
        

        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_cls_stage = val_loss_cls_stage / len(val_loader)
        avg_val_loss_cls_apnea = val_loss_cls_apnea / len(val_loader)
        if ASloss: avg_val_loss_AS = val_loss_AS / len(val_loader) 
        if HingeLoss: avg_val_loss_Hinge = val_loss_Hinge / len(val_loader)
        if supcon_stage: avg_val_loss_SupCon_stage = val_loss_SupCon_stage / len(val_loader)
        if supcon_apnea: avg_val_loss_SupCon_apnea = val_loss_SupCon_apnea / len(val_loader)


        if threhold == 'F1':
            val_probs_pos_stage = np.array(val_probs_stage)[:, 1]
            threshold_results_stage = find_best_threshold_F1(
                y_true=np.array(val_labels_stage),
                y_prob=val_probs_pos_stage,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold_stage = threshold_results_stage['best_threshold']
            val_bacc_stage = threshold_results_stage['best_bacc']
            val_f1_stage = threshold_results_stage['best_f1']
            val_conf_mat_stage = threshold_results_stage['confusion_matrix']

            val_probs_pos_apnea = np.array(val_probs_apnea)[:, 1]
            threshold_results_apnea = find_best_threshold_F1(
                y_true=np.array(val_labels_apnea),
                y_prob=val_probs_pos_apnea,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold_apnea = threshold_results_apnea['best_threshold']
            val_bacc_apnea = threshold_results_apnea['best_bacc']   
            val_f1_apnea = threshold_results_apnea['best_f1']
            val_conf_mat_apnea = threshold_results_apnea['confusion_matrix']
        elif threhold == '0.5':
            val_best_threshold_stage, val_best_threshold_apnea = 0.5, 0.5
            
            val_bacc_stage = balanced_accuracy_score(val_labels_stage, val_preds_stage)
            val_f1_stage = f1_score(val_labels_stage, val_preds_stage, average='weighted')
            val_conf_mat_stage = confusion_matrix(val_labels_stage, val_preds_stage)
            
            val_bacc_apnea = balanced_accuracy_score(val_labels_apnea, val_preds_apnea)
            val_f1_apnea = f1_score(val_labels_apnea, val_preds_apnea, average='weighted')
            val_conf_mat_apnea = confusion_matrix(val_labels_apnea, val_preds_apnea)


        print(f'\nEpoch {epoch} Results:')
        print(f'Training - Loss: {avg_train_loss*1e4:.4f}, Stage Loss: {avg_train_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_train_loss_cls_apnea*1e4:.6f}')
        if ASloss:
            print(f'           AS Loss: {avg_train_loss_AS*1e4:.6f}')
        if HingeLoss:
            print(f'           Hinge Loss: {avg_train_loss_Hinge*1e4:.6f}')
        if supcon_stage:
            print(f'           SupCon Stage Loss: {avg_train_loss_SupCon_stage*gama_stage*1e4:.6f}')
        if supcon_apnea:
            print(f'           SupCon Apnea Loss: {avg_train_loss_SupCon_apnea*gama_apnea*1e4:.6f}')
        print(f'Training - Stage Bacc : {train_bacc_stage:.4f}, Apnea Bacc: {train_bacc_apnea:.4f}')
        print(f'Validation - Loss: {avg_val_loss*1e4:.4f}, Stage Loss: {avg_val_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_val_loss_cls_apnea*1e4:.6f}')
        if ASloss:
            print(f'           AS Loss: {avg_val_loss_AS*1e4:.6f}')
        if HingeLoss:
            print(f'           Hinge Loss: {avg_val_loss_Hinge*1e4:.6f}')
        if supcon_stage:
            print(f'           SupCon Stage Loss: {avg_val_loss_SupCon_stage*gama_stage*1e4:.6f}')
        if supcon_apnea:
            print(f'           SupCon Apnea Loss: {avg_val_loss_SupCon_apnea*gama_apnea*1e4:.6f}')
        print(f'Validation - Stage Bacc: {val_bacc_stage:.4f}, Stage F1: {val_f1_stage:.4f}')
        print(f'Validation - Apnea Bacc: {val_bacc_apnea:.4f}, F1: {val_f1_apnea:.4f}')

        print('\nValidation Stage CM:')
        print(val_conf_mat_stage)

        print('\nValidation Apnea CM:')
        print(val_conf_mat_apnea)


        if train_bacc_apnea < save_bacc and train_bacc_stage < save_bacc: continue

        if (val_bacc_stage + val_bacc_apnea) / 2 > best_val_bacc:
            print(f'✅ Best model found at epoch {epoch} with Stage Threshold: {val_best_threshold_stage:.4f}, Apnea Threshold: {val_best_threshold_apnea:.4f}')
            
            best_val_bacc = (val_bacc_stage +  val_bacc_apnea) / 2
            best_val_f1 = (val_f1_stage + val_f1_apnea) / 2

            model_name = f'epoch{epoch}_{int(val_bacc_stage*100)}_{int(val_f1_stage*100)}_{int(val_bacc_apnea*100)}_{int(val_f1_apnea*100)}.pth'
            latest_model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bacc_stage': val_bacc_stage,
                'val_f1_stage': val_f1_stage,
                'val_bacc_apnea': val_bacc_apnea,
                'val_f1_apnea': val_f1_apnea
            }, latest_model_path)




def train_classifier_MTL_standard(model, train_loader, val_loader, 
                     optimizer,  device, epochs, save_dir, 
                     save_bacc, 
                     gama_stage, supcon_stage, 
                     gama_apnea, supcon_apnea,
                     weight_stage, weight_apnea, weight_as, weight_hinge,
                     weight_strategy='effective_number', 
                     scheduler=None, tta_method=None, threhold='0.5', ASloss=False, HingeLoss=False):
    """
    Train the classifier with validation and model saving using Focal Loss
    
    Args:
        weight_strategy (str): Strategy for class weights:
            - 'balanced': inverse frequency weighting
            - 'equal': equal weights for all classes
            - 'effective_number': weights based on effective number of samples
    """
    os.makedirs(save_dir, exist_ok=True)
    labels_stage, labels_apnea = train_loader.dataset.labels_stage_np, train_loader.dataset.labels_apnea_np
    class_weights_stage = compute_class_weights_new(labels_stage, weight_strategy)
    classification_loss_fn_stage = FocalLoss(alpha=class_weights_stage, gamma=2.0)
    mask = (labels_stage == 0)  & (labels_apnea != -1)
    class_weights_apnea = compute_class_weights_new(labels_apnea[mask], weight_strategy)
    classification_loss_fn_apnea = FocalLoss(alpha=class_weights_apnea, gamma=2.0)

    supconloss_stage = SupConLoss()
    supconloss_apnea = SupConLoss()

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()

        train_preds_stage, train_labels_stage = [], []
        train_preds_apnea, train_labels_apnea = [], []
        
        train_loss, train_loss_cls_stage, train_loss_cls_apnea = 0, 0, 0
        train_loss_AS = 0
        train_loss_Hinge = 0
        train_loss_SupCon_stage = 0
        train_loss_SupCon_apnea = 0

        for batch_idx, (data, labels_stage, labels_apnea) in enumerate(train_loader):
            # print(f'O in labels, 1 in labels: {(labels==0).sum().item()}, {(labels==1).sum().item()}')
            # data, labels = data.to(device), labels.to(device)
            data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
            
            optimizer.zero_grad()
            outputs_stage, outputs_apnea, proj_F, _, _ = model(data)

            loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
            
            stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)      # Sleep 的样本位置
            if stage_apnea_mask.any():                      # 这一 batch 至少有一个 sleep
                loss_apnea_batch = classification_loss_fn_apnea(
                    outputs_apnea[stage_apnea_mask],        # 只取 sleep 的样本
                    labels_apnea[stage_apnea_mask]
                )                                     # FocalLoss 默认 reduction='mean' 就行
                loss_cls_apnea = weight_apnea * loss_apnea_batch
            else:
                # 没有 sleep 样本，就让 apnea loss=0，但图还在
                loss_cls_apnea = 0.0 * loss_cls_stage

            loss = loss_cls_stage + loss_cls_apnea


            if ASloss:
                if stage_apnea_mask.any():
                    p_wake = F.softmax(outputs_stage, dim=1)[stage_apnea_mask, 1]
                    p_apnea = F.softmax(outputs_apnea, dim=1)[stage_apnea_mask, 1]
                    loss_as = (p_wake * p_apnea).mean() * weight_as
                else:
                    loss_as = 0.0 * loss_cls_stage
                train_loss_AS += loss_as.item()
                loss = loss + loss_as
                

            if supcon_stage:
                loss_supcon_stage = supconloss_stage(proj_F, labels_stage)
                train_loss_SupCon_stage += loss_supcon_stage.item()
                loss = loss + gama_stage * loss_supcon_stage

            if supcon_apnea:
                if stage_apnea_mask.any():
                    loss_supcon_apnea = supconloss_apnea(proj_F[stage_apnea_mask], labels_apnea[stage_apnea_mask])
                else:
                    loss_supcon_apnea = 0.0 * loss_cls_stage
                train_loss_SupCon_apnea += loss_supcon_apnea.item()
                loss = loss + gama_apnea * loss_supcon_apnea

            if HingeLoss:
                p_wake = F.softmax(outputs_stage, dim=1)[:, 1].detach()
                p_apnea = F.softmax(outputs_apnea, dim=1)[:, 1]
                base_mask = (labels_stage == 0) & (labels_apnea != -1)     # 只在 GT sleep 上约束（推荐）
                m0 = base_mask.float()
                tau_w, tau_a = 0.6, 0.6
                gate = ((p_wake.detach() > tau_w) & (p_apnea.detach() > tau_a)).float()  # detach 很关键

                margin = 0
                viol = F.relu(p_wake + p_apnea - (1.0 - margin))   # hinge

                den = (m0 * gate).sum() + 1e-6
                loss_hinge = (m0 * gate * viol).sum() / den
                loss_hinge = loss_hinge * weight_hinge

                train_loss_Hinge += loss_hinge.item()
                loss = loss + loss_hinge

                

            # print(f'Losses - Classification Stage: {loss_cls_stage.item():.6f}, Classification Apnea: {loss_cls_apnea.item():.6f}')
            train_loss += loss.item()
            train_loss_cls_stage += loss_cls_stage.item()
            train_loss_cls_apnea += loss_cls_apnea.item()
            
            loss.backward()
            optimizer.step()
            
            _, predicted_stage = torch.max(outputs_stage.data, 1)
            train_preds_stage.extend(predicted_stage.cpu().numpy())
            train_labels_stage.extend(labels_stage.cpu().numpy())

            _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            train_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
            train_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())



        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_cls_stage = train_loss_cls_stage / len(train_loader)
        avg_train_loss_cls_apnea = train_loss_cls_apnea / len(train_loader)
        if ASloss: avg_train_loss_AS = train_loss_AS / len(train_loader)
        if HingeLoss: avg_train_loss_Hinge = train_loss_Hinge / len(train_loader)
        if supcon_stage: avg_train_loss_SupCon_stage = train_loss_SupCon_stage / len(train_loader)
        if supcon_apnea: avg_train_loss_SupCon_apnea = train_loss_SupCon_apnea / len(train_loader)

        train_bacc_stage = balanced_accuracy_score(train_labels_stage, train_preds_stage)
        # train_f1_stage = f1_score(train_labels_stage, train_preds_stage, average='weighted')
        train_bacc_apnea = balanced_accuracy_score(train_labels_apnea, train_preds_apnea)
        # train_f1_apnea = f1_score(train_labels_apnea, train_preds_apnea, average='weighted')

        if scheduler is not None:
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            # print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")


        model.eval()
        val_preds_stage, val_labels_stage, val_probs_stage = [], [], []
        val_preds_apnea, val_labels_apnea, val_probs_apnea = [], [], []
        val_loss, val_loss_cls_stage, val_loss_cls_apnea = 0, 0, 0
        val_loss_AS = 0
        val_loss_Hinge = 0
        val_loss_SupCon_stage = 0
        val_loss_SupCon_apnea = 0
        with torch.no_grad():
            for data, labels_stage, labels_apnea, _ in val_loader:
                data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
                if tta_method is not None:
                    # B, C, T
                    outputs1_stage, outputs1_apnea, proj_F, _, _ = model(data)
                    outputs2_stage, outputs2_apnea, _, _, _ = model(torch.flip(data, dims=[2]))
                    outputs3_stage, outputs3_apnea, _, _, _ = model(-data)
                    
                    if tta_method == 'avg':
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
                        _, predicted_stage = torch.max(outputs_stage.data, 1) 
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)  
                    if tta_method == 'avgnew':
                        outputs4_stage, outputs4_apnea, _, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage + outputs4_stage) / 4
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea + outputs4_apnea) / 4
                        _, predicted_stage = torch.max(outputs_stage.data, 1)
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)
                else:
                    outputs_stage, outputs_apnea, proj_F, _, _ = model(data)
                    _, predicted_stage = torch.max(outputs_stage.data, 1)
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)


                loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
 
                # ===== Apnea: sample-level mask =====
                stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)
                if stage_apnea_mask.any():
                    loss_apnea_batch = classification_loss_fn_apnea(
                        outputs_apnea[stage_apnea_mask],
                        labels_apnea[stage_apnea_mask]
                    )
                    loss_cls_apnea = weight_apnea * loss_apnea_batch
                else:
                    loss_cls_apnea = 0.0 * loss_cls_stage

                loss = loss_cls_stage + loss_cls_apnea  

                if ASloss:
                    if stage_apnea_mask.any():
                        p_wake = F.softmax(outputs_stage, dim=1)[stage_apnea_mask, 1]
                        p_apnea = F.softmax(outputs_apnea, dim=1)[stage_apnea_mask, 1]
                        loss_as = (p_wake * p_apnea).mean() * weight_as
                    else:
                        loss_as = 0.0 * loss_cls_stage
                    val_loss_AS += loss_as.item()
                    loss = loss + loss_as
                
                if HingeLoss:
                    p_wake = F.softmax(outputs_stage, dim=1)[:, 1].detach()
                    p_apnea = F.softmax(outputs_apnea, dim=1)[:, 1]
                    base_mask = (labels_stage == 0) & (labels_apnea != -1)     # 只在 GT sleep 上约束（推荐）
                    m0 = base_mask.float()
                    tau_w, tau_a = 0.6, 0.6
                    gate = ((p_wake.detach() > tau_w) & (p_apnea.detach() > tau_a)).float()  # detach 很关键

                    margin = 0
                    viol = F.relu(p_wake + p_apnea - (1.0 - margin))   # hinge

                    den = (m0 * gate).sum() + 1e-6
                    loss_hinge = (m0 * gate * viol).sum() / den
                    loss_hinge = loss_hinge * weight_hinge

                    val_loss_Hinge += loss_hinge.item()
                    loss = loss + loss_hinge
                
                if supcon_stage:
                    loss_supcon_stage = supconloss_stage(proj_F, labels_stage)
                    val_loss_SupCon_stage += loss_supcon_stage.item()
                    loss = loss + gama_stage * loss_supcon_stage
                
                if supcon_apnea:
                    if stage_apnea_mask.any():
                        loss_supcon_apnea = supconloss_apnea(proj_F[stage_apnea_mask], labels_apnea[stage_apnea_mask])
                    else:
                        loss_supcon_apnea = 0.0 * loss_cls_stage
                    val_loss_SupCon_apnea += loss_supcon_apnea.item()
                    loss = loss + gama_apnea * loss_supcon_apnea


                val_loss += loss.item()
                val_loss_cls_stage += loss_cls_stage.item()
                val_loss_cls_apnea += loss_cls_apnea.item()

                

                val_preds_stage.extend(predicted_stage.cpu().numpy())
                val_probs_stage.extend(F.softmax(outputs_stage.data, dim=1).cpu().numpy())
                val_labels_stage.extend(labels_stage.cpu().numpy())
                val_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
                val_probs_apnea.extend(F.softmax(outputs_apnea.data, dim=1)[stage_apnea_mask].cpu().numpy())
                val_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())
        

        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_cls_stage = val_loss_cls_stage / len(val_loader)
        avg_val_loss_cls_apnea = val_loss_cls_apnea / len(val_loader)
        if ASloss: avg_val_loss_AS = val_loss_AS / len(val_loader) 
        if HingeLoss: avg_val_loss_Hinge = val_loss_Hinge / len(val_loader)
        if supcon_stage: avg_val_loss_SupCon_stage = val_loss_SupCon_stage / len(val_loader)
        if supcon_apnea: avg_val_loss_SupCon_apnea = val_loss_SupCon_apnea / len(val_loader)



        val_best_threshold_stage, val_best_threshold_apnea = 0.5, 0.5
        
        val_bacc_stage = balanced_accuracy_score(val_labels_stage, val_preds_stage)
        val_f1_stage = f1_score(val_labels_stage, val_preds_stage, average='weighted')
        val_conf_mat_stage = confusion_matrix(val_labels_stage, val_preds_stage)
        
        val_bacc_apnea = balanced_accuracy_score(val_labels_apnea, val_preds_apnea)
        val_f1_apnea = f1_score(val_labels_apnea, val_preds_apnea, average='weighted')
        val_conf_mat_apnea = confusion_matrix(val_labels_apnea, val_preds_apnea)


        print(f'\nEpoch {epoch} Results:')
        print(f'Training - Loss: {avg_train_loss*1e4:.4f}, Stage Loss: {avg_train_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_train_loss_cls_apnea*1e4:.6f}')
        if ASloss:
            print(f'           AS Loss: {avg_train_loss_AS*1e4:.6f}')
        if HingeLoss:
            print(f'           Hinge Loss: {avg_train_loss_Hinge*1e4:.6f}')
        if supcon_stage:
            print(f'           SupCon Stage Loss: {avg_train_loss_SupCon_stage*gama_stage*1e4:.6f}')
        if supcon_apnea:
            print(f'           SupCon Apnea Loss: {avg_train_loss_SupCon_apnea*gama_apnea*1e4:.6f}')
        print(f'Training - Stage Bacc : {train_bacc_stage:.4f}, Apnea Bacc: {train_bacc_apnea:.4f}')
        print(f'Validation - Loss: {avg_val_loss*1e4:.4f}, Stage Loss: {avg_val_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_val_loss_cls_apnea*1e4:.6f}')
        if ASloss:
            print(f'           AS Loss: {avg_val_loss_AS*1e4:.6f}')
        if HingeLoss:
            print(f'           Hinge Loss: {avg_val_loss_Hinge*1e4:.6f}')
        if supcon_stage:
            print(f'           SupCon Stage Loss: {avg_val_loss_SupCon_stage*gama_stage*1e4:.6f}')
        if supcon_apnea:
            print(f'           SupCon Apnea Loss: {avg_val_loss_SupCon_apnea*gama_apnea*1e4:.6f}')
        print(f'Validation - Stage Bacc: {val_bacc_stage:.4f}, Stage F1: {val_f1_stage:.4f}')
        print(f'Validation - Apnea Bacc: {val_bacc_apnea:.4f}, F1: {val_f1_apnea:.4f}')

        print('\nValidation Stage CM:')
        print(val_conf_mat_stage)

        print('\nValidation Apnea CM:')
        print(val_conf_mat_apnea)


        if train_bacc_apnea < save_bacc and train_bacc_stage < save_bacc: continue

        if avg_val_loss < best_val_loss:
            print(f'✅ Best model found at epoch {epoch} with Stage Threshold: {val_best_threshold_stage:.4f}, Apnea Threshold: {val_best_threshold_apnea:.4f}')
            
            best_val_loss = avg_val_loss

            model_name = f'epoch{epoch}_{int(val_bacc_stage*100)}_{int(val_f1_stage*100)}_{int(val_bacc_apnea*100)}_{int(val_f1_apnea*100)}.pth'
            latest_model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bacc_stage': val_bacc_stage,
                'val_f1_stage': val_f1_stage,
                'val_bacc_apnea': val_bacc_apnea,
                'val_f1_apnea': val_f1_apnea
            }, latest_model_path)




def train_classifier_MTL_REC(model, train_loader, val_loader, 
                     optimizer,  device, epochs, save_dir, 
                     save_bacc, 
                     weight_stage, weight_apnea, weight_rec,
                     weight_strategy='effective_number', 
                     scheduler=None, tta_method=None, threhold='0.5'):
    """
    Train the classifier with validation and model saving using Focal Loss
    
    Args:
        weight_strategy (str): Strategy for class weights:
            - 'balanced': inverse frequency weighting
            - 'equal': equal weights for all classes
            - 'effective_number': weights based on effective number of samples
    """
    os.makedirs(save_dir, exist_ok=True)
    labels_stage, labels_apnea = train_loader.dataset.labels_stage_np, train_loader.dataset.labels_apnea_np
    class_weights_stage = compute_class_weights_new(labels_stage, weight_strategy)
    classification_loss_fn_stage = FocalLoss(alpha=class_weights_stage, gamma=2.0)
    mask = (labels_stage == 0)  & (labels_apnea != -1)
    class_weights_apnea = compute_class_weights_new(labels_apnea[mask], weight_strategy)
    classification_loss_fn_apnea = FocalLoss(alpha=class_weights_apnea, gamma=2.0)


    best_val_bacc = -1
    for epoch in range(1, epochs + 1):
        model.train()

        train_preds_stage, train_labels_stage = [], []
        train_preds_apnea, train_labels_apnea = [], []
        
        train_loss, train_loss_cls_stage, train_loss_cls_apnea = 0, 0, 0
        train_loss_rec = 0
        for batch_idx, (data, labels_tho, labels_abd, labels_stage, labels_apnea) in enumerate(train_loader):
            # print(f'O in labels, 1 in labels: {(labels==0).sum().item()}, {(labels==1).sum().item()}')
            # data, labels = data.to(device), labels.to(device)
            random_idx = torch.randint(0, data.size(0), (1,)).item()
            idx = random_idx
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 1, figsize=(10, 6))
            axes[0].plot(labels_tho[idx])
            axes[0].set_title('Original Thoracic Signal')
            axes[1].plot(labels_abd[idx])
            axes[1].set_title('Original Abdominal Signal')
            axes[2].plot(data[idx,0,:].cpu().numpy())
            axes[2].set_title('X')
            axes[3].plot(data[idx,1,:].cpu().numpy())
            axes[3].set_title('Y')
            plt.tight_layout()
            plt.savefig(f'/home/jiayu/SleepApnea4Ubicomp/test/debug_signals_{epoch}_{batch_idx}.png')
            plt.close()
            data, labels_tho, labels_abd, labels_stage, labels_apnea = data.to(device), labels_tho.to(device), labels_abd.to(device), labels_stage.to(device), labels_apnea.to(device)
            
            optimizer.zero_grad()
            outputs_stage, outputs_apnea, proj_F, _, _, rec_tho, rec_abd = model(data)

            loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
            
            stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)      # Sleep 的样本位置
            if stage_apnea_mask.any():                      # 这一 batch 至少有一个 sleep
                loss_apnea_batch = classification_loss_fn_apnea(
                    outputs_apnea[stage_apnea_mask],        # 只取 sleep 的样本
                    labels_apnea[stage_apnea_mask]
                )                                     # FocalLoss 默认 reduction='mean' 就行
                loss_cls_apnea = weight_apnea * loss_apnea_batch
            else:
                # 没有 sleep 样本，就让 apnea loss=0，但图还在
                loss_cls_apnea = 0.0 * loss_cls_stage

            # ===== Reconstruction Loss =====
            loss_rec_tho = F.mse_loss(rec_tho, labels_tho, reduction='mean')
            loss_rec_abd = F.mse_loss(rec_abd, labels_abd, reduction='mean')
            loss_rec = weight_rec * (loss_rec_tho + loss_rec_abd)
            
            loss = loss_cls_stage + loss_cls_apnea + loss_rec

            # print(f'Losses - Classification Stage: {loss_cls_stage.item():.6f}, Classification Apnea: {loss_cls_apnea.item():.6f}')
            train_loss += loss.item()
            train_loss_cls_stage += loss_cls_stage.item()
            train_loss_cls_apnea += loss_cls_apnea.item()
            train_loss_rec += loss_rec.item()
            
            loss.backward()
            optimizer.step()
            
            _, predicted_stage = torch.max(outputs_stage.data, 1)
            train_preds_stage.extend(predicted_stage.cpu().numpy())
            train_labels_stage.extend(labels_stage.cpu().numpy())

            _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            train_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
            train_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())


        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_cls_stage = train_loss_cls_stage / len(train_loader)
        avg_train_loss_cls_apnea = train_loss_cls_apnea / len(train_loader)
        avg_train_loss_rec = train_loss_rec / len(train_loader)
  
        train_bacc_stage = balanced_accuracy_score(train_labels_stage, train_preds_stage)
        # train_f1_stage = f1_score(train_labels_stage, train_preds_stage, average='weighted')
        train_bacc_apnea = balanced_accuracy_score(train_labels_apnea, train_preds_apnea)
        # train_f1_apnea = f1_score(train_labels_apnea, train_preds_apnea, average='weighted')

        if scheduler is not None:
            scheduler.step()
            last_lrs = scheduler.get_last_lr()
            # print(f"Current Learning Rate (from scheduler's get_last_lr()): {last_lrs[0]}")


        model.eval()
        val_preds_stage, val_labels_stage, val_probs_stage = [], [], []
        val_preds_apnea, val_labels_apnea, val_probs_apnea = [], [], []
        val_loss, val_loss_cls_stage, val_loss_cls_apnea = 0, 0, 0
        val_loss_rec = 0
        with torch.no_grad():
            for data, labels_tho, labels_abd, labels_stage, labels_apnea, _ in val_loader:
                data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
                labels_tho, labels_abd = labels_tho.to(device), labels_abd.to(device)
                if tta_method is not None:
                    # B, C, T
                    outputs1_stage, outputs1_apnea, _, _, _, rec_tho, rec_abd = model(data)
                    outputs2_stage, outputs2_apnea, _, _, _, _, _ = model(torch.flip(data, dims=[2]))
                    outputs3_stage, outputs3_apnea, _, _, _, _, _ = model(-data)
                    
                    if tta_method == 'avg':
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
                        _, predicted_stage = torch.max(outputs_stage.data, 1) 
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)  
                    if tta_method == 'avgnew':
                        outputs4_stage, outputs4_apnea, _, _, _, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                        outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage + outputs4_stage) / 4
                        outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea + outputs4_apnea) / 4
                        _, predicted_stage = torch.max(outputs_stage.data, 1)
                        _, predicted_apnea = torch.max(outputs_apnea.data, 1)
                else:
                    outputs_stage, outputs_apnea, _, _, _, rec_tho, rec_abd = model(data)
                    _, predicted_stage = torch.max(outputs_stage.data, 1)
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)


                loss_cls_stage = weight_stage * classification_loss_fn_stage(outputs_stage, labels_stage)
 
                # ===== Apnea: sample-level mask =====
                stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)
                if stage_apnea_mask.any():
                    loss_apnea_batch = classification_loss_fn_apnea(
                        outputs_apnea[stage_apnea_mask],
                        labels_apnea[stage_apnea_mask]
                    )
                    loss_cls_apnea = weight_apnea * loss_apnea_batch
                else:
                    loss_cls_apnea = 0.0 * loss_cls_stage

                # ===== Reconstruction Loss =====
                loss_rec_tho = F.mse_loss(rec_tho, labels_tho, reduction='mean')
                loss_rec_abd = F.mse_loss(rec_abd, labels_abd, reduction='mean')
                loss_rec = weight_rec * (loss_rec_tho + loss_rec_abd)
                
                loss = loss_cls_stage + loss_cls_apnea + loss_rec  


                
                val_loss += loss.item()
                val_loss_cls_stage += loss_cls_stage.item()
                val_loss_cls_apnea += loss_cls_apnea.item()
                val_loss_rec += loss_rec.item()

                

                val_preds_stage.extend(predicted_stage.cpu().numpy())
                val_probs_stage.extend(F.softmax(outputs_stage.data, dim=1).cpu().numpy())
                val_labels_stage.extend(labels_stage.cpu().numpy())
                val_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
                val_probs_apnea.extend(F.softmax(outputs_apnea.data, dim=1)[stage_apnea_mask].cpu().numpy())
                val_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())
        

        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_cls_stage = val_loss_cls_stage / len(val_loader)
        avg_val_loss_cls_apnea = val_loss_cls_apnea / len(val_loader)
        avg_val_loss_rec = val_loss_rec / len(val_loader)


        if threhold == 'F1':
            val_probs_pos_stage = np.array(val_probs_stage)[:, 1]
            threshold_results_stage = find_best_threshold_F1(
                y_true=np.array(val_labels_stage),
                y_prob=val_probs_pos_stage,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold_stage = threshold_results_stage['best_threshold']
            val_bacc_stage = threshold_results_stage['best_bacc']
            val_f1_stage = threshold_results_stage['best_f1']
            val_conf_mat_stage = threshold_results_stage['confusion_matrix']

            val_probs_pos_apnea = np.array(val_probs_apnea)[:, 1]
            threshold_results_apnea = find_best_threshold_F1(
                y_true=np.array(val_labels_apnea),
                y_prob=val_probs_pos_apnea,
                num_thresholds=500,
                average='weighted'
            )
            val_best_threshold_apnea = threshold_results_apnea['best_threshold']
            val_bacc_apnea = threshold_results_apnea['best_bacc']   
            val_f1_apnea = threshold_results_apnea['best_f1']
            val_conf_mat_apnea = threshold_results_apnea['confusion_matrix']
        elif threhold == '0.5':
            val_best_threshold_stage, val_best_threshold_apnea = 0.5, 0.5
            
            val_bacc_stage = balanced_accuracy_score(val_labels_stage, val_preds_stage)
            val_f1_stage = f1_score(val_labels_stage, val_preds_stage, average='weighted')
            val_conf_mat_stage = confusion_matrix(val_labels_stage, val_preds_stage)
            
            val_bacc_apnea = balanced_accuracy_score(val_labels_apnea, val_preds_apnea)
            val_f1_apnea = f1_score(val_labels_apnea, val_preds_apnea, average='weighted')
            val_conf_mat_apnea = confusion_matrix(val_labels_apnea, val_preds_apnea)


        print(f'\nEpoch {epoch} Results:')
        print(f'Training - Loss: {avg_train_loss*1e4:.4f}, Stage Loss: {avg_train_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_train_loss_cls_apnea*1e4:.6f}')
        print(f'           Rec Loss: {avg_train_loss_rec*1e4:.6f}')
        print(f'Training - Stage Bacc : {train_bacc_stage:.4f}, Apnea Bacc: {train_bacc_apnea:.4f}')
        print(f'Validation - Loss: {avg_val_loss*1e4:.4f}, Stage Loss: {avg_val_loss_cls_stage*1e4:.6f}, Apnea Loss: {avg_val_loss_cls_apnea*1e4:.6f}')
        print(f'           Rec Loss: {avg_val_loss_rec*1e4:.6f}')
        print(f'Validation - Stage Bacc: {val_bacc_stage:.4f}, Stage F1: {val_f1_stage:.4f}')
        print(f'Validation - Apnea Bacc: {val_bacc_apnea:.4f}, F1: {val_f1_apnea:.4f}')

        print('\nValidation Stage CM:')
        print(val_conf_mat_stage)

        print('\nValidation Apnea CM:')
        print(val_conf_mat_apnea)


        if train_bacc_apnea < save_bacc and train_bacc_stage < save_bacc: continue

        if (val_bacc_stage + val_bacc_apnea) / 2 > best_val_bacc:
            print(f'✅ Best model found at epoch {epoch} with Stage Threshold: {val_best_threshold_stage:.4f}, Apnea Threshold: {val_best_threshold_apnea:.4f}')
            
            best_val_bacc = (val_bacc_stage +  val_bacc_apnea) / 2
            best_val_f1 = (val_f1_stage + val_f1_apnea) / 2

            model_name = f'epoch{epoch}_{int(val_bacc_stage*100)}_{int(val_f1_stage*100)}_{int(val_bacc_apnea*100)}_{int(val_f1_apnea*100)}.pth'
            latest_model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bacc_stage': val_bacc_stage,
                'val_f1_stage': val_f1_stage,
                'val_bacc_apnea': val_bacc_apnea,
                'val_f1_apnea': val_f1_apnea
            }, latest_model_path)

def inference_MTL_REC(model, data_loader, device, tta_method=None):
    model.eval()


    all_preds_stage, all_labels_stage = [], []
    all_preds_apnea, all_labels_apnea = [], []
    all_probs_stage, all_probs_apnea = [], []
    # all_rec_tho, all_rec_abd = [], []
    all_others = []
    all_masks = []


    with torch.no_grad():
        for data, labels_stage, labels_apnea, others in data_loader:
            data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
            if tta_method is not None:
                # B, C, T
                outputs1_stage, outputs1_apnea, proj_F, _, _, rec_tho, rec_abd = model(data)
                outputs2_stage, outputs2_apnea, proj_F, _, _, _, _ = model(torch.flip(data, dims=[2]))
                outputs3_stage, outputs3_apnea, proj_F, _, _, _, _ = model(-data)
                
                if tta_method == 'avg':
                    outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
                    outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
                    _, predicted_stage = torch.max(outputs_stage.data, 1) 
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)  
                if tta_method == 'avgnew':
                    outputs4_stage, outputs4_apnea, _, _, _, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                    outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage + outputs4_stage) / 4
                    outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea + outputs4_apnea) / 4
                    _, predicted_stage = torch.max(outputs_stage.data, 1)
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            else:
                outputs_stage, outputs_apnea, proj_F, _, _, rec_tho, rec_abd = model(data)
                _, predicted_stage = torch.max(outputs_stage.data, 1)
                _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            
            stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)
            
            probs_stage = F.softmax(outputs_stage.data, dim=1)
            probs_apnea = F.softmax(outputs_apnea.data, dim=1)[stage_apnea_mask]
            all_preds_stage.extend(predicted_stage.cpu().numpy())
            all_labels_stage.extend(labels_stage.cpu().numpy())
            all_probs_stage.extend(probs_stage.cpu().numpy())
            all_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
            all_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())
            all_probs_apnea.extend(probs_apnea.cpu().numpy())
            # all_rec_tho.extend(rec_tho.cpu().numpy())
            # all_rec_abd.extend(rec_abd.cpu().numpy())
            all_others.extend(others.numpy())
            all_masks.extend(stage_apnea_mask.cpu().numpy())
        all_labels_stage = np.array(all_labels_stage)
        all_preds_stage = np.array(all_preds_stage)
        all_probs_stage = np.array(all_probs_stage) 
        all_labels_apnea = np.array(all_labels_apnea)
        all_preds_apnea = np.array(all_preds_apnea)
        all_probs_apnea = np.array(all_probs_apnea)
        all_others = np.array(all_others)
        all_masks = np.array(all_masks)
        # all_rec_tho = np.array(all_rec_tho)
        # all_rec_abd = np.array(all_rec_abd)
        return all_preds_stage, all_labels_stage, all_probs_stage, all_preds_apnea, all_labels_apnea, all_probs_apnea, all_others, all_masks



def inference_MTL(model, data_loader, device, tta_method=None):
    model.eval()


    all_preds_stage, all_labels_stage = [], []
    all_preds_apnea, all_labels_apnea = [], []
    all_probs_stage, all_probs_apnea = [], []
    all_others = []
    all_masks = []


    with torch.no_grad():
        for data, labels_stage, labels_apnea, others in data_loader:
            data, labels_stage, labels_apnea = data.to(device), labels_stage.to(device), labels_apnea.to(device)
            if tta_method is not None:
                # B, C, T
                outputs1_stage, outputs1_apnea, proj_F, _, _ = model(data)
                outputs2_stage, outputs2_apnea, proj_F, _, _ = model(torch.flip(data, dims=[2]))
                outputs3_stage, outputs3_apnea, proj_F, _, _ = model(-data)
                
                if tta_method == 'avg':
                    outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage) / 3
                    outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea) / 3
                    _, predicted_stage = torch.max(outputs_stage.data, 1) 
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)  
                if tta_method == 'avgnew':
                    outputs4_stage, outputs4_apnea, _, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                    outputs_stage = (outputs1_stage + outputs2_stage + outputs3_stage + outputs4_stage) / 4
                    outputs_apnea = (outputs1_apnea + outputs2_apnea + outputs3_apnea + outputs4_apnea) / 4
                    _, predicted_stage = torch.max(outputs_stage.data, 1)
                    _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            else:
                outputs_stage, outputs_apnea, proj_F, _, _ = model(data)
                _, predicted_stage = torch.max(outputs_stage.data, 1)
                _, predicted_apnea = torch.max(outputs_apnea.data, 1)
            


            stage_apnea_mask = (labels_stage == 0)  & (labels_apnea != -1)

            
            probs_stage = F.softmax(outputs_stage.data, dim=1)
            probs_apnea = F.softmax(outputs_apnea.data, dim=1)[stage_apnea_mask]
            all_preds_stage.extend(predicted_stage.cpu().numpy())
            all_labels_stage.extend(labels_stage.cpu().numpy())
            all_probs_stage.extend(probs_stage.cpu().numpy())
            all_preds_apnea.extend(predicted_apnea[stage_apnea_mask].cpu().numpy())
            all_labels_apnea.extend(labels_apnea[stage_apnea_mask].cpu().numpy())
            all_probs_apnea.extend(probs_apnea.cpu().numpy())
            all_others.extend(others.numpy())
            all_masks.extend(stage_apnea_mask.cpu().numpy())
        all_labels_stage = np.array(all_labels_stage)
        all_preds_stage = np.array(all_preds_stage)
        all_probs_stage = np.array(all_probs_stage) 
        all_labels_apnea = np.array(all_labels_apnea)
        all_preds_apnea = np.array(all_preds_apnea)
        all_probs_apnea = np.array(all_probs_apnea)
        all_others = np.array(all_others)
        all_masks = np.array(all_masks)
        return all_preds_stage, all_labels_stage, all_probs_stage, all_preds_apnea, all_labels_apnea, all_probs_apnea, all_others, all_masks


def inference_TriClass(model, data_loader, device, tta_method=None):
    model.eval()


    all_labels= []
    all_preds = []
    all_probs = []
    all_others = []
    all_masks = []


    with torch.no_grad():
        for data, labels, others in data_loader:
            data, labels = data.to(device), labels.to(device)
            if tta_method is not None:
                # B, C, T
                outputs1, proj_F, _ = model(data)
                outputs2, proj_F, _ = model(torch.flip(data, dims=[2]))
                outputs3, proj_F, _ = model(-data)
                
                if tta_method == 'avg':
                    outputs = (outputs1 + outputs2 + outputs3) / 3
                    _, predicted = torch.max(outputs.data, 1)   
                if tta_method == 'avgnew':
                    outputs4, _, _ = model(modify_magnitude_with_gaussian_noise_batch(data))
                    outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
                    _, predicted = torch.max(outputs.data, 1)
            else:
                outputs, proj_F, _ = model(data)
                _, predicted = torch.max(outputs.data, 1)
            
            probs = F.softmax(outputs.data, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_others.extend(others.numpy())
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs) 
        all_others = np.array(all_others)
        return all_preds, all_labels, all_probs, all_others



def inference(model, data_loader, device, tta_method=None):

    model.eval()

    all_preds = []
    all_labels = []
    all_others = []
    all_data = []
    all_proj_F = []
    all_conf = []
    all_rep = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels, others in data_loader:
            data, labels = data.to(device), labels.to(device)

            if tta_method is not None:
                outputs1, proj_F, rep = model(data)
                outputs2, proj_F, rep = model(torch.flip(data, dims=[2]))
                outputs3, proj_F, rep = model(-data)
            
                if tta_method == 'avg':
                    outputs = (outputs1 + outputs2 + outputs3) / 3
                    _, predicted = torch.max(outputs.data, 1)   
                if tta_method == 'avgnew':
                    outputs4, prej_F, rep = model(modify_magnitude_with_gaussian_noise_batch(data))
                    outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
                    _, predicted = torch.max(outputs.data, 1) 
            else:
                outputs, proj_F, rep = model(data)
            # conf, predicted = torch.max(outputs.data, 1)
            
            probs = F.softmax(outputs.data, dim=1)
            conf, predicted = torch.max(probs, dim=1)  # confs = max prob
                        
            all_data.extend(data.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_others.extend(others.cpu().numpy())
            all_proj_F.extend(proj_F.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())
            all_rep.extend(rep.cpu().numpy())

        all_others = np.array(all_others)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_data = np.array(all_data)
        all_proj_F = np.array(all_proj_F)
        all_probs = np.array(all_probs)
        all_conf = np.array(all_conf)
        all_rep = np.array(all_rep)

        all_data = np.reshape(all_data, (all_data.shape[0], -1))
        all_data_others = np.concatenate([all_data, all_others], axis=1)
        print(f'all_data_others.shape: {all_data_others.shape}')
        return all_preds, all_labels, all_data_others, all_proj_F, all_probs, all_conf, all_rep




