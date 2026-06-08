"""
main.py — EXP-021
-----------------
4-ch 3-model ensemble with grid-searched weights + TTA + stage-specific thresholds.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from tool import ApneaResNet, ApneaTCN, build_4ch, augment, tta_predict

ROOT    = os.path.dirname(os.path.abspath(__file__))
DS      = os.path.join(ROOT, 'dataset')
TRAIN   = os.path.join(DS, 'train')
VAL     = os.path.join(DS, 'val')

DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH       = 256
EPOCHS      = 80
LR          = 3e-4
SEED        = 42
FOCAL_GAMMA = 2.0
PATIENCE    = 10


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        return (bce * ((1 - p_t) ** self.gamma)).mean()


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    bacc = 0.5 * (tp / (tp + fn + 1e-12) + tn / (tn + fp + 1e-12))
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-12)
    return float(bacc), float(f1), int(tp), int(fp), int(tn), int(fn)


def print_metrics(split, bacc, f1, tp, fp, tn, fn):
    print(f'{split}_bacc: {bacc:.4f}')
    print(f'{split}_f1:   {f1:.4f}')
    print(f'{split}_cm:   TP={tp} FP={fp} TN={tn} FN={fn}')


def best_threshold(probs, labels):
    best_t, best_b = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        b, *_ = compute_metrics(labels.astype(int), preds)
        if b > best_b:
            best_b, best_t = b, t
    return best_t, best_b


def stage_specific_threshold(probs, labels, stages):
    """Find independent optimal thresholds for stage=0 and stage=1 subsets."""
    thresholds = {}
    for s in [0, 1]:
        mask = stages == s
        if mask.sum() == 0:
            thresholds[s] = 0.5
            continue
        t, _ = best_threshold(probs[mask], labels[mask])
        thresholds[s] = t
    return thresholds


def apply_stage_thresholds(probs, stages, thresholds):
    preds = np.zeros(len(probs), dtype=int)
    for s, t in thresholds.items():
        mask = stages == s
        preds[mask] = (probs[mask] >= t).astype(int)
    return preds


@torch.no_grad()
def predict_proba(model, loader, tta=1):
    model.eval()
    probs, labels = [], []
    for xb, sb, yb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        if tta > 1:
            p = tta_predict(model, xb, sb, n_tta=tta)
        else:
            p = torch.sigmoid(model(xb, sb).squeeze(1))
        probs.append(p.cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one(model, tr_loader, val_loader, pos_w, tag):
    criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_bacc = 0.0
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_total = 0.0, 0
        for xb, sb, yb in tr_loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            xb = augment(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb, sb).squeeze(1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_total    += len(xb)
        scheduler.step()

        val_probs, val_true = predict_proba(model, val_loader, tta=1)
        _, vb = best_threshold(val_probs, val_true)
        print(f'  [{tag}] epoch {epoch:02d}  loss={total_loss/n_total:.4f}  val_bacc={vb:.4f}')

        if vb > best_val_bacc + 1e-4:
            best_val_bacc = vb
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  [{tag}] Early stopping at epoch {epoch}')
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f'  [{tag}] best val_bacc={best_val_bacc:.4f}')
    return model


def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    print('Loading data...')
    trX  = np.load(os.path.join(TRAIN, 'X.npy')).astype(np.float32)
    trY  = np.load(os.path.join(TRAIN, 'Y.npy')).astype(np.float32)
    trSL = np.load(os.path.join(TRAIN, 'stage_label.npy')).astype(np.float32)
    trAL = np.load(os.path.join(TRAIN, 'apnea_label.npy')).astype(np.float32)

    vX   = np.load(os.path.join(VAL,   'X.npy')).astype(np.float32)
    vY   = np.load(os.path.join(VAL,   'Y.npy')).astype(np.float32)
    vSL  = np.load(os.path.join(VAL,   'stage_label.npy')).astype(np.float32)
    vAL  = np.load(os.path.join(VAL,   'apnea_label.npy')).astype(np.float32)

    print('Building 4-channel inputs (X, Y, env(X), env(Y))...')
    Xtr  = torch.from_numpy(build_4ch(trX, trY))
    Str  = torch.from_numpy(trSL)
    Ytr  = torch.from_numpy(trAL)
    Xval = torch.from_numpy(build_4ch(vX, vY))
    Sval = torch.from_numpy(vSL)
    Yval = torch.from_numpy(vAL)

    tr_loader  = DataLoader(TensorDataset(Xtr, Str, Ytr),   batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(Xval, Sval, Yval), batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    n_neg = int((trAL == 0).sum())
    n_pos = int((trAL == 1).sum())
    pos_w = torch.tensor([n_neg / (n_pos + 1e-12)], device=DEVICE)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    print('\n=== Training ResNet (seed=42) ===')
    torch.manual_seed(42)
    rn1 = ApneaResNet(in_channels=4, use_stage=True).to(DEVICE)
    rn1 = train_one(rn1, tr_loader, val_loader, pos_w, 'RN1')

    print('\n=== Training ResNet (seed=123) ===')
    torch.manual_seed(123)
    rn2 = ApneaResNet(in_channels=4, use_stage=True).to(DEVICE)
    rn2 = train_one(rn2, tr_loader, val_loader, pos_w, 'RN2')

    print('\n=== Training TCN (seed=42) ===')
    torch.manual_seed(42)
    tcn = ApneaTCN(in_channels=4, hidden=128, dilations=(1,2,4,8,16,32), kernel=5, use_stage=True).to(DEVICE)
    tcn = train_one(tcn, tr_loader, val_loader, pos_w, 'TCN')

    print('\n=== Ensemble (3 models, TTA=5) ===')
    print('Running TTA inference...')
    r1_tr,  tr_true  = predict_proba(rn1, tr_loader,  tta=5)
    r1_val, val_true = predict_proba(rn1, val_loader, tta=5)
    r2_tr,  _        = predict_proba(rn2, tr_loader,  tta=5)
    r2_val, _        = predict_proba(rn2, val_loader, tta=5)
    tc_tr,  _        = predict_proba(tcn, tr_loader,  tta=5)
    tc_val, _        = predict_proba(tcn, val_loader, tta=5)

    # Grid-search optimal ensemble weights on val
    print('Grid-searching ensemble weights...')
    best_weights = (1/3, 1/3, 1/3)
    best_vb_w    = 0.0
    grid = np.arange(0.1, 0.81, 0.1)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            p_val = w1 * r1_val + w2 * r2_val + w3 * tc_val
            _, vb_w = best_threshold(p_val, val_true)
            if vb_w > best_vb_w:
                best_vb_w = vb_w
                best_weights = (w1, w2, w3)

    w1, w2, w3 = best_weights
    print(f'Best weights: RN1={w1:.2f}, RN2={w2:.2f}, TCN={w3:.2f} → val_bACC={best_vb_w:.4f}')

    ens_tr  = w1 * r1_tr  + w2 * r2_tr  + w3 * tc_tr
    ens_val = w1 * r1_val + w2 * r2_val + w3 * tc_val

    # Stage arrays for threshold split
    val_stages = Sval.numpy().astype(int)
    tr_stages  = Str.numpy().astype(int)

    stage_thrs = stage_specific_threshold(ens_val, val_true.astype(int), val_stages)
    print(f'Stage thresholds: stage0={stage_thrs[0]:.2f}, stage1={stage_thrs[1]:.2f}')

    single_thr, single_vb = best_threshold(ens_val, val_true)
    print(f'Single threshold val bACC={single_vb:.4f} at thr={single_thr:.2f}')

    val_pred_st = apply_stage_thresholds(ens_val, val_stages, stage_thrs)
    val_m_st    = compute_metrics(val_true.astype(int), val_pred_st)
    print(f'Stage-specific threshold val bACC={val_m_st[0]:.4f}')

    if val_m_st[0] >= single_vb:
        print('Using stage-specific thresholds')
        tr_pred  = apply_stage_thresholds(ens_tr, tr_stages, stage_thrs)
        val_pred = val_pred_st
    else:
        print('Reverting to single threshold')
        tr_pred  = (ens_tr  >= single_thr).astype(int)
        val_pred = (ens_val >= single_thr).astype(int)

    print()
    print_metrics('train', *compute_metrics(tr_true.astype(int), tr_pred))
    print()
    print_metrics('val',   *compute_metrics(val_true.astype(int), val_pred))
    print()
    vram = torch.cuda.max_memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0
    print(f'peak_vram_mb: {vram:.1f}')
    print(f'elapsed_s: {time.time() - t0:.1f}')


if __name__ == '__main__':
    main()
