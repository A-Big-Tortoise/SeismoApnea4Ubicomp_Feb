"""
tool.py — EXP-019
-----------------
6-channel input: X, Y, env(X), env(Y), inst_freq(X), inst_freq(Y).
Instantaneous frequency (from Hilbert phase) directly encodes respiratory rate —
rate collapses to near-zero during apnea.
Ensemble: ResNet(42) + ResNet(123) + TCN(42) with binary stage + TTA + stage-specific thr.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert


# ── Signal helpers ───────────────────────────────────────────────────────────

def compute_envelope(sig_2d):
    analytic = hilbert(sig_2d, axis=1)
    env = np.abs(analytic).astype(np.float32)
    mean = env.mean(axis=1, keepdims=True)
    std  = env.std(axis=1, keepdims=True) + 1e-8
    return (env - mean) / std


def compute_inst_freq(sig_2d, fs=10):
    """Instantaneous frequency via Hilbert phase derivative (N, T) → (N, T)."""
    analytic = hilbert(sig_2d, axis=1)
    phase    = np.unwrap(np.angle(analytic), axis=1)
    # Central difference, replicate edges
    inst_f   = np.gradient(phase, axis=1) * fs / (2 * np.pi)
    # Clip to physiological range [0.05, 1 Hz] and z-score
    inst_f   = np.clip(inst_f, 0.05, 1.0).astype(np.float32)
    mean     = inst_f.mean(axis=1, keepdims=True)
    std      = inst_f.std(axis=1,  keepdims=True) + 1e-8
    return (inst_f - mean) / std


def build_6ch(X, Y):
    """6 channels: X, Y, env(X), env(Y), inst_freq(X), inst_freq(Y)."""
    eX  = compute_envelope(X)
    eY  = compute_envelope(Y)
    fX  = compute_inst_freq(X)
    fY  = compute_inst_freq(Y)
    return np.stack([X, Y, eX, eY, fX, fY], axis=1)


def build_4ch(X, Y):
    eX = compute_envelope(X)
    eY = compute_envelope(Y)
    return np.stack([X, Y, eX, eY], axis=1)


# ── ResNet ───────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class AttentionPool(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attn = nn.Linear(ch, 1)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        w   = torch.softmax(self.attn(x_t), dim=1)
        return (x_t * w).sum(dim=1)


class ApneaResNet(nn.Module):
    """4-ch ResNet + binary stage scalar concatenated to pooled embedding."""
    def __init__(self, in_channels=4, dropout=0.4, use_stage=True):
        super().__init__()
        self.use_stage = use_stage
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
        )
        self.layer1 = nn.Sequential(ResBlock1D(32, 64, 9, 2, 0.2), ResBlock1D(64, 64, 9, 1, 0.2))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 7, 2, 0.2), ResBlock1D(128, 128, 7, 1, 0.2))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, 5, 2, 0.2), ResBlock1D(256, 256, 5, 1, 0.2))
        self.pool = AttentionPool(256)
        emb_dim = 257 if use_stage else 256
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x, stage=None):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        emb = self.pool(x)
        if self.use_stage and stage is not None:
            emb = torch.cat([emb, stage.float().unsqueeze(1)], dim=1)
        return self.head(emb)


# ── TCN ──────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel=5, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x):
        T = x.size(-1)
        h = self.act(self.bn1(self.conv1(x)[..., :T]))
        h = self.drop(h)
        h = self.bn2(self.conv2(h)[..., :T])
        return self.act(h + x)


class ApneaTCN(nn.Module):
    """Dilated TCN + binary stage scalar."""
    def __init__(self, in_channels=4, hidden=128,
                 dilations=(1,2,4,8,16,32), kernel=5, dropout=0.3, use_stage=True):
        super().__init__()
        self.use_stage = use_stage
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList([TCNBlock(hidden, kernel, d, dropout) for d in dilations])
        self.attn_w = nn.Conv1d(hidden, 1, 1)
        emb_dim = hidden + 1 if use_stage else hidden
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x, stage=None):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        w = torch.softmax(self.attn_w(x), dim=-1)
        emb = (x * w).sum(dim=-1)
        if self.use_stage and stage is not None:
            emb = torch.cat([emb, stage.float().unsqueeze(1)], dim=1)
        return self.head(emb)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(x, noise_std=0.05, scale_range=(0.8, 1.2), channel_swap_p=0.3):
    """
    Augment 6-ch BCG input (B, 6, T): X, Y, env(X), env(Y), freq(X), freq(Y).
    Channel swap: swap X↔Y and all paired derived channels.
    """
    if channel_swap_p > 0 and torch.rand(1).item() < channel_swap_p:
        x = x[:, [1, 0, 3, 2], :]          # swap X↔Y, env(X)↔env(Y)
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    if scale_range is not None:
        lo, hi = scale_range
        scale = lo + (hi - lo) * torch.rand(x.size(0), 1, 1, device=x.device)
        x = x * scale
    return x


@torch.no_grad()
def tta_predict(model, xb, sb, n_tta=5, noise_std=0.03):
    """Test-Time Augmentation: average N noisy inference passes."""
    preds = torch.sigmoid(model(xb, sb).squeeze(1))
    for _ in range(n_tta - 1):
        xb_aug = xb + noise_std * torch.randn_like(xb)
        preds  = preds + torch.sigmoid(model(xb_aug, sb).squeeze(1))
    return preds / n_tta
