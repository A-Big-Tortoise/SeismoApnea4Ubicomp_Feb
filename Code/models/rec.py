import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.inception import Inception
from models.patchtst import PatchTST_backbone


class PatchTST_REC(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, stride, n_layers, d_model,
                  n_heads, d_ff, axis, dropout):
        super(PatchTST_REC, self).__init__()
        self.encoder = PatchTST_backbone(c_in = input_size, 
                                        context_window = seq_len,
                                        patch_len = patch_len,
                                        stride = stride,
                                        max_seq_len = int(seq_len*1.5),
                                        n_layers = n_layers,
                                        d_model = d_model,
                                        n_heads = n_heads,
                                        d_ff = d_ff,
                                        attn_dropout = dropout,
                                        dropout = dropout,
                                        act = "gelu")


        self.generator_tho = nn.Sequential(
            nn.Linear(d_model*axis, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, seq_len)
        )

        self.generator_abd = nn.Sequential(
            nn.Linear(d_model*axis, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, seq_len)
        )




    def forward(self, x):
        """
        x: [B, C, T]
        """
        features_X = self.encoder(x)  # [B, C, d_model, patch_num]
        B, C, D, N = features_X.shape
        
        # === 下游处理 ===
        pooled_X = features_X.reshape(B, -1)  # [B, C*d_model]
        
        reconstructed_tho = self.generator_tho(pooled_X)  # [B, seq_len]
        reconstructed_abd = self.generator_abd(pooled_X)  # [B, seq_len]

        return reconstructed_tho, reconstructed_abd



class VTCN_ED(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, seq_len, dropout):
        super(VTCN_ED, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.decoder_tho = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]//4),
            nn.ReLU(),  
            nn.Linear(num_channels[-1]//4, num_channels[-1]//2),
            nn.ReLU(),
            nn.Linear(num_channels[-1]//2, num_channels[-1]),
            nn.ReLU(),
            nn.Linear(num_channels[-1], output_size)
        )
        
        self.decoder_abd = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1]//4),
            nn.ReLU(),  
            nn.Linear(num_channels[-1]//4, num_channels[-1]//2),
            nn.ReLU(),
            nn.Linear(num_channels[-1]//2, num_channels[-1]),
            nn.ReLU(),
            nn.Linear(num_channels[-1], output_size)
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_channels[-1], num_channels[-1])
        
        self.init_weights()

    def init_weights(self):
        for m in self.decoder_tho.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
        for m in self.decoder_abd.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x =  x.unsqueeze(1)
        # print(f'Input shape: {x.shape}')
        y = self.tcn(x)
        y_latent = self.linear(y.transpose(1, 2))
        y_latent = self.relu(y_latent)

        y_rec_tho = self.decoder_tho(y_latent).squeeze(-1)
        y_rec_abd = self.decoder_abd(y_latent).squeeze(-1)
        return y_rec_tho, y_rec_abd