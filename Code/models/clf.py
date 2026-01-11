import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.inception import Inception
# from models.patchtst import PatchTST_backbone
from models.patchtst_exportable import PatchTST_backbone_exp
from models.medformer import Medformer

class ApneaClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size, seq_len, dropout, projection_dim=128):
        super(ApneaClassifier, self).__init__()
        self.tcn_encoder_1 = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn_encoder_2 = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(seq_len, 1)
        
        # Projection layer
        encoder_output_dim = num_channels[-1]
        self.projection = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1]*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x [B, C, T]
        signal_X = x[:, 0, :].unsqueeze(1)  # [B, 1, T]
        signal_Y = x[:, 1, :].unsqueeze(1)
        
        features_X = self.tcn_encoder_1(signal_X)
        features_Y = self.tcn_encoder_2(signal_Y)

        # pooled_X = self.pool(features_X).squeeze(-1) # [B, nhid]
        # pooled_Y = self.pool(features_Y).squeeze(-1) # [B, nhid]

        # pooled_X = features_X[:, :, -1]  # [B, nhid]
        # pooled_Y = features_Y[:, :, -1]
        pooled_X = self.pool(features_X).squeeze(-1)  # [B, nhid]
        pooled_Y = self.pool(features_Y).squeeze(-1)  # [B, nhid]

        # Apply projection

        projected_X = self.projection(pooled_X)  # [B, projection_dim]
        projected_Y = self.projection(pooled_Y)  # [B, projection_dim]
        # # print(f'projected_X: {projected_X.shape}, projected_Y: {projected_Y.shape}')
        
        combined_features = torch.cat((pooled_X, pooled_Y), dim=1)
        y = self.classifier(combined_features)
        
        return y, F.normalize(projected_X, dim=1), F.normalize(projected_Y, dim=1)



class ApneaClassifier_One(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size, seq_len, dropout, projection_dim=128):
        super(ApneaClassifier_One, self).__init__()
        self.tcn_encoder = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
       
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x [B, C, T]
        features_X = self.tcn_encoder(x)
        pooled_X = self.pool(features_X).squeeze(-1)  # [B, nhid]
        y = self.classifier(pooled_X)
        
        return y, -1, -1 
    

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim, num_layers, dropout, projection_dim=128):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # BiLSTM for channel X and Y
        self.bilstm_encoder_X = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.bilstm_encoder_Y = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)  # Will be applied after permuting to [B, F, T]

        # Projection head
        encoder_output_dim = hidden_dim * 2  # Because of bidirection
        self.projection = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 2, T]
        signal_X = x[:, 0, :].unsqueeze(-1)  # [B, T, 1]
        signal_Y = x[:, 1, :].unsqueeze(-1)  # [B, T, 1]

        # Encode
        out_X, _ = self.bilstm_encoder_X(signal_X)  # [B, T, 2*hidden]
        out_Y, _ = self.bilstm_encoder_Y(signal_Y)

        # Pool along time axis
        out_X = out_X.permute(0, 2, 1)  # [B, 2*hidden, T]
        out_Y = out_Y.permute(0, 2, 1)
        pooled_X = self.pool(out_X).squeeze(-1)  # [B, 2*hidden]
        pooled_Y = self.pool(out_Y).squeeze(-1)

        # Projection head
        projected_X = self.projection(pooled_X)  # [B, projection_dim]
        projected_Y = self.projection(pooled_Y)

        # Concatenate and classify
        combined = torch.cat([pooled_X, pooled_Y], dim=1)  # [B, 4*hidden]
        y = self.classifier(combined)

        return y, F.normalize(projected_X, dim=1), F.normalize(projected_Y, dim=1)


class ApneaClassifier_InceptionTime(nn.Module):
    def __init__(self, input_size, num_classes, n_filters, kernel_sizes, bottleneck_channels, dropout, projection_dim=128):
        super(ApneaClassifier_InceptionTime, self).__init__()
        self.tcn_encoder_1 = Inception(input_size, n_filters=n_filters, kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels)
        self.tcn_encoder_2 = Inception(input_size, n_filters=n_filters, kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # # Projection layer
        # encoder_output_dim = n_filters * 4
        # self.projection = nn.Sequential(
        #     nn.Linear(encoder_output_dim, encoder_output_dim),
        #     nn.ReLU(),
        #     nn.Linear(encoder_output_dim, projection_dim),
        # )
        
        self.classifier = nn.Sequential(
            nn.Linear(n_filters*4*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x [B, C, T]
        signal_X = x[:, 0, :].unsqueeze(1)  # [B, 1, T]
        signal_Y = x[:, 1, :].unsqueeze(1)
        
        features_X = self.tcn_encoder_1(signal_X)
        features_Y = self.tcn_encoder_2(signal_Y)

        pooled_X = self.pool(features_X).squeeze(-1)  # [B, nhid]
        pooled_Y = self.pool(features_Y).squeeze(-1)  # [B, nhid]
        
        combined_features = torch.cat((pooled_X, pooled_Y), dim=1)
        y = self.classifier(combined_features)
        
        return y, 1, 1




class ChannelSE(nn.Module):
    """
    SE over channels on [B, C, D, P]. Squeeze over (D,P), excite over C.
    """
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, z):                     # z: [B, C, D]
        s = z.mean(dim=2)                 # [B, C]
        s = s.reshape(s.size(0), s.size(1))  #
        # print(f's: {s.shape}')
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))  # [B, C]
        w = w.view(w.size(0), w.size(1), 1)
        return z * w                          


class ApneaClassifier_PatchTST(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, stride, n_layers, d_model, n_heads, d_ff, num_classes, axis, dropout, mask_ratio):
        super(ApneaClassifier_PatchTST, self).__init__()
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

        self.mask_ratio = mask_ratio
        # self.gate = self.se = ChannelSE(channels=axis, reduction=2)

        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.linear = nn.Linear(seq_len//patch_len * 2, 1)
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(d_model*axis, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, d_model))

        # self.projection = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model))



        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model*axis, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )


        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(d_model*axis, d_model//2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model//2, d_model//4),
        # )


        self.classifier = nn.Linear(d_model//4, num_classes)


    # def forward(self, x):
    #     # x [B, C, T]
    #     # print(f'x: {x.shape}')
    #     # x = x[:, :2, :]  # use HR + RR only
    #     features_X = self.encoder(x)
    #     # print(f'features_X: {features_X.shape}')
    #     # don't use [cls] token
    #     features_X = features_X.mean(-1)  # [B, C, nhid]
    #     # use [cls] token
    #     # features_X = features_X[:, :, :, 0]  # [B, C, nhid]
    #     # features_X = self.se(features_X)
    #     pooled_X = features_X.reshape(features_X.shape[0], -1)  # [B, nhid_x + nhid_y]

    #     projected_X = self.projection(pooled_X)  # [B, projection_dim]

    #     y_rep = self.feature_extractor(pooled_X)  # [B, d_model//4]
    #     y = self.classifier(y_rep)
        
    #     return y, projected_X, y_rep
    
    # def forward(self, x):
    #     """
    #     x: [B, C, T]
    #     输出:
    #         y: 分类输出
    #         projected_X: 投影特征 [B, d_model]
    #         y_rep: 中间特征 [B, d_model//4]
    #     """
    #     features_X = self.encoder(x)  # [B, C, d_model, patch_num]
    #     B, C, D, N = features_X.shape
    #     # === 通道独立 Patch Mask ===
    #     if self.training and self.mask_ratio > 0:
            
    #         mask_num = int(N * self.mask_ratio)

    #         # 创建全1的mask
    #         keep_mask = torch.ones(B, C, N, device=x.device)
    #         for b in range(B):
    #             for c in range(C):
    #                 perm = torch.randperm(N, device=x.device)
    #                 mask_idx = perm[:mask_num]  # 随机选出要mask掉的patch
    #                 keep_mask[b, c, mask_idx] = 0.0

    #         # 广播到d_model维度并应用mask
    #         features_X = features_X * keep_mask.unsqueeze(2)  # [B, C, d_model, N]

    #     # === 聚合未mask的patch ===
    #     features_X = features_X.mean(-1)  # [B, C, d_model]
    #     pooled_X = features_X.reshape(B, -1)  # [B, C*d_model]

    #     # === 下游 projection + classifier ===
    #     projected_X = self.projection(pooled_X)  # [B, d_model]
    #     y_rep = self.feature_extractor(pooled_X)
    #     y = self.classifier(y_rep)
    #     return y, projected_X, y_rep

    def forward(self, x):
        """
        x: [B, C, T]
        """
        features_X = self.encoder(x)  # [B, C, d_model, patch_num]
        B, C, D, N = features_X.shape
        
        # === 高效的通道独立 Patch Mask ===
        if self.training and self.mask_ratio > 0:
            mask_num = int(N * self.mask_ratio)
            keep_num = N - mask_num
            
            # 批量生成随机索引 [B, C, N]
            rand_idx = torch.rand(B, C, N, device=x.device).argsort(dim=-1)
            keep_idx = rand_idx[..., :keep_num]  # 保留的patch索引
            
            # 创建mask [B, C, 1, N]
            keep_mask = torch.zeros(B, C, 1, N, device=x.device)
            keep_mask.scatter_(-1, keep_idx.unsqueeze(2), 1.0)
            
            # 应用mask
            features_X = features_X * keep_mask  # [B, C, d_model, N]
            
            # 只对未mask的patch求均值（归一化）
            features_X = features_X.sum(-1) / keep_num  # [B, C, d_model]
        else:
            features_X = features_X.mean(-1)  # [B, C, d_model]
        
        # === 下游处理 ===
        pooled_X = features_X.reshape(B, -1)  # [B, C*d_model]
        
        projected_X = self.projection(pooled_X)  # [B, d_model]
        y_rep = self.feature_extractor(pooled_X)  # [B, d_model//4]
        y = self.classifier(y_rep)
        
        return y, projected_X, y_rep





class ApneaClassifier_PatchTST_MTL(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, stride, n_layers, d_model,
                  n_heads, d_ff, num_classes, axis, dropout, mask_ratio):
        super(ApneaClassifier_PatchTST_MTL, self).__init__()
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

        self.mask_ratio = mask_ratio

        self.projection = nn.Sequential(
            nn.Linear(d_model*axis, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, d_model))

        self.feature_extractor_Stage = nn.Sequential(
            nn.Linear(d_model*axis, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )

        self.classifier_Stage = nn.Linear(d_model//4, num_classes)

        self.feature_extractor_Apnea = nn.Sequential(
            nn.Linear(d_model*axis, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )
        self.classifier_Apnea = nn.Linear(d_model//4, num_classes)



    def forward(self, x):
        """
        x: [B, C, T]
        """
        features_X = self.encoder(x)  # [B, C, d_model, patch_num]
        B, C, D, N = features_X.shape
        
        # === 高效的通道独立 Patch Mask ===
        if self.training and self.mask_ratio > 0:
            mask_num = int(N * self.mask_ratio)
            keep_num = N - mask_num
            
            # 批量生成随机索引 [B, C, N]
            rand_idx = torch.rand(B, C, N, device=x.device).argsort(dim=-1)
            keep_idx = rand_idx[..., :keep_num]  # 保留的patch索引
            
            # 创建mask [B, C, 1, N]
            keep_mask = torch.zeros(B, C, 1, N, device=x.device)
            keep_mask.scatter_(-1, keep_idx.unsqueeze(2), 1.0)
            
            # 应用mask
            features_X = features_X * keep_mask  # [B, C, d_model, N]
            
            # 只对未mask的patch求均值（归一化）
            features_X = features_X.sum(-1) / keep_num  # [B, C, d_model]
        else:
            features_X = features_X.mean(-1)  # [B, C, d_model]
        
        # === 下游处理 ===
        pooled_X = features_X.reshape(B, -1)  # [B, C*d_model]
        
        projected_X = self.projection(pooled_X)  # [B, d_model]
        
        
        # y_rep = self.feature_extractor(pooled_X)  # [B, d_model//4]
        # y = self.classifier(y_rep)
        y_rep_Stage = self.feature_extractor_Stage(pooled_X)  # [B, d_model//4]
        y_Stage = self.classifier_Stage(y_rep_Stage)
        y_rep_Apnea = self.feature_extractor_Apnea(pooled_X)  # [B, d_model//4]
        y_Apnea = self.classifier_Apnea(y_rep_Apnea)
        
        return y_Stage, y_Apnea, projected_X, y_rep_Stage, y_rep_Apnea


class ApneaClassifier_PatchTST_MTL_REC(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, stride, n_layers, d_model,
                  n_heads, d_ff, num_classes, axis, dropout, mask_ratio):
        super(ApneaClassifier_PatchTST_MTL_REC, self).__init__()
        self.encoder = PatchTST_backbone_exp(c_in = input_size, 
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

        self.mask_ratio = mask_ratio

        self.projection = nn.Sequential(
            nn.Linear(d_model*axis, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, d_model))
        
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

        self.feature_extractor_Stage = nn.Sequential(
            nn.Linear(d_model*axis, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )

        self.classifier_Stage = nn.Linear(d_model//4, num_classes)

        self.feature_extractor_Apnea = nn.Sequential(
            nn.Linear(d_model*axis, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )
        self.classifier_Apnea = nn.Linear(d_model//4, num_classes)



    def forward(self, x):
        """
        x: [B, C, T]
        """
        features_X = self.encoder(x)  # [B, C, d_model, patch_num]
        B, C, D, N = features_X.shape
        
        # === 高效的通道独立 Patch Mask ===
        if self.training and self.mask_ratio > 0:
            mask_num = int(N * self.mask_ratio)
            keep_num = N - mask_num
            
            # 批量生成随机索引 [B, C, N]
            rand_idx = torch.rand(B, C, N, device=x.device).argsort(dim=-1)
            keep_idx = rand_idx[..., :keep_num]  # 保留的patch索引
            
            # 创建mask [B, C, 1, N]
            keep_mask = torch.zeros(B, C, 1, N, device=x.device)
            keep_mask.scatter_(-1, keep_idx.unsqueeze(2), 1.0)
            
            # 应用mask
            features_X = features_X * keep_mask  # [B, C, d_model, N]
            
            # 只对未mask的patch求均值（归一化）
            features_X = features_X.sum(-1) / keep_num  # [B, C, d_model]
        else:
            features_X = features_X.mean(-1)  # [B, C, d_model]
        
        # === 下游处理 ===
        pooled_X = features_X.reshape(B, -1)  # [B, C*d_model]
        
        projected_X = self.projection(pooled_X)  # [B, d_model]
        reconstructed_tho = self.generator_tho(pooled_X)  # [B, seq_len]
        reconstructed_abd = self.generator_abd(pooled_X)  # [B, seq_len]
        
        # y_rep = self.feature_extractor(pooled_X)  # [B, d_model//4]
        # y = self.classifier(y_rep)
        y_rep_Stage = self.feature_extractor_Stage(pooled_X)  # [B, d_model//4]
        y_Stage = self.classifier_Stage(y_rep_Stage)
        y_rep_Apnea = self.feature_extractor_Apnea(pooled_X)  # [B, d_model//4]
        y_Apnea = self.classifier_Apnea(y_rep_Apnea)
        
        return y_Stage, y_Apnea, projected_X, y_rep_Stage, y_rep_Apnea, reconstructed_tho, reconstructed_abd



class ApneaClassifier_Medformer_MTL(nn.Module):
    def __init__(self, input_size, seq_len, n_layers, output_attention, patch_len_lst, augmentations, d_model, activation, no_inter_attn, n_heads, d_ff, num_classes, dropout):
        super(ApneaClassifier_Medformer_MTL, self).__init__()


        self.model = Medformer(
            configs=type('configs', (object,), {
                'enc_in': input_size,
                'output_attention': output_attention,
                'patch_len_list': patch_len_lst,
                'seq_len': seq_len,
                'augmentations': augmentations,
                'd_model': d_model,
                'n_heads': n_heads,
                'e_layers': n_layers,
                'd_ff': d_ff,
                'dropout': dropout,
                'single_channel': False,
                'activation': activation,
                'num_class': num_classes,
                'no_inter_attn': no_inter_attn,
            })()
        ) 

    def forward(self, x):
        """
        x: [B, C, T]
        """
        y_Stage, y_Apnea, _, y_rep_Stage, y_rep_Apnea = self.model(x)  # [B, C, d_model, patch_num]
        return y_Stage, y_Apnea, None, y_rep_Stage, y_rep_Apnea



class ApneaClassifier_Dual_PatchTST(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, stride, n_layers, d_model, n_heads, d_ff, num_classes, axis, dropout):
        super(ApneaClassifier_Dual_PatchTST, self).__init__()
        self.encoder_R = PatchTST_backbone(c_in = input_size, 
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


        self.encoder_H = PatchTST_backbone(c_in = input_size, 
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
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model*axis*2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
        )

        self.projection = nn.Sequential(
            nn.Linear(d_model*axis*2, d_model*axis),
            nn.ReLU(),
            nn.Linear(d_model*axis, d_model))


        self.classifier = nn.Linear(d_model//4, num_classes)


    def forward(self, x):
        # x [B, 2 * C, T]
        # print(f'x: {x.shape}')
        features_R = self.encoder_R(x[:, :2, :])  # RR
        features_H = self.encoder_H(x[:, 2:, :])  # HR
        # print(f'features_R: {features_R.shape}, features_H: {features_H.shape}')

        features_R = features_R.mean(-1)  # [B, C, nhid]
        features_H = features_H.mean(-1)  # [B, C, nhid]
        # print(f'features_R after mean: {features_R.shape}, features_H after mean: {features_H.shape}')
        pooled_R = features_R.reshape(features_R.shape[0], -1)  # [B, nhid_x + nhid_y]
        pooled_H = features_H.reshape(features_H.shape[0], -1)  # [B, nhid_x + nhid_y]
        # print(f'pooled_R: {pooled_R.shape}, pooled_H: {pooled_H.shape}')
        pooled_X = torch.cat((pooled_R, pooled_H), dim=1)
        # print(f'pooled_X: {pooled_X.shape}')

        projected_X = self.projection(pooled_X)  # [B, projection_dim]
        y_rep = self.feature_extractor(pooled_X)  # [B, d_model//4]
        # print(f'y_rep: {y_rep.shape}')

        y = self.classifier(y_rep)

        return y, projected_X, y_rep
   

