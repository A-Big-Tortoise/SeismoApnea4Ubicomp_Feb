
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import random



class TriangularCausalMask:
	def __init__(self, B, L, device="cpu"):
		mask_shape = [B, 1, L, L]
		with torch.no_grad():
			self._mask = torch.triu(
				torch.ones(mask_shape, dtype=torch.bool), diagonal=1
			).to(device)

	@property
	def mask(self):
		return self._mask



class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
		super(EncoderLayer, self).__init__()
		d_ff = d_ff or 4 * d_model
		self.attention = attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == "relu" else F.gelu

	def forward(self, x, attn_mask=None, tau=None, delta=None):
		new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
		x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

		y = x = [self.norm1(_x) for _x in x]
		y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
		y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

		return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
	def __init__(self, attn_layers, norm_layer=None):
		super(Encoder, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.norm = norm_layer

	def forward(self, x, attn_mask=None, tau=None, delta=None):
		# x [[B, L1, D], [B, L2, D], ...]
		attns = []
		for attn_layer in self.attn_layers:
			x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
			attns.append(attn)

		# concat all the outputs
		"""x = torch.cat(
			x, dim=1
		)  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)"""

		# concat all the routers
		x = torch.cat([x[:, -1, :].unsqueeze(1) for x in x], dim=1)  # (batch_size, len(patch_len_list), d_model)

		if self.norm is not None:
			x = self.norm(x)

		return x, attns
	

class FullAttention(nn.Module):
	def __init__(
		self,
		mask_flag=True,
		factor=5,
		scale=None,
		attention_dropout=0.1,
		output_attention=False,
	):
		super(FullAttention, self).__init__()
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		scale = self.scale or 1.0 / sqrt(E)

		scores = torch.einsum("blhe,bshe->bhls", queries, keys)

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B, L, device=queries.device)

			scores.masked_fill_(attn_mask.mask, -np.inf)

		A = self.dropout(
			torch.softmax(scale * scores, dim=-1)
		)  # Scaled Dot-Product Attention
		V = torch.einsum("bhls,bshd->blhd", A, values)

		if self.output_attention:
			return V.contiguous(), A
		else:
			return V.contiguous(), None



class AttentionLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
		super(AttentionLayer, self).__init__()

		d_keys = d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)

		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)

		out, attn = self.inner_attention(
			queries, keys, values, attn_mask, tau=tau, delta=delta
		)
		out = out.view(B, L, -1)

		return self.out_projection(out), attn


class MedformerLayer(nn.Module):
	def __init__(
		self,
		num_blocks,
		d_model,
		n_heads,
		dropout=0.1,
		output_attention=False,
		no_inter=False,
	):
		super().__init__()

		self.intra_attentions = nn.ModuleList(
			[
				AttentionLayer(
					FullAttention(
						False,
						factor=1,
						attention_dropout=dropout,
						output_attention=output_attention,
					),
					d_model,
					n_heads,
				)
				for _ in range(num_blocks)
			]
		)
		if no_inter or num_blocks <= 1:
			# print("No inter attention for time")
			self.inter_attention = None
		else:
			self.inter_attention = AttentionLayer(
				FullAttention(
					False,
					factor=1,
					attention_dropout=dropout,
					output_attention=output_attention,
				),
				d_model,
				n_heads,
			)
	def forward(self, x, attn_mask=None, tau=None, delta=None):
		attn_mask = attn_mask or ([None] * len(x))
		# Intra attention
		x_intra = []
		attn_out = []
		for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
			_x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
			x_intra.append(_x_out)  # (B, Li, D)
			attn_out.append(_attn)
		if self.inter_attention is not None:
			# Inter attention
			routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
			x_inter, attn_inter = self.inter_attention(
				routers, routers, routers, attn_mask=None, tau=tau, delta=delta
			)
			x_out = [
				torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
				for i, x in enumerate(x_intra)
			]
			attn_out += [attn_inter]
		else:
			x_out = x_intra
		return x_out, attn_out


class Jitter(nn.Module):
	# apply noise on each element
	def __init__(self, scale=0.1):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		if self.training:
			x += torch.randn_like(x) * self.scale
		return x


class Scale(nn.Module):
	# scale each channel by a random scalar
	def __init__(self, scale=0.1):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		if self.training:
			B, C, T = x.shape
			x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
		return x


class Flip(nn.Module):
	# left-right flip
	def __init__(self, prob=0.5):
		super().__init__()
		self.prob = prob

	def forward(self, x):
		if self.training and torch.rand(1) < self.prob:
			return torch.flip(x, [-1])
		return x


class TemporalMask(nn.Module):
	# Randomly mask a portion of timestamps across all channels
	def __init__(self, ratio=0.1):
		super().__init__()
		self.ratio = ratio

	def forward(self, x):
		if self.training:
			B, C, T = x.shape
			num_mask = int(T * self.ratio)
			mask_indices = torch.randperm(T)[:num_mask]
			x[:, :, mask_indices] = 0
		return x


class ChannelMask(nn.Module):
	# Randomly mask a portion of channels across all timestamps
	def __init__(self, ratio=0.1):
		super().__init__()
		self.ratio = ratio

	def forward(self, x):
		if self.training:
			B, C, T = x.shape
			num_mask = int(C * self.ratio)
			mask_indices = torch.randperm(C)[:num_mask]
			x[:, mask_indices, :] = 0
		return x


class FrequencyMask(nn.Module):
	def __init__(self, ratio=0.1):
		super().__init__()
		self.ratio = ratio

	def forward(self, x):
		if self.training:
			B, C, T = x.shape
			# Perform rfft
			x_fft = torch.fft.rfft(x, dim=-1)
			# Generate random indices for masking
			mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
			# Apply mask
			x_fft = x_fft * mask
			# Perform inverse rfft
			x = torch.fft.irfft(x_fft, n=T, dim=-1)
		return x


def get_augmentation(augmentation):
	if augmentation.startswith("jitter"):
		if len(augmentation) == 6:
			return Jitter()
		return Jitter(float(augmentation[6:]))
	elif augmentation.startswith("scale"):
		if len(augmentation) == 5:
			return Scale()
		return Scale(float(augmentation[5:]))
	elif augmentation.startswith("drop"):
		if len(augmentation) == 4:
			return nn.Dropout(0.1)
		return nn.Dropout(float(augmentation[4:]))
	elif augmentation.startswith("flip"):
		if len(augmentation) == 4:
			return Flip()
		return Flip(float(augmentation[4:]))
	elif augmentation.startswith("frequency"):
		if len(augmentation) == 9:
			return FrequencyMask()
		return FrequencyMask(float(augmentation[9:]))
	elif augmentation.startswith("mask"):
		if len(augmentation) == 4:
			return TemporalMask()
		return TemporalMask(float(augmentation[4:]))
	elif augmentation.startswith("channel"):
		if len(augmentation) == 7:
			return ChannelMask()
		return ChannelMask(float(augmentation[7:]))
	elif augmentation == "none":
		return nn.Identity()
	else:
		raise ValueError(f"Unknown augmentation {augmentation}")


class CrossChannelTokenEmbedding(nn.Module):
	def __init__(self, c_in, l_patch, d_model, stride=None):
		super().__init__()
		if stride is None:
			stride = l_patch
		self.tokenConv = nn.Conv2d(
			in_channels=1,
			out_channels=d_model,
			kernel_size=(c_in, l_patch),
			stride=(1, stride),
			padding=0,
			padding_mode="circular",
			bias=False,
		)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(
					m.weight, mode="fan_in", nonlinearity="leaky_relu"
				)

	def forward(self, x):
		x = self.tokenConv(x)
		return x



class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEmbedding, self).__init__()
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (
			torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
		).exp()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer("pe", pe)

	def forward(self, x):
		return self.pe[:, : x.size(1)]





class ListPatchEmbedding(nn.Module):
	def __init__(
		self,
		enc_in,
		d_model,
		seq_len,
		patch_len_list,
		stride_list,
		dropout,
		augmentation=["none"],
		single_channel=False,
	):
		super().__init__()
		self.patch_len_list = patch_len_list
		self.stride_list = stride_list
		self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
		self.single_channel = single_channel

		linear_layers = [
			CrossChannelTokenEmbedding(
				c_in=enc_in if not single_channel else 1,
				l_patch=patch_len,
				d_model=d_model,
			)
			for patch_len in patch_len_list
		]
		self.value_embeddings = nn.ModuleList(linear_layers)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.channel_embedding = PositionalEmbedding(d_model=seq_len)
		self.dropout = nn.Dropout(dropout)
		self.augmentation = nn.ModuleList(
			[get_augmentation(aug) for aug in augmentation]
		)

		self.learnable_embeddings = nn.ParameterList(
			[nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list]
		)


	def forward(self, x):  # (batch_size, seq_len, enc_in)
		# x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
		if self.single_channel:
			B, C, L = x.shape
			x = torch.reshape(x, (B * C, 1, L))

		x_list = []
		for padding, value_embedding in zip(self.paddings, self.value_embeddings):
			x_copy = x.clone()
			# per granularity augmentation
			aug_idx = random.randint(0, len(self.augmentation) - 1)
			x_new = self.augmentation[aug_idx](x_copy)
			# add positional embedding to tag each channel
			x_new = x_new + self.channel_embedding(x_new)
			x_new = padding(x_new).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
			x_new = value_embedding(x_new)  # (batch_size, d_model, 1, patch_num)
			x_new = x_new.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
			x_list.append(x_new)

		x = [
			x + cxt + self.position_embedding(x)
			for x, cxt in zip(x_list, self.learnable_embeddings)
		]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
		return x

class Medformer(nn.Module):
	"""
	Paper link: https://arxiv.org/pdf/2405.19363
	"""

	def __init__(self, configs):
		super(Medformer, self).__init__()
		self.task_name = "classification"
		self.output_attention = configs.output_attention
		self.enc_in = configs.enc_in
		self.single_channel = configs.single_channel
		# Embedding
		patch_len_list = list(map(int, configs.patch_len_list.split(",")))
		stride_list = patch_len_list
		seq_len = configs.seq_len
		patch_num_list = [
			int((seq_len - patch_len) / stride + 2)
			for patch_len, stride in zip(patch_len_list, stride_list)
		]
		augmentations = configs.augmentations.split(",")

		self.enc_embedding = ListPatchEmbedding(
			configs.enc_in,
			configs.d_model,
			configs.seq_len,
			patch_len_list,
			stride_list,
			configs.dropout,
			augmentations,
			configs.single_channel,
		)
		# Encoder
		self.encoder = Encoder(
			[
				EncoderLayer(
					MedformerLayer(
						len(patch_len_list),
						configs.d_model,
						configs.n_heads,
						configs.dropout,
						configs.output_attention,
						configs.no_inter_attn,
					),
					configs.d_model,
					configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation,
				)
				for l in range(configs.e_layers)
			],
			norm_layer=torch.nn.LayerNorm(configs.d_model),
		)
		# Decoder
		if self.task_name == "classification":
			self.act = F.gelu
			self.dropout = nn.Dropout(configs.dropout)
			
			enc_output_dim = configs.d_model * len(patch_num_list) * (1 if not self.single_channel else configs.enc_in)

			self.feature_extractor_Stage = nn.Sequential(
				nn.Linear(enc_output_dim, enc_output_dim),
				nn.ReLU(),
				nn.Dropout(configs.dropout),
				nn.Linear(enc_output_dim, enc_output_dim//2),
				nn.ReLU(),
				nn.Dropout(configs.dropout),
				nn.Linear(enc_output_dim//2, enc_output_dim//4),
			)
			self.classifier_Stage = nn.Linear(enc_output_dim//4, configs.num_class)

			self.feature_extractor_Apnea = nn.Sequential(
				nn.Linear(enc_output_dim, enc_output_dim),
				nn.ReLU(),
				nn.Dropout(configs.dropout),
				nn.Linear(enc_output_dim, enc_output_dim//2),
				nn.ReLU(),
				nn.Dropout(configs.dropout),
				nn.Linear(enc_output_dim//2, enc_output_dim//4),
			)
			self.classifier_Apnea = nn.Linear(enc_output_dim//4, configs.num_class)


	def forward(self, x_enc):
		# Embedding
		enc_out = self.enc_embedding(x_enc)
		enc_out, attns = self.encoder(enc_out, attn_mask=None)
		if self.single_channel:
			enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

		# Output
		output = self.act(
			enc_out
		)  # the output transformer encoder/decoder embeddings don't include non-linearity
		output = self.dropout(output)
		output = output.reshape(
			output.shape[0], -1
		)  # (batch_size, seq_length * d_model)
		rep_stage = self.feature_extractor_Stage(output)  # (batch_size, hidden_size)
		output_stage = self.classifier_Stage(rep_stage)  # (batch_size, num_classes)
		rep_apnea = self.feature_extractor_Apnea(output)  # (batch_size, hidden_size)
		output_apnea = self.classifier_Apnea(rep_apnea)  # (batch_size, num_classes)
		
		return output_stage, output_apnea, None, rep_stage, rep_apnea

