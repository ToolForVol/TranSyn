"""
@desc: a transformer with alternative component
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def positional_encoding(position, d_model):
	angle_rates = 1 / torch.pow(10000, (2 * torch.arange(d_model) // 2).float() / d_model)
	positions = torch.arange(position).unsqueeze(1)
	angle_rads = positions * angle_rates.unsqueeze(0)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads.unsqueeze(0)  # (1, position, d_model)
	return pos_encoding


def scaled_dot_product_attention(q, k, v, mask=None):
	matmul_qk = torch.matmul(q, k.transpose(-2, -1))
	dk = q.size()[-1]
	scaled_attention_logits = matmul_qk / math.sqrt(dk)

	if mask is not None:
		# mask: should broadcast to (batch, heads, q_len, k_len) or (batch, 1, 1, k_len)
		scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

	attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
	output = torch.matmul(attention_weights, v)
	return output, attention_weights


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, num_heads):
		super().__init__()
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.num_heads = num_heads
		self.depth = d_model // num_heads

		self.wq = nn.Linear(d_model, d_model)
		self.wk = nn.Linear(d_model, d_model)
		self.wv = nn.Linear(d_model, d_model)
		self.dense = nn.Linear(d_model, d_model)
	def split_heads(self, x):
		# x: (B, L, d_model) -> (B, heads, L, depth)
		B, L, _ = x.size()
		return x.view(B, L, self.num_heads, self.depth).transpose(1, 2)
	def forward(self, v, k, q, mask=None):
		batch_size = q.size(0)
		q = self.wq(q)  # (B, Lq, d_model)
		k = self.wk(k)  # (B, Lk, d_model)
		v = self.wv(v)  # (B, Lv, d_model)

		q = self.split_heads(q)  # (B, heads, Lq, depth)
		k = self.split_heads(k)
		v = self.split_heads(v)

		scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)
		# scaled_attention: (B, heads, Lq, depth)
		scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (B, Lq, heads, depth)
		concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.depth)  # (B, Lq, d_model)

		output = self.dense(concat_attention)  # (B, Lq, d_model)
		return output, attn_weights


def point_wise_feed_forward_network(d_model, dff):
	return nn.Sequential(
		nn.Linear(d_model, dff),
		nn.ReLU(),
		nn.Linear(dff, d_model)
	)


class EncoderLayer(nn.Module):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super().__init__()
		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)

		self.dropout1 = nn.Dropout(rate)
		self.dropout2 = nn.Dropout(rate)

	def forward(self, x, mask=None):
		attn_output, _ = self.mha(x, x, x, mask)
		out1 = self.layernorm1(x + self.dropout1(attn_output))
		ffn_output = self.ffn(out1)
		out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
		return out2


class Bottleneck(nn.Module):
	def __init__(self, in_dim=768, out_dim=256):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, out_dim)
	def forward(self, x):
		return F.relu((self.fc1(x)))


class DyPositionTransformer(nn.Module):
	def __init__(self,
				 input_dim: int = 768,  
				 num_layers: int = 4,
				 d_model: int = 512,
				 num_heads: int = 8,
				 dff: int = 1024,
				 dropout_rate: float = 0.1,
				 max_len: int = 1024,
				 head_hidden: int = 64,
				 pretrain_weight: str = None,
				 freeze_backbone: bool = False):
		super().__init__()
		# TRSR settings
		self.noise_std = 0.01
		self.inject_layer = 2
		self.measure_from = 2

		# Transformer
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.input_dim = input_dim
		self.num_layers = num_layers
		self.d_model = d_model
		self.num_heads = num_heads
		self.dff = dff
		self.dropout_rate = dropout_rate
		self.head_hidden = head_hidden
		self.bottleneck = Bottleneck(in_dim=input_dim, out_dim=d_model)

		# positional encoding buffer (max_len)
		pos_enc = positional_encoding(max_len, d_model)
		self.register_buffer('pos_encoding', pos_enc, persistent=False)  

		# encoder layers
		self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate=dropout_rate) for _ in range(num_layers)])
		self.dropout = nn.Dropout(dropout_rate)

		# final output dimension 
		self.output_dim = d_model  

		# classifier head 
		self.head = nn.Sequential(
			nn.Linear(self.output_dim, head_hidden),
			nn.ReLU(),
			nn.Dropout(p=dropout_rate),
			nn.Linear(head_hidden, 1)
		)

		# load pretrained weight
		if pretrain_weight:
			state_dict = torch.load(pretrain_weight, map_location="cpu")
			filtered_dict = {k.replace("enc_layers.", ""): v for k, v in state_dict.items() if "enc_layers" in k}
			incompatible = self.enc_layers.load_state_dict(filtered_dict, strict=False)
			if incompatible.missing_keys:
				print("[WARN] Missing keys:", incompatible.missing_keys)
			if incompatible.unexpected_keys:
				print("[WARN] Unexpected keys:", incompatible.unexpected_keys)
			else:
				print("[INFO] All weights loaded successfully")

		# if freeze the weight
		if freeze_backbone:
			for p in self.enc_layers.parameters():
				p.requires_grad = False


	def forward(self, seq_x, mask=None, backbone=False, use_tsrs=False):
		"""
		Args:		
			seq_x: (B, L, C_in)
			mask: optional attention mask
			backbone: If True, skip the head layer. If False, go through the head layer.
			return_deep_features: Return features before the head layer.
			use_tsrs: Apply TSRS noise for stability regularization (noise used only for computing noise_outputs)
		Returns:
			if use_tsrs=False: feature tensor [B, C_out] (clean)
            if use_tsrs=True: (feature tensor [B, C_out] clean, list of noise stability scalars)
		"""
		# Enter the bottleneck
		seq_x = self.bottleneck(seq_x)
		# Position encoding
		seq_len = seq_x.size(1)
		pe = self.pos_encoding[:, :seq_len, :].to(seq_x.dtype)
		seq_x = seq_x + pe
		seq_x = self.dropout(seq_x)

		device = seq_x.device
		clean_x = seq_x.clone() 
		noise_outputs = None

		if use_tsrs:
			# TSRS mode
			noise_x = clean_x.clone()  # noise path
			noise_outputs = []

			for i, layer in enumerate(self.enc_layers):
				# Inject noise
				if i == self.inject_layer:
					noise = self.noise_std * torch.randn_like(noise_x, device=device)
					noise_x = noise_x + noise

				# two separately paths
				clean_x = layer(clean_x, mask)
				noise_x = layer(noise_x, mask)

				# Stability measurement
				if i >= self.measure_from:
					stability = torch.norm(noise_x - clean_x, p=2) / (torch.norm(clean_x, p=2) + 1e-8)
					noise_outputs.append(stability)
		else:
			# Standard mode
			for layer in self.enc_layers:
				clean_x = layer(clean_x, mask)

		# Final outoput
		idseq_x = (seq_len - 1) // 2
		seq_out = clean_x[:, idseq_x, :]

		if not backbone: # Enter head classifier or use as backbone
			out = self.head(seq_out)
		else:
			out = seq_out

		# Return deep feature and noise
		if use_tsrs: 
			return seq_out, noise_outputs
		else: # Return deep feature / logits
			return out