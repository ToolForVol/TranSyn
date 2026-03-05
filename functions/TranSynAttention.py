"""
@description: The main module of TranSyn
@author: ye chen
@email: q23101020@stu.ahu.edu.cn
"""


import torch
import torch.nn as nn
from functions.Fusion import CrossGateBlock


class TranSynAttention(nn.Module):
	def __init__(
		self,
		dna_module: nn.Module = None,
		rna_module: nn.Module = None,
		bio_module: nn.Module = None,

		hidden_layer: int = None,
		dropout: float = 0.3,
		freeze_source: bool=False
	):
		"""
		@Args:
			dna_module, backbone to extract features from dna feature
			rna_module, backbone to extract features from rna feature
			bio_module, backbone to extract features from handcrafted feature

			hidden_layer, hidden layer of final FC layers
			dropout, dropout of final FC layers
			freeze_source, freeze the weight of source or not
		"""
		super().__init__()
		# attribute
		self.hidden_layer = hidden_layer
		self.dropout = dropout

		# init the backbones
		self.dna_module = dna_module
		self.rna_module = rna_module
		self.bio_module = bio_module

		# init the output dims
		self.dna_output_dim = dna_module.output_dim
		self.rna_output_dim = rna_module.output_dim
		self.bio_output_dim = bio_module.output_dim

		self.cg_dna = CrossGateBlock(
			bio_dim=self.bio_output_dim,
			target_dim=self.dna_output_dim,
			proj_dim=max(self.bio_output_dim, self.dna_output_dim),
			dropout=dropout
		)

		self.cg_rna = CrossGateBlock(
			bio_dim=self.bio_output_dim,
			target_dim=self.rna_output_dim,
			proj_dim=max(self.bio_output_dim, self.rna_output_dim),
			dropout=dropout
		)
		
		# ❄ freeze the weight if True
		if freeze_source:
			for param in self.dna_module.parameters():
				param.requires_grad = False
			for param in self.rna_module.parameters():
				param.requires_grad = False
		
		# normalize layers
		self.norm_dna = nn.LayerNorm(self.dna_output_dim)
		self.norm_rna = nn.LayerNorm(self.rna_output_dim)
		self.norm_bio = nn.LayerNorm(self.bio_output_dim)
		self.final_features_dim = self.dna_output_dim + self.rna_output_dim + self.bio_module.output_dim

		# classifier head
		self.head = nn.Sequential(
			nn.Linear(self.final_features_dim, hidden_layer),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(hidden_layer, 1)
		)

	def forward(self, dna_x, rna_x, bio_x, train=True):
		if train:
			dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
			rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
		else: # evaluate或test mode
			dna_output = self.dna_module(dna_x, use_tsrs=False)
			rna_output = self.rna_module(rna_x, backbone=True)
		bio_output = self.bio_module(bio_x)

		# run through the normalizations
		dna_output = self.norm_dna(dna_output)
		rna_output = self.norm_rna(rna_output)
		bio_output = self.norm_bio(bio_output)

		# fusion
		dna_output = self.cg_dna(bio_output, dna_output)
		rna_output = self.cg_rna(bio_output, rna_output)

		# run through the classifier head
		final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
		outputs = self.head(final_feature)
		if train:
			return outputs, dna_output, dna_noise_outputs, rna_output, rna_noise_outputs
		else:
			return outputs
	

class TranSynAttentionAblation(nn.Module):
	def __init__(
		self,
		dna_module: nn.Module = None,
		rna_module: nn.Module = None,
		bio_module: nn.Module = None,

		hidden_layer: int = None,
		dropout: float = 0.3,
		freeze_source: bool=False,
		mode: str='dna'
	):
		"""
		@Args:
			dna_module, backbone to extract features from dna feature
			rna_module, backbone to extract features from rna feature
			bio_module, backbone to extract features from handcrafted feature

			hidden_layer, hidden layer of final FC layers
			dropout, dropout of final FC layers
			freeze_source, freeze the weight of source or not
		"""
		super().__init__()
		# attribute
		self.hidden_layer = hidden_layer
		self.dropout = dropout
		self.mode = mode

		# init the backbones
		self.dna_module = dna_module
		self.rna_module = rna_module
		self.bio_module = bio_module

		# init the output dims
		self.dna_output_dim = dna_module.output_dim
		self.rna_output_dim = rna_module.output_dim
		self.bio_output_dim = bio_module.output_dim

		self.cg_dna = CrossGateBlock(
			bio_dim=self.bio_output_dim,
			target_dim=self.dna_output_dim,
			proj_dim=max(self.bio_output_dim, self.dna_output_dim),
			dropout=dropout
		)

		self.cg_rna = CrossGateBlock(
			bio_dim=self.bio_output_dim,
			target_dim=self.rna_output_dim,
			proj_dim=max(self.bio_output_dim, self.rna_output_dim),
			dropout=dropout
		)
		
		# ❄ freeze the weight if True
		if freeze_source:
			for param in self.dna_module.parameters():
				param.requires_grad = False
			for param in self.rna_module.parameters():
				param.requires_grad = False
		
		# normalize layers
		self.norm_dna = nn.LayerNorm(self.dna_output_dim)
		self.norm_rna = nn.LayerNorm(self.rna_output_dim)
		self.norm_bio = nn.LayerNorm(self.bio_output_dim)
		if self.mode == 'dna':
			self.final_features_dim = self.dna_output_dim
		elif self.mode == 'rna':
			self.final_features_dim = self.rna_output_dim
		elif self.mode == 'bio':
			self.final_features_dim = self.bio_output_dim
		elif self.mode == 'dna+rna':
			self.final_features_dim = self.dna_output_dim + self.rna_output_dim
		elif self.mode == 'dna+bio':
			self.final_features_dim = self.dna_output_dim + self.bio_output_dim
		elif self.mode == 'rna+bio':
			self.final_features_dim = self.rna_output_dim + self.bio_output_dim
		else:
			self.final_features_dim = (
				self.dna_output_dim + self.rna_output_dim + self.bio_output_dim
			)

		# classifier head
		self.head = nn.Sequential(
			nn.Linear(self.final_features_dim, hidden_layer),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(hidden_layer, 1)
		)

	def forward(self, dna_x, rna_x, bio_x, train=True):

		if self.mode == 'dna':
			if train:
				dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
				dna_output = self.norm_dna(dna_output)
				outputs = self.head(dna_output)
				return outputs, dna_output, dna_noise_outputs
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)  # NOTE inference mode, no noise
				dna_output = self.norm_dna(dna_output)
				outputs = self.head(dna_output)
				return outputs

		if self.mode == 'rna':
			if train:
				rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
				rna_output = self.norm_rna(rna_output)
				outputs = self.head(rna_output)
				return outputs, rna_output, rna_noise_outputs
			else:
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				rna_output = self.norm_rna(rna_output)
				outputs = self.head(rna_output)
				return outputs
		
		if self.mode == 'bio':
			bio_output = self.bio_module(bio_x)
			bio_output = self.norm_bio(bio_output)
			outputs = self.head(bio_output)
			return outputs
		
		if self.mode == 'dna+rna':
			if train:
				dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
				dna_output = self.norm_dna(dna_output)
				rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
				rna_output = self.norm_rna(rna_output)
				# Concatenate
				final_feature = torch.cat((dna_output, rna_output), 1)
				outputs = self.head(final_feature)
				return outputs, dna_output, dna_noise_outputs, rna_output, rna_noise_outputs
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				dna_output = self.norm_dna(dna_output)
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				rna_output = self.norm_rna(rna_output)
				# Concatenate
				final_feature = torch.cat((dna_output, rna_output), 1)
				outputs = self.head(final_feature)
				return outputs
		
		if self.mode == 'dna+bio':
			if train:
				dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
				dna_output = self.norm_dna(dna_output)
				bio_output = self.bio_module(bio_x)
				bio_output = self.norm_bio(bio_output)
				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				final_feature = torch.cat((dna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs, dna_output, dna_noise_outputs
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				dna_output = self.norm_dna(dna_output)
				bio_output = self.bio_module(bio_x)
				bio_output = self.norm_bio(bio_output)
				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				final_feature = torch.cat((dna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs
		
		if self.mode == 'rna+bio':
			if train:
				rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.bio_module(bio_x)
				bio_output = self.norm_bio(bio_output)
				# Fusion
				rna_output = self.cg_rna(bio_output, rna_output)
				final_feature = torch.cat((rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs, rna_output, rna_noise_outputs
			else:
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.bio_module(bio_x)
				bio_output = self.norm_bio(bio_output)
				# Fusion
				rna_output = self.cg_rna(bio_output, rna_output)
				final_feature = torch.cat((rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs

		if self.mode == 'no_fusion': # XXX No Gate Fusion is used
			if train:
				dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
				rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
				bio_output = self.bio_module(bio_x)

				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# XXX Skip Fusion

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs, dna_output, dna_noise_outputs, rna_output, rna_noise_outputs
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				bio_output = self.bio_module(bio_x)

				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# XXX Skip Fusion

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs

		if self.mode in {'no_regularization', 'no_transfer'}: # XXX No Regularization is used 
			dna_output = self.dna_module(dna_x, use_tsrs=False)
			rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
			bio_output = self.bio_module(bio_x)

			# Run through the normalizations
			dna_output = self.norm_dna(dna_output)
			rna_output = self.norm_rna(rna_output)
			bio_output = self.norm_bio(bio_output)

			# Fusion
			dna_output = self.cg_dna(bio_output, dna_output)
			rna_output = self.cg_rna(bio_output, rna_output)

			# Run through the classifier head
			final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
			outputs = self.head(final_feature)
			return outputs
		
		if self.mode == 'no_bss': # XXX No BSS is used
			if train:
				dna_output, dna_noise_outputs = self.dna_module(dna_x, use_tsrs=True)
				rna_output, rna_noise_outputs = self.rna_module(rna_x, backbone=True, use_tsrs=True)
				bio_output = self.bio_module(bio_x)

				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				rna_output = self.cg_rna(bio_output, rna_output)

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs, dna_noise_outputs, rna_noise_outputs
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				bio_output = self.bio_module(bio_x)

				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				rna_output = self.cg_rna(bio_output, rna_output)

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs
		
		if self.mode == 'no_trsr': # XXX No TRSR is used
			if train:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				bio_output = self.bio_module(bio_x)
				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				rna_output = self.cg_rna(bio_output, rna_output)

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs, dna_output, rna_output
			else:
				dna_output = self.dna_module(dna_x, use_tsrs=False)
				rna_output = self.rna_module(rna_x, backbone=True, use_tsrs=False)
				bio_output = self.bio_module(bio_x)
				# Run through the normalizations
				dna_output = self.norm_dna(dna_output)
				rna_output = self.norm_rna(rna_output)
				bio_output = self.norm_bio(bio_output)

				# Fusion
				dna_output = self.cg_dna(bio_output, dna_output)
				rna_output = self.cg_rna(bio_output, rna_output)

				# Run through the classifier head
				final_feature = torch.cat((dna_output, rna_output, bio_output), 1)
				outputs = self.head(final_feature)
				return outputs
		else:
			print(f"[Error] invalid combination (•_•)")
			raise ValueError("invalid combination (Accept dna | rna | bio | dna+bio | rna+bio | dna+rna | no_regularization| no_fusion | no_bss | no_trsr ONLY)")