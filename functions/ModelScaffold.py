"""
@description: build the model though a scaffold
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import os, sys
import torch.nn as nn


from functions.PositionTransformer import TransformerEncoder
from functions.ResNet_TSRS import ResNet1D_TSRS, ResidualBlock
from functions.SequenceClassifier import SequenceClassifier
from functions.Bottleneck import Bottleneck
from functions.FullyConnect import FullyConnectFeatureExtractor


def build_backbone(name, seq_len=None, embed_dim=None, output_dim=None):
	if name == "ResNet1D":
		return ResNet1D_TSRS(block=ResidualBlock, layers=[2, 2, 2, 2], input_channels=embed_dim)
	elif name == "PositionTransformer":
		return TransformerEncoder(num_layers=4, d_model=256, num_heads=4, dff=512, rate=0.1)
	elif name == "MLP":
		return FullyConnectFeatureExtractor(in_dim=embed_dim, out_dim=output_dim)
	else:
		raise ValueError(f"Unknown backbone: {name}")


def model_fn(backbone_name="ResNet1D", seq_len=129, embed_dim=768):
	backbone = build_backbone(name=backbone_name, seq_len=seq_len, embed_dim=embed_dim)
	bottleneck = Bottleneck(in_dim=embed_dim) if backbone_name == 'PositionTransformer' else nn.Identity()
	return SequenceClassifier(
		bottleneck=bottleneck,
		backbone=backbone,
		hidden_layer1=backbone.output_dim,
		hidden_layer2=128,
		dropout=0.3,
		num_classes=1,
		is_transformer=(backbone_name == "Transformer")
	)