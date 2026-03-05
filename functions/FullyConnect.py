import torch
import torch.nn as nn

class FullyConnectFeatureExtractor(nn.Module):
    def __init__(self, in_dim=148, out_dim=64, dropout=0.3, num_classes=1):
        super().__init__()
        self.output_dim = out_dim
        # 简单的两层全连接 + ReLU
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        """
        x: tensor of shape (B, in_dim)
        returns: tensor of shape (B, out_dim)
        """
        return self.net(x)