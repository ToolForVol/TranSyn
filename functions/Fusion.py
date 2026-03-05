"""
@description: the feature fusion on TranSyn
@author: ye chen
@email: q23101020@stu.ahu.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGateBlock(nn.Module):
    """
    Cross-gated fusion between bio and another modality (e.g., dna or rna).
    Supports either one-way gating (bio -> target) or two-way gating (bio <-> target).
    Inputs:
      - bio: (B, bio_dim)
      - target: (B, target_dim)
    Output:
      - fused_target: (B, target_dim)
      - fused_bio: (B, bio_dim)     # only if bidirectional=True
    """

    def __init__(self,
                 bio_dim,
                 target_dim,
                 proj_dim=None,
                 bidirectional=False,
                 dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        pd = proj_dim or max(bio_dim, target_dim)

        # project both to same proj_dim
        self.bio_proj = nn.Sequential(nn.Linear(bio_dim, pd), nn.ReLU(), nn.Dropout(dropout))
        self.target_proj = nn.Sequential(nn.Linear(target_dim, pd), nn.ReLU(), nn.Dropout(dropout))

        # gating networks
        self.gate_bio2t = nn.Sequential(nn.Linear(pd * 2, pd), nn.ReLU(), nn.Linear(pd, pd), nn.Sigmoid())
        if bidirectional:
            self.gate_t2bio = nn.Sequential(nn.Linear(pd * 2, pd), nn.ReLU(), nn.Linear(pd, pd), nn.Sigmoid())

        # final re-projection to original dims
        self.to_target_dim = nn.Linear(pd, target_dim)
        if bidirectional:
            self.to_bio_dim = nn.Linear(pd, bio_dim)

        self.t_out_norm = nn.LayerNorm(target_dim)
        if bidirectional:
            self.b_out_norm = nn.LayerNorm(bio_dim)

    def forward(self, bio, target):
        """
        bio: (B, bio_dim)
        target: (B, target_dim)
        """
        b = self.bio_proj(bio)        # (B, pd)
        t = self.target_proj(target)  # (B, pd)

        # concat context
        bt = torch.cat([b, t], dim=1)           # (B, 2*pd)
        g_b2t = self.gate_bio2t(bt)             # (B, pd) in (0,1)
        t_mod = g_b2t * t + (1.0 - g_b2t) * b   # gated fusion in proj space
        t_out = self.to_target_dim(t_mod)
        t_out = self.t_out_norm(t_out)

        if not self.bidirectional:  
            return t_out

        # if bidirectional
        g_t2b = self.gate_t2bio(bt)
        b_mod = g_t2b * b + (1.0 - g_t2b) * t
        b_out = self.to_bio_dim(b_mod)
        b_out = self.b_out_norm(b_out)
        return t_out, b_out