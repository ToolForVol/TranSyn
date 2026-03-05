"""
@description: a common classifier
"""
import torch.nn as nn
import torch
from typing import Tuple, Optional, List, Dict

class SequenceClassifier(nn.Module):
    def __init__(
        self,
        use_embedding: bool = False,
        vocab_size: int = None,
        embed_dim: int = None,
        bottleneck: nn.Module = None,
        backbone: nn.Module = None,
        hidden_layer1: int = None,
        hidden_layer2: int = None,
        dropout: float = 0.3,
        num_classes: int = 1,
        is_transformer: bool = False,
    ):
        super().__init__()

        self.use_embedding = use_embedding
        self.is_transformer = is_transformer

        if use_embedding:
            assert vocab_size is not None and embed_dim is not None, "vocab_size and embed_dim must be provided when using embedding."
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # <pad>: 0
        else:
            self.embedding = nn.Identity()  

        self.bottleneck = bottleneck 
        self.backbone = backbone
        self._features_dim = hidden_layer1
        self.head = nn.Sequential(
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layer2, num_classes)
        )

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x, return_deep_features=False, use_tsrs=False):
        padding_mask = (x == 0) if self.use_embedding else None

        # ------ 1 进入嵌入层
        x = self.embedding(x)

        # ------ 2 进入瓶颈层
        if self.bottleneck:
            x = self.bottleneck(x)

        # ------ 3 进入主干网络
        # 判断是否为 TransformerBackbone 实例
        if self.is_transformer:
            deep_repr = self.backbone(x, src_key_padding_mask=padding_mask)
        else:
            if use_tsrs:
                deep_repr, noise_outputs = self.backbone(x, train=True)  # 添加噪声正则
            else:
                deep_repr = self.backbone(x)  # 不添加噪声正则

        # ------ 4 进入分类头
        out = self.head(deep_repr)
        if use_tsrs: # 包括TSRS
            return out, deep_repr, noise_outputs
        if return_deep_features: # 包括深度输出
            return out, deep_repr
        else: # 包括预测输出
            return out

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params