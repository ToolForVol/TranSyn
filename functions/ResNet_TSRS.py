"""
@desc: 在原有ResNet的基础上添加TSRS
"""
import torch
import torch.nn as nn


def conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D_TSRS(nn.Module):
    """
    ResNet1D with TSRS-style noise injection and stability monitoring.

    Args:
        block: residual block class
        layers: list, e.g., [2, 2, 2, 2]
        input_channels: input feature dimension
        noise_std: float, noise magnitude (default 0.01)
        inject_layer: int, which layer to inject noise (0-3)
        measure_from: int, start layer index to compute stability (default = inject_layer)
    """
    def __init__(self, block, layers, input_channels=4, noise_std=0.01, inject_layer=2, measure_from=None):
        super().__init__()
        self.inplanes = 64
        self.noise_std = noise_std
        self.inject_layer = inject_layer
        self.measure_from = inject_layer if measure_from is None else measure_from

        self.conv1 = conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = 512

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    # NOTE 深度特征添加噪声
    # def forward(self, x, train=False):
    #     """
    #     Args:
    #         x: [B, L, C]
    #         train: whether to apply TSRS noise and return stability values
    #     Returns:
    #         if train=False: feature tensor [B, 512]
    #         if train=True: (feature tensor, list of noise stability scalars)
    #     """
    #     # Stem conv
    #     x = x.permute(0, 2, 1)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     # ---- Standard forward ----
    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    #     # Residual blocks
    #     if not train:
    #         for layer in layers:
    #             x = layer(x)
    #         x = self.avgpool(x).squeeze(-1)
    #         return x

    #     # ---- TSRS Mode ----
    #     noise_x = x
    #     noise_outputs = []
    #     device = x.device

    #     for i, layer in enumerate(layers):
    #         if i == self.inject_layer: # 当第二层时，制造噪声
    #             noise = self.noise_std * torch.randn_like(x)
    #             noise_x = x + noise.to(device)

    #         x_out = layer(x) # 无噪声的输出
    #         noise_out = layer(noise_x) # 增加噪声的输出
    #         x, noise_x = x_out, noise_out

    #         if i >= self.measure_from:
    #             stability = torch.norm(noise_x - x, p=2) / (torch.norm(x, p=2) + 1e-8)
    #             noise_outputs.append(stability)  # <- 保留 tensor

    #     x = noise_x # NOTE 保留噪声输出
    #     x = self.avgpool(x).squeeze(-1)
    #     return x, noise_outputs

    # NOTE 深度特征不添加噪声
    def forward(self, x, use_tsrs=False):
        """
        Args:
            x: [B, L, C]
            train: whether to apply TSRS noise and return stability values
        Returns:
            if train=False: feature tensor [B, 512] (clean)
            if train=True: (feature tensor [B, 512] clean, list of noise stability scalars)
        """
        # Stem conv
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        # ---- Standard Mode----
        if not use_tsrs:
            for layer in layers:
                x = layer(x)
            x = self.avgpool(x).squeeze(-1)
            return x

        # ---- TSRS Mode ----
        clean_x = x.clone()  # clean path
        noise_x = x.clone()  # noise path
        noise_outputs = []
        device = x.device

        for i, layer in enumerate(layers):
            if i == self.inject_layer:
                noise = self.noise_std * torch.randn_like(noise_x)
                noise_x = noise_x + noise.to(device)

            # two paths
            clean_x_out = layer(clean_x)
            noise_x_out = layer(noise_x)
            clean_x, noise_x = clean_x_out, noise_x_out

            if i >= self.measure_from:
                stability = torch.norm(noise_x - clean_x, p=2) / (torch.norm(clean_x, p=2) + 1e-8)
                noise_outputs.append(stability)

        # ---- Final outoput ----
        final_feature = self.avgpool(clean_x).squeeze(-1)  # use the clean path output, noise path is only used to calculate the noise
        return final_feature, noise_outputs

    # NOTE 深度特征也添加噪声 Official
    # def forward(self, x, use_tsrs=False):
    #     x = x.permute(0, 2, 1)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    #     # -------- Standard mode (test / eval) --------
    #     if not use_tsrs:
    #         for layer in layers:
    #             x = layer(x)
    #         x = self.avgpool(x).squeeze(-1)
    #         return x

    #     # -------- TSRS mode (train) --------
    #     clean_x = x
    #     noise_x = x
    #     noise_outputs = []

    #     for i, layer in enumerate(layers):
    #         if i == self.inject_layer:
    #             noise = self.noise_std * torch.randn_like(noise_x)
    #             noise_x = clean_x + noise

    #         clean_x = layer(clean_x)
    #         noise_x = layer(noise_x)

    #         if i >= self.measure_from:
    #             stability = torch.norm(noise_x - clean_x, p=2) / (
    #                 torch.norm(clean_x, p=2) + 1e-8
    #             )
    #             noise_outputs.append(stability)

    #     final_feature = self.avgpool(noise_x).squeeze(-1)
    #     return final_feature, noise_outputs
