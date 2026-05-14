import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class RainNet(nn.Module):
    """基于 ResNet34 的雨强回归网络。"""

    def __init__(
        self,
        in_channels: int = 1,
        dropout_p: float = 0.30,
        pretrained: bool = True,
        torch_home: Optional[str] = None,
    ):
        super().__init__()

        if in_channels not in (1, 3):
            raise ValueError(f"in_channels 只支持 1 或 3，当前为 {in_channels}")

        if torch_home:
            os.environ["TORCH_HOME"] = torch_home

        resnet = self._load_resnet34(pretrained=pretrained)
        self.conv1 = self._build_first_conv(resnet, in_channels)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    @staticmethod
    def _load_resnet34(pretrained: bool):
        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            return models.resnet34(weights=weights)
        except AttributeError:
            return models.resnet34(pretrained=pretrained)

    @staticmethod
    def _build_first_conv(resnet, in_channels: int):
        if in_channels == 3:
            return resnet.conv1

        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
        return conv

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
