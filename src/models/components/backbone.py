from torch import nn, Tensor
from torchvision.models import alexnet, efficientnet_b0, efficientnet_b1, efficientnet_b3, efficientnet_b7
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B3_Weights, EfficientNet_B7_Weights
import torch


class AlexNet(nn.Module):
    def __init__(
        self,
        out_ch: int
    ) -> None:
        super().__init__()
        self.out_ch = out_ch

        self.backbone: nn.Module = alexnet().features

        # 最終畳み込み層のchを変更
        last_conv_layer = self.backbone[10]
        new_last_conv_layer = nn.Conv2d(
            last_conv_layer.in_channels,
            self.out_ch, kernel_size=last_conv_layer.kernel_size,
            stride=last_conv_layer.stride, padding=last_conv_layer.padding
            )
        self.backbone[10] = new_last_conv_layer

        self.avg = nn.AdaptiveAvgPool2d(1)  # GlobalAveragePooling
        self.relu = nn.ReLU(inplace=True)  # 必要？

        # encoder
        self.backbone = nn.Sequential(*self.backbone, self.avg, self.relu)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.backbone(x)

        return x


class EfficientNetB3(nn.Module):
    def __init__(self, out_ch, weights: EfficientNet_B3_Weights = EfficientNet_B3_Weights):
        super().__init__()
        self.out_ch = out_ch
        self.weights = weights

        self.backbone: nn.Module = efficientnet_b3(weights=self.weights)

        # 最終畳み込み層のchを変更
        last_layer = self.backbone.classifier[1]
        new_last_layer = nn.Linear(
            last_layer.in_features,
            out_features=self.out_ch)
        self.backbone.classifier[1] = new_last_layer

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(self.backbone, self.relu)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.backbone(x)

        return x


class EfficientNetB7(nn.Module):
    def __init__(self, out_ch, weights: EfficientNet_B7_Weights = EfficientNet_B7_Weights):
        super().__init__()
        self.out_ch = out_ch
        self.weights = weights

        self.backbone: nn.Module = efficientnet_b7(weights=self.weights)

        # 最終畳み込み層のchを変更
        last_layer = self.backbone.classifier[1]
        new_last_layer = nn.Linear(
            last_layer.in_features,
            out_features=self.out_ch)
        self.backbone.classifier[1] = new_last_layer

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(self.backbone, self.relu)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.backbone(x)

        return x

class Identity(nn.Module):
    def __init__(
        self,
        out_ch: int
    ) -> None:
        super().__init__()
        self.out_ch = out_ch

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = torch.zeros([x.size(0), self.out_ch], device=torch.device('cuda'))

        return x
