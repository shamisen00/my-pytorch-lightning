from torch import nn, Tensor
from torchvision.models import alexnet, densenet121
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
