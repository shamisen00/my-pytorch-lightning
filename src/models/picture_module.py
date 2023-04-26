from typing import Any, Dict, List

import torch
from torch import nn, Tensor
from lightning import LightningModule
from torchmetrics import MeanMetric, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import densenet121

from src.utils.utils import to_rgb

# from src.models.components.backbone import AlexNet, Identity


class PictureModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        backbone,
        feature_size: int = 16,
        hidden_ch: List[int] = [8, 4]   # パラメータを保存するようににしていると、特定のクラスを渡せない（nn.sequnetial、transformなど）
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        # self.net = net
        # self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                           in_channels=3, out_channels=3, init_features=32)
        # self.net = densenet121(num_classes = 3)

        self.backbone = backbone
        # decoder
        self.layers = []

        feature_size = self.hparams.feature_size
        image_ch = 3
        for ch in self.hparams.hidden_ch:
            out_ch = ch
            self.layers.append(Block(feature_size=feature_size, image_ch=image_ch, out_ch=ch))
            feature_size = out_ch
            image_ch = out_ch

        self.blocks = nn.Sequential(*self.layers)

        self.conv_final = nn.Conv2d(in_channels=out_ch, out_channels=3, kernel_size=1)

        # loss function
        self.criterion = torch.nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        f: Tensor = self.backbone(x)
        images: Tensor = x
        images = x

        d = {"x": x, "f": f}

        d = self.blocks(d)

        x = self.conv_final(d["x"])

        x = x + images
        # print("x", x[0, 1, :, :])
        # x = torch.clamp(x, 0, 1)  # (B, C, H, W)

        return x

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def val_model_step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def training_step(self, batch: Any, batch_idx: int):
        loss, targets, y_hat = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets, y_hat = self.val_model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for i in range(len(batch[0])):
            if True:
                targets[i] = to_rgb(targets[i])
                y_hat[i] = to_rgb(y_hat[i])

        self.ssim(targets, y_hat)
        self.log("val/ssim", self.ssim, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets, "y_hat": y_hat}

    def test_step(self, batch: Any, batch_idx: int):
        loss, targets = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets}

    # def configure_optimizers(self):
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     Examples:
    #         https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    #     """
    #     optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size)
    #     optimizer = self.hparams.optimizer(self.parameters())
    #     scheduler = self.hparams.scheduler(self.parameters())

    #     return ([optimizer], [scheduler])


class Block(nn.Module):
    def __init__(
        self,
        feature_size: int,
        image_ch: int,
        out_ch: int
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(image_ch, out_ch, kernel_size=1)
        self.linear = nn.Linear(feature_size, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d) -> Dict:
        f = d["f"].view(d["f"].size(0), -1)
        f = self.linear(f).view(f.size(0), -1, 1, 1)
        x = self.conv(d["x"]) + f
        x = self.relu(x)

        d["x"] = x
        d["f"] = f

        return d
