""" Full assembly of the parts to form the complete network """
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import nn, Tensor
from lightning import LightningModule
from torchmetrics import MeanMetric, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.components.unet_parts import DoubleConv, Down, Up, OutConv


class UnetModule(LightningModule):
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

    def __init__(self):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = UNet(bilinear=True)

        # loss function
        self.criterion = torch.nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

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

        self.ssim(targets, y_hat)
        self.log("val/ssim", self.ssim, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets, "y_hat": y_hat}

    def test_step(self, batch: Any, batch_idx: int):
        loss, targets, y_hat = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets}


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, h):
        x1 = self.inc(h)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.clamp(logits+h, 0, 1)
