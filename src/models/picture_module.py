from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchvision.models import densenet121


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
        net,   # パラメータを保存するようににしていると、特定のクラスを渡せない（nn.sequnetial、transformなど）
        scheduler,
        lr: float,
        step_size: int,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        # self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                           in_channels=3, out_channels=3, init_features=32)
        #self.net = densenet121(num_classes = 3)

        # loss function
        self.criterion = torch.nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
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
        #print("x", x[0,1,:,:], "x\n,y", y, "y\n", y_hat, "y_hat\n")
        #print("x", x[0,1,:,:])
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
        loss, targets, y_hat = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        #print("input_validation", batch[0][0])

        return {"loss": loss, "targets": targets, "y_hat": y_hat}

    def test_step(self, batch: Any, batch_idx: int):
        loss, targets = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size)

        return ([optimizer], [scheduler])
