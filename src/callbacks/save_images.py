from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class SaveImages(pl.callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 16,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images = self._to_grid(outputs["targets"])
        save_image(images, f"./data/output/image{trainer.current_epoch}.jpg")
    # def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

    #     images, _ = next(iter(DataLoader(trainer.datamodule.val, batch_size=self.num_samples)))
    #     images_flattened = images.view(images.size(0), -1)

    #     # generate images
    #     with torch.no_grad():
    #         pl_module.eval()
    #         images_generated = pl_module(images_flattened.to(pl_module.device))
    #         pl_module.train()

    #     str_title = f"{pl_module.__class__.__name__}_images"
    #     if trainer.current_epoch == 0:
    #         trainer.logger.log_image("original", images=[self._to_grid(images)])
                    
    #     trainer.logger.log_image(str_title + f"{trainer.current_epoch}", images=[self._to_grid(images_generated.reshape(images.shape))])
