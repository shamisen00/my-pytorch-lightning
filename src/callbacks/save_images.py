from typing import Optional, Tuple

from lightning.pytorch.callbacks import Callback
import torchvision
from torchvision import transforms
import torch

from src.utils.utils import lab2rgb_torch, de_normalize


class SaveImages(Callback):
    def __init__(
        self,
        lab: bool = True,
        num_samples: int = 16,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0
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
        self.lab = lab

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

    def _to_rgb(self, image):
        image = de_normalize(image)
        image = lab2rgb_torch(image.cpu())
        image = self._to_grid(image)

        return image

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for i in range(len(batch[0])):
            if i % 5 == 0:
                input_grid = batch[0][i]
                prediction_grid = outputs["y_hat"][i]
                target_grid = outputs["targets"][i]

                # # Arrange images vertically

                combined_grid = torch.cat((input_grid.cpu(), prediction_grid.cpu(), target_grid.cpu()), dim=-1)

                to_pil = transforms.ToPILImage()
                combined_grid = to_pil(combined_grid)

                trainer.logger.experiment.log_image(
                    trainer.logger.run_id,
                    image=combined_grid,
                    artifact_file=f"image{trainer.current_epoch}_batchidx{batch_idx}_batchNum{i}.jpg"
                    )
