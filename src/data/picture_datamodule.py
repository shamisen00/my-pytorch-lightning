from typing import Any, Dict, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


from src.data.components.picture_data import PictureDataset


class PictureDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 18,
        pin_memory: bool = False,
        transform: Optional[List] = None,

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_set = PictureDataset('training')
        self.val_set = PictureDataset('validation')
        self.test_set = PictureDataset('validation')

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = PictureDataModule()
    print(_.test_dataloader())
    #print(next(iter(_.test_dataloader())))
    print(_.test_set)
    #print(_.test_set.__getitem__(0))
    # a = PictureDataset(mode="training", lab=True)
    # print(next(iter(a)))
    # print(DataLoader(a, batch_size=16, num_workers=18).dataset[0])
