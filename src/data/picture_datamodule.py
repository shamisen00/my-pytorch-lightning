from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


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
        dataset: Dataset = None,
        data_dir: str = "/workspace/data",
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = False,
        transform: Optional[transforms.Compose] = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((480, 480)),
             transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        self.image_dataset = self.hparams.dataset

        val_size = int(len(self.image_dataset) * self.hparams.val_split)
        test_size = int(len(self.image_dataset) * self.hparams.test_split)
        train_size = len(self.image_dataset) - val_size - test_size
        self.train_set, self.val_set, self.test_set = random_split(
            self.image_dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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
