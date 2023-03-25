from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import webdataset as wds

from src.utils.utils import create_wds


class ImageGenerationDataModule(LightningDataModule):
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
        shard_path: str = "data/shards",
        batch_size: int = 32,
        num_workers: int = 1,
        shard_size: int = 20,
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
    
    def prepare_data(self):
        """Download data if needed.
        """
        create_wds(self.hparams.shard_path, self.hparams.shard_size)

    def setup(self, stage=None):

        shards_list = [str(path) for path in self.hparams.shard_path.glob('*.tar')]

        preproc = transforms.Compose([
            transforms.ToTensor()
        ])

        # TODO:loadも分散処理できるらしい https://webdataset.github.io/webdataset/multinode/
        # https://qiita.com/tttamaki/items/dd0c4b14c6f699d28377
        # TODO:__len()__が定義されていないせいでmulti-gpuでの学習がかなり面倒になる。
        # torchdataのwebdatasetを使うといいのかもしれないけど一旦保留。https://pytorch.org/data/main/generated/torchdata.datapipes.iter.WebDataset.html
        self.dataset = wds.WebDataset(shards_list
                                      ).decode(
                                        "rgb"
                                      ).to_tuple(
                                        "gt.jpg", "train.jpg"
                                      ).map_tuple(
                                        self.hparams.transform, preproc
                                      )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
