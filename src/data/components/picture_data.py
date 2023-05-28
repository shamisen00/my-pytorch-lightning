"""
Dataset class: The Dataset class should focus on loading and preprocessing individual samples. Specifically, it should:

Load the raw data (e.g., images, text, audio) and labels.
Define any preprocessing or transformation steps
that should be applied to the samples (e.g., resizing images, tokenizing text, etc.).
Implement the __len__ method to return the total number of samples in the dataset.
Implement the __getitem__ method to return a single sample (preprocessed data and label) given an index.
"""
from typing import Optional, List

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms
import torch


class PictureDataset(Dataset):
    def __init__(self,
                 mode: str,
                 data_dir: str = "/workspace/data"
                 ) -> None:
        assert mode in {"training", 'validation'}

        self.data_dir = Path(data_dir)

        self.mode = mode
        self.train_transform = [transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5)
                                ]
        self.val_transform = [transforms.ToTensor()]

        self.input_dir = self.data_dir / self.mode / "INPUT_IMAGES"
        self.gt_dir = self.data_dir / self.mode / "GT_IMAGES"

        self.input_paths = list(self.input_dir.glob("*"))
        # self.input_paths = [p for p in input_paths if p.stem.rsplit("_", 1)[1] == "P1.5"]

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        input_img_path = self.input_paths[index]
        f_name = Path(input_img_path.stem.rsplit("_", 1)[0])
        gt_img_path = self.gt_dir / f_name.with_suffix('.jpg')

        train_image = Image.open(input_img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")

        image_transforms = transforms.Compose(self.train_transform)
        if self.mode == "training":
            image_transforms = transforms.Compose(self.train_transform)
        else:
            image_transforms = transforms.Compose(self.val_transform)

        seed = torch.randint(high=100000, size=(1,))
        torch.manual_seed(seed)
        train_image = image_transforms(train_image)
        torch.manual_seed(seed)
        gt_image = image_transforms(gt_image)

        return train_image, gt_image


if __name__ == "__main__":
    _ = PictureDataset(mode="training")
    print(_.input_paths)
    print(next(iter(_))[0][0].shape)
    print(len(_))
