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

from src.utils.utils import rgb2lab_torch, normalize


class PictureDataset(Dataset):
    def __init__(self,
                 mode: str,
                 lab: bool = True,
                 data_dir: str = "/workspace/data",
                 default_transforms: List = [
                                            transforms.ToTensor(),
                                            transforms.Resize((224, 224), antialias=True),
                                            # transforms.RandomHorizontalFlip(p=0.5)
                                            ],
                 val_transforms: List = [
                                            transforms.ToTensor(),
                                            transforms.Resize((224, 224), antialias=True)
                                            ],
                 ) -> None:
        assert mode in {"train", 'validation'}

        self.data_dir = Path(data_dir)

        self.mode = mode
        self.lab = lab
        self.default_transform = default_transforms
        self.val_transform = val_transforms

        self.input_dir = self.data_dir / self.mode / "input"
        self.gt_dir = self.data_dir / self.mode / "gt"

        self.input_paths = list(self.input_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        seed = torch.randint(high=100000, size=(1,))
        torch.manual_seed(seed)

        input_img_path = self.input_paths[index]
        f_name = Path(input_img_path.stem.rsplit("_", 1)[0])
        gt_img_path = self.gt_dir / f_name.with_suffix('.jpg')

        train_image = Image.open(input_img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")

        if self.mode == "train":
            transform = transforms.Compose(self.default_transform)
            gt_transform = transforms.Compose(self.default_transform)
        else:
            transform = transforms.Compose(self.val_transform)
            gt_transform = transforms.Compose(self.val_transform)

#         resize = transforms.Resize((224, 224), antialias=True)
        if self.lab:
            torch.manual_seed(seed)
            train_image = rgb2lab_torch(transform(train_image))
            torch.manual_seed(seed)
            gt_image = rgb2lab_torch(gt_transform(gt_image))

            train_image = normalize(train_image)
            gt_image = normalize(gt_image)
            # print(train_image.min(), train_image.max())
        else:
            train_image = transform(train_image)
            gt_image = gt_transform(gt_image)

        # print("train\n", train_image)
        # print("train", train_image[1, :, :])
        return train_image, gt_image


if __name__ == "__main__":
    _ = PictureDataset(mode="train")
    print(next(iter(_)))
    print(len(_))
