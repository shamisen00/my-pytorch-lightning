"""
Dataset class: The Dataset class should focus on loading and preprocessing individual samples. Specifically, it should:

Load the raw data (e.g., images, text, audio) and labels.
Define any preprocessing or transformation steps
that should be applied to the samples (e.g., resizing images, tokenizing text, etc.).
Implement the __len__ method to return the total number of samples in the dataset.
Implement the __getitem__ method to return a single sample (preprocessed data and label) given an index.
"""
from typing import Optional

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms


class PictureDataset(Dataset):
    def __init__(self,
                 data_dir: str = "/workspace/data",
                 train_dir: str = "train",
                 gt_dir: str = "gt",
                 transform: Optional[transforms.Compose] =
                 transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize((480, 480)),
                     transforms.Normalize((0.1307,), (0.3081,))]
                 ),):
        self.data_dir = Path(data_dir)
        self.train_img_dir = self.data_dir / train_dir
        self.gt_img_dir = self.data_dir / gt_dir

        self.transform = transform

        self.train_image_paths = list(self.train_img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.train_image_paths)

    def __getitem__(self, index):
        train_img_path = self.train_image_paths[index]
        gt_img_path = self.gt_img_dir / self.train_image_paths[index].name

        train_image = Image.open(train_img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")

        if self.transform:
            train_image = self.transform(train_image)
            gt_image = self.transform(gt_image)

        return train_image, gt_image

if __name__ == "__main__":
    _ = PictureDataset()
