from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool, img_size: int = 384) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),  # slight oversize then center-crop
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


class CarDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        self.transform = transform
        self.num_classes = int(self.df["label_idx"].max()) + 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"])
