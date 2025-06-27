import pathlib, json, numpy as np, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

MAP = np.array([255, 0, 255, 255, 1, 255, 2, 255, 255], dtype=np.uint8)

IGNORE_INDEX = 255
JSON_PATH = r"LULCmodelTraining\splits_unordered.json"

class NPZDataset(Dataset):
    def __init__(self, npz_path="lulc_dataset_5band.npz", split="train", augment=True):
        self.path = npz_path
        self.ids = json.load(open(JSON_PATH))[split]
        self.augment = augment and split == "train"
        self._archive = None  # Lazy-load per worker

        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=30, border_mode=0, p=0.5),
            ToTensorV2()
        ]) if self.augment else None

    def _get_archive(self):
        if self._archive is None:
            self._archive = np.load(self.path, mmap_mode="r")
        return self._archive

    def __getitem__(self, idx):
        uid = f"{self.ids[idx]:06d}"
        data = self._get_archive()
        img = data[f"img_{uid}"]
        raw_mask = data[f"msk_{uid}"]

        mask = MAP[raw_mask]

        if self.augment:
            sample = self.augs(image=img.copy(), mask=mask.copy())
            x = sample["image"]
            y = sample["mask"].long()
        else:
            x = torch.from_numpy(img.transpose(2, 0, 1)).float()
            y = torch.from_numpy(mask).long()

        return x, y

    def __len__(self):
        return len(self.ids)
