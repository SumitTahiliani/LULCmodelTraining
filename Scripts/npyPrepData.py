import pathlib, json, numpy as np, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LULCDataset(Dataset):
    def __init__(self, root="dataset", split="train", augment=True):
        # ---------- one-time path cache ----------
        split_ids = json.load(open("splits.json"))[split]          # list[int]
        root      = pathlib.Path(root)
        self.img_paths = [root / "images" / f"{uid:06d}.npy" for uid in split_ids]
        self.msk_paths = [root / "masks"  / f"{uid:06d}.npy" for uid in split_ids]

        # ---------- augmentation pipeline ----------
        self.augment = augment and split == "train"
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=30, border_mode=0, p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx], mmap_mode="r").copy()   # (H,W,3) float32
        msk = np.load(self.msk_paths[idx], mmap_mode="r").copy()   # (H,W)   uint8

        if self.augment:
            sample = self.augs(image=img, mask=msk)
            img, msk = sample["image"], sample["mask"].long()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            msk = torch.from_numpy(msk).long()

        return img, msk
