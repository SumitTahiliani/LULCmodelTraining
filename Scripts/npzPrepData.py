import pathlib, json, numpy as np, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

MAP = np.array([0, 1, 2, 255, 3, 4, 5, 255, 255], dtype=np.uint8)

IGNORE_INDEX = 255
JSON_PATH = r"LULCmodelTraining\splits_unordered.json"

# class NPZDataset(Dataset):
#     def __init__(self, npz_path="LULCmodelTraining\lulc_dataset_5band.npz", split="train", augment=True):
#         self.path = npz_path
#         self.ids = json.load(open(JSON_PATH))[split]
#         self.augment = augment and split == "train"
#         self._archive = None  # Lazy-load per worker

#         self.augs = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.RandomBrightnessContrast(p=0.4),
#             A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
#                                rotate_limit=30, border_mode=0, p=0.5),
#             ToTensorV2()
#         ]) if self.augment else None

#     def _get_archive(self):
#         if self._archive is None:
#             self._archive = np.load(self.path, mmap_mode="r")
#         return self._archive

#     def __getitem__(self, idx):
#         uid = f"{self.ids[idx]:06d}"
#         data = self._get_archive()
#         img = data[f"img_{uid}"]
#         raw_mask = data[f"msk_{uid}"]

#         mask = MAP[raw_mask]

#         if self.augment:
#             sample = self.augs(image=img.copy(), mask=mask.copy())
#             x = sample["image"]
#             y = sample["mask"].long()
#         else:
#             x = torch.from_numpy(img.transpose(2, 0, 1)).float()
#             y = torch.from_numpy(mask).long()

#         return x, y

#     def __len__(self):
#         return len(self.ids)
    

class NPZDataset(Dataset):
    def __init__(self, npz_path="LULCmodelTraining/lulc_dataset_5band.npz", split="train", augment=True, rare_classes=[2, 4], oversample_factor=3):
        self.path = npz_path
        self.split = split
        self.augment = augment and split == "train"
        self._archive = None  # Lazy-load per worker

        self.rare_classes = set(rare_classes)
        self.oversample_factor = oversample_factor

        # Load split file
        raw_ids = json.load(open(JSON_PATH))[split]

        if split == "train":
            print("Scanning for rare-class samples...")
            # Load just the masks for class inspection
            archive = np.load(self.path, mmap_mode="r")

            rare_ids = []
            for uid in raw_ids:
                mask = archive[f"msk_{uid:06d}"]
                mapped = MAP[mask]
                if np.any(np.isin(mapped, list(self.rare_classes))):
                    rare_ids.append(uid)

            print(f"Found {len(rare_ids)} rare-class samples out of {len(raw_ids)}")

            # Oversample rare UIDs
            self.ids = raw_ids + rare_ids * (self.oversample_factor - 1)
        else:
            self.ids = raw_ids

        print(f"Final sample count for split '{split}': {len(self.ids)}")

        # Define augmentations
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            ToTensorV2()
        ]) if self.augment else None

    def _get_archive(self):
        if self._archive is None:
            self._archive = np.load(self.path, mmap_mode="r")
        return self._archive

    def __getitem__(self, idx):
        uid = f"{self.ids[idx]:06d}"
        data = self._get_archive()

        # ---------- raw inputs ----------
        img5      = data[f"img_{uid}"]          # (H,W,5)  [B11, B08, B04, B03, B02]
        raw_mask  = data[f"msk_{uid}"]
        mask      = MAP[raw_mask]               # remap 9→3 (+255 ignore)

        # ---------- derive indices ----------
        # unpack for clarity
        B11, B08, B04, B03, _ = [img5[..., i] for i in range(5)]
        eps = 1e-6
        ndvi  = (B08 - B04) / (B08 + B04 + eps)
        mndwi = (B03 - B11) / (B03 + B11 + eps)
        nbi   = (B11 - B04) / (B11 + B04 + eps)

        # stack -> (H,W,8)
        img = np.dstack([img5, ndvi, mndwi, nbi]).astype("float32")

        # ---------- augment or tensor ----------
        if self.augment:
            sample = self.augs(image=img.copy(), mask=mask.copy())
            x = sample["image"]                 # already (8,H,W) tensor from ToTensorV2
            y = sample["mask"].long()
        else:
            x = torch.from_numpy(img.transpose(2, 0, 1)).float()
            y = torch.from_numpy(mask).long()

        return x, y

    def __len__(self):
        return len(self.ids)

