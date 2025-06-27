from pathlib import Path
import numpy as np

img_dir = Path("dataset/images")
msk_dir = Path("dataset/masks")

for f in img_dir.glob("*.npy"):
    uid = f.stem
    img = np.load(f)
    msk = np.load(msk_dir / f"{uid}.npy")
    if img.shape[:2] != msk.shape:
        print(f"{uid}: img {img.shape}, mask {msk.shape}")
