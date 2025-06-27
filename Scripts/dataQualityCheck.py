import random, pathlib, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

DATA_DIR = pathlib.Path("dataset")
IMG_DIR  = DATA_DIR / "images"
MSK_DIR  = DATA_DIR / "masks"
SAMPLES  = 6

dw_colors = np.array([
    [ 51, 160, 255], [  0, 115,   0], [  0, 255,   0],
    [170, 255, 255], [255, 255,   0], [160,  82,  45],
    [220,  20,  60], [210, 180, 140], [255, 255, 255],
]) / 255.0
cmap = ListedColormap(dw_colors)
norm = BoundaryNorm(range(10), cmap.N)

ids = sorted(int(p.stem) for p in IMG_DIR.glob("*.npy"))
chosen = random.sample(ids, min(SAMPLES, len(ids)))

for uid in chosen:
    patch = np.load(IMG_DIR / f"{uid:06d}.npy")   # (H,W,5) float32
    mask  = np.load(MSK_DIR / f"{uid:06d}.npy")   # (H,W)   uint8

    rgb_true  = patch[..., [2, 3, 4]]
    rgb_false = patch[..., [1, 2, 3]]             # NIR-R-G composite

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))

    axs[0].imshow((rgb_true * 255).astype("uint8"))
    axs[0].set_title("True colour")
    axs[0].axis("off")

    axs[1].imshow((rgb_false / rgb_false.max()) ** 0.5)  # gamma boost
    axs[1].set_title("False colour (NIR)")
    axs[1].axis("off")

    axs[2].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    axs[2].set_title("Mask")
    axs[2].axis("off")

    axs[3].imshow((rgb_true * 255).astype("uint8"))
    axs[3].imshow(mask, cmap=cmap, norm=norm, alpha=0.4)
    axs[3].set_title("Overlay")
    axs[3].axis("off")

    fig.tight_layout(); plt.show()
