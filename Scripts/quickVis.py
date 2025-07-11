# import random, numpy as np, torch, matplotlib.pyplot as plt
# from npzPrepData import NPZDataset
# from trainUnet import build_model

# CHECKPOINT  = r"LULCmodelTraining\best2.pt"
# SPLIT       = "test"
# SAMPLES     = 15

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ds = NPZDataset(split=SPLIT, augment=False)
# model = build_model()
# model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
# model.to(DEVICE).eval()

# # --- add / replace ----------------------------------------------------------
# # Dynamic World class names, just for reference
# CLASS_NAMES = [
#     'Water', 'Trees', 'Grass', 'Flooded Vegetation', 'Crops',
#     'Shrub & Scrub', 'Built‑up', 'Bare Ground', 'Snow & Ice'
# ]

# COLORS = np.array([
#     [ 65, 155, 223],   # 0  Water           (#419bdf)
#     [ 57, 125,  73],   # 1  Trees           (#397d49)
#     [136, 176,  83],   # 2  Grass           (#88b053)
#     [122, 135, 198],   # 3  Flooded Veg.    (#7a87c6)
#     [228, 150,  53],   # 4  Crops           (#e49635)
#     [223, 195,  90],   # 5  Shrub & Scrub   (#dfc35a)
#     [196,  40,  27],   # 6  Built‑up        (#c4281b)
#     [165, 155, 143],   # 7  Bare Ground     (#a59b8f)
#     [179, 159, 225],   # 8  Snow & Ice      (#b39fe1)
# ], dtype=np.uint8)
# # ---------------------------------------------------------------------------

# def colourise(mask):
#     """Map label integers → RGB colours.
#        Works for 0‑8, ignores 255."""
#     out = np.zeros((*mask.shape, 3), dtype=np.uint8)
#     valid = (mask != 255) & (mask < len(COLORS))   # guard against stray values
#     out[valid] = COLORS[mask[valid]]
#     return out

# ids = random.sample(range(len(ds)), SAMPLES)
# for idx in ids:
#     x, y = ds[idx]
#     img  = x.numpy().transpose(1,2,0)
#     rgb  = (img[..., [2,3,4]] * 255).astype(np.uint8)

#     with torch.no_grad():
#         pred = model(x.unsqueeze(0).to(DEVICE))
#         pred = pred.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
#         pred = np.where(y.numpy()==255, 255, pred)

#     def overlay(back, mask, alpha=0.45):
#         color = colourise(mask)
#         blend = back.copy()
#         m = mask != 255
#         blend[m] = (alpha*color[m] + (1-alpha)*back[m]).astype(np.uint8)
#         return blend

#     fig, ax = plt.subplots(1, 3, figsize=(12,4))
#     ax[0].imshow(rgb);                 ax[0].set_title("RGB");         ax[0].axis("off")
#     ax[1].imshow(overlay(rgb, y.numpy()));  ax[1].set_title("GT overlay"); ax[1].axis("off")
#     ax[2].imshow(overlay(rgb, pred));  ax[2].set_title("Pred overlay"); ax[2].axis("off")
#     plt.tight_layout(); plt.show()

import os, glob, random, numpy as np, torch, matplotlib.pyplot as plt
from trainUnet import build_model  # Keep your model builder

# Constants
CHECKPOINT = r"LULCmodelTraining\best2.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES  = 15
NPY_FOLDER = "."  # or replace with path to your .npy files

# Dynamic World class colours & names
CLASS_NAMES = [
    'Water', 'Trees', 'Grass', 'Flooded Vegetation', 'Crops',
    'Shrub & Scrub', 'Built‑up', 'Bare Ground', 'Snow & Ice'
]

COLORS = np.array([
    [ 65, 155, 223], [ 57, 125,  73], [136, 176,  83],
    [122, 135, 198], [228, 150,  53], [223, 195,  90],
    [196,  40,  27], [165, 155, 143], [179, 159, 225],
], dtype=np.uint8)

def colourise(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask != 255) & (mask < len(COLORS))
    out[valid] = COLORS[mask[valid]]
    return out

def overlay(back, mask, alpha=0.45):
    color = colourise(mask)
    blend = back.copy()
    m = mask != 255
    blend[m] = (alpha*color[m] + (1-alpha)*back[m]).astype(np.uint8)
    return blend

# Load model
model = build_model()
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

# List .npy files
npy_files = sorted(glob.glob(os.path.join(NPY_FOLDER, "*.npy")))
sample_files = random.sample(npy_files, min(N_SAMPLES, len(npy_files)))

for path in sample_files:
    arr = np.load(path)  # shape: (10, H, W)

    # Normalize and convert to tensor
    x = torch.from_numpy(arr).float() / 10000.0  # assume Sentinel-2 scaling
    x = x.clamp(0, 1)  # optional but often useful
    x = x.unsqueeze(0).to(DEVICE)  # (1, 10, H, W)

    with torch.no_grad():
        pred = model(x)
        pred = pred.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    # Convert to RGB using B04 (red), B03 (green), B02 (blue)
    rgb = (arr[[3, 2, 1]] / 10000.0 * 255).clip(0, 255).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(rgb);           ax[0].set_title("RGB");         ax[0].axis("off")
    ax[1].imshow(overlay(rgb, pred)); ax[1].set_title("Prediction"); ax[1].axis("off")
    plt.tight_layout(); plt.show()

