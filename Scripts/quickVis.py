import random, numpy as np, torch, matplotlib.pyplot as plt
from npzPrepData import NPZDataset, MAP
from trainUnet import build_model

CHECKPOINT  = r"LULCmodelTraining\best.pt"
SPLIT       = "test"
SAMPLES     = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ds = NPZDataset(split=SPLIT, augment=False)
model = build_model()
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

COLORS = np.array([[ 34,139, 34], [255,255,  0], [220, 20, 60]], dtype=np.uint8)

def colourise(mask):
    out = np.zeros((*mask.shape,3), dtype=np.uint8)
    valid = mask != 255
    out[valid] = COLORS[mask[valid]]
    return out

ids = random.sample(range(len(ds)), SAMPLES)
for idx in ids:
    x, y = ds[idx]
    img  = x.numpy().transpose(1,2,0)
    rgb  = (img[..., [2,3,4]] * 255).astype(np.uint8)

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(DEVICE))
        pred = pred.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        pred = np.where(y.numpy()==255, 255, pred)

    def overlay(back, mask, alpha=0.45):
        color = colourise(mask)
        blend = back.copy()
        m = mask != 255
        blend[m] = (alpha*color[m] + (1-alpha)*back[m]).astype(np.uint8)
        return blend

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(rgb);                 ax[0].set_title("RGB");         ax[0].axis("off")
    ax[1].imshow(overlay(rgb, y.numpy()));  ax[1].set_title("GT overlay"); ax[1].axis("off")
    ax[2].imshow(overlay(rgb, pred));  ax[2].set_title("Pred overlay"); ax[2].axis("off")
    plt.tight_layout(); plt.show()
