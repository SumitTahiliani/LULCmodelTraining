import pathlib, json, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from npzPrepData import NPZDataset

# ---------------- hyper-params ----------------
EPOCHS       = 100
BATCH_SIZE   = 16
NUM_CLASSES  = 3
IGNORE       = 255
LR           = 1e-3
NUM_WORKERS  = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------
JSON_PATH = r"LULCmodelTraining\class_weights.json"
w_tensor = None
if pathlib.Path(JSON_PATH).exists():
    w_dict = json.load(open(JSON_PATH))
    w_tensor = torch.tensor([w_dict[str(c)] for c in range(NUM_CLASSES)],
                            dtype=torch.float, device=DEVICE)

def build_model():
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=5,
        classes=NUM_CLASSES
    )

def train():
    ds_train = NPZDataset(split="train")
    ds_val   = NPZDataset(split="val", augment=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,  prefetch_factor=4)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE, weight=w_tensor)
    # criterion = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=LR)
    sched= optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    best_iou, patience = 0, 15
    for epoch in range(EPOCHS):
        # ---- train ----
        model.train(); running_loss = 0
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); opt.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(ds_train)

        # ---- validate ----
        model.eval()
        int_sum = torch.zeros(NUM_CLASSES, dtype=torch.float, device=DEVICE)    
        uni_sum = torch.zeros(NUM_CLASSES, dtype=torch.float, device=DEVICE)

        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)                     # (B,H,W)
                valid = (y != IGNORE)
                pred, y = pred[valid], y[valid]
                for cls in range(NUM_CLASSES):
                    pred_c = (pred == cls)
                    y_c    = (y    == cls)
                    inter  = (pred_c & y_c).sum()
                    union  = (pred_c | y_c).sum()
                    int_sum[cls] += inter
                    uni_sum[cls] += union

        # --- derive per-class IoU & mean IoU ---
        eps = 1e-6
        class_iou = (int_sum + eps) / (uni_sum + eps)
        miou = class_iou.mean().item()
        print(f"[{epoch:02d}] | loss: {train_loss} | mIoU: {miou*100:5.2f}%  |  " +
            " ".join(f"{c}:{class_iou[c]*100:4.1f}%" for c in range(NUM_CLASSES)))

        sched.step(miou)

        # ---- early-stop ----
        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), "best2.pt")
            patience = 15
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping"); break

    print("Best mIoU:", best_iou)

# ---------- Windows entry-point ----------
if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    train()
