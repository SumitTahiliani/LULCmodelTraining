import json, pathlib, numpy as np, tqdm, collections

ROOT = pathlib.Path("dataset")
MSK_DIR = ROOT / "masks"
NUM_CLASSES = 6
IGNORE = 255

json_path = r"LULCmodelTraining\splits_unordered.json"
train_ids = json.load(open(json_path))["train"]
hist = collections.Counter()

for uid in tqdm.tqdm(train_ids, desc="Counting pixels"):
    arr = np.load(MSK_DIR / f"{uid:06d}.npy")
    arr = arr[arr != IGNORE]
    vals, cnts = np.unique(arr, return_counts=True)
    hist.update(dict(zip(vals, cnts)))

total = sum(hist.values())
freq = np.array([hist.get(c, 0) for c in range(NUM_CLASSES)], dtype=float)
freq[freq == 0] = 1e-6
# weights = total / (NUM_CLASSES * freq)                 # 1/f
weights = np.log1p(total / (NUM_CLASSES * freq))
weights = np.clip(weights, 1.0, 25.0)                  # cap extreme

# json.dump({str(k): int(v) for k, v in hist.items()}, open("class_hist.json", "w"),  indent=2)
json.dump({str(i): float(w) for i, w in enumerate(weights)}, open("LULCmodelTraining/class_weights.json", "w"), indent=2)

print("Saved class_hist.json & class_weights.json")
