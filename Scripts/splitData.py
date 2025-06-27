import pathlib, json, random

ROOT = pathlib.Path("dataset")
IMG_DIR = ROOT / "images"
VAL_FRAC, TEST_FRAC = 0.10, 0.05

# ---- load and shuffle IDs ----
ids = sorted(int(p.stem) for p in IMG_DIR.glob("*.npy"))
random.seed(42)
random.shuffle(ids)

# ---- split by percentage ----
n = len(ids)
n_val = int(n * VAL_FRAC)
n_test = int(n * TEST_FRAC)

val_ids  = ids[:n_val]
test_ids = ids[n_val : n_val + n_test]
train_ids= ids[n_val + n_test :]

splits = {"train": train_ids, "val": val_ids, "test": test_ids}
json.dump(splits, open("splits_unordered.json", "w"), indent=2)
print(f"Saved splits_unorered.json  [train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}]")
