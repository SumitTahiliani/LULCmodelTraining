##Generate NPZ file from a dataset consiting of npy files

import numpy as np, pathlib, zipfile, tqdm, io, gc

IMG_DIR = pathlib.Path("dataset/images")
MSK_DIR = pathlib.Path("dataset/masks")
OUT_ZIP = "LULCmodelTraining/lulc_dataset_5band.npz"

ids = sorted(int(p.stem) for p in IMG_DIR.glob("*.npy"))

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for uid in tqdm.tqdm(ids, desc="Writing .npz"):
        img = np.load(IMG_DIR / f"{uid:06d}.npy")
        buf = io.BytesIO(); np.save(buf, img); buf.seek(0)
        zf.writestr(f"img_{uid:06d}.npy", buf.read())
        buf.close(); del img

        # ---- mask ----
        msk = np.load(MSK_DIR / f"{uid:06d}.npy")
        buf = io.BytesIO(); np.save(buf, msk); buf.seek(0)
        zf.writestr(f"msk_{uid:06d}.npy", buf.read())
        buf.close(); del msk

        gc.collect()
print("Rebuilt", OUT_ZIP)
