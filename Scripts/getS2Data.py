import os, pathlib, numpy as np, rasterio
from rasterio.windows import Window
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType,
)
from tqdm import tqdm
import numpy.random as npr
from dotenv import load_dotenv
load_dotenv()

# ---------------- Sentinel-Hub auth ----------------
cfg = SHConfig()
if os.getenv("SH_INSTANCE_ID"):
    cfg.instance_id = os.getenv("SH_INSTANCE_ID")

# ---------------- dataset parameters ---------------
YEARS = [2016, 2018, 2019, 2020, 2022, 2023, 2025]
PATCHES_PER_YEAR = 1000
PATCH_PX  = 256
RESOLUTION = 10            # metres per pixel (all bands resampled to 10 m)
DW_FOLDER = pathlib.Path(r"C:\Users\sumit\Documents\GIS\ForestTemp\gis_app\dw_up_rasters")
OUT_DIR   = pathlib.Path("dataset")
DW_FILES  = {y: DW_FOLDER / f"dw_up_{y}.tif" for y in YEARS}

OUT_IMG = OUT_DIR / "images"; OUT_MSK = OUT_DIR / "masks"
OUT_IMG.mkdir(parents=True, exist_ok=True); OUT_MSK.mkdir(parents=True, exist_ok=True)

# ---------------- new 5-band evalscript ------------
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands:["B11","B08","B04","B03","B02"], units:"REFLECTANCE"}],
    output:{bands:5, sampleType:"FLOAT32"}
  };
}
function evaluatePixel(s) {
  return [s.B11, s.B08, s.B04, s.B03, s.B02];
}
"""

# ---------------- helpers --------------------------
def sample_windows(src, n):
    w, h = src.width, src.height
    cols = npr.randint(0, w - PATCH_PX, n)
    rows = npr.randint(0, h - PATCH_PX, n)
    return [Window(c, r, PATCH_PX, PATCH_PX) for c, r in zip(cols, rows)]

def win_to_bbox(src, win):
    b = rasterio.windows.bounds(win, src.transform)
    if src.crs.to_epsg() != 4326:
        b = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *b)
    return BBox(b, crs=CRS.WGS84)

def fetch_s2(bbox, year):
    req = SentinelHubRequest(
        evalscript       = EVALSCRIPT,
        input_data       = [SentinelHubRequest.input_data(
                               DataCollection.SENTINEL2_L2A,
                               time_interval=(f"{year}-01-01", f"{year}-12-31"),
                               mosaicking_order="leastCC")],
        responses        = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox             = bbox,
        size             = (PATCH_PX, PATCH_PX),   # 256×256
        config           = cfg,
    )
    return req.get_data()[0]                        # (256,256,5) float32

# ---------------- main loop ------------------------
uid = 0
for year in YEARS:
    tif_path = DW_FILES[year]
    if not tif_path.exists():
        print(f"{tif_path} missing; skipping {year}")
        continue

    with rasterio.open(tif_path) as src:
        for win in tqdm(sample_windows(src, PATCHES_PER_YEAR),
                        desc=f"{year}", unit="patch"):
            mask = src.read(1, window=win)
            if np.all(mask == 0):            # drop fully-zero tiles
                continue
            try:
                rgbnir = fetch_s2(win_to_bbox(src, win), year)
            except Exception as e:
                print("skip:", e)
                continue

            np.save(OUT_IMG / f"{uid:06d}.npy", rgbnir)  # 5-channel patch
            np.save(OUT_MSK / f"{uid:06d}.npy", mask)
            uid += 1

print(f"Finished. Total pairs: {uid}")