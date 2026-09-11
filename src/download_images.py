import csv, io
from pathlib import Path
import requests
from PIL import Image
from tqdm import tqdm
from .config import PROCESSED_DIR, IMAGE_DIR, LOG_DIR

def download_images() -> list[int]:
    import pandas as pd
    df = pd.read_csv(PROCESSED_DIR / "cultural_objects.csv")
    errors = []
    for row in tqdm(df.to_dict("records"), desc="Downloading images"):
        oid = str(row["object_id"]); path = IMAGE_DIR / f"{oid}.jpg"
        if path.exists(): continue
        url = row.get("image_small_url") or row.get("image_url")
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB"); img.save(path, "JPEG", quality=92)
        except Exception as exc: errors.append({"object_id": oid, "url": url, "error": str(exc)})
    with (LOG_DIR / "image_download_errors.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["object_id","url","error"]); writer.writeheader(); writer.writerows(errors)
    return errors

if __name__ == "__main__": print(f"Image errors: {len(download_images())}")

