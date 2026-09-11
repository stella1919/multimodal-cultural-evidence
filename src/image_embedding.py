"""Extract real image features with OpenCLIP; no synthetic embedding fallback."""
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from .config import PROCESSED_DIR, IMAGE_DIR, EMBEDDING_DIR, RANDOM_SEED

def extract(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> int:
    import torch, open_clip
    torch.manual_seed(RANDOM_SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval(); df = pd.read_csv(PROCESSED_DIR / "cultural_objects.csv"); vectors=[]; index=[]
    with torch.no_grad():
        for row in tqdm(df.to_dict("records"), desc="Visual embeddings"):
            path = IMAGE_DIR / f"{row['object_id']}.jpg"
            if not path.exists(): continue
            try:
                vec = model.encode_image(preprocess(Image.open(path)).unsqueeze(0).to(device)).float().cpu().numpy()[0]
                vectors.append(vec); index.append({"object_id": row["object_id"], "image_path": str(path)})
            except Exception as exc: print(f"Skipping {path}: {exc}")
    if not vectors: raise RuntimeError("No valid images were available for visual embedding")
    np.save(EMBEDDING_DIR / "visual_embeddings.npy", np.asarray(vectors)); pd.DataFrame(index).to_csv(EMBEDDING_DIR / "visual_embedding_index.csv", index=False)
    return len(vectors)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model", default="ViT-B-32"); p.add_argument("--pretrained", default="laion2b_s34b_b79k"); a=p.parse_args(); print(f"Embedded {extract(a.model, a.pretrained)} images")

