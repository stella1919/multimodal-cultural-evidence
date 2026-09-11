"""Extract multilingual semantic features with sentence-transformers."""
import numpy as np, pandas as pd
from tqdm import tqdm
from .config import PROCESSED_DIR, EMBEDDING_DIR, RANDOM_SEED

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def extract(model_name: str = MODEL_NAME) -> int:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu"); df = pd.read_csv(PROCESSED_DIR / "cultural_objects.csv")
    texts = df.text_for_embedding.fillna("").tolist(); vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False)
    np.save(EMBEDDING_DIR / "text_embeddings.npy", np.asarray(vectors)); df[["object_id"]].to_csv(EMBEDDING_DIR / "text_embedding_index.csv", index=False)
    return len(vectors)

if __name__ == "__main__": print(f"Embedded {extract()} texts")

