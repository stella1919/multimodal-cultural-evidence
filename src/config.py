from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_DIR = DATA_DIR / "images"
EMBEDDING_DIR = DATA_DIR / "embeddings"
CACHE_DIR = DATA_DIR / "cache"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULT_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 300
VISUAL_WEIGHT = 0.5
TEXT_WEIGHT = 0.5
TOP_K_NETWORK = 5
MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_KEYWORDS = ["Buddha", "Buddhist", "Bodhisattva", "Avalokiteshvara", "Guanyin", "Tara", "Vajra", "Mandala", "Thangka", "Stupa", "Sutra", "Ritual", "Tibet", "Nepal", "India", "China"]

for _p in (RAW_DIR, PROCESSED_DIR, IMAGE_DIR, EMBEDDING_DIR, CACHE_DIR, FIGURE_DIR, RESULT_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

