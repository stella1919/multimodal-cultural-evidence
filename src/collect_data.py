"""Collect and cache real objects from The Met Collection API."""
import argparse, csv, json, time
from pathlib import Path
from typing import Any
import requests
from tqdm import tqdm
from .config import CACHE_DIR, RAW_DIR, MET_BASE, SEARCH_KEYWORDS
from .utils import get_logger, save_json

LOGGER = get_logger("collection", Path(__file__).resolve().parents[1] / "logs/collection.log")
FIELDS = ["object_id", "title", "object_name", "culture", "period", "date_begin", "date_end", "country", "region", "city", "geography", "medium", "classification", "description", "tags", "image_url", "image_small_url", "object_url", "is_public_domain", "source", "search_keywords", "department", "dynasty", "reign", "credit_line"]

def request_json(url: str, retries: int = 3) -> dict[str, Any] | None:
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("request failed attempt=%s url=%s error=%s", attempt + 1, url, exc)
            time.sleep(1.5 * (attempt + 1))
    return None

def search_ids(keywords: list[str], limit_per_keyword: int | None = None) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for keyword in keywords:
        data = request_json(f"{MET_BASE}/search",) if False else request_json(f"{MET_BASE}/search?q={requests.utils.quote(keyword)}&hasImages=true")
        ids = (data or {}).get("objectIDs") or []
        mapping[keyword] = ids[:limit_per_keyword] if limit_per_keyword else ids
        time.sleep(0.2)
    return mapping

def normalize(obj: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
    return {"object_id": obj.get("objectID"), "title": obj.get("title", ""), "object_name": obj.get("objectName", ""), "culture": obj.get("culture", ""), "period": obj.get("period", ""), "date_begin": obj.get("objectBeginDate"), "date_end": obj.get("objectEndDate"), "country": obj.get("country", ""), "region": obj.get("region", ""), "city": obj.get("city", ""), "geography": obj.get("geographyType", ""), "medium": obj.get("medium", ""), "classification": obj.get("classification", ""), "description": obj.get("creditLine", "") if not obj.get("description") else obj.get("description", ""), "tags": "; ".join(t.get("term", "") for t in (obj.get("tags") or []) if isinstance(t, dict)), "image_url": obj.get("primaryImage", ""), "image_small_url": obj.get("primaryImageSmall", ""), "object_url": obj.get("objectURL", ""), "is_public_domain": bool(obj.get("isPublicDomain", False)), "source": "The Metropolitan Museum of Art Collection API", "search_keywords": "; ".join(keywords), "department": obj.get("department", ""), "dynasty": obj.get("dynasty", ""), "reign": obj.get("reign", ""), "credit_line": obj.get("creditLine", "")}

def collect(keywords: list[str], test_limit: int | None = None, max_objects: int = 1200) -> list[dict[str, Any]]:
    mapping = search_ids(keywords, test_limit)
    reverse: dict[int, list[str]] = {}
    for kw, ids in mapping.items():
        for oid in ids: reverse.setdefault(int(oid), []).append(kw)
    records = []
    for oid, kws in tqdm(list(reverse.items())[:max_objects], desc="Fetching Met objects"):
        cache = CACHE_DIR / f"{oid}.json"
        obj = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else request_json(f"{MET_BASE}/objects/{oid}")
        if obj:
            save_json(obj, cache)
            records.append(normalize(obj, kws))
        time.sleep(0.12)
    out = RAW_DIR / "met_objects_raw.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(records)
    save_json(mapping, RAW_DIR / "search_results.json")
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--test", action="store_true"); parser.add_argument("--max-objects", type=int, default=1200); args = parser.parse_args()
    records = collect(SEARCH_KEYWORDS, test_limit=3 if args.test else None, max_objects=20 if args.test else args.max_objects)
    print(f"Collected {len(records)} records")

