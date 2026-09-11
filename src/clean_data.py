"""Quality assessment and transparent pilot selection."""
import argparse, json
from pathlib import Path
import pandas as pd
from .config import RAW_DIR, PROCESSED_DIR, RESULT_DIR, FIGURE_DIR, RANDOM_SEED, TARGET_SAMPLE_SIZE
from .cultural_context import infer_context

def quality_and_select(max_records: int = TARGET_SAMPLE_SIZE) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "met_objects_raw.csv")
    df = df.drop_duplicates("object_id").copy()
    df["cultural_context_reason"] = df.apply(lambda r: infer_context(r.to_dict())[1], axis=1)
    df["cultural_context"] = df.apply(lambda r: infer_context(r.to_dict())[0], axis=1)
    df["text_for_embedding"] = df.apply(lambda r: " ".join(f"{label}: {r.get(col, '')}" for label, col in [("Title","title"),("Object type","object_name"),("Culture","culture"),("Period","period"),("Country","country"),("Region","region"),("Medium","medium"),("Classification","classification"),("Tags","tags")] if pd.notna(r.get(col)) and str(r.get(col)).strip()), axis=1)
    df["date_begin"] = pd.to_numeric(df["date_begin"], errors="coerce"); df["date_end"] = pd.to_numeric(df["date_end"], errors="coerce")
    df["mid_year"] = df[["date_begin", "date_end"]].mean(axis=1)
    df.loc[(df["mid_year"] == 0) | (df["mid_year"] < -5000) | (df["mid_year"] > 2100), "mid_year"] = pd.NA
    required = ["title", "image_url", "object_url"]
    usable = df.dropna(subset=required).query("title != '' and image_url != '' and object_url != ''")
    usable = usable.sort_values(["cultural_context", "object_id"], kind="stable").groupby("cultural_context", group_keys=False).head(max_records)
    if len(usable) > max_records: usable = usable.sample(max_records, random_state=RANDOM_SEED).sort_values("object_id")
    usable.to_csv(PROCESSED_DIR / "cultural_objects.csv", index=False)
    ambiguous = df[df["cultural_context"] == "Other"]; ambiguous.to_csv(PROCESSED_DIR / "ambiguous_context.csv", index=False)
    missing = pd.DataFrame({"column": df.columns, "missing_count": [int(df[c].isna().sum() + (df[c] == "").sum()) if df[c].dtype == object else int(df[c].isna().sum()) for c in df.columns]})
    missing["missing_percentage"] = missing.missing_count / max(1, len(df)) * 100
    missing.to_csv(RESULT_DIR / "data_quality_report.csv", index=False)
    report = [f"Total objects: {len(df)}", f"Selected processed objects: {len(usable)}", f"Objects with images: {int(df.image_url.fillna('').ne('').sum())}", f"Objects with public-domain flag: {int(df.is_public_domain.fillna(False).sum())}", f"Objects with cultural metadata: {int(df.culture.fillna('').ne('').sum())}", f"Objects with geographic metadata: {int((df.country.fillna('').ne('') | df.region.fillna('').ne('')).sum())}", f"Objects with temporal metadata: {int(df.mid_year.notna().sum())}", f"Duplicate count removed: {len(pd.read_csv(RAW_DIR / 'met_objects_raw.csv')) - len(df)}", "", missing.to_string(index=False)]
    (RESULT_DIR / "data_quality_report.txt").write_text("\n".join(report), encoding="utf-8")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = missing.set_index("column").missing_percentage.sort_values().plot.barh(figsize=(8, max(4, len(missing)*.22)), color="#6688aa"); ax.set_xlabel("Missing (%)"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "data_missingness.png", dpi=180); plt.close()
        usable.cultural_context.value_counts().reindex(["East_Asia","Himalayan","South_Asia","Other"], fill_value=0).plot.bar(color="#4c78a8", figsize=(7,4)); plt.ylabel("Objects"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "dataset_distribution.png", dpi=180); plt.close()
    except Exception as exc: (RESULT_DIR / "NOT_GENERATED.md").write_text(f"Some figures were not generated: {exc}", encoding="utf-8")
    return usable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--max-records", type=int, default=TARGET_SAMPLE_SIZE); args = parser.parse_args(); print(f"Processed {len(quality_and_select(args.max_records))} records")
