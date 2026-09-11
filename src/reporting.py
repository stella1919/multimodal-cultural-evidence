import json, platform, sys
from datetime import datetime, timezone
import pandas as pd
from .config import *
from .utils import runtime_info
def run():
    df=pd.read_csv(PROCESSED_DIR/"cultural_objects.csv"); image_count=sum((IMAGE_DIR/f"{x}.jpg").exists() for x in df.object_id); ctx=df.cultural_context.value_counts().to_dict(); pairs=pd.read_csv(RESULT_DIR/"cross_modal_similarity_pairs.csv") if (RESULT_DIR/"cross_modal_similarity_pairs.csv").exists() else pd.DataFrame()
    summary={"run_date":datetime.now(timezone.utc).isoformat(),"number_raw_objects":int(len(pd.read_csv(RAW_DIR/"met_objects_raw.csv"))),"number_processed_objects":int(len(df)),"number_images":int(image_count),"context_distribution":{str(k):int(v) for k,v in ctx.items()},"visual_model":"OpenCLIP ViT-B-32","text_model":"paraphrase-multilingual-MiniLM-L12-v2","random_seed":RANDOM_SEED,**runtime_info()}; (RESULT_DIR/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if not pairs.empty:
        lookup=df.set_index("object_id")
        candidates=pairs[(pairs.context_1!=pairs.context_2)].copy()
        candidates["case_type"]="multimodal_cross_context_link"
        candidates.loc[(candidates.visual_similarity-candidates.semantic_similarity).abs()>=candidates.alignment_difference.quantile(.9),"case_type"]="visual_semantic_mismatch"
        candidates["title_a"]=candidates.object_1.map(lookup.title); candidates["title_b"]=candidates.object_2.map(lookup.title); candidates["period_a"]=candidates.object_1.map(lookup.period); candidates["period_b"]=candidates.object_2.map(lookup.period); candidates["source_url_a"]=candidates.object_1.map(lookup.object_url); candidates["source_url_b"]=candidates.object_2.map(lookup.object_url); candidates["reason_for_review"]="Cross-context computational similarity; requires qualitative historical interpretation"
        candidates[["object_1","object_2","title_a","title_b","context_1","context_2","period_a","period_b","visual_similarity","semantic_similarity","multimodal_similarity","case_type","reason_for_review","source_url_a","source_url_b"]].head(200).to_csv(RESULT_DIR/"humanities_candidate_cases.csv",index=False)
        top=pairs.sort_values("multimodal_similarity",ascending=False).head(5); lines=["# Results Summary","",f"## Dataset Overview\nThe pilot contains **{len(df)}** processed objects and **{image_count}** locally validated images. Context distribution: {ctx}.","","## RQ1 Findings\nComputational similarity tables and separate visual/semantic embedding spaces provide evidence for comparing visual forms and textual semantics. The output is descriptive and does not establish historical influence.","","## RQ2 Findings\nThe cross-modal alignment table contains %d object pairs. Cases at the tails of visual-semantic disagreement are exported for qualitative review."%len(pairs),"","## RQ3 Exploratory Findings\nTemporal summaries use only valid museum-provided date fields. Any cross-context pattern is a potential association requiring historical corroboration.","","## Representative Cases\nSee `cross_modal_cases.csv` and `humanities_candidate_cases.csv`; these are cases requiring qualitative interpretation.","","## Limitations\nThe pilot is limited by The Met collection/search bias, uneven metadata, missing dates/images, simplified contexts, and the difference between digital records and full historical context.","","## Next Steps\nAdd independent museum collections, entity alignment, expert validation, and provenance-aware qualitative interpretation."]
        (RESULT_DIR/"results_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return summary
if __name__ == "__main__": print(json.dumps(run(),indent=2))
