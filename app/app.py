from pathlib import Path
import zipfile
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/processed/cultural_objects.csv"; EMB=ROOT/"data/embeddings"; IMAGE_DIR=ROOT/"data/images"; IMAGE_ARCHIVE=ROOT/"data/images.zip"

def ensure_demo_images():
    """Expand the compact demo-image archive when the repository is cloned without loose images."""
    if any(IMAGE_DIR.glob("*.jpg")) or not IMAGE_ARCHIVE.exists():
        return
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(IMAGE_ARCHIVE) as archive:
        for member in archive.infolist():
            name=Path(member.filename)
            if member.is_dir() or name.is_absolute() or ".." in name.parts or name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            (IMAGE_DIR/name.name).write_bytes(archive.read(member))

ensure_demo_images()
st.set_page_config(page_title="Cultural Evidence Explorer",layout="wide")
st.markdown("""<style>
.hero{padding:2.2rem 2.5rem;border-radius:24px;background:linear-gradient(135deg,#172554 0%,#1e3a5f 55%,#537b8e 100%);color:white;margin-bottom:1.4rem}.hero h1{font-size:2.5rem;margin:0 0 .45rem;font-weight:700}.hero p{font-size:1.05rem;margin:0;color:#dbeafe}.eyebrow{letter-spacing:.13em;text-transform:uppercase;font-size:.74rem;color:#93c5fd;font-weight:700}.section{font-size:1.25rem;font-weight:700;color:#172554;margin-top:1.4rem}.card{padding:1rem 1.15rem;border:1px solid #dbe4ee;border-radius:16px;background:#fbfdff}.metric{font-size:1.55rem;font-weight:700;color:#172554}.muted{color:#64748b;font-size:.86rem}
</style>
<div class="hero"><div class="eyebrow">Computational Humanities · Pilot Study</div><h1>Multimodal Cultural Evidence Explorer</h1><p>Tracing visual, semantic, temporal, and contextual relationships across museum objects.</p></div>
""",unsafe_allow_html=True)
st.caption("A research prototype for computational discovery and humanities interpretation — not a tool for asserting historical causality.")
if not DATA.exists(): st.error("Run the data pipeline first."); st.stop()
df=pd.read_csv(DATA); ctx=st.sidebar.multiselect("Cultural Context",sorted(df.cultural_context.dropna().unique()),default=sorted(df.cultural_context.dropna().unique())); query=st.sidebar.text_input("Object type / keyword"); period=st.sidebar.text_input("Period contains")
st.sidebar.markdown("### About this pilot")
st.sidebar.caption("300 real objects from The Met Collection API. Embeddings are frozen feature representations; similarity is computational proximity.")
view=df[df.cultural_context.isin(ctx)].copy()
if query: view=view[view.astype(str).apply(lambda c:c.str.contains(query,case=False,na=False)).any(axis=1)]
if period: view=view[view.period.fillna("").str.contains(period,case=False,na=False)]
mc1,mc2,mc3,mc4=st.columns(4)
for col, value, label in [(mc1,len(df),"Objects"),(mc2,df.cultural_context.nunique(),"Contexts"),(mc3,300,"Validated images"),(mc4,"44.8k","Object pairs")]:
    col.markdown(f'<div class="card"><div class="metric">{value}</div><div class="muted">{label}</div></div>',unsafe_allow_html=True)
st.markdown(f'<div class="section">Object browser <span class="muted">· showing {len(view)} of {len(df)}</span></div>',unsafe_allow_html=True)
if view.empty: st.warning("No objects match the filters."); st.stop()
labels=view.apply(lambda r:f"{r.object_id} — {r.title}",axis=1); selected=st.selectbox("Select an object",labels.tolist()); oid=int(selected.split(" — ")[0]); row=df[df.object_id==oid].iloc[0]
c1,c2=st.columns([1,2])
with c1:
    image=ROOT/"data/images"/f"{oid}.jpg"
    if image.exists(): st.image(str(image),use_container_width=True)
with c2:
    st.markdown(f"## {row.title}"); st.markdown(f"`{row.cultural_context}`  ·  **{row.object_name or 'Cultural object'}**"); st.write(f"**Period:** {row.period or 'Unknown'}"); st.write(f"**Culture:** {row.culture or 'Unknown'} · **Region:** {row.country or row.region or 'Unknown'}"); st.write(f"**Medium:** {row.medium or 'Unknown'}"); st.markdown(f"[Open source object page ↗]({row.object_url})")
try:
    idx=df.object_id.tolist().index(oid); outputs=[]
    st.markdown('<div class="section">Similarity lenses</div>',unsafe_allow_html=True)
    tabs=st.tabs(["Visual form","Semantic context","Multimodal evidence"])
    for label, filename, score_name in [("Visual","visual_embeddings.npy","visual_similarity"),("Semantic","text_embeddings.npy","semantic_similarity"),("Multimodal","multimodal_embeddings.npy","multimodal_similarity")]:
        emb=np.load(EMB/filename); emb=emb/np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12); sims=emb@emb[idx]; order=np.argsort(sims)[::-1][1:11]; rec=df.iloc[order].copy(); rec[score_name]=sims[order]; outputs.append((label,rec,score_name))
    for tab,(label,rec,score_name) in zip(tabs,outputs):
        with tab: st.dataframe(rec[["object_id","title","cultural_context","period",score_name,"object_url"]],hide_index=True,use_container_width=True)
    st.markdown('<div class="section">Evidence profile</div>',unsafe_allow_html=True); a,b,c=st.columns(3)
    for col,(label,rec,score_name) in zip((a,b,c),outputs): col.metric(label,f"{rec.iloc[0][score_name]:.3f}","top match")
except Exception as exc: st.info(f"Similarity results unavailable: {exc}")
if {"visual_x","visual_y"}.issubset(df.columns):
    st.markdown('<div class="section">Visual embedding landscape</div>',unsafe_allow_html=True); st.plotly_chart(px.scatter(df,x="visual_x",y="visual_y",color="cultural_context",hover_name="title",hover_data=["object_id","period"],template="plotly_white",height=520),use_container_width=True)
st.info("Similarity indicates computational proximity and should not be interpreted as evidence of direct historical transmission.")
