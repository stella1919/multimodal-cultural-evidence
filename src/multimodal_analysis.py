"""Transparent similarity, alignment, clustering and multimodal representation analyses."""
import itertools, json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .config import *
from .utils import l2_normalize, save_json

def reduce_plot(X, df, name, xcol, ycol):
    try:
        if os.environ.get("MCE_USE_PCA"):
            raise ImportError("MCE_USE_PCA requested")
        import umap; coords=umap.UMAP(random_state=RANDOM_SEED, n_neighbors=min(15,max(2,len(X)-1))).fit_transform(X)
    except Exception:
        coords=PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    df[xcol]=coords[:,0]; df[ycol]=coords[:,1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(8,6))
    for ctx, g in df.groupby("cultural_context"):
        ax.scatter(g[xcol],g[ycol],s=22,alpha=.78,label=ctx)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURE_DIR/f"{name}.png",dpi=200); plt.close(fig)

def all_pairs(ids, visual, text, df):
    vm=l2_normalize(visual); tm=l2_normalize(text); rows=[]
    lookup=df.set_index("object_id")
    for i,j in itertools.combinations(range(len(ids)),2):
        rows.append({"object_1":ids[i],"object_2":ids[j],"visual_similarity":float(vm[i]@vm[j]),"semantic_similarity":float(tm[i]@tm[j])})
    out=pd.DataFrame(rows); out["context_1"]=out.object_1.map(lookup.cultural_context); out["context_2"]=out.object_2.map(lookup.cultural_context); out["same_context"]=out.context_1==out.context_2; out["alignment_difference"]=(out.visual_similarity-out.semantic_similarity).abs(); return out

def run() -> dict:
    df=pd.read_csv(PROCESSED_DIR/"cultural_objects.csv"); vi=pd.read_csv(EMBEDDING_DIR/"visual_embedding_index.csv"); ti=pd.read_csv(EMBEDDING_DIR/"text_embedding_index.csv")
    v=np.load(EMBEDDING_DIR/"visual_embeddings.npy"); t=np.load(EMBEDDING_DIR/"text_embeddings.npy"); vm=l2_normalize(v); tm=l2_normalize(t)
    common=set(vi.object_id)&set(ti.object_id); order=[x for x in df.object_id if x in common]; vi_idx={x:i for i,x in enumerate(vi.object_id)}; ti_idx={x:i for i,x in enumerate(ti.object_id)}; v=np.array([v[vi_idx[x]] for x in order]); t=np.array([t[ti_idx[x]] for x in order]); vm=l2_normalize(v); tm=l2_normalize(t); sub=df.set_index("object_id").loc[order].reset_index()
    coords=sub.copy(); reduce_plot(vm,coords,"visual_embedding_space","visual_x","visual_y"); sub=sub.merge(coords[["object_id","visual_x","visual_y"]],on="object_id"); coords=sub.copy(); reduce_plot(tm,coords,"semantic_embedding_space","semantic_x","semantic_y"); sub=sub.merge(coords[["object_id","semantic_x","semantic_y"]],on="object_id"); sub.to_csv(PROCESSED_DIR/"cultural_objects.csv",index=False)
    multi=l2_normalize(np.hstack([VISUAL_WEIGHT*vm,TEXT_WEIGHT*tm])); np.save(EMBEDDING_DIR/"multimodal_embeddings.npy",multi); reduce_plot(multi,sub.copy(),"multimodal_embedding_space","multimodal_x","multimodal_y")
    pairs=all_pairs(order,v,t,sub); pairs["multimodal_similarity"]=[float(multi[i]@multi[j]) for i,j in itertools.combinations(range(len(order)),2)]; pairs.to_csv(RESULT_DIR/"cross_modal_similarity_pairs.csv",index=False)
    pairs.sort_values("visual_similarity",ascending=False).head(100).to_csv(RESULT_DIR/"top_visual_similar_pairs.csv",index=False); pairs.sort_values("semantic_similarity",ascending=False).head(100).to_csv(RESULT_DIR/"top_semantic_similar_pairs.csv",index=False); pairs.sort_values("multimodal_similarity",ascending=False).head(100).to_csv(RESULT_DIR/"top_multimodal_similar_pairs.csv",index=False)
    threshold_v=pairs.visual_similarity.quantile(.9); threshold_s=pairs.semantic_similarity.quantile(.9); cases=pairs[((pairs.visual_similarity>=threshold_v)&(pairs.semantic_similarity<=pairs.semantic_similarity.quantile(.5)))|((pairs.semantic_similarity>=threshold_s)&(pairs.visual_similarity<=pairs.visual_similarity.quantile(.5)))].copy(); cases.to_csv(RESULT_DIR/"cross_modal_cases.csv",index=False)
    sample=pairs.sample(min(5000,len(pairs)),random_state=RANDOM_SEED); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(sample.visual_similarity,sample.semantic_similarity,c=sample.same_context.map({True:"#999999",False:"#d95f02"}),s=8,alpha=.35); ax.set(xlabel="Visual similarity",ylabel="Semantic similarity"); fig.tight_layout(); fig.savefig(FIGURE_DIR/"cross_modal_alignment.png",dpi=200); plt.close(fig)
    evals=[]
    for k in range(2,min(8,len(multi)-1)+1):
        labels=KMeans(n_clusters=k,random_state=RANDOM_SEED,n_init=20).fit_predict(multi); evals.append({"k":k,"silhouette_score":float(silhouette_score(multi,labels))})
    edf=pd.DataFrame(evals); edf.to_csv(RESULT_DIR/"clustering_evaluation.csv",index=False); best=int(edf.loc[edf.silhouette_score.idxmax(),"k"]); labels=KMeans(n_clusters=best,random_state=RANDOM_SEED,n_init=20).fit_predict(multi); sub["exploratory_cluster"]=labels; pd.crosstab(sub.exploratory_cluster,sub.cultural_context).to_csv(RESULT_DIR/"cluster_context_table.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; edf.plot(x="k",y="silhouette_score",marker="o",legend=False,figsize=(7,4)); plt.ylabel("Silhouette score"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"clustering_silhouette.png",dpi=200); plt.close()
    return {"objects":len(order),"best_k":best,"pairs":len(pairs),"cross_context_pairs":int((~pairs.same_context).sum()),"visual_model":"OpenCLIP ViT-B-32","text_model":"paraphrase-multilingual-MiniLM-L12-v2"}

if __name__ == "__main__": print(json.dumps(run(),indent=2))
