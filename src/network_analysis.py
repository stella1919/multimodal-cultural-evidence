import numpy as np, pandas as pd, networkx as nx
from .config import *
def run():
    df=pd.read_csv(PROCESSED_DIR/"cultural_objects.csv"); emb=np.load(EMBEDDING_DIR/"multimodal_embeddings.npy"); ids=df.object_id.tolist(); sim=emb@emb.T; G=nx.Graph();
    for _,r in df.iterrows(): G.add_node(r.object_id,title=r.title,culture=r.culture,cultural_context=r.cultural_context,period=r.period,country=r.country)
    for i,oid in enumerate(ids):
        for j in np.argsort(sim[i])[::-1][1:TOP_K_NETWORK+1]:
            if oid != ids[j]: G.add_edge(oid,ids[j],multimodal_similarity=float(sim[i,j]))
    edges=pd.DataFrame([{"object_id_1":a,"object_id_2":b,**d} for a,b,d in G.edges(data=True)]); edges.to_csv(RESULT_DIR/"network_edges.csv",index=False); nodes=pd.DataFrame([{"object_id":n,**d} for n,d in G.nodes(data=True)]); nodes.to_csv(RESULT_DIR/"network_nodes.csv",index=False); cent=pd.DataFrame({"object_id":list(G),"degree_centrality":[nx.degree_centrality(G)[n] for n in G],"betweenness_centrality":[nx.betweenness_centrality(G,normalized=True,seed=RANDOM_SEED)[n] for n in G]}); cent.to_csv(RESULT_DIR/"network_centrality.csv",index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; top=cent.nlargest(min(40,len(cent)),"betweenness_centrality").object_id; H=G.subgraph(top); pos=nx.spring_layout(H,seed=RANDOM_SEED); colors=[{"East_Asia":"#4c78a8","Himalayan":"#f58518","South_Asia":"#54a24b","Other":"#999999"}.get(G.nodes[n].get("cultural_context"),"#999999") for n in H]; plt.figure(figsize=(10,8)); nx.draw_networkx(H,pos,node_size=70,node_color=colors,with_labels=False,edge_color="#bbbbbb",alpha=.75); plt.axis("off"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"cultural_similarity_network.png",dpi=200); plt.close(); return len(G),len(edges)
if __name__ == "__main__": print(run())
