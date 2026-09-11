import pandas as pd
from .config import *
def run():
    df=pd.read_csv(PROCESSED_DIR/"cultural_objects.csv"); valid=df.dropna(subset=["mid_year"]).copy(); valid["century"]=(valid.mid_year//100).astype(int)*100; summary=valid.groupby(["century","cultural_context"],dropna=False).size().reset_index(name="object_count"); summary.to_csv(RESULT_DIR/"temporal_summary.csv",index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; fig,ax=plt.subplots(figsize=(9,5));
    for c,g in summary.groupby("cultural_context"): ax.plot(g.century,g.object_count,marker="o",label=c)
    ax.set(xlabel="Approximate century start",ylabel="Object count"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURE_DIR/"temporal_distribution.png",dpi=200); plt.close(fig); return len(valid)
if __name__ == "__main__": print(f"Temporal records: {run()}")
