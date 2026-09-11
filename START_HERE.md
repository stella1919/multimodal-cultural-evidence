# Start Here / 使用说明

This is the lightweight GitHub and doctoral-application edition. It contains the complete source code, 300 processed cultural-object records, validated images, embeddings, analysis tables, figures, documentation, notebooks, and the Streamlit interface. The local Python environment, API cache, and demonstration video are intentionally excluded.

这是用于 GitHub 和博士申请材料的轻量最终版，包含完整代码、300 条处理数据、已验证图片、embedding、分析结果、图表、研究文档、Notebook 和 Streamlit 页面。为减小体积，本地 Python 环境、API 缓存和演示视频未包含。

## Browse without installation / 无需安装即可查看

- Read `README.md` for the research overview.
- Open `figures/` for the main visual results.
- Open `results/results_summary.md` for the generated findings.
- Open `docs/` for Chinese/English application materials and methodology.

## Run the interactive demo / 运行交互页面

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Windows activation command:

```powershell
.venv\Scripts\activate
```

Then open `http://localhost:8501` if the browser does not open automatically.

