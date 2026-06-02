# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # CMEG — Content Recommendation Demo
# MAGIC
# MAGIC <img src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png" style="float: right; height: 100px; margin-left: 16px;"/>
# MAGIC
# MAGIC **A plug-and-play Databricks demo of a production-representative content recommendation system for media/OTT customers.**
# MAGIC
# MAGIC This is the entry notebook. Run it once to install everything. Then follow the chapters in order.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What you'll build
# MAGIC
# MAGIC By the end of this demo your workspace will host a complete content recommendation pipeline:
# MAGIC
# MAGIC 1. **Medallion lakehouse** (DLT bronze → silver → gold) with data-quality expectations and Liquid Clustering.
# MAGIC 2. **Feature Store + Vector Search** — user behavioral features and an index over item embeddings.
# MAGIC 3. **Two-tower retrieval + LightGBM ranker** — the architecture used at YouTube, Spotify, Netflix.
# MAGIC 4. **Real-time serving endpoint** with diversity reranking + a Foundation Model generating "why we recommend this".
# MAGIC 5. **Lakehouse Monitoring** on the inference table + UC tags + sample audit query.
# MAGIC 6. **AI/BI Genie space** so business stakeholders can ask plain-English questions of the gold tables.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐       ┌─────────────────────────────────────────────┐
# MAGIC │ Synthetic    │       │   Databricks Lakehouse (Unity Catalog)      │
# MAGIC │ events       │──────▶│                                             │
# MAGIC │ (users,      │       │  ┌─Bronze─┐   ┌─Silver─┐   ┌─Gold──────┐    │
# MAGIC │  items,      │       │  │ raw    │──▶│ deduped│──▶│ user_360  │    │
# MAGIC │  interactions│       │  │ events │   │ + DQ   │   │ interact. │    │
# MAGIC │  → parquet   │       │  └────────┘   └────────┘   └─────┬─────┘    │
# MAGIC │  → UC Volume)│       │                                  │          │
# MAGIC │              │       │                ┌─────────────────┘          │
# MAGIC │              │       │                ▼                            │
# MAGIC │              │       │  ┌─Features Store──┐  ┌─Vector Search─┐     │
# MAGIC │              │       │  │ user_features   │  │ item_index    │     │
# MAGIC │              │       │  │ item_features   │  │ (synopsis emb)│     │
# MAGIC │              │       │  └────────┬────────┘  └──────┬────────┘     │
# MAGIC │              │       │           │                  │              │
# MAGIC │              │       │           ▼                  ▼              │
# MAGIC │              │       │       ┌─MLflow──────────────────┐           │
# MAGIC │              │       │       │ Two-Tower retrieval     │           │
# MAGIC │              │       │       │ LightGBM ranker @champ. │           │
# MAGIC │              │       │       └────────────┬────────────┘           │
# MAGIC │              │       │                    ▼                        │
# MAGIC │              │       │       ┌─Model Serving (chained)──┐          │
# MAGIC │              │       │       │ retrieval → ranker →     │          │
# MAGIC │              │       │       │ diversity → GenAI explain│          │
# MAGIC │              │       │       └──┬──────────────┬────────┘          │
# MAGIC │              │       │          │              │                   │
# MAGIC │              │       │          ▼              ▼                   │
# MAGIC │              │       │   Inference table   App/API                 │
# MAGIC │              │       │          │                                  │
# MAGIC │              │       │          ▼                                  │
# MAGIC │              │       │   ┌─Lakehouse Monitor─┐                     │
# MAGIC │              │       │   │ drift, prediction │                     │
# MAGIC │              │       │   │ distribution      │                     │
# MAGIC │              │       │   └───────────────────┘                     │
# MAGIC │              │       │                                             │
# MAGIC │              │       │   ┌─Genie space (Gold) ──── for biz users──┐│
# MAGIC │              │       │   │ "Top 10 shows last week?"              ││
# MAGIC │              │       │   └────────────────────────────────────────┘│
# MAGIC │              │       └─────────────────────────────────────────────┘
# MAGIC └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Before you run
# MAGIC
# MAGIC Open `config.py` in this folder and confirm:
# MAGIC - **`CATALOG`** — defaults to `main` (works in every workspace). Change if you have a dedicated catalog.
# MAGIC - **`SCHEMA`** — defaults to `cmeg_demo`. All demo tables will live here.
# MAGIC - **`DATA_SCALE`** — `small` (default, ~15 min), `medium` (~45 min), or `large`.
# MAGIC
# MAGIC When you're ready, click **Run all** on this notebook.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 — Generate synthetic data, create catalog/schema/volume

# COMMAND ----------
# MAGIC %run ./_resources/01-generate-data

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 — Create the DLT pipeline and orchestrator/cleanup jobs

# COMMAND ----------
# MAGIC %run ./_resources/02-create-resources

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Your demo is ready. Pick how to run it:
# MAGIC
# MAGIC | Path | Best for |
# MAGIC | --- | --- |
# MAGIC | **Click through chapters manually** (recommended for SE-led demos) | You want to explain each step to the customer |
# MAGIC | **Run the orchestrator job end-to-end** (~20-30 min on small data) | You want a hands-off smoke test or unattended run |

# COMMAND ----------
# Render the live TOC (also re-renderable later by re-running this last cell)
import html as _html
assets = list_assets(spark, OPS_TABLE).toPandas()

CHAPTERS = [
    (1, "DLT Medallion (bronze → silver → gold with expectations)", "01-DLT-Medallion"),
    (2, "Features and Vectors (Feature Store + Vector Search)", "02-Features-and-Vectors"),
    (3, "Train and Register (Two-Tower + LightGBM, MLflow + Registry)", "03-Train-and-Register"),
    (4, "Serve and Explain (chained serving + GenAI 'why we recommend')", "04-Serve-and-Explain"),
    (5, "Monitor and Govern (Lakehouse Monitoring + UC tags)", "05-Monitor-and-Govern"),
    (6, "Genie Space (plain-English questions for business users)", "06-Genie-Space"),
]

# Resource summary block
resource_rows = []
for _, r in assets[assets["chapter"] == 0].iterrows():
    resource_rows.append(
        f"<tr><td style='padding:4px 12px 4px 0;color:#666;'>{_html.escape(r['asset_type'])}</td>"
        f"<td><a href='{_html.escape(r['url'])}' target='_blank'>{_html.escape(r['name'])} &#8599;</a></td></tr>"
    )

# Chapter list
chapter_rows = []
for ch, title, nb in CHAPTERS:
    chap_assets = assets[assets["chapter"] == ch]
    status = "&#10003;" if not chap_assets.empty else "&#9711;"
    chapter_rows.append(
        f"<li style='margin:8px 0;'>{status} <b>Chapter {ch}</b> &mdash; {_html.escape(title)} "
        f"&mdash; <a href='./{nb}' target='_blank'>Open &#8599;</a></li>"
    )

orchestrator_url = ""
orch = assets[(assets["asset_type"] == "job") & (assets["name"] == "cmeg_orchestrator")]
if not orch.empty:
    orchestrator_url = orch.iloc[0]["url"]

displayHTML(f"""
<div style='font-family:Inter,sans-serif;'>
  <div style='border:2px solid #1f6feb;border-radius:10px;padding:20px;background:#f0f7ff;margin-bottom:16px;'>
    <h2 style='margin:0 0 12px 0;'>&#10003; Install complete</h2>
    <table style='border-collapse:collapse;'>{''.join(resource_rows)}</table>
    {f"<p style='margin:16px 0 0 0;'><a href='{orchestrator_url}' target='_blank' style='background:#1f6feb;color:white;padding:10px 18px;border-radius:6px;text-decoration:none;font-weight:600;'>Run all chapters end-to-end &rarr;</a></p>" if orchestrator_url else ""}
  </div>
  <div style='border:1px solid #d0d7de;border-radius:8px;padding:16px;'>
    <h3 style='margin:0 0 8px 0;'>Chapters</h3>
    <ol style='margin:0;padding-left:20px;'>{''.join(chapter_rows)}</ol>
  </div>
</div>
""")
