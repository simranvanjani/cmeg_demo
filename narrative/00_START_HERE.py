# Databricks notebook source
# MAGIC %md
# MAGIC # CMEG Content Recommendation Demo — Start Here
# MAGIC
# MAGIC A 7-chapter walkthrough of a production-representative recommendation pipeline.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC events (synthetic) -> Volume -> DLT (bronze/silver/gold + expectations)
# MAGIC                                  |
# MAGIC                                  v
# MAGIC                       Feature Store + Online Table + Vector Search
# MAGIC                                  |
# MAGIC                                  v
# MAGIC                Two-Tower retrieval + GBT ranker (MLflow @champion)
# MAGIC                                  |
# MAGIC                                  v
# MAGIC          Model Serving (chained) + GenAI explanation (Foundation Model)
# MAGIC                                  |
# MAGIC                                  v
# MAGIC          Lakehouse Monitor + AI/BI dashboard + Genie space
# MAGIC ```

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from cmeg.config import set_widgets, from_widgets
from cmeg.state import list_assets

set_widgets(
    dbutils,
    {
        "catalog_name": "cmeg_demo",
        "bronze_schema": "bronze",
        "silver_schema": "silver",
        "gold_schema": "gold",
        "ml_schema": "ml",
        "ops_schema": "ops",
        "data_scale": "small",
        "serving_endpoint_enabled": "true",
        "genai_model": "databricks-meta-llama-3-3-70b-instruct",
    },
)
cfg = from_widgets(dbutils)
w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md ## Chapters

# COMMAND ----------
CHAPTERS = [
    (1, "Setup and synthetic data", "01_setup_and_data"),
    (2, "DLT medallion (bronze/silver/gold + expectations)", "02_dlt_medallion"),
    (3, "Features and vector index", "03_features_and_vectors"),
    (4, "Train and register models", "04_train_and_register"),
    (5, "Serve and explain", "05_serve_and_explain"),
    (6, "Monitor and govern", "06_monitor_and_govern"),
    (7, "Genie space for business users", "07_genie_space"),
]

# COMMAND ----------
assets = list_assets(spark, cfg.ops_table).toPandas()

import html as _html

rows = []
for chap, title, nb in CHAPTERS:
    chap_assets = assets[assets["chapter"] == chap]
    status = "&#10003;" if not chap_assets.empty else "&#9711;"
    asset_html = ""
    if not chap_assets.empty:
        asset_html = "<ul>" + "".join(
            f"<li><b>{_html.escape(r['asset_type'])}</b>: {_html.escape(r['name'])} "
            f"&mdash; <a href='{_html.escape(r['url'])}' target='_blank'>Open &#8599;</a></li>"
            for _, r in chap_assets.iterrows()
        ) + "</ul>"
    rows.append(
        f"<li style='margin:8px 0;'>{status} <b>Chapter {chap}</b> &mdash; {_html.escape(title)} "
        f"&mdash; <a href='./{nb}' target='_blank'>Open notebook &#8599;</a>{asset_html}</li>"
    )

displayHTML(f"""
<div style='font-family:Inter,sans-serif;border:1px solid #d0d7de;padding:16px;border-radius:8px;'>
  <h2>CMEG Demo &mdash; Table of Contents</h2>
  <p>Catalog: <b>{cfg.catalog}</b> &middot; Scale: <b>{cfg.data_scale}</b></p>
  <ol style='margin:0;padding-left:20px;'>{''.join(rows)}</ol>
</div>
""")
