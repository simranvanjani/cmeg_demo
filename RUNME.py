# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # ▶ CMEG Content Recommendation Accelerator — RUNME
# MAGIC
# MAGIC **This is the only notebook you run manually.** It:
# MAGIC 1. Validates that the catalog in `config.py` exists.
# MAGIC 2. Creates the demo schema and landing volume (idempotent).
# MAGIC 3. Generates synthetic data into the volume.
# MAGIC 4. Creates the **`cmeg_orchestrator`** workflow (a multi-task job that runs all 7 chapters in order).
# MAGIC 5. Creates the **`cmeg_cleanup`** workflow.
# MAGIC 6. Displays a clickable button to open the orchestrator and run it.
# MAGIC
# MAGIC ## Before you click Run All
# MAGIC
# MAGIC Open **`config.py`** in this folder and set:
# MAGIC - `CATALOG` — **must already exist**. Defaults to `main` (present in every workspace).
# MAGIC - `SCHEMA` — will be created under `CATALOG`. Defaults to `cmeg_demo`.
# MAGIC - `DATA_SCALE` — `small` (default), `medium`, or `large`.
# MAGIC
# MAGIC When done, click **Run all** above.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %md ## Step 1 — Validate environment, generate data

# COMMAND ----------
# MAGIC %run ./_resources/01-generate-data

# COMMAND ----------
# MAGIC %md ## Step 2 — Create the DLT pipeline and workflow jobs

# COMMAND ----------
# MAGIC %run ./_resources/02-create-resources

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Run the demo
# MAGIC
# MAGIC The orchestrator workflow runs all 6 chapters in dependency order (~20-30 min on `small` data).
# MAGIC You can either click the button below to run it end-to-end, or open the individual chapter
# MAGIC notebooks listed under "Chapters" and click through them manually.

# COMMAND ----------
import html as _html
assets = list_assets(spark, OPS_TABLE).toPandas()

# Find the orchestrator job we just created
orch_row = assets[(assets["asset_type"] == "job") & (assets["name"] == "cmeg_orchestrator")]
orch_url = orch_row.iloc[0]["url"] if not orch_row.empty else workspace_url

# Resource summary
resource_html = "".join(
    f"<tr><td style='padding:4px 16px 4px 0;color:#666;'>{_html.escape(r['asset_type'])}</td>"
    f"<td><a href='{_html.escape(r['url'])}' target='_blank'>{_html.escape(r['name'])} &#8599;</a></td></tr>"
    for _, r in assets[assets["chapter"] == 0].iterrows()
)

# Chapter list
CHAPTERS = [
    (2, "DLT Medallion — bronze→silver→gold with expectations + Liquid Clustering", "02_dlt_medallion"),
    (3, "Features and Vectors — Feature Store + Vector Search", "03_features_and_vectors"),
    (4, "Train and Register — two-tower retrieval + LightGBM ranker", "04_train_and_register"),
    (5, "Serve and Explain — chained inference + GenAI explanation", "05_serve_and_explain"),
    (6, "Monitor and Govern — Lakehouse Monitoring + UC tags", "06_monitor_and_govern"),
    (7, "Genie Space — plain-English Q&A for business users", "07_genie_space"),
]
chapters_html = "".join(
    f"<li style='margin:6px 0;'>"
    f"<a href='./{nb}' target='_blank'><b>{nb}</b></a> &mdash; {_html.escape(title)}"
    f"</li>"
    for _, title, nb in CHAPTERS
)

displayHTML(f"""
<div style='font-family:Inter,sans-serif;'>
  <div style='border:2px solid #1f6feb;border-radius:10px;padding:24px;background:#f0f7ff;margin-bottom:16px;'>
    <h2 style='margin:0 0 12px 0;'>&#10003; Accelerator installed</h2>
    <p style='margin:0 0 12px 0;color:#555;'>Catalog <b>{CATALOG}</b>, schema <b>{SCHEMA}</b>. Resources created:</p>
    <table style='border-collapse:collapse;margin-bottom:18px;'>{resource_html}</table>
    <p style='margin:8px 0 0 0;'>
      <a href='{orch_url}' target='_blank'
         style='background:#1f6feb;color:white;padding:12px 22px;border-radius:6px;
                text-decoration:none;font-weight:600;font-size:15px;'>
        &#9654;&nbsp; Run the cmeg_orchestrator workflow
      </a>
    </p>
  </div>

  <div style='border:1px solid #d0d7de;border-radius:8px;padding:16px;'>
    <h3 style='margin:0 0 8px 0;'>Chapters (or open the ones you want to walk through)</h3>
    <ol style='margin:0;padding-left:20px;'>{chapters_html}</ol>
  </div>
</div>
""")
