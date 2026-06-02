# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 1 of 6 — The Lakehouse Medallion with DLT
# MAGIC
# MAGIC > 🕐 **5 min to read · 3 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - How to ingest streaming events into a lakehouse with **Auto Loader** (schema evolution included)
# MAGIC - How to enforce data quality with **DLT expectations** (`@dlt.expect_or_drop`)
# MAGIC - How **Liquid Clustering** replaces partitioning + Z-ORDER for high-cardinality join keys
# MAGIC - How Unity Catalog tracks **lineage automatically** so you can see exactly what feeds your ML model
# MAGIC
# MAGIC ## Where we are in the demo
# MAGIC
# MAGIC ```
# MAGIC   ┌─Volume (parquet from RUNME)
# MAGIC ┌►│   users / items / interactions
# MAGIC │ └───────────────┬────────────────
# MAGIC │                 │ Auto Loader (this chapter)
# MAGIC │                 ▼
# MAGIC │   ┌─Bronze─┐ ┌─Silver─┐ ┌─Gold──────┐
# MAGIC │   │ raw    │►│ deduped│►│ user_360  │── feeds chapter 2 (features)
# MAGIC │   │ events │ │ + DQ   │ │ interact. │
# MAGIC │   └────────┘ └────────┘ └───────────┘
# MAGIC └──── we are HERE ────────────────────
# MAGIC ```

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 of 3 — Find the pre-created DLT pipeline
# MAGIC
# MAGIC `RUNME` already created the pipeline (`cmeg_dlt_pipeline`) and pointed it at our pipeline
# MAGIC definition file `_resources/_dlt_pipeline.py`. **Open that file in a new tab** to see what
# MAGIC the bronze / silver / gold tables look like in DLT's declarative Python.
# MAGIC
# MAGIC Below, we just look it up via the SDK so we can trigger an update.

# COMMAND ----------
pipelines = [p for p in w.pipelines.list_pipelines(filter="name LIKE 'cmeg_dlt_pipeline'")]
assert pipelines, "Pipeline not found. Run RUNME.py first."
pipeline = pipelines[0]
pipeline_url = format_asset_url(workspace_url, "pipeline", pipeline.pipeline_id)
displayHTML(f"<p>✓ Pipeline found: <a href='{pipeline_url}' target='_blank'>cmeg_dlt_pipeline &#8599;</a></p>")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 3 — Trigger an update and watch it run
# MAGIC
# MAGIC We kick off the pipeline with `start_update()` and poll until it finishes (or fails).
# MAGIC
# MAGIC **🔍 What to look for while it runs:** Open the pipeline link above in a new tab. You'll see
# MAGIC the **pipeline graph** light up node-by-node — first bronze tables, then silver (with green
# MAGIC "data quality" badges), then gold. Each node shows row counts, processing time, and any
# MAGIC expectations that were dropped.

# COMMAND ----------
import time
w.pipelines.start_update(pipeline.pipeline_id)
print("Pipeline update started. Polling every 20s, max 30 min total...")
deadline = time.time() + 60 * 30
while time.time() < deadline:
    state = w.pipelines.get(pipeline.pipeline_id).state
    print(f"  state: {state}")
    if str(state).endswith("IDLE") or str(state).endswith("FAILED"):
        break
    time.sleep(20)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 3 — Inspect what got built
# MAGIC
# MAGIC The pipeline produced 8 tables. Let's list them, then sample the most important one
# MAGIC (`gold_interactions` — events enriched with user country/age and item genre/year).

# COMMAND ----------
display(spark.sql(f"""
    SELECT table_name, table_type, comment
    FROM {CATALOG}.information_schema.tables
    WHERE table_schema = '{SCHEMA}'
      AND (table_name LIKE 'bronze\\_%' ESCAPE '\\'
           OR table_name LIKE 'silver\\_%' ESCAPE '\\'
           OR table_name LIKE 'gold\\_%' ESCAPE '\\')
    ORDER BY table_name
"""))

# COMMAND ----------
display(spark.table(FQ("gold_interactions")).limit(20))

# COMMAND ----------
# MAGIC %md
# MAGIC ### 🔍 Try this in the UI
# MAGIC
# MAGIC 1. **Open the pipeline link above.** Notice the auto-generated DAG of tables.
# MAGIC 2. **Click any silver table in the graph → "Data quality" tab.** You'll see expectation pass rates
# MAGIC    (e.g., `valid_age = 100%`, `watch_seconds_non_negative = 100%`).
# MAGIC 3. **Click a gold table → "Lineage" tab.** Unity Catalog automatically tracked that
# MAGIC    `gold_interactions` was built from `silver_interactions` + `silver_items` + `silver_users`.
# MAGIC    Later, when chapter 2 builds feature tables from these gold tables, the lineage extends
# MAGIC    automatically — and chapter 3's MLflow model registration adds itself to the chain.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - A 3-layer **medallion lakehouse** (bronze → silver → gold) via Delta Live Tables
# MAGIC - **5 data quality expectations** enforced at the silver layer (bad rows dropped, not silently corrupted)
# MAGIC - **Liquid Clustering** on `(user_id, content_id)` for fast joins in the next chapter
# MAGIC - **Lineage** captured automatically — every gold row knows where it came from
# MAGIC
# MAGIC ## Up next — Chapter 2: Features and Vector Search
# MAGIC
# MAGIC We'll turn the gold tables into **feature tables** (the Databricks Feature Store), and build a
# MAGIC **Vector Search index** over item synopsis embeddings. The two-tower retrieval model in chapter 3
# MAGIC will use both.

# COMMAND ----------
for tbl in ["silver_users", "silver_items", "silver_interactions", "gold_user_360", "gold_interactions"]:
    fq = FQ(tbl)
    record_asset(spark, OPS_TABLE, AssetRecord(
        chapter=1, asset_type="table", name=fq, id=fq,
        url=format_asset_url(workspace_url, "table", fq),
        description=f"DLT-produced {tbl}",
    ))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=1, asset_type="pipeline", name="cmeg_dlt_pipeline", id=pipeline.pipeline_id,
    url=pipeline_url, description="Medallion DLT pipeline",
))

chapter_complete(
    chapter=1, title="DLT Medallion",
    created=[
        ("pipeline", "cmeg_dlt_pipeline", pipeline_url),
        ("table", FQ("gold_interactions"), format_asset_url(workspace_url, "table", FQ("gold_interactions"))),
        ("table", FQ("gold_user_360"), format_asset_url(workspace_url, "table", FQ("gold_user_360"))),
    ],
    next_label="03_features_and_vectors",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/03_features_and_vectors",
)
