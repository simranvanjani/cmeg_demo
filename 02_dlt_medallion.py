# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 1 — Lakehouse Medallion with Delta Live Tables
# MAGIC
# MAGIC ## Why this matters for content recommendation
# MAGIC
# MAGIC OTT platforms ingest **billions of events per day** from app/web/TV clients. Before any ML can
# MAGIC happen, raw events need to be:
# MAGIC - **Reliably ingested** from cloud storage (S3/ADLS/GCS) with schema evolution as the app changes.
# MAGIC - **Deduplicated and validated** so bad events don't poison downstream models.
# MAGIC - **Materialized into gold tables** that feed feature engineering, ML training, and BI.
# MAGIC
# MAGIC **Delta Live Tables (DLT)** gives us all three with declarative Python decorators.
# MAGIC
# MAGIC ## The medallion pattern
# MAGIC
# MAGIC ```
# MAGIC bronze  ── raw, append-only, exactly what came in
# MAGIC   │
# MAGIC silver  ── conformed: deduped, type-cast, validated with expectations
# MAGIC   │
# MAGIC gold    ── business-ready: user_360, gold_interactions (joined, enriched)
# MAGIC ```
# MAGIC
# MAGIC ## Best practices applied in this chapter
# MAGIC
# MAGIC - **Auto Loader** for ingestion with `cloudFiles.schemaEvolutionMode = addNewColumns` (handles new columns gracefully)
# MAGIC - **DLT expectations** for data quality: `@dlt.expect_or_drop("user_id_not_null", ...)`, `@dlt.expect("watch_seconds_non_negative", ...)`
# MAGIC - **Liquid Clustering** on `(user_id, content_id)` — current best practice over partitioning + Z-ORDER
# MAGIC - **Serverless DLT** with Photon — no cluster sizing needed
# MAGIC - **Lineage** tracked automatically by Unity Catalog (see the lineage graph in the table UI)
# MAGIC
# MAGIC > **Sidebar — CDC ingestion** (not used here, but shown for completeness)
# MAGIC > For change-data-capture sources (Debezium, Fivetran), DLT supports SCD2:
# MAGIC > `dlt.apply_changes(target='silver_users', source='bronze_cdc', keys=['user_id'], sequence_by='ts', stored_as_scd_type=2)`

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %md
# MAGIC ## Trigger the pipeline
# MAGIC
# MAGIC The DLT pipeline definition lives at `_resources/_dlt_pipeline`. The pipeline itself
# MAGIC was created during install (`RUNME` → `_resources/02-create-resources`).
# MAGIC We now find it and trigger an update.

# COMMAND ----------
import time

pipelines = [p for p in w.pipelines.list_pipelines(filter="name LIKE 'cmeg_dlt_pipeline'")]
assert pipelines, "Pipeline not found. Run RUNME first."
pipeline = pipelines[0]
print(f"Pipeline id: {pipeline.pipeline_id}")

w.pipelines.start_update(pipeline.pipeline_id)
print("Pipeline update started. Polling until done (max 30 min)...")
deadline = time.time() + 60 * 30
while time.time() < deadline:
    state = w.pipelines.get(pipeline.pipeline_id).state
    print(f"  state: {state}")
    if str(state).endswith("IDLE") or str(state).endswith("FAILED"):
        break
    time.sleep(20)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect what got built
# MAGIC
# MAGIC The pipeline produced:
# MAGIC - 3 bronze tables (raw users/items/interactions, append-only)
# MAGIC - 3 silver tables (deduped, validated, Liquid Clustered)
# MAGIC - 2 gold tables (`gold_user_360` user aggregates, `gold_interactions` enriched events)

# COMMAND ----------
display(spark.sql(f"SELECT table_name, table_type FROM {CATALOG}.information_schema.tables WHERE table_schema = '{SCHEMA}' AND table_name LIKE 'bronze_%' OR table_name LIKE 'silver_%' OR table_name LIKE 'gold_%' ORDER BY table_name"))

# COMMAND ----------
# MAGIC %md ### Sample of the gold interactions table

# COMMAND ----------
display(spark.table(FQ("gold_interactions")).limit(20))

# COMMAND ----------
# MAGIC %md ### Data quality — expectation pass rates
# MAGIC
# MAGIC DLT records expectation metrics on every update. Open the pipeline UI (link below)
# MAGIC and look at the **Pipeline graph → click any silver table → "Data quality" tab**.

# COMMAND ----------
# MAGIC %md ## Wrap up

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
    url=format_asset_url(workspace_url, "pipeline", pipeline.pipeline_id),
    description="Medallion DLT pipeline",
))

chapter_complete(
    chapter=1,
    title="DLT Medallion",
    created=[
        ("pipeline", "cmeg_dlt_pipeline", format_asset_url(workspace_url, "pipeline", pipeline.pipeline_id)),
        ("table", FQ("gold_interactions"), format_asset_url(workspace_url, "table", FQ("gold_interactions"))),
        ("table", FQ("gold_user_360"), format_asset_url(workspace_url, "table", FQ("gold_user_360"))),
    ],
    next_label="03_features_and_vectors",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/03_features_and_vectors",
)
