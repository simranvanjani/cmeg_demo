# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 02 — DLT medallion
# MAGIC
# MAGIC Starts the DLT pipeline `cmeg_dlt_pipeline` (declared in `resources/pipelines.yml`).
# MAGIC The pipeline reads parquet from the landing volume via Auto Loader, applies
# MAGIC expectations on silver, and produces gold tables with Liquid Clustering on
# MAGIC `(user_id, content_id)`.
# MAGIC
# MAGIC > **Sidebar — `APPLY CHANGES INTO` for CDC**
# MAGIC > For change-data-capture sources, the same DLT pipeline can ingest CDC events with
# MAGIC > `dlt.apply_changes(target='silver_users', source='bronze_users_cdc', keys=['user_id'],
# MAGIC > sequence_by='ts', stored_as_scd_type=2)`. Not used here (synthetic full snapshots).

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import time
from databricks.sdk import WorkspaceClient
from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset

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
workspace_url = w.config.host

# COMMAND ----------
# MAGIC %md ## Find and start the DLT pipeline

# COMMAND ----------
pipelines = [p for p in w.pipelines.list_pipelines(filter="name LIKE 'cmeg_dlt_pipeline'")]
assert pipelines, "Pipeline not found. Run `databricks bundle deploy` first."
pipeline = pipelines[0]
print(f"Pipeline id: {pipeline.pipeline_id}")

w.pipelines.start_update(pipeline.pipeline_id)
print("Pipeline update started. Polling until done...")
deadline = time.time() + 60 * 30
while time.time() < deadline:
    state = w.pipelines.get(pipeline.pipeline_id)
    print(f"State: {state.state}")
    if str(state.state).endswith("IDLE") or str(state.state).endswith("FAILED"):
        break
    time.sleep(20)

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=2, asset_type="pipeline", name="cmeg_dlt_pipeline",
    id=pipeline.pipeline_id,
    url=format_asset_url(workspace_url, "pipeline", pipeline.pipeline_id),
    description="Medallion DLT pipeline",
))
for tbl in ["silver_users", "silver_items", "silver_interactions"]:
    fq = f"{cfg.catalog}.{cfg.silver_schema}.{tbl}"
    record_asset(spark, cfg.ops_table, AssetRecord(
        chapter=2, asset_type="table", name=fq, id=fq,
        url=format_asset_url(workspace_url, "table", fq),
        description=f"DLT silver: {tbl}",
    ))
for tbl in ["gold_user_360", "gold_interactions"]:
    fq = f"{cfg.catalog}.{cfg.gold_schema}.{tbl}"
    record_asset(spark, cfg.ops_table, AssetRecord(
        chapter=2, asset_type="table", name=fq, id=fq,
        url=format_asset_url(workspace_url, "table", fq),
        description=f"DLT gold: {tbl}",
    ))

chapter_complete(
    chapter=2, title="DLT medallion",
    created=[
        ("pipeline", "cmeg_dlt_pipeline", format_asset_url(workspace_url, "pipeline", pipeline.pipeline_id)),
        ("table", f"{cfg.catalog}.{cfg.gold_schema}.gold_interactions",
         format_asset_url(workspace_url, "table", f"{cfg.catalog}.{cfg.gold_schema}.gold_interactions")),
    ],
    next_label="03_features_and_vectors",
    next_url=f"{workspace_url}/#workspace/03_features_and_vectors",
)
