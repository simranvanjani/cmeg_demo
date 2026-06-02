# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 06 — Monitor and govern
# MAGIC
# MAGIC Create a Lakehouse Monitor on the inference table (TimeSeries profile),
# MAGIC apply UC tags for PII + ownership, and run one sample audit query.
# MAGIC
# MAGIC > **Sidebar — Champion/Challenger pattern**
# MAGIC > To A/B between two model versions, register a second version, assign `@challenger`,
# MAGIC > and route a fraction of traffic via `served_entities` traffic_percentage on the
# MAGIC > endpoint config. Promote with `client.set_registered_model_alias(..., 'champion', ver)`.

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries

from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset
from cmeg.governance import apply_pii_tags, apply_table_owner_tag
from cmeg.monitoring import build_monitor_dir

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
user_name = w.current_user.me().user_name

# COMMAND ----------
# MAGIC %md ## Apply governance tags

# COMMAND ----------
user_feat = cfg.fq(cfg.ml_schema, "user_features")
apply_pii_tags(spark, user_feat, ["fav_genre"])
apply_table_owner_tag(spark, user_feat, owner="cmeg_demo", cost_center="cmeg_demo")

# COMMAND ----------
# MAGIC %md ## Lakehouse Monitor on inference table

# COMMAND ----------
inference_table = f"{cfg.catalog}.{cfg.ops_schema}.cmeg_inference_payload"
try:
    w.quality_monitors.create(
        table_name=inference_table,
        assets_dir=build_monitor_dir(user_name),
        output_schema_name=f"{cfg.catalog}.{cfg.ops_schema}",
        time_series=MonitorTimeSeries(timestamp_col="timestamp_ms", granularities=["1 day"]),
    )
except Exception as e:
    print(f"Monitor create may have failed (table may not exist yet, or monitor already exists): {e}")

# COMMAND ----------
# MAGIC %md ## Sample audit query

# COMMAND ----------
try:
    audit = spark.sql(f"""
        SELECT event_time, action_name, request_params
        FROM system.access.audit
        WHERE event_time > current_timestamp() - INTERVAL 1 DAY
          AND request_params:full_name_arg LIKE '{cfg.catalog}.%'
        ORDER BY event_time DESC
        LIMIT 20
    """)
    display(audit)
except Exception as e:
    print(f"system.access.audit may not be available in this workspace: {e}")

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=6, asset_type="monitor", name=inference_table, id=inference_table,
    url=format_asset_url(workspace_url, "monitor", inference_table),
    description="Lakehouse Monitor on inference table",
))

chapter_complete(
    chapter=6, title="Monitor and govern",
    created=[
        ("monitor", inference_table, format_asset_url(workspace_url, "monitor", inference_table)),
    ],
    next_label="07_genie_space",
    next_url=f"{workspace_url}/#workspace/07_genie_space",
)
