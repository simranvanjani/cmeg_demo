# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 5 — Monitor and Govern
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC A deployed recommender is a moving target. Two things go wrong silently:
# MAGIC 1. **Input drift** — viewing habits shift (e.g., new World Cup season). Features the model trained on no longer represent reality.
# MAGIC 2. **Prediction drift** — model outputs skew (e.g., suddenly recommending the same 5 items to everyone).
# MAGIC
# MAGIC **Lakehouse Monitoring** sits on top of the inference table (auto-captured in chapter 4) and
# MAGIC computes drift metrics on a schedule. You'll see them as charts on the table page.
# MAGIC
# MAGIC ## What we build
# MAGIC
# MAGIC 1. **Lakehouse Monitor** on the inference table with a `TimeSeries` profile (daily granularity).
# MAGIC 2. **UC tags** for PII columns and ownership.
# MAGIC 3. **One sample audit query** showing recent access events against `system.access.audit`.
# MAGIC
# MAGIC > **Sidebar — Champion/Challenger pattern**
# MAGIC > To A/B between model versions, register a second version, assign `@challenger`, and route
# MAGIC > a fraction of traffic via `served_entities[].traffic_percentage` on the endpoint config.
# MAGIC > Promote with `client.set_registered_model_alias(name, 'champion', version)`.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
from databricks.sdk.service.catalog import MonitorTimeSeries
from cmeg.governance import apply_pii_tags, apply_table_owner_tag
from cmeg.monitoring import build_monitor_dir

# COMMAND ----------
# MAGIC %md ## Apply UC tags

# COMMAND ----------
user_feat = FQ("user_features")
apply_pii_tags(spark, user_feat, ["fav_genre"])
apply_table_owner_tag(spark, user_feat, owner="cmeg_demo", cost_center="cmeg_demo")
print(f"✓ tagged {user_feat} with pii + owner + cost_center")

# COMMAND ----------
# MAGIC %md ## Create the Lakehouse Monitor on the inference table

# COMMAND ----------
inference_table = FQ("cmeg_inference_payload")
try:
    w.quality_monitors.create(
        table_name=inference_table,
        assets_dir=build_monitor_dir(current_user),
        output_schema_name=f"{CATALOG}.{SCHEMA}",
        time_series=MonitorTimeSeries(timestamp_col="timestamp_ms", granularities=["1 day"]),
    )
    print(f"✓ monitor created on {inference_table}")
except Exception as e:
    print(f"○ monitor may already exist or inference table not yet populated: {e}")

# COMMAND ----------
# MAGIC %md ## Sample audit query
# MAGIC
# MAGIC `system.access.audit` records every UC operation. Filter to our catalog to see who touched
# MAGIC the demo's tables in the last 24h.

# COMMAND ----------
try:
    display(spark.sql(f"""
        SELECT event_time, action_name, request_params:full_name_arg AS table_arg
        FROM system.access.audit
        WHERE event_time > current_timestamp() - INTERVAL 1 DAY
          AND request_params:full_name_arg LIKE '{CATALOG}.%'
        ORDER BY event_time DESC
        LIMIT 20
    """))
except Exception as e:
    print(f"○ system.access.audit not available in this workspace yet: {e}")

# COMMAND ----------
# MAGIC %md ## Wrap up

# COMMAND ----------
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=5, asset_type="monitor", name=inference_table, id=inference_table,
    url=format_asset_url(workspace_url, "monitor", inference_table),
    description="Lakehouse Monitor on inference table",
))

chapter_complete(
    chapter=5, title="Monitor and Govern",
    created=[
        ("monitor", inference_table, format_asset_url(workspace_url, "monitor", inference_table)),
    ],
    next_label="07_genie_space",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/07_genie_space",
)
