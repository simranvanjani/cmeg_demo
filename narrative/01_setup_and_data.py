# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 01 — Setup and synthetic data
# MAGIC
# MAGIC Creates schemas/volume (if not already created by the installer), generates synthetic
# MAGIC users, items, and interactions, applies UC tags for ownership, and lands parquet files.

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset, ensure_table
from cmeg import data_gen

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
# MAGIC %md ## Ensure catalog/schemas/volume exist (idempotent)

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{sch}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.bronze_schema}.landing")
ensure_table(spark, cfg.ops_table)

# COMMAND ----------
# MAGIC %md ## Generate synthetic data

# COMMAND ----------
params = cfg.scale_params()
print(f"Scale: {cfg.data_scale} -> {params}")

users = data_gen.build_users(n=params["n_users"], seed=1)
items = data_gen.build_items(n=params["n_items"], seed=2)
interactions = data_gen.build_interactions(users=users, items=items, n=params["n_interactions"], seed=3)

# COMMAND ----------
# MAGIC %md ## Write parquet to the landing volume

# COMMAND ----------
vol = f"/Volumes/{cfg.catalog}/{cfg.bronze_schema}/landing"
data_gen.write_parquet(spark, users, f"{vol}/users")
data_gen.write_parquet(spark, items, f"{vol}/items")
data_gen.write_parquet(spark, interactions, f"{vol}/interactions")

# COMMAND ----------
# MAGIC %md ## Apply UC tags for ownership

# COMMAND ----------
for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    try:
        spark.sql(
            f"ALTER SCHEMA {cfg.catalog}.{sch} "
            f"SET TAGS ('business_owner' = 'cmeg_demo', 'cost_center' = 'cmeg_demo')"
        )
    except Exception as e:
        print(f"tag {sch}: {e}")

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
vol_fq = f"{cfg.catalog}.{cfg.bronze_schema}.landing"
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=1, asset_type="volume", name=vol_fq, id=vol_fq,
    url=format_asset_url(workspace_url, "volume", vol_fq),
    description="Synthetic data landing volume",
))

chapter_complete(
    chapter=1,
    title="Setup and synthetic data",
    created=[
        ("volume", vol_fq, format_asset_url(workspace_url, "volume", vol_fq)),
    ],
    next_label="02_dlt_medallion",
    next_url=f"{workspace_url}/#workspace/02_dlt_medallion",
)
