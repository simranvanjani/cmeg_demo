# Databricks notebook source
# MAGIC %md
# MAGIC # CMEG Demo — Installer
# MAGIC
# MAGIC This notebook creates the catalog, schemas, volume, and asset-tracking table for the demo.
# MAGIC Fill in the widgets below, then **Run All**.
# MAGIC
# MAGIC When install finishes, click the link to open `00_START_HERE`.

# COMMAND ----------
# MAGIC %pip install -q databricks-sdk pyyaml
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
from pathlib import Path

from databricks.sdk import WorkspaceClient

from cmeg.config import set_widgets, from_widgets
from cmeg.state import AssetRecord, ensure_table, record_asset
from cmeg.companion import format_asset_url

# COMMAND ----------
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
print(f"Installing demo into catalog: {cfg.catalog}, data_scale={cfg.data_scale}")

# COMMAND ----------
w = WorkspaceClient()
workspace_url = w.config.host

# COMMAND ----------
# MAGIC %md ## Step 1 — Catalog, schemas, volume

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{sch}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.bronze_schema}.landing")

# COMMAND ----------
# MAGIC %md ## Step 2 — Predictive Optimization

# COMMAND ----------
try:
    spark.sql(f"ALTER CATALOG {cfg.catalog} ENABLE PREDICTIVE OPTIMIZATION")
except Exception as e:
    print(f"PO may already be enabled or unavailable: {e}")

# COMMAND ----------
# MAGIC %md ## Step 3 — Initialize asset tracking table

# COMMAND ----------
ensure_table(spark, cfg.ops_table)

record_asset(
    spark, cfg.ops_table,
    AssetRecord(
        chapter=0, asset_type="catalog", name=cfg.catalog,
        id=cfg.catalog, url=format_asset_url(workspace_url, "catalog", cfg.catalog),
        description="Demo catalog",
    ),
)
for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    record_asset(
        spark, cfg.ops_table,
        AssetRecord(
            chapter=0, asset_type="schema", name=f"{cfg.catalog}.{sch}",
            id=f"{cfg.catalog}.{sch}",
            url=format_asset_url(workspace_url, "schema", f"{cfg.catalog}.{sch}"),
            description=f"{sch} schema",
        ),
    )

# COMMAND ----------
# MAGIC %md ## Step 4 — Open the entry notebook

# COMMAND ----------
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
bundle_root = "/".join(nb_path.split("/")[:-2])
start_here = f"{bundle_root}/narrative/00_START_HERE"

displayHTML(f"""
<div style='border:1px solid #d0d7de;border-radius:8px;padding:20px;background:#dafbe1;font-family:Inter,sans-serif;'>
  <h2>&#10003; Install complete</h2>
  <p>Catalog: <b>{cfg.catalog}</b></p>
  <p>Schemas: {', '.join([cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema])}</p>
  <p style='margin-top:16px;'>
    <a href='{workspace_url}/#workspace{start_here}'
       target='_blank'
       style='background:#1f6feb;color:white;padding:10px 16px;border-radius:6px;text-decoration:none;'>
      Open 00_START_HERE &rarr;
    </a>
  </p>
</div>
""")
