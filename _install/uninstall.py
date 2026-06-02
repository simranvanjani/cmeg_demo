# Databricks notebook source
# MAGIC %md
# MAGIC # CMEG Demo — Uninstall
# MAGIC
# MAGIC Deletes SDK-created assets: serving endpoint, vector index/endpoint, monitor,
# MAGIC registered models, and (optionally) drops the catalog.
# MAGIC
# MAGIC Run this BEFORE `databricks bundle destroy`.

# COMMAND ----------
# MAGIC %pip install -q -e ../ databricks-vectorsearch

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from cmeg.config import set_widgets, from_widgets

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
        "drop_catalog": "false",
    },
)
cfg = from_widgets(dbutils)
w = WorkspaceClient()


def safe(call, label):
    try:
        call()
        print(f"deleted: {label}")
    except Exception as e:
        print(f"skip {label}: {e}")


# COMMAND ----------
safe(lambda: w.serving_endpoints.delete(name="cmeg_rec_endpoint"), "serving endpoint")

# COMMAND ----------
vs = VectorSearchClient(disable_notice=True)
safe(lambda: vs.delete_index(endpoint_name="cmeg_vs_endpoint", index_name=cfg.fq(cfg.ml_schema, "item_index")), "vector index")
safe(lambda: vs.delete_endpoint(name="cmeg_vs_endpoint"), "vector search endpoint")

# COMMAND ----------
inference_table = f"{cfg.catalog}.{cfg.ops_schema}.cmeg_inference_payload"
safe(lambda: w.quality_monitors.delete(table_name=inference_table), "monitor")

# COMMAND ----------
for m in ["cmeg_rec_chain", "cmeg_two_tower", "cmeg_ranker"]:
    safe(lambda m=m: spark.sql(f"DROP TABLE IF EXISTS {cfg.fq(cfg.ml_schema, m)}"), f"registered model {m}")

# COMMAND ----------
if dbutils.widgets.get("drop_catalog") == "true":
    safe(lambda: spark.sql(f"DROP CATALOG IF EXISTS {cfg.catalog} CASCADE"), "catalog (cascade)")

# COMMAND ----------
print("Uninstall complete. Run `databricks bundle destroy --profile DEFAULT --target dev` to remove jobs and DLT pipeline.")
