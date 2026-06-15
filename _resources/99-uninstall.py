# Databricks notebook source
# MAGIC %md
# MAGIC # Uninstall — clean up SDK-created resources
# MAGIC
# MAGIC Deletes the serving endpoint, vector index/endpoint, Lakehouse Monitor, and registered models.
# MAGIC Does NOT drop the catalog or schema unless you set `DROP_SCHEMA = True` below.
# MAGIC
# MAGIC Run this from the `cmeg_cleanup` job or open it manually and click Run All.

# COMMAND ----------
# MAGIC %run ./00-setup

# COMMAND ----------
# Set to True to also drop the demo schema (irreversible)
DROP_SCHEMA = False

# COMMAND ----------
# MAGIC %pip install -q --no-deps databricks-vectorsearch

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00-setup

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient


def safe(call, label):
    try:
        call()
        print(f"✓ deleted: {label}")
    except Exception as e:
        print(f"○ skip {label}: {e}")


safe(lambda: w.serving_endpoints.delete(name="cmeg_rec_endpoint"), "serving endpoint cmeg_rec_endpoint")

vs = VectorSearchClient(disable_notice=True)
safe(lambda: vs.delete_index(endpoint_name="cmeg_vs_endpoint", index_name=FQ("item_index")), "vector index item_index")
safe(lambda: vs.delete_endpoint(name="cmeg_vs_endpoint"), "vector search endpoint cmeg_vs_endpoint")

safe(lambda: w.quality_monitors.delete(table_name=FQ("cmeg_inference_payload")), "lakehouse monitor")

for m in ["cmeg_rec_chain", "cmeg_two_tower", "cmeg_ranker"]:
    safe(lambda m=m: spark.sql(f"DROP TABLE IF EXISTS {FQ(m)}"), f"registered model {m}")

if DROP_SCHEMA:
    safe(lambda: spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE"), f"schema {CATALOG}.{SCHEMA}")

# COMMAND ----------
print("Uninstall complete.")
print("To also remove the DLT pipeline and jobs created at install time, delete them via the Workspace UI")
print("(Jobs → cmeg_orchestrator, cmeg_cleanup; Workflows → cmeg_dlt_pipeline).")
