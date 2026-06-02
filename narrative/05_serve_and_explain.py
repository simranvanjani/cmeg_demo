# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 05 — Serve and explain
# MAGIC
# MAGIC Deploy a chained Model Serving endpoint that runs: two-tower retrieval ->
# MAGIC GBT ranker -> diversity rerank -> Foundation Model explanation copy. Inference
# MAGIC table is enabled and auto-captures every request. A dynamic view masks PII on
# MAGIC user_features for non-admin readers.

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os, tempfile
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput,
)
from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset
from cmeg.serving import RecChain

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

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------
# MAGIC %md ## Build and register the chained pyfunc

# COMMAND ----------
tt_uri = f"models:/{cfg.fq(cfg.ml_schema, 'cmeg_two_tower')}@champion"
r_uri = f"models:/{cfg.fq(cfg.ml_schema, 'cmeg_ranker')}@champion"

tmp = tempfile.mkdtemp()
item_meta_path = os.path.join(tmp, "items.parquet")
user_meta_path = os.path.join(tmp, "users.parquet")
spark.table(cfg.fq(cfg.ml_schema, "item_features")).toPandas().to_parquet(item_meta_path)
spark.table(cfg.fq(cfg.ml_schema, "user_features")).toPandas().to_parquet(user_meta_path)

chain = RecChain()
input_ex = pd.DataFrame({"user_id": ["u_0000001"]})
chain_name = cfg.fq(cfg.ml_schema, "cmeg_rec_chain")

with mlflow.start_run(run_name="rec_chain"):
    mlflow.pyfunc.log_model(
        artifact_path="chain",
        python_model=chain,
        artifacts={"two_tower": tt_uri, "ranker": r_uri, "item_meta": item_meta_path, "user_meta": user_meta_path},
        signature=infer_signature(input_ex, [[]]),
        input_example=input_ex,
        model_config={"genai_model": cfg.genai_model, "top_k": 5},
        registered_model_name=chain_name,
        pip_requirements=["mlflow", "pandas", "lightgbm", "scikit-learn", "databricks-sdk"],
    )

chain_latest = client.get_registered_model(chain_name).latest_versions[0].version
client.set_registered_model_alias(chain_name, "champion", version=chain_latest)

# COMMAND ----------
# MAGIC %md ## Create the serving endpoint with auto-capture inference table

# COMMAND ----------
endpoint_name = "cmeg_rec_endpoint"
config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=chain_name,
            entity_version=chain_latest,
            scale_to_zero_enabled=True,
            workload_size="Small",
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=cfg.catalog,
        schema_name=cfg.ops_schema,
        table_name_prefix="cmeg_inference",
        enabled=True,
    ),
)

try:
    w.serving_endpoints.create(name=endpoint_name, config=config)
except Exception as e:
    print(f"Updating existing endpoint: {e}")
    try:
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=config.served_entities,
            auto_capture_config=config.auto_capture_config,
        )
    except Exception as e2:
        print(f"Update also failed: {e2}")

# COMMAND ----------
# MAGIC %md ## Dynamic view masking PII

# COMMAND ----------
view = cfg.fq(cfg.gold_schema, "user_features_masked")
src = cfg.fq(cfg.ml_schema, "user_features")
spark.sql(f"DROP VIEW IF EXISTS {view}")
spark.sql(f"""
    CREATE VIEW {view} AS
    SELECT
      user_id,
      CASE WHEN is_account_group_member('cmeg_pii_readers') THEN fav_genre ELSE 'REDACTED' END AS fav_genre,
      watch_count_7d, avg_session_seconds, p50_session_seconds, last_active_ts
    FROM {src}
""")

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
inference_table = f"{cfg.catalog}.{cfg.ops_schema}.cmeg_inference_payload"
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=5, asset_type="endpoint", name=endpoint_name, id=endpoint_name,
    url=format_asset_url(workspace_url, "endpoint", endpoint_name),
    description="Chained recommendation serving endpoint (retrieval -> ranker -> rerank -> LLM explain)",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=5, asset_type="table", name=inference_table, id=inference_table,
    url=format_asset_url(workspace_url, "table", inference_table),
    description="Inference table (auto-captured)",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=5, asset_type="table", name=view, id=view,
    url=format_asset_url(workspace_url, "table", view),
    description="PII-masked dynamic view",
))

chapter_complete(
    chapter=5, title="Serve and explain",
    created=[
        ("endpoint", endpoint_name, format_asset_url(workspace_url, "endpoint", endpoint_name)),
        ("table", view, format_asset_url(workspace_url, "table", view)),
    ],
    next_label="06_monitor_and_govern",
    next_url=f"{workspace_url}/#workspace/06_monitor_and_govern",
)
