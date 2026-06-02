# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 4 — Serve and Explain (chained inference + GenAI)
# MAGIC
# MAGIC ## What we build
# MAGIC
# MAGIC A single **Model Serving endpoint** that, on each request, runs a 4-stage chain:
# MAGIC
# MAGIC ```
# MAGIC POST /serving-endpoints/cmeg_rec_endpoint/invocations
# MAGIC   { "user_id": "u_0001234" }
# MAGIC
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │  1. Two-tower retrieval  →  100 candidate content_ids                │
# MAGIC │  2. LightGBM ranker      →  scored candidates (P(completed))         │
# MAGIC │  3. Diversity rerank     →  genre-deduped top 5                      │
# MAGIC │  4. Foundation Model     →  "Why we recommend this" copy per item    │
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC   returns: [{title, genre, score, why}, ...]
# MAGIC ```
# MAGIC
# MAGIC ## Best practices applied
# MAGIC
# MAGIC - **Inference table** auto-captures every request and response → drives the Lakehouse Monitor in chapter 5
# MAGIC - **Diversity reranking** prevents the "5 dramas in a row" failure mode — genre dedupe + top-K cap
# MAGIC - **GenAI explanation** generated per-card with a Foundation Model — gives the customer a "why" they can show end-users
# MAGIC - **Dynamic view** masks PII columns unless the caller is in the `cmeg_pii_readers` group — demonstrates UC fine-grained access

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
import os, tempfile
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput
from cmeg.serving import RecChain

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------
# MAGIC %md ## Build the chained pyfunc and register it

# COMMAND ----------
tt_uri = f"models:/{FQ('cmeg_two_tower')}@champion"
r_uri = f"models:/{FQ('cmeg_ranker')}@champion"

# Snapshot item and user metadata for the pyfunc (so serving has no dep on Spark)
tmp = tempfile.mkdtemp()
item_meta_path = os.path.join(tmp, "items.parquet")
user_meta_path = os.path.join(tmp, "users.parquet")
spark.table(FQ("item_features")).toPandas().to_parquet(item_meta_path)
spark.table(FQ("user_features")).toPandas().to_parquet(user_meta_path)

chain = RecChain()
input_ex = pd.DataFrame({"user_id": ["u_0000001"]})
chain_name = FQ("cmeg_rec_chain")

with mlflow.start_run(run_name="rec_chain"):
    mlflow.pyfunc.log_model(
        artifact_path="chain", python_model=chain,
        artifacts={"two_tower": tt_uri, "ranker": r_uri, "item_meta": item_meta_path, "user_meta": user_meta_path},
        signature=infer_signature(input_ex, [[]]),
        input_example=input_ex,
        model_config={"genai_model": GENAI_MODEL, "top_k": 5},
        registered_model_name=chain_name,
        pip_requirements=["mlflow", "pandas", "lightgbm", "scikit-learn", "databricks-sdk"],
    )
chain_latest = client.get_registered_model(chain_name).latest_versions[0].version
client.set_registered_model_alias(chain_name, "champion", version=chain_latest)
print(f"✓ {chain_name} v{chain_latest} @champion")

# COMMAND ----------
# MAGIC %md ## Create the serving endpoint with inference-table auto-capture

# COMMAND ----------
endpoint_name = "cmeg_rec_endpoint"
config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=chain_name, entity_version=chain_latest,
            scale_to_zero_enabled=True, workload_size="Small",
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix="cmeg_inference", enabled=True,
    ),
)

if SERVING_ENDPOINT_ENABLED:
    try:
        w.serving_endpoints.create(name=endpoint_name, config=config)
        print(f"✓ endpoint {endpoint_name} created")
    except Exception as e:
        print(f"○ endpoint exists, updating: {e}")
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=config.served_entities,
            auto_capture_config=config.auto_capture_config,
        )
else:
    print("○ skipping endpoint creation (SERVING_ENDPOINT_ENABLED=False in config.py)")

# COMMAND ----------
# MAGIC %md ## Create the PII-masking dynamic view
# MAGIC
# MAGIC Members of `cmeg_pii_readers` (an account-level group an admin would create) see real values;
# MAGIC everyone else sees `REDACTED`. This is the UC pattern for column-level access control.

# COMMAND ----------
view = FQ("user_features_masked")
src = FQ("user_features")
spark.sql(f"DROP VIEW IF EXISTS {view}")
spark.sql(f"""
    CREATE VIEW {view} AS
    SELECT
      user_id,
      CASE WHEN is_account_group_member('cmeg_pii_readers') THEN fav_genre ELSE 'REDACTED' END AS fav_genre,
      watch_count_7d, avg_session_seconds, p50_session_seconds, last_active_ts
    FROM {src}
""")
print(f"✓ created masked view {view}")

# COMMAND ----------
# MAGIC %md ## Wrap up

# COMMAND ----------
inference_table = FQ("cmeg_inference_payload")
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=4, asset_type="endpoint", name=endpoint_name, id=endpoint_name,
    url=format_asset_url(workspace_url, "endpoint", endpoint_name),
    description="Chained inference: retrieval → ranker → diversity → GenAI explain",
))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=4, asset_type="table", name=inference_table, id=inference_table,
    url=format_asset_url(workspace_url, "table", inference_table),
    description="Inference table (auto-captured request/response)",
))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=4, asset_type="table", name=view, id=view,
    url=format_asset_url(workspace_url, "table", view),
    description="PII-masked dynamic view over user_features",
))

chapter_complete(
    chapter=4, title="Serve and Explain",
    created=[
        ("endpoint", endpoint_name, format_asset_url(workspace_url, "endpoint", endpoint_name)),
        ("table", view, format_asset_url(workspace_url, "table", view)),
    ],
    next_label="05-Monitor-and-Govern",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/05-Monitor-and-Govern",
)
