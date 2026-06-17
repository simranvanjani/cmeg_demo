# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 4 of 6 — Serve and Explain (chained inference + GenAI)
# MAGIC
# MAGIC > 🕐 **6 min to read · 4 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - How **Databricks Vector Search** (AI semantic search) generates the candidate shows
# MAGIC - How to **chain multiple models behind a single serving endpoint** so the app makes one API call
# MAGIC - The **diversity rerank** pattern — preventing the "5 dramas in a row" failure mode
# MAGIC - How to add a **GenAI explanation** to every recommendation card via Foundation Models
# MAGIC - How **inference table auto-capture** records every request/response automatically for chapter 5's monitoring
# MAGIC - **Dynamic views** for column-level PII masking in UC
# MAGIC
# MAGIC ## What we're building
# MAGIC
# MAGIC ```
# MAGIC POST /serving-endpoints/cmeg_rec_endpoint/invocations
# MAGIC   { "user_id": "u_0001234" }
# MAGIC
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │  Stage 1 — Vector Search retrieval →  100 semantically-similar shows  │
# MAGIC │  Stage 2 — LightGBM ranker         →  scored candidates (P(completed))│
# MAGIC │  Stage 3 — Diversity rerank        →  genre-deduped top 5             │
# MAGIC │  Stage 4 — Foundation Model        →  "Why we recommend this" per card│
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC                  ▼ auto-captured ▼
# MAGIC          ┌─inference table──────────┐
# MAGIC          │ request, response, ts    │── monitored in chapter 5
# MAGIC          └──────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC One HTTP request → 4-stage pipeline → ranked recommendations with human-readable explanations.

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
# MAGIC %md
# MAGIC ## Step 1 of 5 — Retrieval with Databricks Vector Search (AI semantic search)
# MAGIC
# MAGIC This is the **AI-search core** of the recommender. For a user, we take the synopsis of the show
# MAGIC they watched most ("seed") and ask the `item_index` (BGE embeddings) for the most **semantically
# MAGIC similar** shows. Those candidates are what the ranker then scores — so Vector Search, not a
# MAGIC hardcoded query, is producing the recommended shows.

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
import time as _time

vs = VectorSearchClient(disable_notice=True)
vs_index = vs.get_index(endpoint_name="cmeg_vs_endpoint", index_name=FQ("item_index"))

# wait until the index is ready to serve queries
for _ in range(40):
    try:
        _st = vs_index.describe().get("status", {})
        if _st.get("ready") or "ONLINE" in str(_st.get("detailed_state", "")):
            break
    except Exception:
        pass
    _time.sleep(15)

# pick a sample user, find the show they watched most, search for similar shows
sample_uid = "u_0000001"
_seed_id_row = spark.table(FQ("user_features")).filter(f"user_id = '{sample_uid}'").select("seed_content_id").first()
seed_id = _seed_id_row[0] if _seed_id_row else None
seed = spark.table(FQ("item_features")).filter(f"content_id = '{seed_id}'").select("title", "synopsis", "genre").first()
print(f"User {sample_uid} watched most: {seed['title'] if seed else 'n/a'} ({seed['genre'] if seed else ''})")

if seed:
    res = vs_index.similarity_search(
        query_text=seed["synopsis"],
        columns=["content_id", "title", "genre"],
        num_results=10,
    )
    rows = (res.get("result", {}) or {}).get("data_array", []) or []
    display(pd.DataFrame(rows, columns=["content_id", "title", "genre", "similarity_score"]))
    print("☝ These candidate shows came from Databricks Vector Search (semantic similarity).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 5 — Build the chained pyfunc
# MAGIC
# MAGIC A pyfunc model is just a Python class with a `predict()` method. Ours wraps the two registered
# MAGIC models and adds the diversity rerank + GenAI explanation logic. The code lives in
# MAGIC `lib/cmeg/serving.py` so it's importable + testable.
# MAGIC
# MAGIC We log it to MLflow with **artifacts** pointing at the two `@champion` model versions. At
# MAGIC serving time, the endpoint will resolve those references and load all three models into the
# MAGIC same container.

# COMMAND ----------
tt_uri = f"models:/{FQ('cmeg_two_tower')}@champion"
r_uri = f"models:/{FQ('cmeg_ranker')}@champion"

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
        model_config={
            "genai_model": GENAI_MODEL, "top_k": 5, "num_candidates": 100,
            "vs_endpoint": "cmeg_vs_endpoint", "vs_index": FQ("item_index"),
        },
        registered_model_name=chain_name,
        pip_requirements=["mlflow", "pandas", "lightgbm", "scikit-learn",
                          "databricks-sdk", "databricks-vectorsearch", "protobuf>=5.29.4,<6"],
    )
chain_latest = client.get_registered_model(chain_name).latest_versions[0].version
client.set_registered_model_alias(chain_name, "champion", version=chain_latest)
print(f"✓ {chain_name} v{chain_latest} @champion")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 5 — Deploy to a Model Serving endpoint with inference-table capture
# MAGIC
# MAGIC `auto_capture_config` is the key piece: **every request and response gets written to a Delta
# MAGIC table automatically**, with no extra code. That table powers the Lakehouse Monitor in chapter 5.
# MAGIC
# MAGIC `scale_to_zero_enabled=True` means the endpoint costs nothing when idle — it spins up on first
# MAGIC request, then drops back to zero replicas after a few minutes of no traffic.
# MAGIC
# MAGIC **⏳ This step takes 5-10 minutes** the first time as the endpoint provisions a container.

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
        print(f"○ endpoint exists, updating config: {e}")
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=config.served_entities,
            auto_capture_config=config.auto_capture_config,
        )
else:
    print("○ skipping endpoint creation (SERVING_ENDPOINT_ENABLED=False in config.py)")

# COMMAND ----------
# MAGIC %md
# MAGIC **🔍 Try this in the UI:**
# MAGIC
# MAGIC 1. Open the **Serving** tab in the left nav → click `cmeg_rec_endpoint`
# MAGIC 2. Wait until it shows "Ready" (5-10 min on first deploy)
# MAGIC 3. Use the **Query** tab on the right to send `{"dataframe_records": [{"user_id": "u_0000001"}]}` —
# MAGIC    you'll get back 5 recommendations with `score`, `title`, `genre`, and a `why` field (the GenAI explanation)
# MAGIC 4. Then click **Inference table** → you'll see your test request captured automatically

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 of 5 — Create a dynamic view that masks PII
# MAGIC
# MAGIC Users of the `user_features` table shouldn't see raw `fav_genre` unless they're in the
# MAGIC `cmeg_pii_readers` group. UC dynamic views give us per-row, per-column access control
# MAGIC using `is_account_group_member()` directly in the view definition.
# MAGIC
# MAGIC In production you'd have an admin create the `cmeg_pii_readers` group in account console;
# MAGIC for the demo we just create the view — non-admin readers see `'REDACTED'` for `fav_genre`.

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

display(spark.table(view).limit(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5 of 5 — Confirm everything is wired up

# COMMAND ----------
inference_table = FQ("cmeg_inference_payload")

try:
    ep = w.serving_endpoints.get(name=endpoint_name)
    print(f"endpoint state: {ep.state.ready if ep.state else 'unknown'}")
except Exception as e:
    print(f"endpoint info: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - A single serving endpoint that chains retrieval → ranker → rerank → GenAI explain
# MAGIC - An inference table auto-capturing every request for monitoring
# MAGIC - A PII-masking dynamic view demonstrating UC column-level access control
# MAGIC - Diversity reranking that avoids the "5 of the same genre" failure mode
# MAGIC
# MAGIC ## Up next — Chapter 5: Monitor and Govern
# MAGIC
# MAGIC We attach a **Lakehouse Monitor** to the inference table so drift in features or predictions
# MAGIC triggers an alert. We also apply UC tags and run a sample audit query.

# COMMAND ----------
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
    next_label="06_monitor_and_govern",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/06_monitor_and_govern",
)
