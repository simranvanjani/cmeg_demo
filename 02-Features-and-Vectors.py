# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 2 — Features and Vector Search
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC Production recommenders need **two kinds of lookups** at serving time, each with different
# MAGIC latency and quality requirements:
# MAGIC
# MAGIC | Lookup | Latency target | Mechanism |
# MAGIC | --- | --- | --- |
# MAGIC | "What features does this user have right now?" | < 10ms | **Online Tables** mirror Feature Store tables to a low-latency store |
# MAGIC | "Find me the 100 items most similar to what this user watched" | < 50ms | **Vector Search** indexes item embeddings |
# MAGIC
# MAGIC ## What we build in this chapter
# MAGIC
# MAGIC 1. **User feature table** — behavioral aggregates (watch_count, avg_session_seconds, fav_genre, last_active_ts)
# MAGIC 2. **Item feature table** — content metadata + popularity signals
# MAGIC 3. **Vector Search index** — embeddings over item synopsis text via `databricks-bge-large-en`
# MAGIC
# MAGIC The two-tower retrieval model in chapter 3 will use the item embeddings to find
# MAGIC candidates; the ranker will use both user and item features to score them.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %pip install -q databricks-feature-engineering databricks-vectorsearch

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.vector_search.client import VectorSearchClient
from cmeg.features import build_user_features, build_item_features

fe = FeatureEngineeringClient()

# COMMAND ----------
# MAGIC %md ## Build user features

# COMMAND ----------
gold_inter = spark.table(FQ("gold_interactions"))
silver_items = spark.table(FQ("silver_items"))

user_feats = build_user_features(gold_inter)
display(user_feats.limit(10))

# COMMAND ----------
# MAGIC %md ## Build item features

# COMMAND ----------
item_feats = build_item_features(silver_items, gold_inter)
display(item_feats.limit(10))

# COMMAND ----------
# MAGIC %md ## Register both as Feature Engineering tables
# MAGIC The Feature Engineering client logs lineage in UC: every model that later trains on these
# MAGIC features will be linked back to the source table.

# COMMAND ----------
user_feat_table = FQ("user_features")
item_feat_table = FQ("item_features")

try:
    fe.create_table(name=user_feat_table, primary_keys=["user_id"], df=user_feats,
                    description="Per-user behavioral features (cmeg demo)")
except Exception:
    fe.write_table(name=user_feat_table, df=user_feats, mode="merge")

try:
    fe.create_table(name=item_feat_table, primary_keys=["content_id"], df=item_feats,
                    description="Per-item popularity and metadata features (cmeg demo)")
except Exception:
    fe.write_table(name=item_feat_table, df=item_feats, mode="merge")

# COMMAND ----------
# MAGIC %md ## Vector Search index
# MAGIC
# MAGIC We index the item synopsis text using `databricks-bge-large-en`, a Databricks-hosted
# MAGIC embedding model. Vector Search will keep the index in sync via Delta Change Data Feed.

# COMMAND ----------
vs = VectorSearchClient(disable_notice=True)
vs_endpoint = "cmeg_vs_endpoint"
try:
    vs.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
    print(f"✓ created vector search endpoint {vs_endpoint}")
except Exception as e:
    print(f"○ endpoint exists: {e}")

# Source table for the index — must have CDF enabled
index_source = FQ("item_features_for_index")
spark.sql(f"CREATE OR REPLACE TABLE {index_source} AS SELECT content_id, title, synopsis, genre FROM {item_feat_table}")
spark.sql(f"ALTER TABLE {index_source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

index_name = FQ("item_index")
try:
    vs.create_delta_sync_index(
        endpoint_name=vs_endpoint,
        source_table_name=index_source,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="content_id",
        embedding_source_column="synopsis",
        embedding_model_endpoint_name="databricks-bge-large-en",
    )
    print(f"✓ created vector index {index_name}")
except Exception as e:
    print(f"○ index exists: {e}")

# COMMAND ----------
# MAGIC %md ## Wrap up

# COMMAND ----------
for fq, desc in [(user_feat_table, "User feature table"), (item_feat_table, "Item feature table")]:
    record_asset(spark, OPS_TABLE, AssetRecord(
        chapter=2, asset_type="table", name=fq, id=fq,
        url=format_asset_url(workspace_url, "table", fq), description=desc,
    ))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=2, asset_type="vector_index", name=index_name, id=index_name,
    url=format_asset_url(workspace_url, "vector_index", index_name),
    description="Item Vector Search index over synopsis embeddings",
))

chapter_complete(
    chapter=2, title="Features and Vector Search",
    created=[
        ("table", user_feat_table, format_asset_url(workspace_url, "table", user_feat_table)),
        ("table", item_feat_table, format_asset_url(workspace_url, "table", item_feat_table)),
        ("vector_index", index_name, format_asset_url(workspace_url, "vector_index", index_name)),
    ],
    next_label="03-Train-and-Register",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/03-Train-and-Register",
)
