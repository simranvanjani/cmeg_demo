# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 03 — Features and vector index
# MAGIC
# MAGIC Builds user and item feature tables via the Feature Engineering API, mirrors user
# MAGIC features into an Online Table for low-latency serving lookup, and creates a Vector
# MAGIC Search index over item synopsis embeddings.

# COMMAND ----------
# MAGIC %pip install -q -e ../ databricks-feature-engineering databricks-vectorsearch

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.vector_search.client import VectorSearchClient

from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset
from cmeg.features import build_user_features, build_item_features

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
fe = FeatureEngineeringClient()

# COMMAND ----------
# MAGIC %md ## Build feature tables

# COMMAND ----------
gold_inter = spark.table(cfg.fq(cfg.gold_schema, "gold_interactions"))
silver_items = spark.table(cfg.fq(cfg.silver_schema, "silver_items"))

user_feats = build_user_features(gold_inter)
item_feats = build_item_features(silver_items, gold_inter)

user_feat_table = cfg.fq(cfg.ml_schema, "user_features")
item_feat_table = cfg.fq(cfg.ml_schema, "item_features")

try:
    fe.create_table(
        name=user_feat_table,
        primary_keys=["user_id"],
        df=user_feats,
        description="Per-user behavioral features",
    )
except Exception as e:
    print(f"user feature table exists, writing: {e}")
    fe.write_table(name=user_feat_table, df=user_feats, mode="merge")

try:
    fe.create_table(
        name=item_feat_table,
        primary_keys=["content_id"],
        df=item_feats,
        description="Per-item popularity and metadata features",
    )
except Exception as e:
    print(f"item feature table exists, writing: {e}")
    fe.write_table(name=item_feat_table, df=item_feats, mode="merge")

# COMMAND ----------
# MAGIC %md ## Vector Search index over item embeddings

# COMMAND ----------
vs = VectorSearchClient(disable_notice=True)
vs_endpoint = "cmeg_vs_endpoint"

try:
    vs.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
except Exception as e:
    print(f"vs endpoint exists: {e}")

index_source = f"{item_feat_table}_for_index"
spark.sql(f"CREATE OR REPLACE TABLE {index_source} AS SELECT content_id, title, synopsis, genre FROM {item_feat_table}")
spark.sql(f"ALTER TABLE {index_source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

index_name = cfg.fq(cfg.ml_schema, "item_index")
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
except Exception as e:
    print(f"index exists: {e}")

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=3, asset_type="table", name=user_feat_table, id=user_feat_table,
    url=format_asset_url(workspace_url, "table", user_feat_table),
    description="User feature table (Feature Store)",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=3, asset_type="table", name=item_feat_table, id=item_feat_table,
    url=format_asset_url(workspace_url, "table", item_feat_table),
    description="Item feature table (Feature Store)",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=3, asset_type="vector_index", name=index_name, id=index_name,
    url=format_asset_url(workspace_url, "vector_index", index_name),
    description="Item Vector Search index",
))

chapter_complete(
    chapter=3, title="Features and vector index",
    created=[
        ("table", user_feat_table, format_asset_url(workspace_url, "table", user_feat_table)),
        ("table", item_feat_table, format_asset_url(workspace_url, "table", item_feat_table)),
        ("vector_index", index_name, format_asset_url(workspace_url, "vector_index", index_name)),
    ],
    next_label="04_train_and_register",
    next_url=f"{workspace_url}/#workspace/04_train_and_register",
)
