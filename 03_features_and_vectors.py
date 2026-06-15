# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 2 of 6 — Features and Vector Search
# MAGIC
# MAGIC > 🕐 **5 min to read · 3 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - **Why production recommenders need TWO kinds of lookups** at serving time, with very different SLAs
# MAGIC - How the **Feature Engineering API** tracks model-feature lineage automatically
# MAGIC - How **Vector Search** indexes text embeddings without leaving the lakehouse
# MAGIC - The role of **Online Tables** for sub-10ms feature lookup (prod pattern, mentioned here)
# MAGIC
# MAGIC ## The two lookups
# MAGIC
# MAGIC | Lookup | When | Latency budget | What we use |
# MAGIC | --- | --- | --- | --- |
# MAGIC | "What does this user look like right now?" | every request | **< 10 ms** | Feature Store table (+ Online Table in prod) |
# MAGIC | "Find 100 items semantically similar to what this user enjoys" | every request | **< 50 ms** | Vector Search index over item embeddings |

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %pip install -q --no-deps databricks-vectorsearch "protobuf>=5.29.4,<6"

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
# MAGIC %md
# MAGIC ## Step 1 of 4 — Build user features
# MAGIC
# MAGIC We aggregate per-user behavior from `gold_interactions`: how many sessions, average watch time,
# MAGIC favorite genre, last active time. These are the features the **ranker** in chapter 3 will use.
# MAGIC
# MAGIC **Note:** the SQL logic lives in `lib/cmeg/features.py` so it's importable + testable. The notebook
# MAGIC just calls it.

# COMMAND ----------
gold_inter = spark.table(FQ("gold_interactions"))
silver_items = spark.table(FQ("silver_items"))

user_feats = build_user_features(gold_inter)
display(user_feats.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 4 — Build item features

# COMMAND ----------
item_feats = build_item_features(silver_items, gold_inter)
display(item_feats.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 4 — Register as Feature Engineering tables
# MAGIC
# MAGIC The `FeatureEngineeringClient.create_table` call does two important things:
# MAGIC 1. Writes the data as a Delta table under our schema.
# MAGIC 2. **Registers the table as a Feature Store entity** — every model trained with these features
# MAGIC    will be linked back here in UC lineage.
# MAGIC
# MAGIC **🔍 What to look for after this runs:** Open the **Features** tab in the left nav. You'll see
# MAGIC `user_features` and `item_features` listed with their primary keys.

# COMMAND ----------
user_feat_table = FQ("user_features")
item_feat_table = FQ("item_features")

try:
    fe.create_table(name=user_feat_table, primary_keys=["user_id"], df=user_feats,
                    description="Per-user behavioral features (cmeg demo)")
    print(f"✓ created {user_feat_table}")
except Exception:
    fe.write_table(name=user_feat_table, df=user_feats, mode="merge")
    print(f"✓ updated {user_feat_table}")

try:
    fe.create_table(name=item_feat_table, primary_keys=["content_id"], df=item_feats,
                    description="Per-item popularity and metadata features (cmeg demo)")
    print(f"✓ created {item_feat_table}")
except Exception:
    fe.write_table(name=item_feat_table, df=item_feats, mode="merge")
    print(f"✓ updated {item_feat_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 of 4 — Build the Vector Search index over item synopsis
# MAGIC
# MAGIC Vector Search is Databricks' managed embedding index. We point it at a Delta table
# MAGIC (with Change Data Feed enabled) and tell it which column holds the text we want embedded.
# MAGIC Databricks hosts the `databricks-bge-large-en` embedding model for us, and the index
# MAGIC stays in sync as the source table changes.
# MAGIC
# MAGIC **In chapter 3** the two-tower model will use its own learned embeddings (not these BGE ones)
# MAGIC but the pattern is identical: index embeddings in Vector Search, query them at serving time.
# MAGIC
# MAGIC **🔍 What to look for:** Open the Catalog Explorer → your catalog → schema → `item_index`.
# MAGIC You'll see the index status and a sample query box.

# COMMAND ----------
vs = VectorSearchClient(disable_notice=True)
vs_endpoint = "cmeg_vs_endpoint"
try:
    vs.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
    print(f"✓ created vector search endpoint {vs_endpoint}")
except Exception as e:
    print(f"○ endpoint exists: {e}")

# COMMAND ----------
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
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - `user_features` Delta table with 5 behavioral aggregates per user
# MAGIC - `item_features` Delta table with popularity + metadata per item
# MAGIC - A **Vector Search index** over item synopsis text, refreshed automatically as the source changes
# MAGIC - Both feature tables are registered with the Feature Engineering API → **lineage from gold → feature → model** kicks in automatically when we train in the next chapter
# MAGIC
# MAGIC ## Up next — Chapter 3: Train and Register
# MAGIC
# MAGIC We train a **two-tower retrieval model** (the architecture YouTube and Spotify use) and a
# MAGIC **LightGBM ranker** with Optuna hyperparameter search. Both get registered to Unity Catalog
# MAGIC with `@champion` aliases.

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
    next_label="04_train_and_register",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/04_train_and_register",
)
