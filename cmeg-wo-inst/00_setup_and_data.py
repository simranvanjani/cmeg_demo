# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 0 of 6 — Setup and Synthetic Data
# MAGIC
# MAGIC > 🕐 **2 min to read · 2 min to run**
# MAGIC >
# MAGIC > **This is the installer-free version of the demo.** Run the notebooks in order
# MAGIC > (`00` → `06`). No `RUNME`, no jobs, no orchestrator — each chapter sets up what it needs.
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC - Validates the catalog in `config.py` exists (never creates one)
# MAGIC - Creates the schema + landing UC Volume
# MAGIC - Generates synthetic users / items / interactions and lands them as parquet
# MAGIC - Initializes the `_cmeg_assets` tracking table
# MAGIC
# MAGIC ## Before you run
# MAGIC
# MAGIC Open `config.py` and confirm `CATALOG` points at a catalog you can write to.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %md ## Validate the catalog exists (we never create one)

# COMMAND ----------
_existing = {c.name for c in w.catalogs.list()}
if CATALOG not in _existing:
    raise RuntimeError(
        f"\n\nCatalog '{CATALOG}' does not exist in this workspace.\n"
        f"Edit config.py and set CATALOG to one of:\n  - "
        + "\n  - ".join(sorted(c for c in _existing if c not in ("system",)))
        + "\n"
    )
print(f"✓ using existing catalog: {CATALOG}")

# COMMAND ----------
# MAGIC %md ## Create schema + landing volume

# COMMAND ----------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.landing")
print(f"✓ schema/volume ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# MAGIC %md ## Generate synthetic data

# COMMAND ----------
from cmeg import data_gen

scales = {
    "small":  {"n_users": 10_000,   "n_items": 5_000,   "n_interactions": 500_000},
    "medium": {"n_users": 100_000,  "n_items": 20_000,  "n_interactions": 10_000_000},
    "large":  {"n_users": 1_000_000,"n_items": 100_000, "n_interactions": 100_000_000},
}
params = scales[DATA_SCALE]
print(f"Scale '{DATA_SCALE}': {params}")

users = data_gen.build_users(n=params["n_users"], seed=1)
items = data_gen.build_items(n=params["n_items"], seed=2)
interactions = data_gen.build_interactions(users=users, items=items, n=params["n_interactions"], seed=3)

data_gen.write_parquet(spark, users, f"{VOLUME_PATH}/users")
data_gen.write_parquet(spark, items, f"{VOLUME_PATH}/items")
data_gen.write_parquet(spark, interactions, f"{VOLUME_PATH}/interactions")
print(f"✓ wrote synthetic data to {VOLUME_PATH}")

# COMMAND ----------
# MAGIC %md ## Initialize asset tracking + tag schema

# COMMAND ----------
ensure_table(spark, OPS_TABLE)
try:
    spark.sql(f"ALTER SCHEMA {CATALOG}.{SCHEMA} SET TAGS ('business_owner' = 'cmeg_demo', 'cost_center' = 'cmeg_demo')")
except Exception as e:
    print(f"○ tag schema skipped: {e}")

record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=0, asset_type="volume", name=f"{CATALOG}.{SCHEMA}.landing", id=f"{CATALOG}.{SCHEMA}.landing",
    url=format_asset_url(workspace_url, "volume", f"{CATALOG}.{SCHEMA}.landing"),
    description="Synthetic data landing volume",
))

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✓ Setup complete
# MAGIC
# MAGIC Synthetic data is in the landing volume. **Next: open `01_dlt_medallion` and Run all.**

# COMMAND ----------
chapter_complete(
    chapter=0, title="Setup and Synthetic Data",
    created=[
        ("volume", f"{CATALOG}.{SCHEMA}.landing",
         format_asset_url(workspace_url, "volume", f"{CATALOG}.{SCHEMA}.landing")),
    ],
    next_label="01_dlt_medallion",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/01_dlt_medallion",
)
