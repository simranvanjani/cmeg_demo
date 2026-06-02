# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic data
# MAGIC
# MAGIC This notebook is run **once** by `RUNME` during install.
# MAGIC It creates the schema (if missing), the landing UC Volume, and writes synthetic
# MAGIC users / items / interactions as parquet files. The DLT pipeline (chapter 1) then
# MAGIC reads from this volume via Auto Loader.
# MAGIC
# MAGIC No customer data leaves the workspace — everything here is generated locally.

# COMMAND ----------
# MAGIC %run ./00-setup

# COMMAND ----------
# Validate the catalog exists. We NEVER create catalogs — that requires Default
# Storage or an explicit MANAGED LOCATION, which most customer workspaces don't have.
# The customer must point CATALOG in config.py at an existing catalog they can
# CREATE SCHEMA in.
_existing = {c.name for c in w.catalogs.list()}
if CATALOG not in _existing:
    raise RuntimeError(
        f"\n\nCatalog '{CATALOG}' does not exist in this workspace.\n"
        f"Edit config.py and set CATALOG to one of:\n  - "
        + "\n  - ".join(sorted(c for c in _existing if c not in ("system",)))
        + "\n\n(You can also create a new catalog yourself in the Catalog UI, "
        "then set CATALOG to its name.)\n"
    )
print(f"✓ using existing catalog: {CATALOG}")

# Schema and volume are created if missing (these need only CREATE SCHEMA / CREATE VOLUME
# privileges on the catalog, NOT metastore admin).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.landing")
print(f"✓ schema/volume ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# Tag schema for ownership
try:
    spark.sql(f"ALTER SCHEMA {CATALOG}.{SCHEMA} SET TAGS ('business_owner' = 'cmeg_demo', 'cost_center' = 'cmeg_demo')")
    print("✓ schema tagged with business_owner and cost_center")
except Exception as e:
    print(f"○ tag schema skipped: {e}")

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
ensure_table(spark, OPS_TABLE)
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=0, asset_type="volume", name=f"{CATALOG}.{SCHEMA}.landing", id=f"{CATALOG}.{SCHEMA}.landing",
    url=format_asset_url(workspace_url, "volume", f"{CATALOG}.{SCHEMA}.landing"),
    description="Synthetic data landing volume",
))
print("✓ asset tracking table initialized")
