# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — context bootstrap
# MAGIC
# MAGIC This notebook is `%run` by every chapter. It reads `config.py` at the repo root,
# MAGIC sets the catalog/schema/helpers as globals in the calling notebook, and is otherwise
# MAGIC lightweight (no resource creation here — that lives in `01-generate-data` and
# MAGIC `02-create-resources`, both run once by `00-CMEG-Demo-Intro`).

# COMMAND ----------
import sys, os, importlib

# Find the repo root from this notebook's location
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/".join(_nb_path.split("/")[:-2])  # _resources/00-setup -> repo root

# Make config.py and the lib/ package importable
for _p in (f"/Workspace{_repo_root}", f"/Workspace{_repo_root}/lib"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
importlib.reload(config)

# Hoist config values
CATALOG = config.CATALOG
SCHEMA = config.SCHEMA
DATA_SCALE = config.DATA_SCALE
GENAI_MODEL = config.GENAI_MODEL
SERVING_ENDPOINT_ENABLED = config.SERVING_ENDPOINT_ENABLED

# Convenience: fully-qualified table names
def FQ(table: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{table}"

OPS_TABLE = FQ("_cmeg_assets")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"

# Workspace client
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
workspace_url = w.config.host
current_user = w.current_user.me().user_name

# Helpers (companion cards, asset tracking)
from cmeg.companion import chapter_complete, format_asset_url, render_chapter_card  # noqa: F401
from cmeg.state import AssetRecord, ensure_table, record_asset, list_assets  # noqa: F401

# Useful paths
REPO_ROOT = _repo_root
NARRATIVE_DIR = _repo_root  # chapters live at root in the dbdemos pattern

print(f"✓ Setup complete | catalog={CATALOG} schema={SCHEMA} scale={DATA_SCALE}")
