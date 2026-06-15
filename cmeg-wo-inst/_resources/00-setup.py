# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — context bootstrap
# MAGIC
# MAGIC This notebook is `%run` by every chapter. It locates the repo root (by walking up
# MAGIC from the calling notebook's path until it finds `config.py`), reads `config.py`,
# MAGIC and sets the catalog/schema/helpers as globals in the calling notebook's scope.
# MAGIC
# MAGIC No resource creation happens here — that lives in `01-generate-data` and
# MAGIC `02-create-resources`, both run once by `RUNME`.

# COMMAND ----------
import sys, os, importlib

# When %run is used, dbutils.notebook.entry_point...notebookPath() returns the path
# of the TOP-LEVEL (calling) notebook, not the path of this %run'd notebook. The
# calling notebook can live at the repo root (chapters, intro) or in _resources/
# (the helper notebooks). Walk up looking for config.py to find the repo root.
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()


def _find_repo_root(start_path: str) -> str:
    parts = start_path.split("/")
    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i]) or "/"
        if os.path.exists(f"/Workspace{candidate}/config.py"):
            return candidate
    raise RuntimeError(
        f"Could not find config.py walking up from {start_path}. "
        f"Make sure config.py exists at the repo root."
    )


_repo_root = _find_repo_root(_nb_path)
print(f"repo root resolved to: {_repo_root}")

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
