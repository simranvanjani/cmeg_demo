# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 07 — Genie space
# MAGIC
# MAGIC Create a Genie space scoped to gold + ml schemas so business stakeholders
# MAGIC can ask plain-English questions about watch behavior and recommendation
# MAGIC effectiveness during the demo.

# COMMAND ----------
# MAGIC %pip install -q -e ../

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import requests
from databricks.sdk import WorkspaceClient
from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset

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

# COMMAND ----------
# MAGIC %md ## Pick a warehouse for the Genie space

# COMMAND ----------
warehouses = list(w.warehouses.list())
assert warehouses, "No SQL warehouses available. Create one to host the Genie space."
warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")

# COMMAND ----------
# MAGIC %md ## Create Genie space

# COMMAND ----------
SAMPLE_QUESTIONS = [
    "What are the top 10 most-watched shows last week?",
    "Which content genres have the highest completion rate?",
    "Show me engagement by age segment for drama content.",
    "Which users have the highest 7-day watch time?",
    "What share of viewing comes from recommended vs. self-discovered content?",
    "Which content has the highest cold-start ranker score this week?",
    "Compare weekday vs. weekend watch patterns.",
    "Top 5 content items by recommendations served, last 24 hours.",
]

INSTRUCTIONS = (
    "Domain glossary:\n"
    "- watch_seconds: total seconds played in an interaction; only count when > 30 seconds.\n"
    "- completed: TRUE when watch_seconds >= 85% of duration.\n"
    "- cold_start: users with fewer than 5 interactions.\n"
    "- last 7 days: filter event_ts > current_timestamp() - INTERVAL 7 DAYS.\n\n"
    "Tables: gold_interactions (event-level), gold_user_360 (per-user aggregates), "
    "user_features and item_features (engagement metadata)."
)

token = w.config.authenticate()["Authorization"].split(" ")[1]
host = w.config.host.rstrip("/")

payload = {
    "display_name": "CMEG Demo — Content Analytics",
    "description": "Ask questions about TrueID-style viewing data, content engagement, and recommendation effectiveness.",
    "warehouse_id": warehouse_id,
    "table_identifiers": [
        f"{cfg.catalog}.{cfg.gold_schema}.gold_interactions",
        f"{cfg.catalog}.{cfg.gold_schema}.gold_user_360",
        f"{cfg.catalog}.{cfg.ml_schema}.user_features",
        f"{cfg.catalog}.{cfg.ml_schema}.item_features",
    ],
    "instructions": INSTRUCTIONS,
}

space_id = None
try:
    resp = requests.post(
        f"{host}/api/2.0/genie/spaces",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    print(resp.status_code, resp.text[:500])
    if resp.ok:
        space_id = resp.json().get("space_id") or resp.json().get("id")
except Exception as e:
    print(f"Genie space create failed (API may have changed; create manually): {e}")

if space_id:
    for q in SAMPLE_QUESTIONS:
        try:
            requests.post(
                f"{host}/api/2.0/genie/spaces/{space_id}/sample-questions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"text": q},
                timeout=30,
            )
        except Exception as e:
            print(f"sample question add failed: {e}")

# COMMAND ----------
# MAGIC %md ## Record asset and finish chapter

# COMMAND ----------
if space_id:
    record_asset(spark, cfg.ops_table, AssetRecord(
        chapter=7, asset_type="genie", name="CMEG Content Analytics", id=space_id,
        url=format_asset_url(workspace_url, "genie", space_id),
        description="Genie space scoped to gold + ml schemas with 8 sample questions",
    ))

chapter_complete(
    chapter=7, title="Genie space",
    created=[
        ("genie", "CMEG Content Analytics",
         format_asset_url(workspace_url, "genie", space_id) if space_id else workspace_url + "/genie"),
    ],
    next_label="00_START_HERE",
    next_url=f"{workspace_url}/#workspace/00_START_HERE",
)
