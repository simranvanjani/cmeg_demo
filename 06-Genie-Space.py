# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 6 — Genie Space for business users
# MAGIC
# MAGIC ## Why this matters
# MAGIC
# MAGIC Your customer's content team doesn't want to write SQL. They want to ask things like
# MAGIC *"which dramas had the best completion rate last week?"* in plain English.
# MAGIC
# MAGIC **AI/BI Genie** does exactly that. We scope a Genie space to our gold + feature tables,
# MAGIC seed it with domain glossary and a few example questions, and hand the URL to non-technical
# MAGIC stakeholders.
# MAGIC
# MAGIC ## What we build
# MAGIC
# MAGIC 1. A Genie space scoped to:
# MAGIC    - `gold_interactions` — event-level enriched data
# MAGIC    - `gold_user_360` — per-user aggregates
# MAGIC    - `user_features`, `item_features` — engagement metadata
# MAGIC 2. **Instructions**: domain glossary (e.g., *completed = watch_seconds ≥ 85% of duration*).
# MAGIC 3. **8 sample questions** so the customer sees something interesting before typing anything.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
import requests

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
    "- watch_seconds: total seconds played in an interaction; only count > 30 seconds.\n"
    "- completed: TRUE when watch_seconds >= 85% of duration.\n"
    "- cold_start: users with fewer than 5 interactions.\n"
    "- 'last 7 days': filter event_ts > current_timestamp() - INTERVAL 7 DAYS.\n\n"
    "Tables: gold_interactions (event-level), gold_user_360 (per-user aggregates), "
    "user_features and item_features (engagement metadata)."
)

# COMMAND ----------
# MAGIC %md ## Pick a SQL warehouse

# COMMAND ----------
warehouses = list(w.warehouses.list())
assert warehouses, "No SQL warehouses available. Create one to host the Genie space."
warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")

# COMMAND ----------
# MAGIC %md ## Create the Genie space

# COMMAND ----------
token = w.config.authenticate()["Authorization"].split(" ")[1]
host = workspace_url.rstrip("/")

payload = {
    "display_name": "CMEG Demo — Content Analytics",
    "description": "Ask plain-English questions about TrueID-style viewing data and recommendation effectiveness.",
    "warehouse_id": warehouse_id,
    "table_identifiers": [
        FQ("gold_interactions"),
        FQ("gold_user_360"),
        FQ("user_features"),
        FQ("item_features"),
    ],
    "instructions": INSTRUCTIONS,
}

space_id = None
try:
    resp = requests.post(
        f"{host}/api/2.0/genie/spaces",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    print(resp.status_code, resp.text[:300])
    if resp.ok:
        space_id = resp.json().get("space_id") or resp.json().get("id")
except Exception as e:
    print(f"○ Genie API may have changed; create the space manually under SQL → Genie: {e}")

if space_id:
    for q in SAMPLE_QUESTIONS:
        try:
            requests.post(
                f"{host}/api/2.0/genie/spaces/{space_id}/sample-questions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"text": q}, timeout=30,
            )
        except Exception:
            pass

# COMMAND ----------
# MAGIC %md ## Wrap up

# COMMAND ----------
if space_id:
    record_asset(spark, OPS_TABLE, AssetRecord(
        chapter=6, asset_type="genie", name="CMEG Content Analytics", id=space_id,
        url=format_asset_url(workspace_url, "genie", space_id),
        description="Genie space scoped to gold + ml features, 8 sample questions",
    ))

chapter_complete(
    chapter=6, title="Genie Space",
    created=[
        ("genie", "CMEG Content Analytics",
         format_asset_url(workspace_url, "genie", space_id) if space_id else workspace_url + "/sql/genie"),
    ],
    next_label="00-CMEG-Demo-Intro (back to TOC)",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/00-CMEG-Demo-Intro",
)
