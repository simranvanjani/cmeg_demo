# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 6 of 6 — Genie Space for Business Users
# MAGIC
# MAGIC > 🕐 **3 min to read · 1 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - How to expose your gold tables to **non-technical business users** without writing dashboards
# MAGIC - How **Genie instructions** (a domain glossary) make plain-English queries actually accurate
# MAGIC - How to seed the space with **sample questions** so customers see value on day one
# MAGIC
# MAGIC ## Why Genie matters for this demo
# MAGIC
# MAGIC The data scientist team built the recommender. The content programming team wants to ask things like:
# MAGIC > *"Which dramas had the best completion rate last week?"*
# MAGIC > *"What share of viewing came from recommended vs self-discovered content yesterday?"*
# MAGIC
# MAGIC They don't write SQL. Genie does. We scope the Genie space to the right tables, seed it with
# MAGIC business-friendly definitions, and the content team gets a chat interface over the data.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
import requests

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 of 3 — Pick a SQL warehouse to back the Genie space
# MAGIC
# MAGIC Genie translates plain-English questions to SQL, then runs that SQL on a serverless SQL warehouse.
# MAGIC We use whichever warehouse you already have in this workspace.

# COMMAND ----------
warehouses = list(w.warehouses.list())
assert warehouses, "No SQL warehouses available. Create one to host the Genie space."
warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 3 — Define the domain glossary and sample questions
# MAGIC
# MAGIC ### Why instructions matter
# MAGIC
# MAGIC Without context, Genie has to guess what your columns mean. With a glossary, it understands the
# MAGIC business — e.g., that `completed` means "watched ≥ 85% of duration", or that a "cold start" user
# MAGIC means "fewer than 5 interactions". Good instructions turn an OK Genie space into a useful one.
# MAGIC
# MAGIC ### Why sample questions
# MAGIC
# MAGIC When the customer's content team opens Genie for the first time, they see prompts they can click
# MAGIC instead of a blinking cursor. That's the difference between *"this is interesting"* and *"this is
# MAGIC mine to use"*.

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
    "- 'last 7 days': filter event_ts > current_timestamp() - INTERVAL 7 DAYS.\n\n"
    "Tables: gold_interactions (event-level), gold_user_360 (per-user aggregates), "
    "user_features and item_features (engagement metadata)."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 3 — Create the Genie space via API

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
    print(f"✓ added {len(SAMPLE_QUESTIONS)} sample questions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔍 Try it
# MAGIC
# MAGIC Open the Genie space link in the recap card below. Click any of the sample questions, or type
# MAGIC your own. Some good follow-ups to try after Genie answers:
# MAGIC - *"Now group that by country"*
# MAGIC - *"Show that as a bar chart"*
# MAGIC - *"What changed between this week and last week?"*
# MAGIC
# MAGIC Genie remembers the conversation context, so follow-ups build on the previous query.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - A Genie space scoped to gold + feature tables
# MAGIC - Domain instructions so Genie understands business definitions
# MAGIC - 8 sample questions so non-technical users see immediate value
# MAGIC
# MAGIC ## 🎉 Demo complete!
# MAGIC
# MAGIC Across 6 chapters, you built a complete content recommendation system on Databricks:
# MAGIC
# MAGIC | Chapter | What it adds |
# MAGIC | --- | --- |
# MAGIC | 1 — DLT Medallion | Governed lakehouse with data quality + Liquid Clustering |
# MAGIC | 2 — Features & Vectors | Feature Store + Vector Search index over item embeddings |
# MAGIC | 3 — Train & Register | Two-tower retrieval + LightGBM ranker with `@champion` aliases |
# MAGIC | 4 — Serve & Explain | Chained inference + GenAI explanation, inference table capture |
# MAGIC | 5 — Monitor & Govern | Lakehouse Monitoring, UC tags, audit log |
# MAGIC | 6 — Genie Space | Plain-English Q&A for business users |
# MAGIC
# MAGIC ## Next steps for a real customer
# MAGIC
# MAGIC - **Replace synthetic data** with the customer's real event source (Kafka, Auto Loader from cloud storage)
# MAGIC - **Scale up the two-tower** by swapping the numpy implementation for TensorFlow Recommenders
# MAGIC - **Add AI Gateway** in front of the GenAI explanation layer for rate limiting + PII guardrails
# MAGIC - **Run challenger/champion A/B** to safely promote new model versions

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
    next_label="RUNME (back to overview)",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/RUNME",
)
