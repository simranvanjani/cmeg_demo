# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 5 of 6 — Monitor and Govern
# MAGIC
# MAGIC > 🕐 **4 min to read · 2 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - **Why deployed recommenders silently degrade** and how Lakehouse Monitoring catches it
# MAGIC - How **UC tags** turn ad-hoc table metadata into structured governance data
# MAGIC - How `system.access.audit` gives you a complete audit log of every UC operation
# MAGIC - The **champion/challenger** A/B pattern for safe model promotion (concept only — we don't run it)
# MAGIC
# MAGIC ## Why monitor at all?
# MAGIC
# MAGIC A recommender that worked perfectly at launch slowly degrades. Two things go wrong silently:
# MAGIC
# MAGIC 1. **Input drift** — viewing habits shift (new World Cup season, a viral show drops). The features
# MAGIC    your model trained on no longer match production.
# MAGIC 2. **Prediction drift** — outputs skew (suddenly recommending the same 5 items to everyone, or
# MAGIC    average scores creeping up over time).
# MAGIC
# MAGIC **Lakehouse Monitoring** sits on top of the inference table (logged via AI Gateway in chapter 4) and
# MAGIC computes drift metrics nightly. You see them as charts on the table page — and can wire DBSQL
# MAGIC alerts on them.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
from databricks.sdk.service.catalog import MonitorTimeSeries
from cmeg.governance import apply_pii_tags, apply_table_owner_tag
from cmeg.monitoring import build_monitor_dir

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 of 3 — Apply UC governance tags
# MAGIC
# MAGIC Three tags we apply across the demo:
# MAGIC - **`pii=true`** on columns containing personally identifiable data (here, `fav_genre`)
# MAGIC - **`business_owner=cmeg_demo`** so anyone looking at the table knows who owns it
# MAGIC - **`cost_center=cmeg_demo`** for chargeback reporting
# MAGIC
# MAGIC Tags are searchable in Catalog Explorer ("show me all tables tagged `pii=true`"), filterable
# MAGIC in DBSQL queries, and feed governance dashboards.

# COMMAND ----------
user_feat = FQ("user_features")
apply_pii_tags(spark, user_feat, ["fav_genre"])
apply_table_owner_tag(spark, user_feat, owner="cmeg_demo", cost_center="cmeg_demo")
print(f"✓ tagged {user_feat} with pii + owner + cost_center")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 3 — Create a Lakehouse Monitor on the inference table
# MAGIC
# MAGIC We use the `TimeSeries` profile because the inference table has a natural time column
# MAGIC (`request_time`, from the AI Gateway inference table). The monitor computes daily slices and reports:
# MAGIC - **Numeric drift**: how feature distributions have shifted over time
# MAGIC - **Categorical drift**: how genre/device/country distributions have shifted
# MAGIC - **Data quality**: nulls, type errors
# MAGIC
# MAGIC The monitor creates its own dashboard automatically. You'll see it linked from the inference
# MAGIC table's page in Catalog Explorer.
# MAGIC
# MAGIC **⏳ Note:** The monitor needs at least one row in the inference table to compute meaningful
# MAGIC metrics — if no one has called the endpoint yet, the dashboard will be empty until traffic flows.

# COMMAND ----------
inference_table = FQ("cmeg_inference_payload")
# AI Gateway inference tables use `request_time`; pick whatever timestamp column exists.
try:
    cols = [c.name for c in spark.table(inference_table).schema]
    ts_col = next((c for c in ("request_time", "timestamp_ms", "__db_request_time") if c in cols), None)
except Exception:
    ts_col = "request_time"   # table not created until the endpoint serves a request

try:
    w.quality_monitors.create(
        table_name=inference_table,
        assets_dir=build_monitor_dir(current_user),
        output_schema_name=f"{CATALOG}.{SCHEMA}",
        time_series=MonitorTimeSeries(timestamp_col=ts_col, granularities=["1 day"]),
    )
    print(f"✓ monitor created on {inference_table} (timestamp col: {ts_col})")
except Exception as e:
    print(f"○ monitor may already exist or inference table not yet populated: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 3 — Audit query against system tables
# MAGIC
# MAGIC `system.access.audit` is a built-in UC system table that logs every operation against Unity
# MAGIC Catalog: who read what, who created/dropped what, who changed permissions. Filtering to the
# MAGIC demo catalog gives us a complete activity log for the last 24h.
# MAGIC
# MAGIC **🔍 Try this:** Modify the filter to look at the last 7 days, or filter by `action_name` to
# MAGIC find specific operations (e.g., `LIST_TABLE` for read patterns, `CREATE_TABLE` for new objects).

# COMMAND ----------
try:
    display(spark.sql(f"""
        SELECT event_time, action_name, request_params:full_name_arg AS table_arg
        FROM system.access.audit
        WHERE event_time > current_timestamp() - INTERVAL 1 DAY
          AND request_params:full_name_arg LIKE '{CATALOG}.%'
        ORDER BY event_time DESC
        LIMIT 20
    """))
except Exception as e:
    print(f"○ system.access.audit not yet provisioned in this workspace: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Sidebar — the champion/challenger A/B pattern
# MAGIC
# MAGIC For production model upgrades, never replace your champion outright. Instead:
# MAGIC
# MAGIC 1. **Register the new model version** as the same registered model name.
# MAGIC 2. **Alias it `@challenger`** instead of `@champion`.
# MAGIC 3. **Route a fraction of endpoint traffic** to it via `served_entities[].traffic_percentage`
# MAGIC    on the serving endpoint config.
# MAGIC 4. **Compare champion vs challenger** in the inference table (they're labeled per-row).
# MAGIC 5. **Promote** if challenger wins: `client.set_registered_model_alias(name, 'champion', version)`.
# MAGIC
# MAGIC We don't run this in the demo, but the inference table + monitoring infrastructure already in
# MAGIC place is what makes it safe to do.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - A Lakehouse Monitor on the inference table (drift metrics, auto-dashboard)
# MAGIC - UC tags applied for PII + ownership
# MAGIC - A reusable audit query against system tables
# MAGIC
# MAGIC ## Up next — Chapter 6: Genie Space
# MAGIC
# MAGIC We create an **AI/BI Genie space** so non-technical stakeholders can ask plain-English questions
# MAGIC about the data. *"Which content genres have the highest completion rate?"* → Genie writes the SQL,
# MAGIC runs it, shows a chart.

# COMMAND ----------
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=5, asset_type="monitor", name=inference_table, id=inference_table,
    url=format_asset_url(workspace_url, "monitor", inference_table),
    description="Lakehouse Monitor on inference table",
))

chapter_complete(
    chapter=5, title="Monitor and Govern",
    created=[
        ("monitor", inference_table, format_asset_url(workspace_url, "monitor", inference_table)),
    ],
    next_label="07_genie_space",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/07_genie_space",
)
