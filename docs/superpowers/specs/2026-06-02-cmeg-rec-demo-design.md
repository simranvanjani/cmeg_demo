# CMEG Content Recommendation Demo — Design Spec

**Date:** 2026-06-02
**Owner:** Simran Vanjani (Scale SE, Databricks)
**Audience for the demo:** Communications, Media, and Entertainment (CMEG) customers, primarily OTT/streaming. Inspired by TrueID (True Corp, Thailand).

---

## 1. Problem statement

CMEG customers in OTT/streaming need a content recommendation system that's modern, governed, and observable. There is no plug-and-play Databricks demo today that walks a customer end-to-end from raw event ingestion through a production-representative recommendation model with built-in monitoring, governance, and a non-technical exploration surface.

This spec defines `cmeg_demo` — a notebook-walkthrough demo, packaged as a Databricks Asset Bundle (DAB) and installed via Git Folders, that builds a hybrid two-tower retrieval + GBT ranker + GenAI explanation pipeline on synthetic data, with DE and ML best practices baked in.

## 2. Goals

- **Plug-and-play install.** Customer-facing install path has zero CLI dependency. A non-engineer in the customer org can install the demo in under 5 minutes by adding a Git Folder and running one notebook.
- **Production-representative architecture.** The recommendation model and surrounding data/ML platform mirror what a real OTT platform would run at scale, not a toy example.
- **Best-practice DE and ML patterns applied inline.** Liquid Clustering, Auto Loader, Predictive Optimization, MLflow signatures, Online Tables, hyperparameter tuning, diversity reranking, idempotent writes — all present, with markdown explanations of *why*.
- **Inline monitoring and governance.** UC tags, dynamic views, DLT expectations, Lakehouse Monitoring on the inference table, AI/BI dashboard, sample audit queries — illustrative, not exhaustive.
- **Non-technical exploration surface.** A Genie space scoped to gold tables ships as part of the install, so business stakeholders can ask plain-English questions during the demo.
- **dbdemos-style navigation.** Each notebook ends with a link card showing what was created and a "Next chapter" button. A live table of contents in `00_START_HERE` reads from a persistent `_cmeg_assets` Delta table.
- **Configurable.** Customer can override catalog name, schema name, and data scale via DAB variables / install widgets.
- **End-to-end run time on small data: ~15-20 minutes** on a small cluster.

## 3. Non-goals

- mParticle / external CDP integration (out of scope — cloud-agnostic synthetic demo).
- Real telco, ad-manager, or third-party data ingestion.
- Multi-region / multi-workspace deployment patterns.
- Production-scale load testing of the serving endpoint.
- AI Gateway with PII guardrails on the GenAI layer (illustrative LLM explanation is enough; full gateway adds setup cost without proportional demo value).
- A custom frontend or Databricks App. Genie space covers the non-technical exploration need.
- Mosaic AI Agent / chat-based interface.
- Challenger/champion auto-promotion job (the pattern is taught in markdown; aliases are set manually).
- Multiple Lakehouse Monitors (one on the inference table is sufficient to teach the pattern).
- DBSQL Alerts and full system-tables governance dashboard (one ad-hoc query is enough).

## 4. Target audience and install ergonomics

### Two install paths from a single source of truth

| Path | Audience | Steps |
|---|---|---|
| **No-CLI (default)** | Customer org, business or technical | Workspace → Repos → Add Repo → paste GitHub URL → open `_install/install.py` → fill widgets → Run All |
| **CLI** | Field SEs, internal devs | `git clone …` → `databricks bundle deploy --profile DEFAULT --target dev` |

Both paths produce the same artifacts in the workspace under a configurable folder (default: `cmeg_demo`).

The bootstrapper notebook (`_install/install.py`) reads the DAB YAML files (`databricks.yml` + `resources/*.yml`) and uses the Databricks SDK (pre-installed on every cluster) to create catalog, schemas, jobs, DLT pipeline, and Genie space. Subsequent narrative chapters create models, vector index, serving endpoint, monitors, and dashboards.

## 5. Architecture

```
flowchart TB
  subgraph sources [Synthetic sources]
    APP[App events generator]
    CMS[Content catalog generator]
    USER[User profile generator]
  end

  subgraph ingest [Landing]
    VOL[UC Volume - parquet drops]
  end

  subgraph lake [Databricks Lakehouse — UC governed]
    subgraph dlt [DLT Pipeline]
      BRZ[Bronze - Auto Loader, schema evolution]
      SLV[Silver - conformed + expectations + Liquid Clustering]
      GLD[Gold - user_360 + interactions + recommendations]
    end
    FS[Feature Store - user and item features]
    OT[Online Table - low-latency feature lookup]
    VS[Vector Search - item embeddings index]
    ML[MLflow - 2 registered models with signatures + aliases]
    BS[Batch scoring job - nightly MERGE into gold.recommendations]
    SRV[Model Serving endpoint - chained inference]
    LLM[Foundation Model API - explanation generator]
  end

  subgraph observability [Governance and monitoring]
    LHM[Lakehouse Monitoring - inference table TimeSeries profile]
    TAGS[UC tags - pii, business_owner, cost_center]
    VIEWS[Dynamic view - PII masking on user_features]
    AUDIT[system.access.audit ad-hoc query]
  end

  subgraph out [Consumption surfaces]
    DASH[AI/BI overview dashboard]
    GENIE[Genie space - gold scope, 8 sample questions]
  end

  APP --> VOL
  CMS --> VOL
  USER --> VOL
  VOL --> BRZ
  BRZ --> SLV --> GLD
  GLD --> FS --> OT
  GLD --> VS
  FS --> ML
  VS --> ML
  ML --> BS --> GLD
  ML --> SRV
  OT --> SRV
  SRV --> LLM
  SRV --> LHM
  TAGS -.applied to.-> GLD
  VIEWS -.over.-> GLD
  GLD --> DASH
  GLD --> GENIE
```

## 6. Repo layout

```
cmeg_demo/                            # GitHub repo root
├── databricks.yml                    # DAB entry, variables, targets
├── resources/
│   ├── schemas.yml                   # UC catalog + bronze/silver/gold/ml schemas
│   ├── pipelines.yml                 # DLT pipeline declaration
│   ├── jobs.yml                      # init, nightly batch scoring, cleanup
│   └── dashboards.yml                # AI/BI dashboard refs
├── _install/
│   ├── install.py                    # Bootstrapper notebook — customer entry point
│   └── uninstall.py                  # Cleanup helper notebook
├── narrative/                        # Customer-clickable chapters
│   ├── 00_START_HERE.py
│   ├── 01_setup_and_data.py
│   ├── 02_dlt_medallion.py
│   ├── 03_features_and_vectors.py
│   ├── 04_train_and_register.py
│   ├── 05_serve_and_explain.py
│   ├── 06_monitor_and_govern.py
│   └── 07_genie_space.py
├── lib/cmeg/                         # Python package (wheel built by DAB)
│   ├── __init__.py
│   ├── companion.py                  # chapter_complete card, link helpers
│   ├── state.py                      # _cmeg_assets Delta table read/write
│   ├── config.py                     # Reads DAB variables / widget values
│   ├── data_gen.py                   # Synthetic data generators
│   ├── features.py                   # Feature engineering
│   ├── models.py                     # Two-tower + ranker training
│   ├── monitoring.py                 # Lakehouse Monitor setup helpers
│   ├── governance.py                 # UC tag + dynamic view helpers
│   └── genai.py                      # Foundation Model explanation prompt
├── dashboards/
│   └── cmeg_overview.lvdash.json
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_companion.py
├── .github/workflows/
│   └── validate.yml                  # databricks bundle validate on PRs
└── README.md
```

## 7. Notebook chapter contract

Every narrative notebook follows the same structure:

```
Cell 1: Markdown header — chapter title, what gets built, prerequisites
Cell 2: %run ../lib/cmeg/setup_session  (loads config + companion)
Cell 3..N: Narrated work — markdown + code, one concept per cell
Last cell: companion.chapter_complete(chapter=N, created=[...], next="...")
```

### `chapter_complete()`

Writes one row per created asset to `{catalog}.{ops_schema}._cmeg_assets`:

| Column | Type | Example |
|---|---|---|
| `chapter` | int | 2 |
| `asset_type` | string | `dlt_pipeline` |
| `name` | string | `cmeg_silver_pipeline` |
| `id` | string | `abc-123-def` |
| `url` | string | `https://...#pipeline/abc-123-def` |
| `description` | string | "DLT bronze→silver→gold pipeline" |
| `created_at` | timestamp | `2026-06-02 08:30:00` |

Then renders a `displayHTML()` card with `[Open ↗]` buttons per asset and a `[Next chapter →]` button.

### `00_START_HERE.py`

Queries `_cmeg_assets` and renders a live TOC grouped by chapter. Pending chapters render with `[Run ↗]` buttons that link to the unrun notebook. Completed chapters render with `[Open ↗]` buttons per asset.

## 8. Chapter-by-chapter breakdown

### Chapter 00 — `START_HERE`

- Project pitch + architecture diagram (Mermaid).
- Live TOC from `_cmeg_assets`.
- "Run all chapters" button → triggers `cmeg_orchestrator` job.
- "Uninstall" button → triggers `cmeg_cleanup` job.

### Chapter 01 — `setup_and_data`

- Reads widget values for catalog/schema/data_scale.
- Creates schemas (bronze, silver, gold, ml, ops) if missing.
- Creates UC Volume for landing.
- Applies UC tags: `pii=true` on `user_profiles.email/phone`, `business_owner=cmeg_demo`, `cost_center=cmeg_demo` on all tables.
- Enables Predictive Optimization on the catalog.
- Generates synthetic data: 10K users, 5K content items, 500K interactions, written as parquet to the UC Volume.
- Chapter card: links to catalog, schemas, volume, sample data preview.

### Chapter 02 — `dlt_medallion`

- DLT pipeline definition with:
  - **Bronze:** Auto Loader from UC Volume, `schemaEvolutionMode = "addNewColumns"`.
  - **Silver:** Conformed types, deduplication, 3-4 expectations (`@dlt.expect_or_drop`).
  - **Gold:** Aggregated user 360 view, interactions mart, with Liquid Clustering on `(user_id, content_id)`.
- Markdown sidebar: `APPLY CHANGES INTO` example for SCD2 (teaching only, not executed).
- Triggers the pipeline, polls until success.
- Chapter card: pipeline URL, gold table URLs, lineage graph link.

### Chapter 03 — `features_and_vectors`

- Builds user features (watch_count_7d, fav_genre, avg_session_minutes) and item features (genre, popularity, freshness) via Feature Engineering API.
- Creates an Online Table mirror of `gold.user_features` for low-latency lookup at serving.
- Generates simple content embeddings (e.g., text embedding of title + synopsis via Foundation Models) and indexes them in Vector Search.
- Chapter card: feature table URLs, online table URL, vector index URL.

### Chapter 04 — `train_and_register`

- **Two-tower retrieval model:** TensorFlow Recommenders or scikit-based two-tower. Trains on `gold.interactions`. Logs to MLflow with:
  - `mlflow.models.infer_signature(...)` and an input example.
  - Model description set via `mlflow.set_registered_model_alias(..., "@champion")`.
- **GBT ranker:** LightGBM trained on interaction features + retrieval scores. Optuna sweep, 5-10 trials. Logged with signature + input example. Aliased `@champion`.
- Chapter card: experiment URL, two registered model URLs, run comparison view URL.

### Chapter 05 — `serve_and_explain`

- Creates a chained Model Serving endpoint:
  - Stage 1: Two-tower retrieval — top-100 candidate items from Vector Search.
  - Stage 2: GBT ranker — rescore with full features (Online Table lookup).
  - Stage 3: Diversity reranker — genre dedupe, MMR-style spread.
  - Stage 4: Foundation Model call — generate "why we recommend this" copy per top item.
- Enables inference table on the endpoint (auto-captures requests + responses).
- Creates one dynamic view `gold.user_features_masked` that masks `email` and `phone` unless caller is in group `cmeg_pii_readers`.
- Chapter card: endpoint URL, inference table URL, dynamic view DDL link.

### Chapter 06 — `monitor_and_govern`

- Creates a Lakehouse Monitor on the inference table (`TimeSeries` profile on the request timestamp, monitoring feature drift + prediction distribution).
- Imports and binds the AI/BI overview dashboard (`dashboards/cmeg_overview.lvdash.json`) showing watch behavior + recommendation effectiveness over time.
- Runs one ad-hoc SQL query against `system.access.audit` filtered to the demo catalog — displays the result as a teaching example.
- Markdown sidebar: champion/challenger pattern explained, with example commands (not run).
- Chapter card: monitor URL, dashboard URL, sample audit query link.

### Chapter 07 — `genie_space`

- Creates a Genie space via Databricks SDK, scoped to `{catalog}.gold` (`interactions`, `user_features`, `content_catalog`, `recommendations`).
- Seeds Genie instructions with a CMEG domain glossary (e.g., `watch_seconds = total seconds played; only count if > 30s; cold_start = users with < 5 interactions`).
- Seeds 8 sample questions:
  1. *What are the top 10 most-watched shows last week?*
  2. *Which content genres have the highest completion rate?*
  3. *Show me engagement by age segment for drama content.*
  4. *Which users have the highest 7-day watch time?*
  5. *What share of viewing comes from recommended vs. self-discovered content?*
  6. *Which content has the highest cold-start ranker score this week?*
  7. *Compare weekday vs. weekend watch patterns.*
  8. *Top 5 content items by recommendations served, last 24 hours.*
- Chapter card: Genie space URL, sample questions list.

## 9. DE best practices applied

| Practice | Where |
|---|---|
| Medallion architecture | Chapter 02 (DLT bronze→silver→gold) |
| Auto Loader for ingestion | Chapter 02 (bronze) |
| Schema evolution | Chapter 02 (`schemaEvolutionMode = "addNewColumns"`) |
| DLT expectations for data quality | Chapter 02 (3-4 rules on silver/gold) |
| Liquid Clustering | Chapter 02 (`gold.interactions`, `silver.events` clustered on `user_id, content_id`) |
| Predictive Optimization | Chapter 01 (enabled on catalog) |
| Photon | `resources/jobs.yml` (`photon = true`) |
| Pinned DBR | `resources/jobs.yml` (e.g., `15.4.x-cpu-ml-scala2.12`) |
| Idempotent batch writes | Batch scoring uses `MERGE INTO gold.recommendations USING ...` |
| CDC pattern (taught) | Chapter 02 markdown sidebar on `APPLY CHANGES INTO` |
| UC governance | Chapters 01, 05, 06 |
| Lineage | Surfaced via UC lineage UI; chapter 02 includes link |

## 10. ML best practices applied

| Practice | Where |
|---|---|
| MLflow tracking | Chapter 04 |
| Model signatures + input examples | Chapter 04 (both models) |
| Model Registry with aliases | Chapter 04 (`@champion`) |
| Feature Store / Feature Engineering | Chapter 03 |
| Online Tables for low-latency lookup | Chapter 03 + chapter 05 |
| Vector Search | Chapter 03 |
| Hyperparameter tuning | Chapter 04 (Optuna sweep on ranker) |
| Model description / docs | Chapter 04 |
| Two-stage retrieval + ranking architecture | Chapters 04 + 05 |
| Diversity reranking | Chapter 05 (genre dedupe) |
| Inference tables | Chapter 05 |
| Lakehouse Monitoring on predictions | Chapter 06 |
| Champion/challenger pattern (taught) | Chapter 06 markdown sidebar |
| Reproducible env | Pinned DBR via `jobs.yml` |

## 11. Monitoring and governance specifics

| Layer | What | Where |
|---|---|---|
| Data quality | DLT expectations on silver/gold | Chapter 02 |
| Model inference monitoring | Lakehouse Monitor TimeSeries on inference table | Chapter 06 |
| UC tags | `pii`, `business_owner`, `cost_center` applied during chapter 01 | Chapter 01 |
| Access control | One dynamic view masking PII on `gold.user_features` | Chapter 05 |
| Lineage | UC native (no extra code), surfaced via dashboard link | Chapter 06 |
| Audit | One ad-hoc query against `system.access.audit` for the demo catalog | Chapter 06 |
| AI/BI dashboard | `dashboards/cmeg_overview.lvdash.json` showing watch + recommendation metrics | Chapter 06 |

## 12. DAB variables

```yaml
variables:
  catalog_name:
    description: "UC catalog (created if missing). Customer must have CREATE CATALOG, or pre-create."
    default: "cmeg_demo"
  bronze_schema:
    default: "bronze"
  silver_schema:
    default: "silver"
  gold_schema:
    default: "gold"
  ml_schema:
    default: "ml"
  ops_schema:
    default: "ops"
  data_scale:
    description: "small | medium | large"
    default: "small"
  serving_endpoint_enabled:
    description: "Create the real-time serving endpoint (DBU cost while running)"
    default: true
  genai_model:
    description: "Foundation Model for explanation copy"
    default: "databricks-meta-llama-3-3-70b-instruct"
  dbr_version:
    default: "15.4.x-cpu-ml-scala2.12"
```

The `_install/install.py` bootstrapper exposes the first three (catalog, data_scale, serving_endpoint_enabled) as widgets at the top of the notebook.

## 13. CI/CD

`.github/workflows/validate.yml`:
- Triggers on PR.
- Runs `pip install databricks-cli` then `databricks bundle validate --target dev`.
- Runs `pytest tests/` for the `lib/cmeg/` package.

No deployment automation in the demo — customers install via Git Folder; SEs deploy via CLI. CI is for source-of-truth integrity only.

## 14. Cleanup

Two-step uninstall:

1. **Run `cmeg_cleanup` job** (or open `_install/uninstall.py`). Deletes SDK-created assets: serving endpoint, vector index, online table, monitors, registered models, dashboards, Genie space, `_cmeg_assets` table rows.
2. **`databricks bundle destroy`** (CLI users) or manual catalog drop with cascade (UI users). Removes DAB-declared resources: schemas, DLT pipeline, jobs.

The bootstrapper notebook also offers a "💥 Uninstall everything" cell that runs step 1 and then drops the catalog (with confirmation prompt).

## 15. Open questions

- Exact Foundation Model to default to — `databricks-meta-llama-3-3-70b-instruct` assumed available in customer workspace. Fallback strategy if not?
- Whether to ship the `cmeg_overview.lvdash.json` dashboard as a hand-crafted JSON or to generate it programmatically via the Lakeview API at install time (programmatic = more portable but more code).
- Whether the `_install/install.py` bootstrapper should depend on `databricks-bundles` Python package (if available) or parse YAML manually with PyYAML (more code, no extra dependency).

## 16. Future work (out of this spec)

- AI Gateway with PII guardrails on the GenAI endpoint.
- Full system-tables governance dashboard with cost attribution.
- DBSQL Alerts on monitor drift + cost thresholds.
- Challenger/champion auto-promotion via scheduled A/B job.
- Multi-region / multi-workspace bundle variants.
- Adding Mosaic AI Agent as an alternative non-technical surface.
- CDC pattern from a real source (currently taught in markdown only).
