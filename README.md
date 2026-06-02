# CMEG Content Recommendation Demo

A plug-and-play Databricks demo of a production-representative content recommendation system for media/OTT customers (TrueID-style).

## Install in 60 seconds (no CLI)

1. In your Databricks workspace, click **Workspace** → **Repos** → **Add Repo** and paste this repo's URL.
2. Open the cloned folder. You'll see **`INSTALL.py`** at the top.
3. Open `INSTALL.py`, attach any cluster, and click **Run all**.
4. When it finishes, click the green **Open START_HERE →** button in the output.

That's it. `INSTALL.py` creates the catalog (if missing), schemas, volume, DLT pipeline, orchestrator job, and cleanup job — all from one notebook via the Databricks SDK. No CLI, no `bundle deploy`.

### Catalog choice

The installer auto-detects existing catalogs in your workspace and defaults to `main` (present in every workspace). You can override the `catalog_name` widget at the top of `INSTALL.py` to any catalog you have CREATE SCHEMA privileges on.

## What gets built

7 narrative chapters under `narrative/`:

| # | Chapter | What it shows |
|---|---|---|
| 0 | `00_START_HERE` | Live TOC of every asset created across chapters |
| 1 | `01_setup_and_data` | Synthetic data generation, UC tags for ownership |
| 2 | `02_dlt_medallion` | DLT bronze→silver→gold with expectations + Liquid Clustering |
| 3 | `03_features_and_vectors` | Feature Store + Vector Search index over item embeddings |
| 4 | `04_train_and_register` | Two-tower retrieval + LightGBM ranker (MLflow signatures, `@champion`) |
| 5 | `05_serve_and_explain` | Chained Model Serving endpoint with diversity rerank + GenAI explanation |
| 6 | `06_monitor_and_govern` | Lakehouse Monitor on inference table + PII tags + audit query |
| 7 | `07_genie_space` | Genie space scoped to gold for non-technical exploration |

## Run all chapters at once

After `INSTALL.py` finishes, you can either:
- **Click through chapters manually** — open `narrative/00_START_HERE.py` and follow the links.
- **Run the orchestrator job** — open `cmeg_orchestrator` (linked in the install output) and click **Run now**. Runs all 7 chapters in dependency order (~20-30 min on small data).

## Uninstall

1. Open `cmeg_cleanup` job (created by `INSTALL.py`) and click **Run now**. This deletes the serving endpoint, vector index, monitor, and registered models.
2. (Optional) Drop the catalog manually if you want a fully clean state.

## For SEs / power users

A Databricks Asset Bundle (`databricks.yml` + `resources/`) is included for SE-driven deployments:

```bash
databricks bundle deploy --profile DEFAULT --target dev --var "catalog_name=my_catalog"
databricks bundle run cmeg_orchestrator --profile DEFAULT --target dev --var "catalog_name=my_catalog"
```

The bundle is optional. Customers should use `INSTALL.py` instead.

## Documentation

- Design spec: `docs/superpowers/specs/2026-06-02-cmeg-rec-demo-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-02-cmeg-rec-demo.md`
