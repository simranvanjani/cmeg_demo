# CMEG Content Recommendation Demo

A plug-and-play Databricks demo showing a production-representative content recommendation system for media/OTT customers (TrueID-style).

## Install (no CLI — for customers)

1. In your Databricks workspace: **Workspace → Repos → Add Repo** and paste this repo's URL.
2. Open `_install/install.py` and click **Run All**.
3. Fill in catalog name, schema names, and data scale when prompted.
4. When install finishes, open `narrative/00_START_HERE` and follow the chapters.

## Install (CLI — for SEs/devs)

```bash
databricks bundle deploy --profile DEFAULT --target dev
databricks bundle run cmeg_orchestrator --profile DEFAULT --target dev
```

## What gets built

7 narrative chapters:

| # | Chapter |
|---|---|
| 0 | START_HERE (live TOC of all assets created) |
| 1 | Setup + synthetic data + PII tags |
| 2 | DLT medallion (bronze/silver/gold) with expectations + Liquid Clustering |
| 3 | Feature Store + Online Table + Vector Search index |
| 4 | Two-tower retrieval + LightGBM ranker (MLflow, signatures, @champion) |
| 5 | Chained serving endpoint with diversity rerank + GenAI explanation |
| 6 | Lakehouse Monitor + UC tags + audit query |
| 7 | Genie space scoped to gold |

See `docs/superpowers/specs/2026-06-02-cmeg-rec-demo-design.md` for the full design.

## Uninstall

```bash
databricks bundle run cmeg_cleanup --profile DEFAULT --target dev
databricks bundle destroy --profile DEFAULT --target dev
```
