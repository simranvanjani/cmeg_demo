# CMEG Content Recommendation Accelerator

A plug-and-play Databricks accelerator for content recommendation on media/OTT platforms (TrueID-style). Follows the [Databricks Industry Solution Accelerator](https://www.databricks.com/solutions/accelerators) pattern.

## Install in 3 steps

1. **Clone into your workspace.** Workspace → **Repos** → **Add Repo** → paste this repo's URL.
2. **Edit `config.py`.** Set `CATALOG` to a catalog that already exists in your workspace (defaults to `main`). Optionally set `SCHEMA`, `DATA_SCALE`.
3. **Open `RUNME.py` and click Run All.** It validates the catalog, generates synthetic data, creates the DLT pipeline and the orchestrator workflow, and shows a clickable button to run the workflow end-to-end.

The orchestrator runs all 6 chapters in dependency order (~20-30 min on `small` data). You can also walk through the chapters manually.

## Layout

```
cmeg_demo/
├── RUNME.py                          ← run this once. Validates env, creates the workflow.
├── config.py                         ← edit catalog/schema/scale here
├── 02_dlt_medallion.py               ← chapter 1
├── 03_features_and_vectors.py        ← chapter 2
├── 04_train_and_register.py          ← chapter 3
├── 05_serve_and_explain.py           ← chapter 4
├── 06_monitor_and_govern.py          ← chapter 5
├── 07_genie_space.py                 ← chapter 6
├── _resources/
│   ├── 00-setup.py                   ← %run by every chapter (zero boilerplate per chapter)
│   ├── 01-generate-data.py
│   ├── 02-create-resources.py
│   ├── 99-uninstall.py
│   └── _dlt_pipeline.py
├── lib/cmeg/                         ← shared Python helpers
└── docs/                             ← design spec + implementation plan
```

## The 6 chapters

| # | Chapter | Best practices applied |
|---|---|---|
| 02 | DLT Medallion | Auto Loader, schema evolution, DLT expectations, Liquid Clustering |
| 03 | Features & Vectors | Feature Engineering API, Vector Search with delta-sync index |
| 04 | Train & Register | Two-tower retrieval + LightGBM ranker, Optuna HPO, MLflow signatures, UC Registry @champion |
| 05 | Serve & Explain | Chained inference, diversity rerank, Foundation Model "why" copy, inference table auto-capture, dynamic view PII masking |
| 06 | Monitor & Govern | Lakehouse Monitoring on inference table, UC tags, audit query |
| 07 | Genie Space | Plain-English Q&A for business users scoped to gold tables |

## What if `main` doesn't have permissions?

`config.py` accepts any existing catalog. If `main` doesn't have CREATE SCHEMA for you, set `CATALOG` to a catalog you own. If `RUNME` fails on catalog validation, the error message lists every catalog you can see — copy one and retry.

## Uninstall

1. Run the **`cmeg_cleanup`** workflow (created by `RUNME`) — deletes serving endpoint, vector index, monitor, registered models.
2. (Optional) Drop the schema manually if you want a fully clean state.

## For SE / power users

A Databricks Asset Bundle (`databricks.yml` + `resources/`) is included:

```bash
databricks bundle deploy --profile DEFAULT --target dev --var "catalog_name=my_catalog"
databricks bundle run cmeg_orchestrator --profile DEFAULT --target dev --var "catalog_name=my_catalog"
```

## Docs

- Design spec: `docs/superpowers/specs/2026-06-02-cmeg-rec-demo-design.md`
- Plan: `docs/superpowers/plans/2026-06-02-cmeg-rec-demo.md`
