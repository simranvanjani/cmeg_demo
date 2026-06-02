# CMEG Content Recommendation Demo

A plug-and-play Databricks demo of a production-representative content recommendation system for media/OTT customers (TrueID-style).

## Install (60 seconds, no CLI)

1. **Clone the repo into your Databricks workspace**: Workspace → Repos → **Add Repo** → paste this repo's URL.
2. **Edit `config.py`** in the cloned folder. Set:
   - `CATALOG` — defaults to `main` (works in every workspace). Change to your own catalog if you have one.
   - `SCHEMA` — defaults to `cmeg_demo`. All tables created here.
   - (Optional) `DATA_SCALE` — `small` / `medium` / `large`.
3. **Open `00-CMEG-Demo-Intro.py`** and click **Run all**.

That's it. The intro notebook generates synthetic data, creates the DLT pipeline and jobs, and shows a live table of contents linking to every chapter.

## Structure

```
cmeg_demo/
├── config.py                       ← edit catalog/schema here (no widgets)
├── 00-CMEG-Demo-Intro.py           ← entry: architecture, install, TOC
├── 01-DLT-Medallion.py             ← chapter 1
├── 02-Features-and-Vectors.py      ← chapter 2
├── 03-Train-and-Register.py        ← chapter 3
├── 04-Serve-and-Explain.py         ← chapter 4
├── 05-Monitor-and-Govern.py        ← chapter 5
├── 06-Genie-Space.py               ← chapter 6
├── _resources/
│   ├── 00-setup.py                 ← %run by every chapter (no boilerplate)
│   ├── 01-generate-data.py         ← one-time data generation
│   ├── 02-create-resources.py     ← one-time pipeline/job creation
│   ├── 99-uninstall.py             ← cleanup
│   └── _dlt_pipeline.py            ← DLT pipeline definition
├── lib/cmeg/                       ← shared Python helpers (companion cards, models, features, ...)
├── tests/                          ← pytest unit tests on lib/cmeg/
└── docs/superpowers/               ← design spec + implementation plan
```

## The 6 chapters

| # | Chapter | What you'll see |
|---|---|---|
| 1 | DLT Medallion | Bronze→silver→gold with Auto Loader, expectations, Liquid Clustering |
| 2 | Features & Vectors | Feature Store (user + item) + Vector Search index over item embeddings |
| 3 | Train & Register | Two-tower retrieval + LightGBM ranker (Optuna), both registered with `@champion` aliases |
| 4 | Serve & Explain | Chained serving endpoint: retrieval → ranker → diversity → GenAI explanation |
| 5 | Monitor & Govern | Lakehouse Monitor on inference table + UC tags + sample audit query |
| 6 | Genie Space | Plain-English Q&A for business users, scoped to gold tables |

## Run everything end-to-end

After `00-CMEG-Demo-Intro` finishes, you can run all 6 chapters as a single job:

- **From the UI**: open the `cmeg_orchestrator` job (linked in the install output) → **Run now**.
- **From the CLI** (SEs): `databricks bundle run cmeg_orchestrator --profile DEFAULT --target dev`.

Total runtime: ~20-30 min on `small` data.

## Uninstall

Either:
- **Run the `cmeg_cleanup` job** (created by the install) — deletes endpoint, vector index, monitor, registered models.
- Or open `_resources/99-uninstall.py` and Run All.

Then optionally drop the schema yourself.

## For SEs / power users

A Databricks Asset Bundle is included for SE-driven deploys:

```bash
databricks bundle deploy --profile DEFAULT --target dev
databricks bundle run cmeg_orchestrator --profile DEFAULT --target dev
```

The bundle is optional — customers should use the `00-CMEG-Demo-Intro` notebook instead.

## Design and plan docs

- Spec: `docs/superpowers/specs/2026-06-02-cmeg-rec-demo-design.md`
- Plan: `docs/superpowers/plans/2026-06-02-cmeg-rec-demo.md`
