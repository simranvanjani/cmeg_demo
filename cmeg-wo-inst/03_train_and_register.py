# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chapter 3 of 6 — Train and Register Models
# MAGIC
# MAGIC > 🕐 **7 min to read · 5 min to run**
# MAGIC
# MAGIC ## What you'll learn
# MAGIC
# MAGIC - **Why production recommenders use TWO models** (retrieval + ranker), not one
# MAGIC - How **MLflow** captures every experiment with signatures, input examples, and lineage
# MAGIC - How **Optuna** integrates with MLflow for hyperparameter search
# MAGIC - How **Unity Catalog Model Registry aliases** (`@champion`, `@challenger`) replace stage strings
# MAGIC
# MAGIC ## The two-stage retrieval+ranking architecture
# MAGIC
# MAGIC ```
# MAGIC                       ┌─────────────────┐   ~100 candidates    ┌──────────────┐    top-N    ┌─────┐
# MAGIC POST /recs/u_12345 ──▶│ Two-Tower       │ ─────────────────────▶│ GBT Ranker   │ ───────────▶│ App │
# MAGIC                       │ Retrieval       │   from 5,000 items   │ (LightGBM)   │             │     │
# MAGIC                       │ (this chapter)  │                       │ (this chap.) │             └─────┘
# MAGIC                       └─────────────────┘                       └──────────────┘
# MAGIC                          cheap, fast                              expensive features,
# MAGIC                          item embedding                            rich scoring
# MAGIC                          dot product
# MAGIC ```
# MAGIC
# MAGIC Every large content recommender (YouTube, Spotify, Netflix, TikTok) uses this pattern. The
# MAGIC retriever is cheap and approximate; the ranker is expensive and precise. Together they let you
# MAGIC personalize over millions of items at sub-100ms latency.

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
# MAGIC %pip install -q lightgbm optuna scikit-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_resources/00-setup

# COMMAND ----------
import mlflow, mlflow.pyfunc, mlflow.lightgbm
import pandas as pd
from mlflow.models.signature import infer_signature
from cmeg.models import train_two_tower, train_ranker

mlflow.set_registry_uri("databricks-uc")
experiment_path = f"/Users/{current_user}/cmeg_demo_experiments"
mlflow.set_experiment(experiment_path)
client = mlflow.MlflowClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 of 4 — Load training data
# MAGIC
# MAGIC We pull a sample of `gold_interactions` (capped at 200K for the demo) plus the two feature
# MAGIC tables we built in chapter 2. The interactions provide the (user, item, watch_seconds) signal;
# MAGIC the features provide the side data the ranker uses for scoring.

# COMMAND ----------
inter = spark.table(FQ("gold_interactions")).limit(200_000).toPandas()
user_feats = spark.table(FQ("user_features")).toPandas()
item_feats = spark.table(FQ("item_features")).toPandas()
print(f"interactions={len(inter):,}  users={len(user_feats):,}  items={len(item_feats):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 of 4 — Train the two-tower retrieval model
# MAGIC
# MAGIC ### What is a two-tower model?
# MAGIC
# MAGIC Two-tower = one neural network for users, another for items. Both project into the same
# MAGIC embedding space. At serving time:
# MAGIC 1. Look up the user's embedding (fast)
# MAGIC 2. Find items whose embeddings are closest (dot product, can use Vector Search)
# MAGIC
# MAGIC We're using a lightweight matrix-factorization SGD implementation (numpy, 32-dim embeddings,
# MAGIC 3 epochs) to keep the demo fast. In production you'd swap in TensorFlow Recommenders or
# MAGIC PyTorch. The output shape — a user embedding table and an item embedding table — is identical.
# MAGIC
# MAGIC ### MLflow best practices applied here
# MAGIC
# MAGIC - **`mlflow.models.infer_signature`** — captures input/output schema for the registered model
# MAGIC - **`input_example`** — a 1-row sample logged with the model so future callers can introspect
# MAGIC - **`registered_model_name`** — auto-registers to UC Model Registry on log

# COMMAND ----------
tt_model_name = FQ("cmeg_two_tower")

with mlflow.start_run(run_name="two_tower"):
    art = train_two_tower(inter, n_factors=32, n_epochs=3)

    class TwoTower(mlflow.pyfunc.PythonModel):
        def __init__(self, item_emb_df, user_emb_df):
            import numpy as _np
            self.item_ids = item_emb_df["content_id"].tolist()
            self.item_matrix = _np.array(item_emb_df["embedding"].tolist())
            self.user_emb = user_emb_df.set_index("user_id")["embedding"].to_dict()
            self.dim = self.item_matrix.shape[1]

        def predict(self, context, model_input):
            import numpy as _np
            out = []
            for _, row in model_input.iterrows():
                u = _np.array(self.user_emb.get(row["user_id"], [0.0] * self.dim))
                scores = self.item_matrix @ u
                top_idx = _np.argsort(-scores)[:100]
                out.append([self.item_ids[i] for i in top_idx])
            return out

    model = TwoTower(art.item_embeddings, art.user_embeddings)
    input_ex = pd.DataFrame({"user_id": [inter["user_id"].iloc[0]]})
    sig = infer_signature(input_ex, model.predict(None, input_ex))
    mlflow.pyfunc.log_model(
        artifact_path="model", python_model=model,
        signature=sig, input_example=input_ex,
        registered_model_name=tt_model_name,
    )

tt_latest = client.get_registered_model(tt_model_name).latest_versions[0].version
client.set_registered_model_alias(tt_model_name, "champion", version=tt_latest)
client.update_registered_model(tt_model_name, description="Two-tower retrieval: returns top-100 content_ids for a user_id. Embedding dim=32.")
print(f"✓ {tt_model_name} v{tt_latest} aliased @champion")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 of 4 — Train the LightGBM ranker with Optuna hyperparameter search
# MAGIC
# MAGIC Now we train the second stage. The ranker takes (user_features × item_features) pairs and
# MAGIC predicts `P(completed)` — the probability the user will finish watching this title.
# MAGIC
# MAGIC ### Why Optuna?
# MAGIC
# MAGIC We run 5 trials with Bayesian optimization over `num_leaves`, `learning_rate`, `n_estimators`,
# MAGIC `min_child_samples`. Each trial is logged as a nested MLflow run, and the best one becomes
# MAGIC the registered model. In production you'd run 50-200 trials.
# MAGIC
# MAGIC **🔍 What to look for after this runs:** Open the **Experiments** tab in the left nav, find
# MAGIC `cmeg_demo_experiments`, and you'll see one parent run for the ranker with multiple Optuna
# MAGIC child trials, each with its own metrics and hyperparameter set.

# COMMAND ----------
r_model_name = FQ("cmeg_ranker")

ranker_data = (
    inter.merge(user_feats.add_prefix("u_"), left_on="user_id", right_on="u_user_id", how="left")
    .merge(item_feats.add_prefix("i_"), left_on="content_id", right_on="i_content_id", how="left")
)
drop_cols = ["u_user_id", "i_content_id", "u_fav_genre", "i_genre", "i_title", "i_synopsis",
             "i_language", "device", "interaction_id", "event_ts", "u_last_active_ts"]
ranker_data = ranker_data.drop(columns=[c for c in drop_cols if c in ranker_data.columns], errors="ignore")
ranker_data["completed"] = ranker_data["completed"].astype(int)
ranker_data = ranker_data.select_dtypes(include=["number"]).copy()
ranker_data["completed"] = ranker_data["completed"].astype(int)

with mlflow.start_run(run_name="ranker"):
    model_r, info = train_ranker(ranker_data, target_col="completed", n_trials=5)
    mlflow.log_params(info["best_params"])
    mlflow.log_metric("val_auc", info["val_auc"])
    X_sample = ranker_data[info["feature_cols"]].head(3).fillna(0)
    preds = model_r.predict_proba(X_sample)[:, 1]
    sig_r = infer_signature(X_sample, preds)
    mlflow.lightgbm.log_model(
        model_r, artifact_path="ranker", signature=sig_r,
        input_example=X_sample, registered_model_name=r_model_name,
    )

r_latest = client.get_registered_model(r_model_name).latest_versions[0].version
client.set_registered_model_alias(r_model_name, "champion", version=r_latest)
client.update_registered_model(r_model_name, description="LightGBM ranker: P(completed) for (user, content). Optuna-tuned.")
print(f"✓ {r_model_name} v{r_latest} @champion (val AUC = {info['val_auc']:.3f})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 of 4 — Inspect the registered models in Unity Catalog
# MAGIC
# MAGIC **🔍 Try this:** Open the Catalog Explorer (left nav → Catalog) → your catalog → schema.
# MAGIC You'll see two registered models alongside the tables: **`cmeg_two_tower`** and **`cmeg_ranker`**.
# MAGIC
# MAGIC Click `cmeg_ranker` → notice:
# MAGIC - **Aliases tab** shows `@champion` pointing at version 1
# MAGIC - **Lineage tab** shows the model is connected to `user_features` and `item_features` and (via those)
# MAGIC   back to `gold_interactions` and the DLT pipeline — automatic
# MAGIC - **Description** appears as we set it via `update_registered_model`
# MAGIC
# MAGIC ### The `@champion` alias pattern
# MAGIC
# MAGIC Old Databricks Model Registry had stages: Staging / Production / Archived. UC Model Registry
# MAGIC uses **aliases**: arbitrary string labels you point at specific versions. The convention is:
# MAGIC - `@champion` — the version currently live in serving
# MAGIC - `@challenger` — a candidate version being A/B-tested
# MAGIC - Promote with `client.set_registered_model_alias(name, "champion", version)` — atomic, auditable

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recap — what we just built
# MAGIC
# MAGIC - Two registered models in UC, both with `@champion` aliases and full lineage
# MAGIC - **Retriever**: takes `user_id`, returns top-100 candidate `content_id`s
# MAGIC - **Ranker**: takes user+item features, returns `P(completed)`
# MAGIC - One MLflow experiment with parent runs + Optuna child trials
# MAGIC
# MAGIC ## Up next — Chapter 4: Serve and Explain
# MAGIC
# MAGIC We **chain** the retriever and ranker behind a single Model Serving endpoint. Then we add a
# MAGIC **diversity rerank** (don't recommend 5 dramas in a row) and call a **Foundation Model** to generate
# MAGIC the "why we recommend this" copy your app shows under each card.

# COMMAND ----------
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=3, asset_type="model", name=tt_model_name, id=tt_model_name,
    url=format_asset_url(workspace_url, "model", tt_model_name), description="Two-tower retrieval @champion",
))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=3, asset_type="model", name=r_model_name, id=r_model_name,
    url=format_asset_url(workspace_url, "model", r_model_name), description="LightGBM ranker @champion",
))
exp = mlflow.get_experiment_by_name(experiment_path)
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=3, asset_type="experiment", name="cmeg_demo_experiments", id=exp.experiment_id,
    url=format_asset_url(workspace_url, "experiment", exp.experiment_id),
    description="MLflow experiment for two-tower + ranker",
))

chapter_complete(
    chapter=3, title="Train and Register",
    created=[
        ("model", tt_model_name, format_asset_url(workspace_url, "model", tt_model_name)),
        ("model", r_model_name, format_asset_url(workspace_url, "model", r_model_name)),
    ],
    next_label="04_serve_and_explain",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/04_serve_and_explain",
)
