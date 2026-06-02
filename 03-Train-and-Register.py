# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 3 — Train and Register Models
# MAGIC
# MAGIC ## Why two models?
# MAGIC
# MAGIC YouTube, Spotify, Netflix, TikTok — every large-scale content recommender uses **two-stage retrieval + ranking**:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────┐    ~100 candidates  ┌──────────┐    ranked top-N    ┌──────────┐
# MAGIC │  Retrieval  │ ───────────────────▶│  Ranker  │ ──────────────────▶│   App    │
# MAGIC │ (two-tower) │   from millions     │ (LightGBM│                    │          │
# MAGIC └─────────────┘                     └──────────┘                    └──────────┘
# MAGIC ```
# MAGIC
# MAGIC - **Retrieval** narrows millions of items down to ~100 candidates fast. Cheap dot-product over learned embeddings.
# MAGIC - **Ranker** scores those candidates with rich features (popularity, freshness, user state).
# MAGIC
# MAGIC ## What we build in this chapter
# MAGIC
# MAGIC 1. **Two-tower retrieval** trained with matrix-factorization-style SGD on (user, item, watch_seconds).
# MAGIC    Produces a user embedding table and an item embedding table. (Lightweight numpy implementation;
# MAGIC    in production you'd swap in TensorFlow Recommenders.)
# MAGIC 2. **LightGBM ranker** with a small **Optuna hyperparameter search** (5 trials).
# MAGIC    Predicts `P(completed)` from user/item features.
# MAGIC
# MAGIC ## Best practices applied
# MAGIC
# MAGIC - **MLflow model signatures + input examples** logged with both models — required for prod-grade serving.
# MAGIC - **Unity Catalog Model Registry** with `@champion` alias — clean promotion semantics, no "stage" strings.
# MAGIC - **Hyperparameter tuning** (Optuna) tracked as nested MLflow runs.
# MAGIC - **Model descriptions** set on the registered model so consumers know what it does.

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
# MAGIC %md ## Load training data

# COMMAND ----------
inter = spark.table(FQ("gold_interactions")).limit(200_000).toPandas()
user_feats = spark.table(FQ("user_features")).toPandas()
item_feats = spark.table(FQ("item_features")).toPandas()
print(f"interactions={len(inter)}, users={len(user_feats)}, items={len(item_feats)}")

# COMMAND ----------
# MAGIC %md ## Train two-tower retrieval

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
# MAGIC %md ## Train LightGBM ranker with Optuna hyperparameter search

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
print(f"✓ {r_model_name} v{r_latest} aliased @champion (val AUC = {info['val_auc']:.3f})")

# COMMAND ----------
# MAGIC %md ## Wrap up

# COMMAND ----------
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=3, asset_type="model", name=tt_model_name, id=tt_model_name,
    url=format_asset_url(workspace_url, "model", tt_model_name),
    description="Two-tower retrieval @champion",
))
record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=3, asset_type="model", name=r_model_name, id=r_model_name,
    url=format_asset_url(workspace_url, "model", r_model_name),
    description="LightGBM ranker @champion",
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
    next_label="04-Serve-and-Explain",
    next_url=f"{workspace_url}/#workspace{REPO_ROOT}/04-Serve-and-Explain",
)
