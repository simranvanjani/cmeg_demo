# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 04 — Train and register
# MAGIC
# MAGIC Train a two-tower retrieval model (numpy SGD) and a LightGBM ranker with Optuna
# MAGIC hyperparameter search. Log both to MLflow with signatures + input examples and
# MAGIC register in Unity Catalog with `@champion` aliases.

# COMMAND ----------
# MAGIC %pip install -q -e ../ lightgbm optuna scikit-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import mlflow.pyfunc
import mlflow.lightgbm
import pandas as pd
from mlflow.models.signature import infer_signature

from databricks.sdk import WorkspaceClient
from cmeg.config import set_widgets, from_widgets
from cmeg.companion import chapter_complete, format_asset_url
from cmeg.state import AssetRecord, record_asset
from cmeg.models import train_two_tower, train_ranker

set_widgets(
    dbutils,
    {
        "catalog_name": "cmeg_demo",
        "bronze_schema": "bronze",
        "silver_schema": "silver",
        "gold_schema": "gold",
        "ml_schema": "ml",
        "ops_schema": "ops",
        "data_scale": "small",
        "serving_endpoint_enabled": "true",
        "genai_model": "databricks-meta-llama-3-3-70b-instruct",
    },
)
cfg = from_widgets(dbutils)
w = WorkspaceClient()
workspace_url = w.config.host

mlflow.set_registry_uri("databricks-uc")
user_name = w.current_user.me().user_name
experiment_path = f"/Users/{user_name}/cmeg_demo_experiments"
mlflow.set_experiment(experiment_path)

# COMMAND ----------
# MAGIC %md ## Load training data

# COMMAND ----------
inter = spark.table(cfg.fq(cfg.gold_schema, "gold_interactions")).limit(200_000).toPandas()
user_feats = spark.table(cfg.fq(cfg.ml_schema, "user_features")).toPandas()
item_feats = spark.table(cfg.fq(cfg.ml_schema, "item_features")).toPandas()

# COMMAND ----------
# MAGIC %md ## Train two-tower retrieval

# COMMAND ----------
tt_model_name = cfg.fq(cfg.ml_schema, "cmeg_two_tower")

with mlflow.start_run(run_name="two_tower"):
    art = train_two_tower(inter, n_factors=32, n_epochs=3)

    class TwoTower(mlflow.pyfunc.PythonModel):
        def __init__(self, item_emb_df: pd.DataFrame, user_emb_df: pd.DataFrame):
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
        artifact_path="model",
        python_model=model,
        signature=sig,
        input_example=input_ex,
        registered_model_name=tt_model_name,
    )

client = mlflow.MlflowClient()
tt_latest = client.get_registered_model(tt_model_name).latest_versions[0].version
client.set_registered_model_alias(tt_model_name, "champion", version=tt_latest)
client.update_registered_model(tt_model_name, description="Two-tower retrieval: top-100 content ids for a user_id.")

# COMMAND ----------
# MAGIC %md ## Train LightGBM ranker

# COMMAND ----------
r_model_name = cfg.fq(cfg.ml_schema, "cmeg_ranker")

ranker_data = (
    inter.merge(user_feats.add_prefix("u_"), left_on="user_id", right_on="u_user_id", how="left")
    .merge(item_feats.add_prefix("i_"), left_on="content_id", right_on="i_content_id", how="left")
)
ranker_data = ranker_data.drop(
    columns=[c for c in ["u_user_id", "i_content_id", "u_fav_genre", "i_genre", "i_title", "i_synopsis", "i_language", "device", "interaction_id", "event_ts", "u_last_active_ts"] if c in ranker_data.columns],
    errors="ignore",
)
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
        model_r,
        artifact_path="ranker",
        signature=sig_r,
        input_example=X_sample,
        registered_model_name=r_model_name,
    )

r_latest = client.get_registered_model(r_model_name).latest_versions[0].version
client.set_registered_model_alias(r_model_name, "champion", version=r_latest)
client.update_registered_model(r_model_name, description="LightGBM ranker: P(completed) for (user, content) features.")

# COMMAND ----------
# MAGIC %md ## Record assets and finish chapter

# COMMAND ----------
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=4, asset_type="model", name=tt_model_name, id=tt_model_name,
    url=format_asset_url(workspace_url, "model", tt_model_name),
    description="Two-tower retrieval model",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=4, asset_type="model", name=r_model_name, id=r_model_name,
    url=format_asset_url(workspace_url, "model", r_model_name),
    description="LightGBM ranker",
))
exp = mlflow.get_experiment_by_name(experiment_path)
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=4, asset_type="experiment", name="cmeg_demo_experiments",
    id=exp.experiment_id,
    url=format_asset_url(workspace_url, "experiment", exp.experiment_id),
    description="MLflow experiment for two-tower + ranker runs",
))

chapter_complete(
    chapter=4, title="Train and register",
    created=[
        ("model", tt_model_name, format_asset_url(workspace_url, "model", tt_model_name)),
        ("model", r_model_name, format_asset_url(workspace_url, "model", r_model_name)),
    ],
    next_label="05_serve_and_explain",
    next_url=f"{workspace_url}/#workspace/05_serve_and_explain",
)
