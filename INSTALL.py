# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 CMEG Content Recommendation Demo — Install
# MAGIC
# MAGIC **One-click install. No CLI required.**
# MAGIC
# MAGIC This notebook creates everything you need to run the demo in your workspace:
# MAGIC catalog (if missing), schemas, volume, DLT pipeline, orchestrator job, and the
# MAGIC asset-tracking table.
# MAGIC
# MAGIC ### Steps
# MAGIC 1. Set the widgets at the top of this notebook (defaults work for most cases).
# MAGIC 2. Click **Run all**.
# MAGIC 3. When it finishes, click the green **Open START_HERE →** button.

# COMMAND ----------
# MAGIC %pip install -q databricks-sdk
# MAGIC %pip install -q -e ./

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, JobCluster, TaskDependency, PipelineTask,
)
from databricks.sdk.service.compute import ClusterSpec, DataSecurityMode, RuntimeEngine
from databricks.sdk.service.pipelines import PipelineLibrary, NotebookLibrary

from cmeg.config import set_widgets, from_widgets
from cmeg.state import AssetRecord, ensure_table, record_asset
from cmeg.companion import format_asset_url

# COMMAND ----------
# MAGIC %md ## Configure

# COMMAND ----------
w = WorkspaceClient()
workspace_url = w.config.host

# Discover available catalogs to help customer pick one that already exists
available = [c.name for c in w.catalogs.list() if c.name not in ("system",)]
default_catalog = "main" if "main" in available else (available[0] if available else "cmeg_demo")
print(f"Catalogs available in this workspace: {', '.join(available)}")
print(f"Suggested default: {default_catalog}")

# COMMAND ----------
set_widgets(
    dbutils,
    {
        "catalog_name": default_catalog,
        "bronze_schema": "cmeg_demo_bronze",
        "silver_schema": "cmeg_demo_silver",
        "gold_schema": "cmeg_demo_gold",
        "ml_schema": "cmeg_demo_ml",
        "ops_schema": "cmeg_demo_ops",
        "data_scale": "small",
        "serving_endpoint_enabled": "true",
        "genai_model": "databricks-meta-llama-3-3-70b-instruct",
    },
)
cfg = from_widgets(dbutils)
print(f"Installing into: catalog={cfg.catalog}, scale={cfg.data_scale}")

# COMMAND ----------
# MAGIC %md ## Resolve repo paths

# COMMAND ----------
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = nb_path.rsplit("/", 1)[0]
print(f"Repo root in workspace: {repo_root}")

NARRATIVE = f"{repo_root}/narrative"
CHAPTER_NOTEBOOKS = {
    "01_setup_and_data": f"{NARRATIVE}/01_setup_and_data",
    "02_dlt_medallion": f"{NARRATIVE}/02_dlt_medallion",
    "03_features_and_vectors": f"{NARRATIVE}/03_features_and_vectors",
    "04_train_and_register": f"{NARRATIVE}/04_train_and_register",
    "05_serve_and_explain": f"{NARRATIVE}/05_serve_and_explain",
    "06_monitor_and_govern": f"{NARRATIVE}/06_monitor_and_govern",
    "07_genie_space": f"{NARRATIVE}/07_genie_space",
}
DLT_DEFINITION = f"{NARRATIVE}/_dlt_pipeline_definition"
UNINSTALL = f"{repo_root}/_install/uninstall"

# COMMAND ----------
# MAGIC %md ## Create catalog (if needed), schemas, volume

# COMMAND ----------
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
    print(f"catalog ready: {cfg.catalog}")
except Exception as e:
    print(f"⚠️  Catalog '{cfg.catalog}' could not be created and does not exist.")
    print(f"    {e}")
    print(f"    Available catalogs: {', '.join(available)}")
    print(f"    👉  Set the `catalog_name` widget to an existing catalog and re-run.")
    raise

for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{sch}")
    print(f"schema ready: {cfg.catalog}.{sch}")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.bronze_schema}.landing")
print(f"volume ready: {cfg.catalog}.{cfg.bronze_schema}.landing")

# COMMAND ----------
# MAGIC %md ## Predictive Optimization (best-effort)

# COMMAND ----------
try:
    spark.sql(f"ALTER CATALOG {cfg.catalog} ENABLE PREDICTIVE OPTIMIZATION")
    print("Predictive Optimization enabled")
except Exception as e:
    print(f"Predictive Optimization not enabled (may already be on, or insufficient perms): {e}")

# COMMAND ----------
# MAGIC %md ## Initialize asset tracking table

# COMMAND ----------
ensure_table(spark, cfg.ops_table)
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=0, asset_type="catalog", name=cfg.catalog, id=cfg.catalog,
    url=format_asset_url(workspace_url, "catalog", cfg.catalog),
    description="Demo catalog",
))
for sch in [cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema, cfg.ml_schema, cfg.ops_schema]:
    record_asset(spark, cfg.ops_table, AssetRecord(
        chapter=0, asset_type="schema", name=f"{cfg.catalog}.{sch}", id=f"{cfg.catalog}.{sch}",
        url=format_asset_url(workspace_url, "schema", f"{cfg.catalog}.{sch}"),
        description=f"{sch} schema",
    ))

# COMMAND ----------
# MAGIC %md ## Create DLT pipeline

# COMMAND ----------
pipeline_name = "cmeg_dlt_pipeline"
existing_pipeline = next((p for p in w.pipelines.list_pipelines(filter=f"name LIKE '{pipeline_name}'")), None)
pipeline_config = {
    "cmeg.catalog": cfg.catalog,
    "cmeg.bronze_schema": cfg.bronze_schema,
    "cmeg.silver_schema": cfg.silver_schema,
    "cmeg.gold_schema": cfg.gold_schema,
}
if existing_pipeline:
    print(f"DLT pipeline exists: {existing_pipeline.pipeline_id} — updating")
    w.pipelines.update(
        pipeline_id=existing_pipeline.pipeline_id,
        name=pipeline_name,
        catalog=cfg.catalog,
        target=cfg.silver_schema,
        photon=True,
        serverless=True,
        development=True,
        continuous=False,
        libraries=[PipelineLibrary(notebook=NotebookLibrary(path=DLT_DEFINITION))],
        configuration=pipeline_config,
    )
    pipeline_id = existing_pipeline.pipeline_id
else:
    created = w.pipelines.create(
        name=pipeline_name,
        catalog=cfg.catalog,
        target=cfg.silver_schema,
        photon=True,
        serverless=True,
        development=True,
        continuous=False,
        libraries=[PipelineLibrary(notebook=NotebookLibrary(path=DLT_DEFINITION))],
        configuration=pipeline_config,
    )
    pipeline_id = created.pipeline_id
    print(f"DLT pipeline created: {pipeline_id}")

record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=0, asset_type="pipeline", name=pipeline_name, id=pipeline_id,
    url=format_asset_url(workspace_url, "pipeline", pipeline_id),
    description="Medallion DLT pipeline (bronze → silver → gold)",
))

# COMMAND ----------
# MAGIC %md ## Create orchestrator + cleanup jobs

# COMMAND ----------
demo_cluster = JobCluster(
    job_cluster_key="demo_cluster",
    new_cluster=ClusterSpec(
        spark_version="15.4.x-cpu-ml-scala2.12",
        node_type_id="Standard_D4ds_v5",
        num_workers=1,
        data_security_mode=DataSecurityMode.SINGLE_USER,
        runtime_engine=RuntimeEngine.PHOTON,
    ),
)

def nb_task(key, path, depends=None, cluster_key="demo_cluster"):
    return Task(
        task_key=key,
        notebook_task=NotebookTask(notebook_path=path),
        job_cluster_key=cluster_key,
        depends_on=[TaskDependency(task_key=d) for d in (depends or [])],
    )

orchestrator_tasks = [
    nb_task("setup_and_data", CHAPTER_NOTEBOOKS["01_setup_and_data"]),
    Task(
        task_key="dlt_medallion",
        depends_on=[TaskDependency(task_key="setup_and_data")],
        pipeline_task=PipelineTask(pipeline_id=pipeline_id),
    ),
    nb_task("features_and_vectors", CHAPTER_NOTEBOOKS["03_features_and_vectors"], depends=["dlt_medallion"]),
    nb_task("train_and_register", CHAPTER_NOTEBOOKS["04_train_and_register"], depends=["features_and_vectors"]),
    nb_task("serve_and_explain", CHAPTER_NOTEBOOKS["05_serve_and_explain"], depends=["train_and_register"]),
    nb_task("monitor_and_govern", CHAPTER_NOTEBOOKS["06_monitor_and_govern"], depends=["serve_and_explain"]),
    nb_task("genie_space", CHAPTER_NOTEBOOKS["07_genie_space"], depends=["monitor_and_govern"]),
]

orchestrator_name = "cmeg_orchestrator"
existing_orch = next((j for j in w.jobs.list(name=orchestrator_name)), None)
if existing_orch:
    w.jobs.reset(
        job_id=existing_orch.job_id,
        new_settings={
            "name": orchestrator_name,
            "tasks": [t.as_dict() for t in orchestrator_tasks],
            "job_clusters": [demo_cluster.as_dict()],
        },
    )
    orchestrator_id = existing_orch.job_id
    print(f"orchestrator updated: {orchestrator_id}")
else:
    created_job = w.jobs.create(
        name=orchestrator_name,
        tasks=orchestrator_tasks,
        job_clusters=[demo_cluster],
    )
    orchestrator_id = created_job.job_id
    print(f"orchestrator created: {orchestrator_id}")

cleanup_name = "cmeg_cleanup"
cleanup_tasks = [nb_task("uninstall", UNINSTALL)]
existing_cleanup = next((j for j in w.jobs.list(name=cleanup_name)), None)
if existing_cleanup:
    w.jobs.reset(
        job_id=existing_cleanup.job_id,
        new_settings={
            "name": cleanup_name,
            "tasks": [t.as_dict() for t in cleanup_tasks],
            "job_clusters": [demo_cluster.as_dict()],
        },
    )
    cleanup_id = existing_cleanup.job_id
else:
    cleanup_id = w.jobs.create(name=cleanup_name, tasks=cleanup_tasks, job_clusters=[demo_cluster]).job_id
print(f"cleanup job: {cleanup_id}")

record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=0, asset_type="job", name=orchestrator_name, id=str(orchestrator_id),
    url=format_asset_url(workspace_url, "job", str(orchestrator_id)),
    description="Runs all 7 chapters in order",
))
record_asset(spark, cfg.ops_table, AssetRecord(
    chapter=0, asset_type="job", name=cleanup_name, id=str(cleanup_id),
    url=format_asset_url(workspace_url, "job", str(cleanup_id)),
    description="Removes SDK-created assets",
))

# COMMAND ----------
# MAGIC %md ## Done — open the demo

# COMMAND ----------
start_here = f"{NARRATIVE}/00_START_HERE"
displayHTML(f"""
<div style='border:2px solid #1f6feb;border-radius:10px;padding:24px;background:#f0f7ff;font-family:Inter,sans-serif;'>
  <h1 style='margin:0 0 8px 0;'>&#10003; Install complete</h1>
  <p style='margin:0 0 16px 0;color:#555;'>Everything is ready in catalog <b>{cfg.catalog}</b>.</p>
  <table style='border-collapse:collapse;margin:8px 0 20px 0;'>
    <tr><td style='padding:4px 12px 4px 0;'>DLT pipeline:</td>
        <td><a href='{format_asset_url(workspace_url, 'pipeline', pipeline_id)}' target='_blank'>{pipeline_name} &#8599;</a></td></tr>
    <tr><td style='padding:4px 12px 4px 0;'>Orchestrator job:</td>
        <td><a href='{format_asset_url(workspace_url, 'job', str(orchestrator_id))}' target='_blank'>cmeg_orchestrator &#8599;</a></td></tr>
    <tr><td style='padding:4px 12px 4px 0;'>Cleanup job:</td>
        <td><a href='{format_asset_url(workspace_url, 'job', str(cleanup_id))}' target='_blank'>cmeg_cleanup &#8599;</a></td></tr>
  </table>
  <p style='margin:16px 0;'>
    <a href='{workspace_url}/#workspace{start_here}'
       style='background:#1f6feb;color:white;padding:14px 24px;border-radius:8px;
              text-decoration:none;font-weight:600;font-size:16px;'>
      Open START_HERE &rarr;
    </a>
  </p>
  <p style='margin:16px 0 0 0;color:#888;font-size:13px;'>
    Or click <b>Run now</b> on the orchestrator job to execute all 7 chapters end-to-end (~20-30 min).
  </p>
</div>
""")
