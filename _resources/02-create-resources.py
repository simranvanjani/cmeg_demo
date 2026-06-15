# Databricks notebook source
# MAGIC %md
# MAGIC # Create DLT pipeline + orchestrator + cleanup jobs
# MAGIC
# MAGIC Run **once** by `RUNME` during install. Uses the Databricks SDK to
# MAGIC create the DLT pipeline pointing at `_resources/_dlt_pipeline`, and two jobs:
# MAGIC `cmeg_orchestrator` (runs all chapters in order) and `cmeg_cleanup` (uninstall).
# MAGIC
# MAGIC Idempotent — if the pipeline or jobs already exist, they're updated in place.

# COMMAND ----------
# MAGIC %run ./00-setup

# COMMAND ----------
from databricks.sdk.service.jobs import Task, NotebookTask, TaskDependency, PipelineTask, JobSettings
from databricks.sdk.service.pipelines import PipelineLibrary, NotebookLibrary

DLT_DEFINITION = f"{REPO_ROOT}/_resources/_dlt_pipeline"
UNINSTALL_NB = f"{REPO_ROOT}/_resources/99-uninstall"
CHAPTERS = {
    "dlt_medallion":         f"{REPO_ROOT}/02_dlt_medallion",
    "features_and_vectors":  f"{REPO_ROOT}/03_features_and_vectors",
    "train_and_register":    f"{REPO_ROOT}/04_train_and_register",
    "serve_and_explain":     f"{REPO_ROOT}/05_serve_and_explain",
    "monitor_and_govern":    f"{REPO_ROOT}/06_monitor_and_govern",
    "genie_space":           f"{REPO_ROOT}/07_genie_space",
}

# COMMAND ----------
# DLT pipeline (idempotent)
pipeline_name = "cmeg_dlt_pipeline"
existing = next((p for p in w.pipelines.list_pipelines(filter=f"name LIKE '{pipeline_name}'")), None)
pipeline_args = dict(
    name=pipeline_name,
    catalog=CATALOG,
    target=SCHEMA,
    photon=True,
    serverless=True,
    development=True,
    continuous=False,
    libraries=[PipelineLibrary(notebook=NotebookLibrary(path=DLT_DEFINITION))],
    configuration={
        "cmeg.catalog": CATALOG,
        "cmeg.schema": SCHEMA,
    },
)
if existing:
    w.pipelines.update(pipeline_id=existing.pipeline_id, **pipeline_args)
    pipeline_id = existing.pipeline_id
    print(f"✓ DLT pipeline updated: {pipeline_id}")
else:
    created = w.pipelines.create(**pipeline_args)
    pipeline_id = created.pipeline_id
    print(f"✓ DLT pipeline created: {pipeline_id}")

record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=0, asset_type="pipeline", name=pipeline_name, id=pipeline_id,
    url=format_asset_url(workspace_url, "pipeline", pipeline_id),
    description="Medallion DLT pipeline (bronze → silver → gold)",
))

# COMMAND ----------
# Jobs run on SERVERLESS compute (no job_clusters / job_cluster_key). Notebook tasks
# without a cluster reference automatically run on serverless — the right default for
# a portable accelerator that must work in serverless-only workspaces.

def nb_task(key, path, depends=None):
    return Task(
        task_key=key,
        notebook_task=NotebookTask(notebook_path=path),
        depends_on=[TaskDependency(task_key=d) for d in (depends or [])],
    )

orchestrator_tasks = [
    Task(
        task_key="dlt_medallion",
        pipeline_task=PipelineTask(pipeline_id=pipeline_id),
    ),
    nb_task("features_and_vectors", CHAPTERS["features_and_vectors"], depends=["dlt_medallion"]),
    nb_task("train_and_register",   CHAPTERS["train_and_register"],   depends=["features_and_vectors"]),
    nb_task("serve_and_explain",    CHAPTERS["serve_and_explain"],    depends=["train_and_register"]),
    nb_task("monitor_and_govern",   CHAPTERS["monitor_and_govern"],   depends=["serve_and_explain"]),
    nb_task("genie_space",          CHAPTERS["genie_space"],          depends=["monitor_and_govern"]),
]

# COMMAND ----------
# Orchestrator job (idempotent). reset() takes a JobSettings object, not a dict.
orchestrator_name = "cmeg_orchestrator"
existing_job = next((j for j in w.jobs.list(name=orchestrator_name)), None)
if existing_job:
    w.jobs.reset(
        job_id=existing_job.job_id,
        new_settings=JobSettings(name=orchestrator_name, tasks=orchestrator_tasks),
    )
    orchestrator_id = existing_job.job_id
    print(f"✓ orchestrator updated: {orchestrator_id}")
else:
    orchestrator_id = w.jobs.create(name=orchestrator_name, tasks=orchestrator_tasks).job_id
    print(f"✓ orchestrator created: {orchestrator_id}")

record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=0, asset_type="job", name=orchestrator_name, id=str(orchestrator_id),
    url=format_asset_url(workspace_url, "job", str(orchestrator_id)),
    description="Runs all 6 chapters end-to-end",
))

# COMMAND ----------
# Cleanup job (idempotent)
cleanup_name = "cmeg_cleanup"
cleanup_tasks = [nb_task("uninstall", UNINSTALL_NB)]
existing_cleanup = next((j for j in w.jobs.list(name=cleanup_name)), None)
if existing_cleanup:
    w.jobs.reset(
        job_id=existing_cleanup.job_id,
        new_settings=JobSettings(name=cleanup_name, tasks=cleanup_tasks),
    )
    cleanup_id = existing_cleanup.job_id
else:
    cleanup_id = w.jobs.create(name=cleanup_name, tasks=cleanup_tasks).job_id
print(f"✓ cleanup job ready: {cleanup_id}")

record_asset(spark, OPS_TABLE, AssetRecord(
    chapter=0, asset_type="job", name=cleanup_name, id=str(cleanup_id),
    url=format_asset_url(workspace_url, "job", str(cleanup_id)),
    description="Uninstall: deletes endpoint, vector index, monitor, models",
))
