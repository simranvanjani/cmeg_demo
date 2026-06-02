# Databricks notebook source
# DLT pipeline definition — referenced by resources/pipelines.yml.
# Reads parquet from the landing volume via Auto Loader, applies expectations,
# produces silver + gold tables with Liquid Clustering.

import dlt
from pyspark.sql import functions as F

CATALOG = spark.conf.get("cmeg.catalog")
BRONZE = spark.conf.get("cmeg.bronze_schema")
SILVER = spark.conf.get("cmeg.silver_schema")
GOLD = spark.conf.get("cmeg.gold_schema")

VOLUME = f"/Volumes/{CATALOG}/{BRONZE}/landing"


@dlt.table(name="bronze_users", table_properties={"quality": "bronze"})
def bronze_users():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME}/users")
    )


@dlt.table(name="bronze_items", table_properties={"quality": "bronze"})
def bronze_items():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME}/items")
    )


@dlt.table(name="bronze_interactions", table_properties={"quality": "bronze"})
def bronze_interactions():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME}/interactions")
    )


@dlt.table(name="silver_users", cluster_by=["user_id"])
@dlt.expect_or_drop("user_id_not_null", "user_id IS NOT NULL")
@dlt.expect("valid_age", "age BETWEEN 5 AND 120")
def silver_users():
    return dlt.read_stream("bronze_users").dropDuplicates(["user_id"])


@dlt.table(name="silver_items", cluster_by=["content_id"])
@dlt.expect_or_drop("content_id_not_null", "content_id IS NOT NULL")
@dlt.expect_or_drop("duration_positive", "duration_min > 0")
def silver_items():
    return dlt.read_stream("bronze_items").dropDuplicates(["content_id"])


@dlt.table(name="silver_interactions", cluster_by=["user_id", "content_id"])
@dlt.expect_or_drop("ids_not_null", "user_id IS NOT NULL AND content_id IS NOT NULL")
@dlt.expect("watch_seconds_non_negative", "watch_seconds >= 0")
def silver_interactions():
    return dlt.read_stream("bronze_interactions")


@dlt.table(
    name="gold_user_360",
    cluster_by=["user_id"],
    comment="User-level aggregates: watch counts, favorite genre",
)
def gold_user_360():
    inters = dlt.read("silver_interactions")
    items = dlt.read("silver_items")
    j = inters.join(items.select("content_id", "genre"), "content_id", "left")
    return (
        j.groupBy("user_id")
        .agg(
            F.count("*").alias("interaction_count"),
            F.sum("watch_seconds").alias("total_watch_seconds"),
            F.avg("watch_seconds").alias("avg_watch_seconds"),
            F.countDistinct("content_id").alias("unique_content_watched"),
            F.first("genre", ignorenulls=True).alias("first_genre_seen"),
        )
    )


@dlt.table(name="gold_interactions", cluster_by=["user_id", "content_id"])
def gold_interactions():
    return (
        dlt.read("silver_interactions")
        .join(dlt.read("silver_items").select("content_id", "genre", "release_year"), "content_id", "left")
        .join(dlt.read("silver_users").select("user_id", "country", "age"), "user_id", "left")
    )
