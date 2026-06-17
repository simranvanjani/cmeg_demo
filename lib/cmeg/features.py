"""Feature engineering helpers for the cmeg demo."""

from pyspark.sql import DataFrame, functions as F, Window


def build_user_features(gold_interactions: DataFrame) -> DataFrame:
    base = (
        gold_interactions.groupBy("user_id")
        .agg(
            F.count("*").alias("watch_count_7d"),
            F.avg("watch_seconds").alias("avg_session_seconds"),
            F.expr("percentile_approx(watch_seconds, 0.5)").alias("p50_session_seconds"),
            F.first("genre", ignorenulls=True).alias("fav_genre"),
            F.max("event_ts").alias("last_active_ts"),
        )
    )
    # seed_content_id = the item this user spent the most total watch time on.
    # Used as the query "seed" for Vector Search content-based retrieval
    # ("recommend shows similar to the one you watched most").
    per_item = (
        gold_interactions.groupBy("user_id", "content_id")
        .agg(F.sum("watch_seconds").alias("_ws"))
    )
    rank = Window.partitionBy("user_id").orderBy(F.desc("_ws"))
    seed = (
        per_item.withColumn("_rn", F.row_number().over(rank))
        .filter("_rn = 1")
        .select("user_id", F.col("content_id").alias("seed_content_id"))
    )
    return base.join(seed, "user_id", "left")


def build_item_features(silver_items: DataFrame, gold_interactions: DataFrame) -> DataFrame:
    pop = (
        gold_interactions.groupBy("content_id")
        .agg(
            F.count("*").alias("popularity_7d"),
            F.avg(F.col("completed").cast("int")).alias("completion_rate"),
        )
    )
    return silver_items.join(pop, "content_id", "left").fillna(0, subset=["popularity_7d", "completion_rate"])
