"""Feature engineering helpers for the cmeg demo."""

from pyspark.sql import DataFrame, functions as F


def build_user_features(gold_interactions: DataFrame) -> DataFrame:
    return (
        gold_interactions.groupBy("user_id")
        .agg(
            F.count("*").alias("watch_count_7d"),
            F.avg("watch_seconds").alias("avg_session_seconds"),
            F.expr("percentile_approx(watch_seconds, 0.5)").alias("p50_session_seconds"),
            F.first("genre", ignorenulls=True).alias("fav_genre"),
            F.max("event_ts").alias("last_active_ts"),
        )
    )


def build_item_features(silver_items: DataFrame, gold_interactions: DataFrame) -> DataFrame:
    pop = (
        gold_interactions.groupBy("content_id")
        .agg(
            F.count("*").alias("popularity_7d"),
            F.avg(F.col("completed").cast("int")).alias("completion_rate"),
        )
    )
    return silver_items.join(pop, "content_id", "left").fillna(0, subset=["popularity_7d", "completion_rate"])
