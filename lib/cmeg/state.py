"""Persist asset metadata for the demo so 00_START_HERE can render a live TOC."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AssetRecord:
    chapter: int
    asset_type: str
    name: str
    id: str
    url: str
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_row(self) -> dict:
        return {
            "chapter": self.chapter,
            "asset_type": self.asset_type,
            "name": self.name,
            "id": self.id,
            "url": self.url,
            "description": self.description,
            "created_at": self.created_at,
        }


def build_create_table_sql(fq_table: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {fq_table} (
          chapter INT,
          asset_type STRING,
          name STRING,
          id STRING,
          url STRING,
          description STRING,
          created_at TIMESTAMP
        )
        USING DELTA
    """


def build_insert_sql(fq_table: str) -> str:
    return f"""
        MERGE INTO {fq_table} AS t
        USING (
          SELECT
            :chapter AS chapter,
            :asset_type AS asset_type,
            :name AS name,
            :id AS id,
            :url AS url,
            :description AS description,
            :created_at AS created_at
        ) AS s
        ON t.asset_type = s.asset_type AND t.name = s.name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """


def build_select_sql(fq_table: str) -> str:
    return f"""
        SELECT chapter, asset_type, name, id, url, description, created_at
        FROM {fq_table}
        ORDER BY chapter, asset_type, name
    """


def ensure_table(spark, fq_table: str) -> None:
    spark.sql(build_create_table_sql(fq_table))


def record_asset(spark, fq_table: str, rec: AssetRecord) -> None:
    ensure_table(spark, fq_table)
    row = rec.as_row()
    spark.sql(
        build_insert_sql(fq_table),
        args={
            "chapter": row["chapter"],
            "asset_type": row["asset_type"],
            "name": row["name"],
            "id": row["id"],
            "url": row["url"],
            "description": row["description"],
            "created_at": row["created_at"],
        },
    )


def list_assets(spark, fq_table: str):
    ensure_table(spark, fq_table)
    return spark.sql(build_select_sql(fq_table))
