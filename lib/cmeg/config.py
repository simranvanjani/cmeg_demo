"""Config object for the cmeg demo. Reads widget values or accepts overrides."""

from dataclasses import dataclass


@dataclass
class DemoConfig:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    ml_schema: str = "ml"
    ops_schema: str = "ops"
    data_scale: str = "small"
    serving_endpoint_enabled: bool = True
    genai_model: str = "databricks-meta-llama-3-3-70b-instruct"

    @property
    def ops_table(self) -> str:
        return f"{self.catalog}.{self.ops_schema}._cmeg_assets"

    def fq(self, schema: str, table: str) -> str:
        return f"{self.catalog}.{schema}.{table}"

    def scale_params(self) -> dict:
        return {
            "small": {"n_users": 10_000, "n_items": 5_000, "n_interactions": 500_000},
            "medium": {"n_users": 100_000, "n_items": 20_000, "n_interactions": 10_000_000},
            "large": {"n_users": 1_000_000, "n_items": 100_000, "n_interactions": 100_000_000},
        }[self.data_scale]


def from_widgets(dbutils) -> DemoConfig:
    """Read widget values inside a Databricks notebook."""
    return DemoConfig(
        catalog=dbutils.widgets.get("catalog_name"),
        bronze_schema=dbutils.widgets.get("bronze_schema"),
        silver_schema=dbutils.widgets.get("silver_schema"),
        gold_schema=dbutils.widgets.get("gold_schema"),
        ml_schema=dbutils.widgets.get("ml_schema"),
        ops_schema=dbutils.widgets.get("ops_schema"),
        data_scale=dbutils.widgets.get("data_scale"),
        serving_endpoint_enabled=dbutils.widgets.get("serving_endpoint_enabled") == "true",
        genai_model=dbutils.widgets.get("genai_model"),
    )


def set_widgets(dbutils, defaults: dict) -> None:
    """Initialize widgets at the top of a notebook."""
    for k, v in defaults.items():
        dbutils.widgets.text(k, str(v))
