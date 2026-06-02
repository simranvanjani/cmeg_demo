"""Governance helpers: UC tag application."""

PII_TAG = "pii"
OWNER_TAG = "business_owner"
COST_TAG = "cost_center"


def apply_pii_tags(spark, fq_table: str, pii_columns: list) -> None:
    for c in pii_columns:
        try:
            spark.sql(f"ALTER TABLE {fq_table} ALTER COLUMN {c} SET TAGS ('{PII_TAG}' = 'true')")
        except Exception as e:
            print(f"tag column {c} on {fq_table}: {e}")


def apply_table_owner_tag(spark, fq_table: str, owner: str, cost_center: str) -> None:
    try:
        spark.sql(
            f"ALTER TABLE {fq_table} "
            f"SET TAGS ('{OWNER_TAG}' = '{owner}', '{COST_TAG}' = '{cost_center}')"
        )
    except Exception as e:
        print(f"tag table {fq_table}: {e}")
