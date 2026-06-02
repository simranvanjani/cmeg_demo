from cmeg.state import AssetRecord, build_insert_sql, build_select_sql


def test_asset_record_to_row():
    rec = AssetRecord(
        chapter=2,
        asset_type="pipeline",
        name="cmeg_dlt_pipeline",
        id="abc-123",
        url="https://x/y",
        description="DLT pipeline",
    )
    row = rec.as_row()
    assert row["chapter"] == 2
    assert row["asset_type"] == "pipeline"
    assert row["name"] == "cmeg_dlt_pipeline"


def test_build_insert_sql():
    sql = build_insert_sql("cat.ops._cmeg_assets")
    assert "MERGE INTO cat.ops._cmeg_assets" in sql
    assert "WHEN MATCHED" in sql


def test_build_select_sql():
    sql = build_select_sql("cat.ops._cmeg_assets")
    assert "FROM cat.ops._cmeg_assets" in sql
    assert "ORDER BY chapter" in sql
