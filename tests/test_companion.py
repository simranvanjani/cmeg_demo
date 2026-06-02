from cmeg.companion import format_asset_url, render_chapter_card


def test_format_asset_url_for_table():
    url = format_asset_url(
        workspace_url="https://adb-123.azuredatabricks.net",
        asset_type="table",
        asset_id="catalog.schema.table",
    )
    assert url == "https://adb-123.azuredatabricks.net/explore/data/catalog/schema/table"


def test_format_asset_url_for_pipeline():
    url = format_asset_url(
        workspace_url="https://adb-123.azuredatabricks.net",
        asset_type="pipeline",
        asset_id="abc-def",
    )
    assert url == "https://adb-123.azuredatabricks.net/#joblist/pipelines/abc-def"


def test_render_chapter_card_html_contains_assets_and_next():
    html = render_chapter_card(
        chapter=2,
        title="DLT medallion",
        created=[
            ("dlt_pipeline", "cmeg_dlt_pipeline", "https://x/y/pipeline"),
            ("table", "cmeg_demo.gold.interactions", "https://x/y/table"),
        ],
        next_label="03_features_and_vectors",
        next_url="https://x/y/notebook",
    )
    assert "Chapter 2 complete" in html
    assert "cmeg_dlt_pipeline" in html
    assert "https://x/y/pipeline" in html
    assert "Next: 03_features_and_vectors" in html
