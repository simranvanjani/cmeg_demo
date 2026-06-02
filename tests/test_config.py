from cmeg.config import DemoConfig


def test_demo_config_defaults():
    c = DemoConfig(catalog="cmeg_demo")
    assert c.catalog == "cmeg_demo"
    assert c.bronze_schema == "bronze"
    assert c.ops_table == "cmeg_demo.ops._cmeg_assets"


def test_demo_config_overrides():
    c = DemoConfig(
        catalog="my_catalog",
        bronze_schema="b",
        silver_schema="s",
        gold_schema="g",
        ml_schema="m",
        ops_schema="o",
        data_scale="medium",
    )
    assert c.gold_schema == "g"
    assert c.ml_schema == "m"
    assert c.data_scale == "medium"
    assert c.ops_table == "my_catalog.o._cmeg_assets"


def test_demo_config_scale_params_small():
    c = DemoConfig(catalog="cmeg_demo")
    p = c.scale_params()
    assert p["n_users"] == 10_000
    assert p["n_items"] == 5_000
    assert p["n_interactions"] == 500_000


def test_demo_config_scale_params_medium():
    c = DemoConfig(catalog="cmeg_demo", data_scale="medium")
    p = c.scale_params()
    assert p["n_users"] == 100_000
