from cmeg.data_gen import build_users, build_items, build_interactions


def test_build_users_shape():
    users = build_users(n=100, seed=42)
    assert len(users) == 100
    assert {"user_id", "age", "country", "email", "phone", "signup_ts"} <= set(users[0].keys())


def test_build_items_shape():
    items = build_items(n=50, seed=42)
    assert len(items) == 50
    assert {"content_id", "title", "genre", "release_year", "duration_min"} <= set(items[0].keys())


def test_build_interactions_referential():
    users = build_users(n=10, seed=1)
    items = build_items(n=5, seed=1)
    inters = build_interactions(users=users, items=items, n=100, seed=1)
    assert len(inters) == 100
    u_ids = {u["user_id"] for u in users}
    i_ids = {i["content_id"] for i in items}
    for r in inters:
        assert r["user_id"] in u_ids
        assert r["content_id"] in i_ids
        assert r["watch_seconds"] >= 0
