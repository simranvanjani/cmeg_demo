"""Synthetic data generators for the cmeg demo."""

import random
from datetime import datetime, timedelta, timezone

GENRES = ["drama", "comedy", "action", "documentary", "thriller", "romance", "scifi", "sports"]
COUNTRIES = ["TH", "VN", "ID", "MY", "PH", "SG", "IN", "JP"]


def build_users(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "user_id": f"u_{i:07d}",
            "age": rng.randint(13, 75),
            "gender": rng.choice(["M", "F", "X"]),
            "country": rng.choice(COUNTRIES),
            "email": f"user_{i}@example.invalid",
            "phone": f"+66{rng.randint(100000000, 999999999)}",
            "signup_ts": base + timedelta(days=rng.randint(0, 2000)),
        }
        for i in range(n)
    ]


def build_items(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "content_id": f"c_{i:06d}",
            "title": f"Title {i}",
            "synopsis": f"A {rng.choice(GENRES)} story about character #{i}.",
            "genre": rng.choice(GENRES),
            "release_year": rng.randint(1990, 2025),
            "duration_min": rng.randint(20, 180),
            "language": rng.choice(["th", "en", "ko", "ja", "hi"]),
        }
        for i in range(n)
    ]


def build_interactions(users: list[dict], items: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = []
    for _ in range(n):
        u = rng.choice(users)
        c = rng.choice(items)
        watched = rng.randint(0, c["duration_min"] * 60)
        out.append(
            {
                "interaction_id": f"i_{rng.getrandbits(64):016x}",
                "user_id": u["user_id"],
                "content_id": c["content_id"],
                "event_ts": base + timedelta(seconds=rng.randint(0, 60 * 60 * 24 * 150)),
                "watch_seconds": watched,
                "completed": watched >= int(0.85 * c["duration_min"] * 60),
                "device": rng.choice(["mobile", "tv", "tablet", "web"]),
            }
        )
    return out


def write_parquet(spark, rows: list[dict], path: str) -> None:
    df = spark.createDataFrame(rows)
    df.write.mode("overwrite").parquet(path)
