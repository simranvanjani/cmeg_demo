"""Training functions for two-tower retrieval and GBT ranker."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TwoTowerArtifacts:
    user_embeddings: pd.DataFrame
    item_embeddings: pd.DataFrame


def train_two_tower(
    interactions: pd.DataFrame,
    n_factors: int = 32,
    n_epochs: int = 3,
    seed: int = 42,
) -> TwoTowerArtifacts:
    """Lightweight two-tower via numpy SGD on (user, item, watch_seconds) tuples.

    Produces a user embedding table and an item embedding table that can be
    indexed in Vector Search. Avoids a heavy TF Recommenders dependency.
    """
    rng = np.random.default_rng(seed)
    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(interactions["content_id"].unique())
    u_ix = {u: i for i, u in enumerate(user_ids)}
    i_ix = {c: i for i, c in enumerate(item_ids)}

    U = rng.normal(0, 0.1, (len(user_ids), n_factors))
    V = rng.normal(0, 0.1, (len(item_ids), n_factors))

    pairs = interactions[["user_id", "content_id"]].to_numpy()
    rates = interactions["watch_seconds"].to_numpy().astype(float)
    rates = rates / max(rates.max(), 1.0)
    lr = 0.05
    sample_n = min(50_000, len(pairs))

    for epoch in range(n_epochs):
        order = rng.permutation(len(pairs))[:sample_n]
        loss = 0.0
        for k in order:
            uid, cid = pairs[k]
            iu, ic = u_ix[uid], i_ix[cid]
            pred = U[iu] @ V[ic]
            err = rates[k] - pred
            U[iu] += lr * (err * V[ic] - 0.01 * U[iu])
            V[ic] += lr * (err * U[iu] - 0.01 * V[ic])
            loss += err * err
        print(f"epoch {epoch + 1}/{n_epochs}: loss={loss:.3f}")

    return TwoTowerArtifacts(
        user_embeddings=pd.DataFrame({"user_id": user_ids, "embedding": [u.tolist() for u in U]}),
        item_embeddings=pd.DataFrame({"content_id": item_ids, "embedding": [v.tolist() for v in V]}),
    )


def train_ranker(features: pd.DataFrame, target_col: str = "completed", n_trials: int = 5):
    """LightGBM binary classifier with a tiny Optuna sweep."""
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    feature_cols = [c for c in features.columns if c not in ("user_id", "content_id", target_col)]
    X = features[feature_cols].fillna(0)
    y = features[target_col].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "verbose": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    final = lgb.LGBMClassifier(**best, verbose=-1)
    final.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
    return final, {"best_params": best, "val_auc": auc, "feature_cols": feature_cols}
