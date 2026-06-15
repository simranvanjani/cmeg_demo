"""Chained inference serving pyfunc: retrieval -> ranker -> diversity -> explanation."""

import mlflow.pyfunc
import pandas as pd


class RecChain(mlflow.pyfunc.PythonModel):
    """Calls the two-tower for candidates, ranker for scores, dedupes by genre, attaches LLM explanation."""

    def load_context(self, context):
        import mlflow
        self.tt = mlflow.pyfunc.load_model(context.artifacts["two_tower"])
        self.ranker = mlflow.pyfunc.load_model(context.artifacts["ranker"])
        self.item_meta = pd.read_parquet(context.artifacts["item_meta"])
        self.user_meta = pd.read_parquet(context.artifacts["user_meta"])
        self.genai_model = (context.model_config or {}).get("genai_model", "databricks-meta-llama-3-3-70b-instruct")
        self.top_k = int((context.model_config or {}).get("top_k", 5))

    def _diversity_rerank(self, candidates: list, top_k: int) -> list:
        seen_genres = set()
        out = []
        for c in candidates:
            if c["genre"] in seen_genres and len(out) < top_k:
                continue
            seen_genres.add(c["genre"])
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def _explain(self, fav_genre: str, recent: list, cand_title: str, cand_genre: str) -> str:
        try:
            from databricks.sdk import WorkspaceClient
            from cmeg.genai import build_explanation_messages
            w = WorkspaceClient()
            resp = w.serving_endpoints.query(
                name=self.genai_model,
                messages=build_explanation_messages(fav_genre, recent, cand_title, cand_genre),
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"Recommended based on your taste in {fav_genre}."

    def predict(self, context, model_input):
        out = []
        for _, row in model_input.iterrows():
            uid = row["user_id"]
            candidate_ids = self.tt.predict(pd.DataFrame({"user_id": [uid]}))[0]
            cand_df = self.item_meta[self.item_meta["content_id"].isin(candidate_ids)].copy()
            if cand_df.empty:
                out.append([])
                continue
            user_row = self.user_meta[self.user_meta["user_id"] == uid].head(1)
            scored = []
            for _, c in cand_df.iterrows():
                feats = pd.concat([user_row.reset_index(drop=True), c.to_frame().T.reset_index(drop=True)], axis=1)
                feats = feats.select_dtypes(include="number").fillna(0)
                try:
                    score = float(self.ranker.predict(feats)[0])
                except Exception:
                    score = 0.0
                scored.append({"content_id": c["content_id"], "title": c.get("title", ""), "genre": c.get("genre", ""), "score": score})
            scored.sort(key=lambda x: -x["score"])
            top = self._diversity_rerank(scored, top_k=self.top_k)
            fav_genre = user_row["fav_genre"].iloc[0] if (not user_row.empty and "fav_genre" in user_row.columns) else "drama"
            recent = []
            for t in top:
                t["why"] = self._explain(fav_genre, recent, t["title"], t["genre"])
                recent.append(t["title"])
            out.append(top)
        return out
