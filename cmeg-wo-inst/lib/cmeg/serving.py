"""Chained inference serving pyfunc.

Retrieval -> ranker -> diversity -> GenAI explanation.

Retrieval uses **Databricks Vector Search** (AI semantic search) as the primary
candidate generator: for each user we take the synopsis of the show they watched
most ("seed"), and ask the item_index for the most semantically similar shows.
If Vector Search is unavailable at serving time, we fall back to the in-memory
two-tower dot product so the endpoint still works.
"""

import mlflow.pyfunc
import pandas as pd


class RecChain(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import mlflow
        self.tt = mlflow.pyfunc.load_model(context.artifacts["two_tower"])
        self.ranker = mlflow.pyfunc.load_model(context.artifacts["ranker"])
        self.item_meta = pd.read_parquet(context.artifacts["item_meta"])
        self.user_meta = pd.read_parquet(context.artifacts["user_meta"])

        cfg = context.model_config or {}
        self.genai_model = cfg.get("genai_model", "databricks-meta-llama-3-3-70b-instruct")
        self.top_k = int(cfg.get("top_k", 5))
        self.num_candidates = int(cfg.get("num_candidates", 100))
        self.vs_endpoint = cfg.get("vs_endpoint")
        self.vs_index = cfg.get("vs_index")

        # Try to connect to the Vector Search index up front
        self._index = None
        if self.vs_endpoint and self.vs_index:
            try:
                from databricks.vector_search.client import VectorSearchClient
                self._index = VectorSearchClient(disable_notice=True).get_index(
                    endpoint_name=self.vs_endpoint, index_name=self.vs_index
                )
            except Exception as e:
                print(f"[RecChain] Vector Search unavailable, will fall back to two-tower: {e}")
                self._index = None

    # ---- retrieval ---------------------------------------------------------

    def _retrieve(self, uid, user_row):
        """Return (candidate_content_ids, source). Vector Search first, two-tower fallback."""
        if self._index is not None and not user_row.empty and "seed_content_id" in user_row.columns:
            seed_id = user_row["seed_content_id"].iloc[0]
            seed = self.item_meta[self.item_meta["content_id"] == seed_id]
            if not seed.empty and "synopsis" in seed.columns:
                query_text = str(seed["synopsis"].iloc[0])
                try:
                    res = self._index.similarity_search(
                        query_text=query_text,
                        columns=["content_id"],
                        num_results=self.num_candidates,
                    )
                    rows = (res.get("result", {}) or {}).get("data_array", []) or []
                    ids = [r[0] for r in rows]
                    if ids:
                        return ids, "vector_search"
                except Exception as e:
                    print(f"[RecChain] Vector Search query failed, falling back: {e}")
        # fallback: in-memory two-tower dot product
        return self.tt.predict(pd.DataFrame({"user_id": [uid]}))[0], "two_tower"

    # ---- rerank + explain --------------------------------------------------

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

    # ---- entry point -------------------------------------------------------

    def predict(self, context, model_input):
        out = []
        for _, row in model_input.iterrows():
            uid = row["user_id"]
            user_row = self.user_meta[self.user_meta["user_id"] == uid].head(1)

            candidate_ids, source = self._retrieve(uid, user_row)
            cand_df = self.item_meta[self.item_meta["content_id"].isin(candidate_ids)].copy()
            if cand_df.empty:
                out.append([])
                continue

            scored = []
            for _, c in cand_df.iterrows():
                feats = pd.concat([user_row.reset_index(drop=True), c.to_frame().T.reset_index(drop=True)], axis=1)
                feats = feats.select_dtypes(include="number").fillna(0)
                try:
                    score = float(self.ranker.predict(feats)[0])
                except Exception:
                    score = 0.0
                scored.append({"content_id": c["content_id"], "title": c.get("title", ""),
                               "genre": c.get("genre", ""), "score": score})
            scored.sort(key=lambda x: -x["score"])
            top = self._diversity_rerank(scored, top_k=self.top_k)

            fav_genre = user_row["fav_genre"].iloc[0] if (not user_row.empty and "fav_genre" in user_row.columns) else "drama"
            recent = []
            for t in top:
                t["retrieved_by"] = source     # "vector_search" or "two_tower"
                t["why"] = self._explain(fav_genre, recent, t["title"], t["genre"])
                recent.append(t["title"])
            out.append(top)
        return out
