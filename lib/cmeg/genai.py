"""GenAI explanation prompt construction for the cmeg demo."""

EXPLAIN_SYSTEM_PROMPT = (
    "You are a recommendation explainer for a streaming app. Given a user's "
    "recently watched genres and a candidate title, return ONE short sentence "
    "(max 20 words) explaining why the user might like the candidate. "
    "Do not mention the user by name. Do not invent facts."
)


def build_explanation_messages(fav_genre: str, recent_titles: list, candidate_title: str, candidate_genre: str) -> list:
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recently watched: {', '.join(recent_titles[:3]) or 'unknown'}\n"
                f"Favorite genre: {fav_genre}\n"
                f"Candidate: '{candidate_title}' (genre: {candidate_genre})\n"
                "Why might they like this candidate?"
            ),
        },
    ]
