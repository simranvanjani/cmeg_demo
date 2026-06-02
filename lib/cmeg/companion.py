"""Render chapter-complete cards and asset link helpers for the cmeg demo."""

from html import escape
from typing import Iterable, Tuple

AssetTuple = Tuple[str, str, str]  # (asset_type, name, url)


def format_asset_url(workspace_url: str, asset_type: str, asset_id: str) -> str:
    base = workspace_url.rstrip("/")
    if asset_type == "table":
        parts = asset_id.split(".")
        return f"{base}/explore/data/{parts[0]}/{parts[1]}/{parts[2]}"
    if asset_type == "schema":
        parts = asset_id.split(".")
        return f"{base}/explore/data/{parts[0]}/{parts[1]}"
    if asset_type == "catalog":
        return f"{base}/explore/data/{asset_id}"
    if asset_type == "volume":
        parts = asset_id.split(".")
        return f"{base}/explore/data/volumes/{parts[0]}/{parts[1]}/{parts[2]}"
    if asset_type == "pipeline":
        return f"{base}/#joblist/pipelines/{asset_id}"
    if asset_type == "job":
        return f"{base}/jobs/{asset_id}"
    if asset_type == "experiment":
        return f"{base}/ml/experiments/{asset_id}"
    if asset_type == "model":
        return f"{base}/explore/data/models/{asset_id}"
    if asset_type == "endpoint":
        return f"{base}/ml/endpoints/{asset_id}"
    if asset_type == "vector_index":
        parts = asset_id.split(".")
        return f"{base}/explore/data/{parts[0]}/{parts[1]}/{parts[2]}"
    if asset_type == "monitor":
        parts = asset_id.split(".")
        return f"{base}/explore/data/{parts[0]}/{parts[1]}/{parts[2]}/monitoring"
    if asset_type == "dashboard":
        return f"{base}/dashboardsv3/{asset_id}"
    if asset_type == "genie":
        return f"{base}/genie/rooms/{asset_id}"
    return base


def render_chapter_card(
    chapter: int,
    title: str,
    created: Iterable[AssetTuple],
    next_label: str | None,
    next_url: str | None,
) -> str:
    rows = "".join(
        f'<li><b>{escape(t)}</b>: {escape(n)} '
        f'&mdash; <a href="{escape(u)}" target="_blank">Open &#8599;</a></li>'
        for (t, n, u) in created
    )
    next_html = ""
    if next_label and next_url:
        next_html = (
            f'<p style="margin-top:12px;">'
            f'<a href="{escape(next_url)}" target="_blank" '
            f'style="background:#1f6feb;color:white;padding:8px 14px;'
            f'border-radius:6px;text-decoration:none;">'
            f"Next: {escape(next_label)} &rarr;</a></p>"
        )
    return (
        f'<div style="border:1px solid #d0d7de;border-radius:8px;'
        f'padding:16px;background:#f6f8fa;margin:12px 0;font-family:Inter,sans-serif;">'
        f'<h3 style="margin:0 0 8px 0;">&#10003; Chapter {chapter} complete '
        f'&mdash; {escape(title)}</h3>'
        f'<p style="margin:0 0 6px 0;">Created in this chapter:</p>'
        f'<ul style="margin:0;">{rows}</ul>'
        f"{next_html}"
        f"</div>"
    )


def chapter_complete(
    chapter: int,
    title: str,
    created: list,
    next_label: str | None = None,
    next_url: str | None = None,
):
    """Notebook-only helper. Calls displayHTML if running inside Databricks."""
    html = render_chapter_card(chapter, title, created, next_label, next_url)
    try:
        from IPython.display import display, HTML  # noqa
        display(HTML(html))
    except Exception:
        print(html)
