"""Lakehouse Monitor helper for the cmeg demo inference table."""


def build_monitor_dir(user_name: str) -> str:
    return f"/Workspace/Users/{user_name}/cmeg_monitor"
