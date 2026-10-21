"""Shared experiment directory names, timestamped in Beijing time to seconds."""

from datetime import datetime
from zoneinfo import ZoneInfo


def libero_run_name(backend, commander, *, model=None, started_at=None):
    action_policy = {"openpi": "pi05", "gawm": "gawm"}[backend]
    command_policy = (
        "fake"
        if commander == "fake"
        else "".join(
            char for char in model.lower() if char.isascii() and char.isalnum()
        )
    )
    if not command_policy:
        raise ValueError("A command policy name is required")
    timestamp = (started_at or datetime.now(ZoneInfo("Asia/Shanghai"))).astimezone(
        ZoneInfo("Asia/Shanghai")
    )
    return f"liberopro-{action_policy}-{command_policy}-{timestamp:%Y%m%d-%H%M%S}"
