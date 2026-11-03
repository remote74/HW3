"""JSON persistence for successful churn model training runs."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from repository.model import MODEL_DIRECTORY

TRAINING_HISTORY_PATH = MODEL_DIRECTORY / "training_history.json"


def load_training_history() -> list[dict[str, Any]]:
    """Return history in chronological order; a missing file means no runs yet."""
    if not TRAINING_HISTORY_PATH.exists():
        return []
    try:
        records = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read training history: {exc}") from exc
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Training history has an invalid format")
    return records


def record_training_run(record: dict[str, Any]) -> dict[str, Any]:
    """Append a JSON-compatible record and atomically publish the history file."""
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = load_training_history()
    # Round-trip validates that callers do not persist numpy or other non-JSON values.
    normalized = json.loads(json.dumps(record, allow_nan=False))
    records.append(normalized)
    write_training_history(records)
    return normalized


def write_training_history(records: list[dict[str, Any]]) -> None:
    """Atomically replace history with a validated list of records."""
    normalized = json.loads(json.dumps(records, allow_nan=False))
    if not isinstance(normalized, list) or any(not isinstance(row, dict) for row in normalized):
        raise ValueError("Training history has an invalid format")
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".training-history-", suffix=".tmp", dir=MODEL_DIRECTORY)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, TRAINING_HISTORY_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
