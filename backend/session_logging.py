"""Append structured conversation events to one local file per browser session."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


class SessionEventLogger:
    def __init__(self, log_directory: Path | None = None) -> None:
        self._log_directory = log_directory or Path(__file__).with_name("logs")
        self._lock = threading.Lock()

    def log(
        self,
        *,
        session_id: UUID,
        request_id: UUID,
        event_type: str,
        payload: dict,
    ) -> Path:
        self._log_directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            existing_logs = sorted(
                self._log_directory.glob(f"session-*-{session_id}.jsonl")
            )
            if existing_logs:
                log_path = existing_logs[0]
            else:
                filename_timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S.%fZ"
                )
                log_path = self._log_directory / (
                    f"session-{filename_timestamp}-{session_id}.jsonl"
                )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": str(session_id),
                "request_id": str(request_id),
                "event_type": event_type,
                "payload": payload,
            }
            serialized = json.dumps(record, ensure_ascii=False, default=str)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")
                log_file.flush()
        return log_path
