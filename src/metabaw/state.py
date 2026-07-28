from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    fingerprint: str
    status: str
    started_at: str | None
    ended_at: str | None
    exit_code: int | None
    log_path: str | None
    message: str | None


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                task_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                exit_code INTEGER,
                log_path TEXT,
                message TEXT
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, task_id: str) -> TaskRecord | None:
        row = self.connection.execute(
            "SELECT task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message "
            "FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return TaskRecord(*row) if row else None

    def start(self, task_id: str, fingerprint: str, log_path: Path) -> None:
        self.connection.execute(
            """
            INSERT INTO task_runs(task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message)
            VALUES (?, ?, 'running', ?, NULL, NULL, ?, NULL)
            ON CONFLICT(task_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status='running',
                started_at=excluded.started_at,
                ended_at=NULL,
                exit_code=NULL,
                log_path=excluded.log_path,
                message=NULL
            """,
            (task_id, fingerprint, utc_now(), str(log_path)),
        )
        self.connection.commit()

    def finish(self, task_id: str, status: str, exit_code: int, message: str = "") -> None:
        self.connection.execute(
            "UPDATE task_runs SET status=?, ended_at=?, exit_code=?, message=? WHERE task_id=?",
            (status, utc_now(), exit_code, message, task_id),
        )
        self.connection.commit()

    def adopt(self, task_id: str, fingerprint: str, message: str) -> None:
        """Record complete pre-existing outputs as a successful reusable task."""
        timestamp = utc_now()
        self.connection.execute(
            """
            INSERT INTO task_runs(
                task_id, fingerprint, status, started_at, ended_at,
                exit_code, log_path, message
            )
            VALUES (?, ?, 'success', ?, ?, 0, NULL, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status='success',
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                exit_code=0,
                log_path=NULL,
                message=excluded.message
            """,
            (task_id, fingerprint, timestamp, timestamp, message),
        )
        self.connection.commit()

    def records(self) -> Iterable[TaskRecord]:
        rows = self.connection.execute(
            "SELECT task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message "
            "FROM task_runs ORDER BY task_id"
        )
        for row in rows:
            yield TaskRecord(*row)
