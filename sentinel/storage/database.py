"""Local SQLite store for events, sessions and periodic statistics.

Entirely local - the product requires no cloud backend.  Writes arrive from the
inference thread while reads come from the GUI thread, so the single connection
is opened with ``check_same_thread=False`` and every statement is serialised
through one lock.
"""

from __future__ import annotations

import csv
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sentinel.config import DB_PATH, LOGGER
from sentinel.models import SecurityEvent

__all__ = ["EventDatabase"]


class EventDatabase:
    """Local SQLite store for events, sessions and periodic statistics.

    Entirely local - the product requires no cloud backend.  Writes arrive from
    the inference thread while reads come from the GUI thread, so the single
    connection is opened with ``check_same_thread=False`` and every statement is
    serialised through one lock.
    """

    SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id    TEXT PRIMARY KEY,
            ts          REAL NOT NULL,
            iso_ts      TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            severity    TEXT NOT NULL,
            track_id    INTEGER,
            zone        TEXT,
            description TEXT NOT NULL,
            mode        TEXT,
            source      TEXT,
            session_id  TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            started_at  REAL NOT NULL,
            ended_at    REAL,
            mode        TEXT,
            source      TEXT,
            backend     TEXT,
            device      TEXT,
            frames      INTEGER DEFAULT 0,
            events      INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS statistics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            ts          REAL NOT NULL,
            fps         REAL,
            people      INTEGER,
            vehicles    INTEGER,
            tracks      INTEGER,
            latency_ms  REAL,
            cpu_percent REAL,
            ram_mb      REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)",
        "CREATE INDEX IF NOT EXISTS idx_events_sev ON events(severity)",
        "CREATE INDEX IF NOT EXISTS idx_stats_session ON statistics(session_id)",
    )

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self.session_id: str = uuid.uuid4().hex[:12]
        self.available = False
        self.error: Optional[str] = None
        self._open()

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=5.0
            )
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                for statement in self.SCHEMA:
                    self._conn.execute(statement)
                self._conn.commit()
            self.available = True
            LOGGER.info("Event database ready at %s", self.path)
        except sqlite3.Error as exc:
            self.error = str(exc)
            self.available = False
            self._conn = None
            LOGGER.error("SQLite unavailable (%s) - running without persistence", exc)

    # -- sessions -------------------------------------------------------------
    def start_session(self, mode: str, source: str, backend: str, device: str) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        if not self.available or self._conn is None:
            return self.session_id
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO sessions (session_id, started_at, mode, source, "
                    "backend, device) VALUES (?, ?, ?, ?, ?, ?)",
                    (self.session_id, time.time(), mode, source, backend, device),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("start_session failed: %s", exc)
        return self.session_id

    def end_session(self, frames: int, events: int) -> None:
        if not self.available or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE sessions SET ended_at=?, frames=?, events=? "
                    "WHERE session_id=?",
                    (time.time(), int(frames), int(events), self.session_id),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("end_session failed: %s", exc)

    # -- events ---------------------------------------------------------------
    def insert_events(self, events: Sequence[SecurityEvent]) -> None:
        if not events or not self.available or self._conn is None:
            return
        rows = [event.as_row() + (self.session_id,) for event in events]
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO events (event_id, ts, iso_ts, event_type, "
                    "severity, track_id, zone, description, mode, source, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("insert_events failed: %s", exc)

    def record_statistics(
        self, *, fps: float, people: int, vehicles: int, tracks: int,
        latency_ms: float, cpu_percent: float, ram_mb: float,
    ) -> None:
        if not self.available or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO statistics (session_id, ts, fps, people, vehicles, "
                    "tracks, latency_ms, cpu_percent, ram_mb) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.session_id, time.time(), float(fps), int(people),
                     int(vehicles), int(tracks), float(latency_ms),
                     float(cpu_percent), float(ram_mb)),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.debug("record_statistics failed: %s", exc)

    # -- queries --------------------------------------------------------------
    def query_events(
        self, limit: int = 500, severity: Optional[str] = None,
        event_type: Optional[str] = None, since: Optional[float] = None,
        search: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        if not self.available or self._conn is None:
            return []
        sql = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if since is not None:
            sql += " AND ts >= ?"
            params.append(float(since))
        if search:
            sql += " AND (description LIKE ? OR zone LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like])
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        try:
            with self._lock:
                return list(self._conn.execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            LOGGER.error("query_events failed: %s", exc)
            return []

    def event_counts_by_severity(self, since: Optional[float] = None) -> Dict[str, int]:
        if not self.available or self._conn is None:
            return {}
        sql = "SELECT severity, COUNT(*) AS n FROM events"
        params: List[Any] = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(float(since))
        sql += " GROUP BY severity"
        try:
            with self._lock:
                return {row["severity"]: int(row["n"])
                        for row in self._conn.execute(sql, params).fetchall()}
        except sqlite3.Error as exc:
            LOGGER.error("event_counts_by_severity failed: %s", exc)
            return {}

    def recent_sessions(self, limit: int = 25) -> List[sqlite3.Row]:
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                return list(self._conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall())
        except sqlite3.Error as exc:
            LOGGER.error("recent_sessions failed: %s", exc)
            return []

    def clear_events(self) -> int:
        if not self.available or self._conn is None:
            return 0
        try:
            with self._lock:
                cursor = self._conn.execute("DELETE FROM events")
                self._conn.commit()
                return int(cursor.rowcount or 0)
        except sqlite3.Error as exc:
            LOGGER.error("clear_events failed: %s", exc)
            return 0

    # -- export ---------------------------------------------------------------
    def export_csv(self, destination: Path, rows: Optional[Sequence[sqlite3.Row]] = None) -> int:
        """Write event history to CSV.  Returns the number of exported rows."""
        if rows is None:
            rows = self.query_events(limit=100000)
        destination = Path(destination)
        columns = ["iso_ts", "event_type", "severity", "track_id", "zone",
                   "description", "mode", "source", "session_id"]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([c.upper() for c in columns])
                for row in rows:
                    writer.writerow([row[c] if c in row.keys() else "" for c in columns])
            return len(rows)
        except OSError as exc:
            raise RuntimeError(f"CSV export failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                    LOGGER.info("Event database closed")
                except sqlite3.Error as exc:
                    LOGGER.debug("DB close error: %s", exc)
                finally:
                    self._conn = None
                    self.available = False
