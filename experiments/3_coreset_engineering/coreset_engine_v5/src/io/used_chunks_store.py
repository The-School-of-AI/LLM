"""Used-chunk ID store for enforcing non-overlap across stages in streaming mode.

At 2T-scale, holding all used chunk IDs in memory is not feasible.
This module provides a lightweight disk-backed membership structure.

Implementation: SQLite with a primary-key table of chunk_id.
- Fast enough for batch membership checks via temporary table join.
- Deterministic and restart-safe.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set


@dataclass
class UsedChunksStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._tmp_table_initialized = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # WAL improves concurrency/read performance; safe for single-writer runs.
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._tmp_table_initialized = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS used_chunks (
                chunk_id TEXT PRIMARY KEY
            );
            """
        )
        conn.commit()

    def _ensure_tmp_ids_table(self) -> None:
        if self._tmp_table_initialized:
            return
        conn = self._connect()
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS tmp_ids (chunk_id TEXT PRIMARY KEY);"
        )
        self._tmp_table_initialized = True

    def add_many(self, chunk_ids: Iterable[str]) -> int:
        ids = [(str(cid),) for cid in chunk_ids]
        if not ids:
            return 0
        conn = self._connect()
        conn.executemany("INSERT OR IGNORE INTO used_chunks(chunk_id) VALUES (?);", ids)
        conn.commit()
        # sqlite3 doesn't reliably expose inserted count with OR IGNORE;
        # return input count as an upper bound.
        return len(ids)

    def filter_unused(self, chunk_ids: Iterable[str]) -> Set[str]:
        """Return the subset of chunk_ids that are NOT present in the store."""
        ids_list: List[str] = [str(cid) for cid in chunk_ids]
        if not ids_list:
            return set()

        # Use a temporary table and a join instead of a huge IN (...) list.
        conn = self._connect()
        self._ensure_tmp_ids_table()
        conn.execute("DELETE FROM tmp_ids;")
        conn.executemany(
            "INSERT OR IGNORE INTO tmp_ids(chunk_id) VALUES (?);",
            [(cid,) for cid in ids_list],
        )

        cur = conn.execute(
            """
            SELECT t.chunk_id
            FROM tmp_ids t
            LEFT JOIN used_chunks u ON u.chunk_id = t.chunk_id
            WHERE u.chunk_id IS NULL;
            """
        )
        rows = cur.fetchall()
        conn.commit()
        return {r[0] for r in rows}
