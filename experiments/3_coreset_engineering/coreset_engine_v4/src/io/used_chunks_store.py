"""Used-chunk ID store for enforcing non-overlap across stages in streaming mode.

At 2T-scale, holding all used chunk IDs in memory is not feasible.
This module provides a lightweight disk-backed membership structure.

Implementation: SQLite with a primary-key table of chunk_id.
- Fast enough for batch membership checks via temporary table join.
- Deterministic and restart-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set
import sqlite3


@dataclass
class UsedChunksStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # WAL improves concurrency/read performance; safe for single-writer runs.
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS used_chunks (
                    chunk_id TEXT PRIMARY KEY
                );
                """
            )

    def add_many(self, chunk_ids: Iterable[str]) -> int:
        ids = [(str(cid),) for cid in chunk_ids]
        if not ids:
            return 0
        with self._connect() as conn:
            conn.executemany("INSERT OR IGNORE INTO used_chunks(chunk_id) VALUES (?);", ids)
            # sqlite3 doesn't reliably expose inserted count with OR IGNORE;
            # return input count as an upper bound.
        return len(ids)

    def filter_unused(self, chunk_ids: Iterable[str]) -> Set[str]:
        """Return the subset of chunk_ids that are NOT present in the store."""
        ids_list: List[str] = [str(cid) for cid in chunk_ids]
        if not ids_list:
            return set()

        # Use a temporary table and a join instead of a huge IN (...) list.
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS tmp_ids;")
            conn.execute("CREATE TEMP TABLE tmp_ids (chunk_id TEXT PRIMARY KEY);")
            conn.executemany("INSERT OR IGNORE INTO tmp_ids(chunk_id) VALUES (?);", [(cid,) for cid in ids_list])

            cur = conn.execute(
                """
                SELECT t.chunk_id
                FROM tmp_ids t
                LEFT JOIN used_chunks u ON u.chunk_id = t.chunk_id
                WHERE u.chunk_id IS NULL;
                """
            )
            rows = cur.fetchall()
            return {r[0] for r in rows}
