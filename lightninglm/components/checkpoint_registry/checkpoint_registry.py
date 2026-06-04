"""
Checkpoint Registry — ClickHouse-backed governance layer.

Stores checkpoint metadata (S3 paths, tags, protection status) in the
training_observability.checkpoints table.  Uses ReplacingMergeTree so the
latest row per (run_id, s3_key) wins; queries use FINAL for consistent reads.
Soft-deletes via a `status` column instead of actual DELETEs.
"""

import base64
import json
import os
import ssl
import urllib.error
import urllib.request

PROTECTED_TAGS = frozenset(["growth", "tqp", "release_candidate"])


class CheckpointRegistry:
    """
    Governance layer for checkpoints, backed by ClickHouse.

    Parameters
    ----------
    clickhouse_url : str
        ClickHouse HTTP(S) endpoint, e.g. "https://db-host:8443".
    database : str
        Database name (default: training_observability).
    user : str | None
        ClickHouse user.  Falls back to CLICKHOUSE_USER env var.
    password : str | None
        ClickHouse password.  Falls back to CLICKHOUSE_PASSWORD env var.
    ca_cert : str | None
        Path to CA certificate for TLS verification.  Falls back to
        CLICKHOUSE_CA_CERT env var.  Set to empty string to skip verification.
    """

    def __init__(
        self,
        clickhouse_url: str | None = None,
        database: str = "training_observability",
        user: str | None = None,
        password: str | None = None,
        ca_cert: str | None = None,
        check_connectivity_on_init: bool = False,
    ):
        self.clickhouse_url = (
            clickhouse_url
            or os.environ.get("CLICKHOUSE_HTTPS_ENDPOINT")
            or os.environ.get("CLICKHOUSE_HTTP_ENDPOINT", "http://localhost:8123")
        )
        self.database = database
        self.table = f"{database}.checkpoints"

        # Auth
        self._user = user or os.environ.get("CLICKHOUSE_USER", "")
        self._password = password or os.environ.get("CLICKHOUSE_PASSWORD", "")
        self._auth_header = ""
        if self._user:
            creds = base64.b64encode(f"{self._user}:{self._password}".encode()).decode()
            self._auth_header = f"Basic {creds}"

        # TLS
        ca_path = (
            ca_cert if ca_cert is not None else os.environ.get("CLICKHOUSE_CA_CERT", "")
        )
        self._ssl_ctx = None
        if self.clickhouse_url.startswith("https"):
            self._ssl_ctx = ssl.create_default_context()
            if ca_path and os.path.isfile(ca_path):
                self._ssl_ctx.load_verify_locations(ca_path)
            else:
                # Self-signed cert without CA file: skip verification
                self._ssl_ctx.check_hostname = False
                self._ssl_ctx.verify_mode = ssl.CERT_NONE

        # Optional direct connectivity probe (off by default).
        if check_connectivity_on_init:
            try:
                self._query("SELECT 1")
                print(f"✓ CheckpointRegistry connected to {self.clickhouse_url}")
            except Exception as e:
                print(f"⚠ CheckpointRegistry: ClickHouse not reachable ({e})")

    # ------------------------------------------------------------------
    # Low-level ClickHouse HTTP helpers
    # ------------------------------------------------------------------

    def _make_request(self, req: urllib.request.Request) -> str:
        """Send an HTTP(S) request with auth headers and TLS context."""
        if self._auth_header:
            req.add_header("Authorization", self._auth_header)
        ctx = self._ssl_ctx if self._ssl_ctx else None
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return resp.read().decode()

    def _query(self, sql: str) -> str:
        """Execute a read query and return the response body as a string."""
        url = f"{self.clickhouse_url}/?query={urllib.request.quote(sql)}"
        req = urllib.request.Request(url, method="GET")
        return self._make_request(req)

    def _insert(self, sql: str) -> None:
        """Execute an INSERT statement via POST body."""
        req = urllib.request.Request(
            self.clickhouse_url,
            data=sql.encode(),
            method="POST",
        )
        self._make_request(req)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_checkpoint(
        self,
        run_id: str,
        step: int,
        s3_key: str,
        loss: float = 0.0,
        tag: str = "temporary",
        host: str = "",
        duration_s: float = 0.0,
        size_bytes: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """
        Register a checkpoint after it has been saved/uploaded.
        Auto-protects 'growth', 'tqp', and 'release_candidate' tags.
        """
        is_protected = 1 if tag in PROTECTED_TAGS else 0
        metadata_json = json.dumps(metadata) if metadata else "{}"
        host = host or os.environ.get("HOSTNAME", os.uname().nodename)

        sql = (
            f"INSERT INTO {self.table} "
            f"(run_id, step, s3_key, loss, tag, is_protected, status, host, "
            f"duration_s, size_bytes, metadata_json) VALUES "
            f"('{_esc(run_id)}', {step}, '{_esc(s3_key)}', {loss}, "
            f"'{_esc(tag)}', {is_protected}, 'registered', '{_esc(host)}', "
            f"{duration_s}, {size_bytes}, '{_esc(metadata_json)}')"
        )
        self._insert(sql)
        print(
            f"✓ Registered checkpoint: {s3_key} (tag={tag}, protected={bool(is_protected)})"
        )

    def can_delete(self, s3_key: str) -> bool:
        """
        Policy check: is it safe to delete this checkpoint?
        Returns False if the checkpoint is protected or unknown.
        """
        sql = (
            f"SELECT is_protected, tag, status FROM {self.table} FINAL "
            f"WHERE s3_key = '{_esc(s3_key)}' LIMIT 1"
        )
        result = self._query(sql).strip()
        if not result:
            print(f"⚠️  Unknown checkpoint {s3_key}. Preventing deletion.")
            return False

        parts = result.split("\t")
        is_protected, tag, status = int(parts[0]), parts[1], parts[2]

        if status == "deleted":
            return True  # already soft-deleted

        if is_protected:
            print(f"⛔ Blocked deletion of protected checkpoint {s3_key} (tag={tag})")
            return False

        return True

    def mark_for_deletion(self, s3_key: str) -> None:
        """
        Soft-delete a checkpoint. Appends a new row with status='deleted'.
        Raises ValueError if the checkpoint is protected.
        """
        if not self.can_delete(s3_key):
            raise ValueError(f"Cannot delete protected checkpoint {s3_key}")

        # Read existing row to carry forward its fields
        sql = (
            f"SELECT run_id, step, loss, tag, host "
            f"FROM {self.table} FINAL "
            f"WHERE s3_key = '{_esc(s3_key)}' LIMIT 1"
        )
        result = self._query(sql).strip()
        if not result:
            raise ValueError(f"Checkpoint not found: {s3_key}")

        run_id, step, loss, tag, host = result.split("\t")

        insert_sql = (
            f"INSERT INTO {self.table} "
            f"(run_id, step, s3_key, loss, tag, is_protected, status, host) VALUES "
            f"('{_esc(run_id)}', {step}, '{_esc(s3_key)}', {loss}, "
            f"'{_esc(tag)}', 0, 'deleted', '{_esc(host)}')"
        )
        self._insert(insert_sql)
        print(f"✓ Soft-deleted checkpoint: {s3_key}")

    def get_checkpoint(self, s3_key: str) -> dict | None:
        """Return the latest state of a single checkpoint, or None."""
        sql = (
            f"SELECT run_id, step, s3_key, loss, tag, is_protected, status, "
            f"host, duration_s, size_bytes, event_time "
            f"FROM {self.table} FINAL "
            f"WHERE s3_key = '{_esc(s3_key)}' LIMIT 1"
        )
        result = self._query(sql).strip()
        if not result:
            return None
        return _row_to_dict(result)

    def list_checkpoints(self, run_id: str, status: str = "registered") -> list[dict]:
        """List all checkpoints for a run, filtered by status."""
        sql = (
            f"SELECT run_id, step, s3_key, loss, tag, is_protected, status, "
            f"host, duration_s, size_bytes, event_time "
            f"FROM {self.table} FINAL "
            f"WHERE run_id = '{_esc(run_id)}' AND status = '{_esc(status)}' "
            f"ORDER BY step ASC"
        )
        result = self._query(sql).strip()
        if not result:
            return []
        return [_row_to_dict(line) for line in result.split("\n") if line.strip()]

    def best_checkpoint(
        self, run_id: str, metric: str = "loss", top_n: int = 1
    ) -> list[dict]:
        """Return the top-N checkpoints with the lowest loss for a run."""
        sql = (
            f"SELECT run_id, step, s3_key, loss, tag, is_protected, status, "
            f"host, duration_s, size_bytes, event_time "
            f"FROM {self.table} FINAL "
            f"WHERE run_id = '{_esc(run_id)}' AND status = 'registered' "
            f"ORDER BY loss ASC LIMIT {top_n}"
        )
        result = self._query(sql).strip()
        if not result:
            return []
        return [_row_to_dict(line) for line in result.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(value: str) -> str:
    """Escape single quotes for ClickHouse SQL strings."""
    return str(value).replace("'", "\\'")


def _row_to_dict(tsv_line: str) -> dict:
    """Parse a tab-separated ClickHouse row into a dict."""
    parts = tsv_line.split("\t")
    return {
        "run_id": parts[0],
        "step": int(parts[1]),
        "s3_key": parts[2],
        "loss": float(parts[3]),
        "tag": parts[4],
        "is_protected": bool(int(parts[5])),
        "status": parts[6],
        "host": parts[7],
        "duration_s": float(parts[8]),
        "size_bytes": int(parts[9]),
        "event_time": parts[10],
    }
