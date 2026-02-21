"""
dashboard_server.py
ClickHouse-backed Training Dashboard Server — with auto table discovery
"""

import os
import threading
from collections import defaultdict
from pathlib import Path

import clickhouse_connect
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Load .env file if it exists (no extra packages needed)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# -----------------------------------------------------------------------------
# CONFIGURATION — set these as environment variables, never hardcode them
# -----------------------------------------------------------------------------

CLICKHOUSE_HOST = os.environ["CH_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CH_PORT", 8443))
CLICKHOUSE_USER = os.environ["CH_USER"]
CLICKHOUSE_PASSWORD = os.environ["CH_PASSWORD"]
CLICKHOUSE_DB = os.environ.get("CH_DB", "training_observability")

DASHBOARD_DIR = Path("dashboard")

# -----------------------------------------------------------------------------
# APP SETUP
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

_thread_local = threading.local()


def ch_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB,
            secure=True,
            verify=False,
        )
    return _thread_local.client


# -----------------------------------------------------------------------------
# TABLE DISCOVERY
# Runs once at startup, cached in TABLE_REGISTRY
# Each entry describes what columns the table has so we know how to query it
#
# Roles assigned:
#   "runs"    - has run_id but no step/metric → treated as run metadata
#   "scalar"  - has run_id + step + metric + a float value → time-series charts
#   "array"   - has run_id + step + metric + Array(Float) values → histograms
#   "events"  - has run_id + step + event_type/message → event log
#   "generic" - has run_id + step but unknown structure → still surfaced
#   "ignored" - no run_id column, can't be linked to a run
# -----------------------------------------------------------------------------

TABLE_REGISTRY = {}


def discover_tables():
    result = ch_client().query(
        """
        SELECT table, name, type
        FROM system.columns
        WHERE database = %(db)s
        ORDER BY table, position
    """,
        {"db": CLICKHOUSE_DB},
    )

    # Group columns by table
    tables = defaultdict(dict)
    for row in result.result_rows:
        table, col_name, col_type = row
        tables[table][col_name] = col_type

    registry = {}
    for table, cols in tables.items():
        col_names = set(cols.keys())

        if "run_id" not in col_names:
            registry[table] = {"columns": cols, "role": "ignored"}
            continue

        has_step = "step" in col_names
        has_metric = "metric" in col_names

        array_cols = [
            c
            for c, t in cols.items()
            if t.startswith("Array(Float") or t.startswith("Array(Int")
        ]
        scalar_cols = [
            c
            for c, t in cols.items()
            if any(t.startswith(p) for p in ("Float", "Int", "UInt"))
            and c not in ("step", "rank", "device")
            and not t.startswith("Array")
        ]

        if not has_step and not has_metric:
            role = "runs"
        elif has_step and has_metric and array_cols:
            role = "array"
        elif has_step and has_metric and scalar_cols:
            role = "scalar"
        elif has_step and ("event_type" in col_names or "message" in col_names):
            role = "events"
        elif has_step:
            role = "generic"
        else:
            role = "ignored"

        registry[table] = {
            "columns": cols,
            "role": role,
            "array_cols": array_cols,
            "scalar_cols": scalar_cols,
        }

    return registry


def refresh_table_registry():
    global TABLE_REGISTRY
    TABLE_REGISTRY = discover_tables()
    print(f"📋 Discovered {len(TABLE_REGISTRY)} tables:")
    for t, info in TABLE_REGISTRY.items():
        print(f"   {t:30s} → {info['role']}")


refresh_table_registry()

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS
# -----------------------------------------------------------------------------


def get_runs():
    """
    Query the runs table (role=runs) for rich metadata.
    Falls back to scanning all metric tables for distinct run_ids.
    """
    runs_tables = [t for t, info in TABLE_REGISTRY.items() if info["role"] == "runs"]

    if runs_tables:
        table = runs_tables[0]
        cols = set(TABLE_REGISTRY[table]["columns"].keys())

        select_parts = ["run_id"]
        if "status" in cols:
            select_parts.append("status")
        if "model_name" in cols:
            select_parts.append("model_name")
        if "model_size" in cols:
            select_parts.append("model_size")
        if "source" in cols:
            select_parts.append("source")
        if "cluster" in cols:
            select_parts.append("cluster")
        if "created_time" in cols:
            select_parts.append(
                "formatDateTime(created_time, '%Y-%m-%d %H:%i') AS created_at"
            )

        order = "created_time DESC" if "created_time" in cols else "run_id DESC"
        query = f"""
            SELECT {', '.join(select_parts)}
            FROM {table}
            FINAL
            ORDER BY {order}
            LIMIT 200
        """
        result = ch_client().query(query)

        runs = []
        for row in result.result_rows:
            keys = [p.split(" AS ")[-1] for p in select_parts]
            run = dict(zip(keys, row))
            runs.append(
                {
                    "run_id": run.get("run_id", ""),
                    "status": run.get("status", "unknown"),
                    "model_name": run.get("model_name", ""),
                    "model_size": run.get("model_size", ""),
                    "source": run.get("source", ""),
                    "cluster": run.get("cluster", ""),
                    "created_at": run.get("created_at", ""),
                }
            )
        if runs:
            return runs

    # Fallback: collect run_ids from all data tables
    run_ids = set()
    for table, info in TABLE_REGISTRY.items():
        if info["role"] in ("scalar", "array", "events", "generic"):
            try:
                rows = (
                    ch_client()
                    .query(
                        f"""
                    SELECT DISTINCT run_id FROM {table}
                    WHERE run_id != '' LIMIT 200
                """
                    )
                    .result_rows
                )
                run_ids.update(r[0] for r in rows)
            except Exception:
                pass

    return [
        {
            "run_id": rid,
            "status": "unknown",
            "model_name": "",
            "model_size": "",
            "source": "",
            "cluster": "",
            "created_at": "",
        }
        for rid in sorted(run_ids, reverse=True)
    ]


def discover_metrics(run_id):
    """
    Scan all scalar/array tables for metrics belonging to this run.
    Groups them by the prefix before the first '/'.
    """
    grouped = defaultdict(list)

    for table, info in TABLE_REGISTRY.items():
        if info["role"] not in ("scalar", "array", "generic"):
            continue
        if "metric" not in info["columns"]:
            continue
        try:
            rows = (
                ch_client()
                .query(
                    f"""
                SELECT DISTINCT metric FROM {table}
                WHERE run_id = %(run_id)s
                ORDER BY metric
            """,
                    {"run_id": run_id},
                )
                .result_rows
            )
        except Exception:
            continue

        kind = "array" if info["role"] == "array" else "scalar"
        for (metric,) in rows:
            if metric.startswith("sys.") or metric.startswith("sys/"):
                category = "system"
            elif "/" in metric:
                category = metric.split("/", 1)[0]
            else:
                category = table
            grouped[category].append({"name": metric, "kind": kind, "table": table})

    return dict(grouped)


def get_metric_data(run_id, metric_names, max_points=1000):
    """
    Fetch data for the requested metrics, auto-routing to the correct table.
    Scalar → [{x: step, y: value}]
    Array  → [{step, keys, value: [...]}]
    """
    # Build metric → (table, role) map
    metric_table_map = {}
    for table, info in TABLE_REGISTRY.items():
        if info["role"] not in ("scalar", "array", "generic"):
            continue
        if "metric" not in info["columns"]:
            continue
        try:
            rows = (
                ch_client()
                .query(
                    f"""
                SELECT DISTINCT metric FROM {table}
                WHERE run_id = %(run_id)s
            """,
                    {"run_id": run_id},
                )
                .result_rows
            )
            for (m,) in rows:
                if m in metric_names:
                    metric_table_map[m] = (table, info["role"])
        except Exception:
            pass

    data = {}

    for metric in metric_names:
        if metric not in metric_table_map:
            continue
        table, role = metric_table_map[metric]

        if role == "array":
            try:
                table_cols = TABLE_REGISTRY.get(table, {}).get("columns", {})
                arr_cols = TABLE_REGISTRY.get(table, {}).get("array_cols", [])
                # Detect a keys column: any Array(String) column named 'keys'
                has_keys = "keys" in table_cols and table_cols["keys"].startswith(
                    "Array("
                )
                val_col = arr_cols[0] if arr_cols else "values"

                if has_keys:
                    rows = (
                        ch_client()
                        .query(
                            f"""
                        SELECT step, keys, {val_col} FROM {table}
                        WHERE run_id = %(run_id)s AND metric = %(metric)s
                        ORDER BY step
                    """,
                            {"run_id": run_id, "metric": metric},
                        )
                        .result_rows
                    )
                    if rows:
                        data[metric] = [
                            {"step": int(r[0]), "keys": list(r[1]), "value": list(r[2])}
                            for r in rows
                        ]
                else:
                    rows = (
                        ch_client()
                        .query(
                            f"""
                        SELECT step, {val_col} FROM {table}
                        WHERE run_id = %(run_id)s AND metric = %(metric)s
                        ORDER BY step
                    """,
                            {"run_id": run_id, "metric": metric},
                        )
                        .result_rows
                    )
                    if rows:
                        data[metric] = [
                            {"step": int(r[0]), "keys": [], "value": list(r[1])}
                            for r in rows
                        ]
            except Exception as e:
                print(f"⚠ Array query failed for {metric} in {table}: {e}")
        else:
            try:
                rows = (
                    ch_client()
                    .query(
                        f"""
                    SELECT step, avg(value) AS value
                    FROM {table}
                    WHERE run_id = %(run_id)s AND metric = %(metric)s
                    GROUP BY step ORDER BY step
                """,
                        {"run_id": run_id, "metric": metric},
                    )
                    .result_rows
                )
                if rows:
                    points = [{"x": int(r[0]), "y": float(r[1])} for r in rows]
                    if len(points) > max_points:
                        step_size = max(1, len(points) // max_points)
                        points = points[::step_size]
                    data[metric] = points
            except Exception:
                pass

    return data


def get_run_events(run_id, limit=200):
    """Pull from all event-role tables for this run."""
    events = []
    for table, info in TABLE_REGISTRY.items():
        if info["role"] != "events":
            continue
        cols = set(info["columns"].keys())
        try:
            select_parts = [
                (
                    "formatDateTime(event_time, '%Y-%m-%d %H:%i:%S') AS ts"
                    if "event_time" in cols
                    else "'' AS ts"
                ),
                "step" if "step" in cols else "0 AS step",
                "event_type" if "event_type" in cols else "'' AS event_type",
                "severity" if "severity" in cols else "'' AS severity",
                "message" if "message" in cols else "'' AS message",
                "host" if "host" in cols else "'' AS host",
                "rank" if "rank" in cols else "0 AS rank",
            ]
            rows = (
                ch_client()
                .query(
                    f"""
                SELECT {', '.join(select_parts)}
                FROM {table}
                WHERE run_id = %(run_id)s
                ORDER BY event_time DESC
                LIMIT %(limit)s
            """,
                    {"run_id": run_id, "limit": limit},
                )
                .result_rows
            )
            for r in rows:
                events.append(
                    {
                        "time": r[0],
                        "step": int(r[1]),
                        "event_type": r[2],
                        "severity": r[3],
                        "message": r[4],
                        "host": r[5],
                        "rank": int(r[6]),
                        "source_table": table,
                    }
                )
        except Exception:
            pass

    return sorted(events, key=lambda e: e["time"], reverse=True)


def get_current_metrics(run_id):
    result_map = {}
    for table, info in TABLE_REGISTRY.items():
        if info["role"] not in ("scalar", "generic"):
            continue
        if "metric" not in info["columns"]:
            continue
        try:
            rows = (
                ch_client()
                .query(
                    f"""
                SELECT metric, argMax(value, step)
                FROM {table}
                WHERE run_id = %(run_id)s
                GROUP BY metric
            """,
                    {"run_id": run_id},
                )
                .result_rows
            )
            for r in rows:
                result_map[r[0]] = r[1]
        except Exception:
            pass
    return result_map


# -----------------------------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------------------------


@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": get_runs()})


@app.route("/api/runs/<run_id>/metrics")
def api_discover_metrics(run_id):
    return jsonify({"metrics": discover_metrics(run_id)})


@app.route("/api/runs/<run_id>/data")
def api_metric_data(run_id):
    metrics_param = request.args.get("metrics", "")
    metric_names = [m.strip() for m in metrics_param.split(",") if m.strip()]
    max_points = request.args.get("max_points", 1000, type=int)
    if not metric_names:
        return jsonify({"error": "No metrics specified"}), 400
    return jsonify(get_metric_data(run_id, metric_names, max_points))


@app.route("/api/runs/<run_id>/current")
def api_current(run_id):
    return jsonify(get_current_metrics(run_id))


@app.route("/api/runs/<run_id>/events")
def api_events(run_id):
    limit = request.args.get("limit", 200, type=int)
    return jsonify({"events": get_run_events(run_id, limit)})


@app.route("/api/tables")
def api_tables():
    """See all discovered tables and their assigned roles."""
    return jsonify(
        {
            table: {"role": info["role"], "columns": list(info["columns"].keys())}
            for table, info in TABLE_REGISTRY.items()
            if info["role"] != "ignored"
        }
    )


@app.route("/api/tables/refresh")
def api_refresh_tables():
    """
    Call this after your team adds a new table — no server restart needed.
    Just hit: GET /api/tables/refresh
    """
    refresh_table_registry()
    return jsonify(
        {
            table: {"role": info["role"], "columns": list(info["columns"].keys())}
            for table, info in TABLE_REGISTRY.items()
            if info["role"] != "ignored"
        }
    )


@app.route("/api/health")
def api_health():
    try:
        ch_client().query("SELECT 1")
        return jsonify(
            {
                "status": "ok",
                "clickhouse": "connected",
                "tables_discovered": len(
                    [
                        t
                        for t in TABLE_REGISTRY
                        if TABLE_REGISTRY[t]["role"] != "ignored"
                    ]
                ),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "clickhouse": str(e)}), 500


# -----------------------------------------------------------------------------
# SERVE FRONTEND
# -----------------------------------------------------------------------------


@app.route("/")
def serve_dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(DASHBOARD_DIR, path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n🚀 ClickHouse Dashboard Server Running")
    print(f"   Host:     {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")
    print(f"   Database: {CLICKHOUSE_DB}")
    print(
        f"   Tables:   {len([t for t in TABLE_REGISTRY if TABLE_REGISTRY[t]['role'] != 'ignored'])} active\n"
    )
    app.run(host="0.0.0.0", port=5050, debug=False)
