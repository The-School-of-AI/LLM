# Training Metrics Dashboard

Real-time dashboard for monitoring ML training runs. Streams metrics from ClickHouse via SSE, renders interactive charts with multi-run comparison, run filtering, and timeline controls.

## Features

- **Real-time streaming** -- SSE connection delivers metric updates every 3 seconds with automatic reconnection and exponential backoff
- **Multi-run comparison** -- all runs (active and inactive) displayed simultaneously with color-coded series overlays
- **Run filtering** -- clickable badges to toggle individual runs on/off; bulk All/None controls; green dot = active, gray = inactive
- **Timeline picker** -- scope charts to 1h, 6h, 24h, 7d, or All (live mode) time windows
- **5 metric tabs** -- Training (loss, throughput), Model (MoE, GSA, null ratio), System (GPU/CPU/network), Checkpoints (table), Progress (milestone bars)
- **Efficient data handling** -- server-side LTTB decimation caps each series at 2000 points; client receives only deltas after initial snapshot
- **Connection status** -- header shows live/connecting/disconnected state with colored indicator

## Architecture

```
ClickHouse (metric_points)
        |
        v
  [poller.py]  ----3s----> poll active runs, merge into cache
  [poller.py]  ---30s----> scan all runs metadata
        |
        v
  [cache.py]   in-memory store with LTTB decimation
        |
        v
  [sse.py]     fan-out broadcaster with per-client queues
        |
        v
  [stream.py]  GET /stream  ->  snapshot + delta + runs_meta events
        |
        v
  Browser (EventSource)
        |
        v
  [metricsStore.ts]  Zustand store  ->  React components  ->  uPlot charts
```

Data flows in one direction: ClickHouse -> poller -> cache -> SSE broadcaster -> browser. The poller runs two loops:

1. **Metric poll (3s)** -- queries active runs (data in last 5 min) for new points since last poll. On first poll, fetches all runs to backfill historical data.
2. **Runs metadata poll (30s)** -- queries `GROUP BY run_id` aggregates (start time, last event, max step, active status) and broadcasts a `runs_meta` SSE event.

The SSE stream sends three event types:
- `snapshot` -- full cache state on client connect (metric data + runs metadata)
- `delta` -- incremental metric points from each poll cycle
- `runs_meta` -- periodic refresh of run status information

## Project Structure

```
backend/
  app.py              FastAPI app, lifespan (starts pollers), CORS
  config.py           Pydantic settings (ClickHouse connection, poll interval)
  db.py               ClickHouse client singleton
  cache.py            Versioned in-memory cache, LTTB decimation, RunInfo metadata
  poller.py           Dual-interval async polling (metrics 3s, metadata 30s)
  sse.py              Typed-event broadcaster with per-client async queues
  routes/
    stream.py         GET /stream -- SSE endpoint
    runs.py           GET /runs -- all runs with metadata
    metrics.py        GET /metrics/{run_id} -- historical query with step/time filters
    checkpoints.py    GET /checkpoints/{run_id} -- checkpoint data

frontend/src/
  types/metrics.ts    TypeScript types, tab-metric mapping, RunInfo, TimeRange
  stores/metricsStore.ts  Zustand store (runs, selectedRuns, timeRange, actions)
  hooks/useSSE.ts     EventSource with auto-reconnect, handles snapshot/delta/runs_meta
  components/
    layout/
      Header.tsx      Title, timeline picker (1h/6h/24h/7d/All), connection status
      Sidebar.tsx     Tab navigation
      RunBadges.tsx   Interactive run filter badges with status dots
    charts/
      TimeSeriesChart.tsx  uPlot wrapper with multi-series, resize handling
      MetricCard.tsx       Metric card with title, latest value, chart, time filtering
    tabs/
      TrainingTab.tsx      Loss, throughput, token metrics
      ModelTab.tsx         MoE, GSA, recurrence, null ratio metrics
      SystemTab.tsx        GPU/CPU utilization, idle time, network
      CheckpointsTab.tsx   Checkpoint data table
      ProgressTab.tsx      Milestone progress bars
      useTabMetrics.ts     Hook to find matching metrics for selected runs
```

## ClickHouse Schema

The dashboard reads from a single table:

```sql
CREATE TABLE IF NOT EXISTS training_observability.metric_points
(
    event_time DateTime64(3) DEFAULT now64(3),
    run_id     LowCardinality(String),
    host       LowCardinality(String) DEFAULT '',
    rank       UInt32 DEFAULT 0,
    device     UInt16 DEFAULT 65535,
    step       UInt64 DEFAULT 0,
    metric     LowCardinality(String),
    value      Float64,
    unit       LowCardinality(String) DEFAULT '',
    tags_json  String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (run_id, metric, host, rank, device, step, event_time);
```

The `metric` column contains values like `loss_t1`, `sys.gpu.0.util_percent`, `checkpoint_save`, etc. Metrics are mapped to tabs by prefix matching (see `TAB_METRICS` in `types/metrics.ts`).

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /stream` | SSE stream: `snapshot`, `delta`, `runs_meta` events |
| `GET /runs` | All runs with metadata (`run_id`, `start_time`, `last_event_time`, `latest_step`, `is_active`) |
| `GET /metrics/{run_id}?metric=...&from_step=0&to_step=1000000&from_time=...&to_time=...` | Historical metric query with optional step and time range filters |
| `GET /checkpoints/{run_id}` | Checkpoint metrics for a run |
| `GET /health` | Health check |

## How to Run

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A running ClickHouse instance with the `training_observability.metric_points` table

### Backend

```bash
cd backend
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies `/stream`, `/runs`, `/metrics`, `/checkpoints`, and `/health` to the backend at `localhost:8000`.

### Configuration

ClickHouse connection is configured via environment variables or defaults in `backend/config.py`:

| Variable | Default | Description |
|---|---|---|
| `CH_HOST` | `13.221.29.26` | ClickHouse host |
| `CH_PORT` | `8123` | ClickHouse HTTP port |
| `CH_PROTOCOL` | `http` | Connection protocol |
| `CH_DATABASE` | `training_observability` | Database name |
| `CH_USER` | `default` | Username |
| `CH_PASSWORD` | *(empty)* | Password |
| `POLL_INTERVAL_SEC` | `3.0` | Metric polling interval in seconds |

### Production Build

```bash
cd frontend
npm run build     # outputs to frontend/dist/
```

Serve `frontend/dist/` with any static file server and point API routes to the backend.

## Tech Stack

**Backend:** FastAPI, clickhouse-connect, sse-starlette, Pydantic Settings, uvicorn

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS 4, Zustand, uPlot
