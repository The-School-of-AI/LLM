Assume there is clickhouse database with table:

CREATE TABLE IF NOT EXISTS training_observability.metric_points
(
`event_time` DateTime64(3) DEFAULT now64(3),
`run_id` LowCardinality(String),
`host` LowCardinality(String) DEFAULT '',
`rank` UInt32 DEFAULT 0,
`device` UInt16 DEFAULT 65535,
`step` UInt64 DEFAULT 0,
`metric` LowCardinality(String),
`value` Float64,
`unit` LowCardinality(String) DEFAULT '',
`tags_json` String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (run_id, metric, host, rank, device, step, event_time);

- column "metric" has different string values e.g. loss, sys.cpu_percent, sys.mem_percent, sys.gpu.<idx>.util_percent, sys.disk.<mount>.percent, sys.net.<iface>.sent_bytes_per_s etc

Here’s my incomplete list to publish on Dashboard:

- loss t + 1
- loss t + 2
- NULL Router loss
- MoE Router loss
- validation loss
- token/sec
- batch/sec
- NULL ratio
- GSA stats
- recurrence performance
- mHC stats
- MoE favourite tokens graph
- MoE Fourier subject/Bucket
- current Bucket
- total tokens processed
- time to B1 stage/B2 stage
- time to 3B/8B/70B/SFT
- GPU utilization
- Data transfer speed
- GPU idle time
- CPU idle time
- checkpoint list
- checkpoint stats
- checkpoint benchmarks
- checkpoint hosted list
- generated samples

Backend (FastAPI + ClickHouse + SSE)
┌───────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│ File │ Purpose │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ config.py │ Pydantic Settings: CH_HOST, CH_PORT, CH_DATABASE, CH_USER, CH_PASSWORD, POLL_INTERVAL_SEC │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ db.py │ ClickHouse client singleton via clickhouse-connect │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ cache.py │ Versioned in-memory cache with LTTB decimation (2000-point cap), snapshot/delta methods │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ poller.py │ Async background task polling CH every 3s for active runs + new metrics │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ sse.py │ Broadcaster with per-client asyncio.Queue, fan-out, slow-client eviction │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ app.py │ FastAPI app with lifespan (starts poller, cleans up), CORS middleware │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ routes/stream.py │ GET /stream — SSE endpoint (snapshot on connect, then deltas) │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ routes/runs.py │ GET /runs — active run list from cache │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ routes/metrics.py │ GET /metrics/{run_id} — historical REST query with step range filters │
├───────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ routes/checkpoints.py │ GET /checkpoints/{run_id} — checkpoint data from CH │
└───────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
Frontend (React + TypeScript + Vite + Tailwind + Zustand + uPlot)
┌───────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│ File │ Purpose │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ types/metrics.ts │ TypeScript types, tab-to-metric mapping, metricMatchesTab() helper │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ stores/metricsStore.ts │ Zustand store: applySnapshot/applyDelta, per-run per-metric series │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ hooks/useSSE.ts │ SSE connection with auto-reconnect (exponential backoff), parses snapshot/delta events │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ lib/decimation.ts │ Client-side LTTB downsampling utility │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/charts/TimeSeriesChart.tsx │ Generic uPlot wrapper with multi-run overlay, ResizeObserver │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/charts/MetricCard.tsx │ Card: title + latest value + chart, auto-formatted numbers │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/layout/Header.tsx │ Title bar with connection status indicator (green/yellow/red) │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/layout/Sidebar.tsx │ Tab navigation: Training, Model, System, Checkpoints, Progress │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/layout/RunBadges.tsx │ Color-coded active run pills │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/tabs/TrainingTab.tsx │ Loss curves, throughput, tokens grid │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/tabs/ModelTab.tsx │ MoE, GSA, mHC, null ratio charts │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/tabs/SystemTab.tsx │ GPU/CPU util, idle time, network charts │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/tabs/CheckpointsTab.tsx │ Checkpoint data table │
├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ components/tabs/ProgressTab.tsx │ Milestone timeline/progress bars │
└───────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘
How to run

Backend: cd backend && uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000

Frontend: cd frontend && npm run dev

The frontend dev server proxies /stream, /runs, /metrics, /checkpoints to the backend at :8000.

TEST

```curl -sk --cacert ~/ca.crt \
 "https://18.118.133.106:8443/?user=p12_reader&password=EndGameDashboard" \
 -d "SELECT metric, count(), avg(value) FROM training_observability.metric_points GROUP BY metric"
```

## Revision -1 
 Structural Changes:
  - Replaced the tabbed layout (Header + Sidebar + Tabs) with a single-page scrollable dashboard matching the target design
  - Replaced Tailwind CSS with custom CSS variables and styles from the target HTML
  - Removed the Tailwind Vite plugin from vite.config.ts
  - Added JetBrains Mono + DM Sans fonts in index.html

  New Components (all data-driven from the Zustand store via SSE):
  - Topbar — Shows connection status from the store (Live/Connecting/Disconnected)
  - LossMetrics — Reads loss_t1, loss_t2, null_router_loss, moe_router_loss, validation_loss from the store; charts use MetricChart which pulls real series data
  - ThroughputSection — Reads token_per_sec, batch_per_sec, null_ratio, total_tokens
  - ArchitectureStats — Reads gsa_*, recurrence_*, mhc_* metrics
  - MoEAnalytics — Reads moe_favourite_tokens*, moe_fourier_*, current_bucket* metrics
  - TimelineSection — Reads time_to_b1/b2/3b/8b/70b/sft progress metrics + builds event log from allRuns metadata
  - InfrastructureSection — Reads sys.gpu.* (per-GPU utilization/temp/memory), gpu_idle_time, cpu_idle_time, sys.cpu_percent, sys.net.*
  - CheckpointsSection — Reads all checkpoint_* metrics including benchmarks
  - GeneratedSamples — Reads checkpoint_quality_* and checkpoint_sample_* metrics

  Data Layer (preserved):
  - useSSE hook — Still connects to /stream for real-time SSE data
  - metricsStore — Zustand store unchanged, still handles snapshot/delta/runs_meta
  - useMetricData hook (new) — Provides useLatestMetric(), useMetricsByPrefix(), useMetricSeries(), useCheckpointMetrics() for extracting values from the store
  - MetricChart component — Reads directly from the Zustand store to build SVG charts from real time-series data

  When no data is available for a metric, components show "—" or "Awaiting data" placeholders instead of static values.

  ## Revision -2
  
    ┌───────────────────────┬─────────────────────────────────┬────────────────────────────┐                                                                                         
  │       Component       │             Old Key             │    New Key (ClickHouse)    │                                                                                         
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤                                                                                         
  │ LossMetrics           │ loss_t1                         │ loss/train_t_plus_1        │                                                                                         
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ LossMetrics           │ loss_t2                         │ loss/train_t_plus_2        │                                                                                         
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤                                                                                         
  │ LossMetrics           │ null_router_loss                │ loss/router_null           │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ LossMetrics           │ moe_router_loss                 │ loss/router_moe            │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ LossMetrics           │ validation_loss                 │ loss                       │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ LossMetrics           │ (new)                           │ loss/train                 │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ token_per_sec                   │ throughput/tokens_per_sec  │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ batch_per_sec                   │ throughput/batches_per_sec │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ total_tokens                    │ tokens/processed_total     │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ null_ratio                      │ router/null_ratio          │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ (new)                           │ lr                         │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ ThroughputSection     │ (new)                           │ step_time_ms               │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ InfrastructureSection │ cpu_idle_time / sys.cpu_percent │ cpu/idle_percent           │
  ├───────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ InfrastructureSection │ (new)                           │ step_time_ms (chart)       │
  └───────────────────────┴─────────────────────────────────┴────────────────────────────┘

  New charts added:
  - Learning Rate Schedule — lr time series in Throughput
  - Step Time — step_time_ms time series in both Throughput and Infrastructure
  - CPU Idle % Over Time — cpu/idle_percent time series in Infrastructure
  - Training Loss — loss/train added to the loss curves chart