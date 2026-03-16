CREATE TABLE IF NOT EXISTS training_observability.logs
(
  `event_time` DateTime64(3) DEFAULT now64(3),
  `timestamp` String,
  `step` UInt64,
  `metrics` String,
  `context` String,
  `host` LowCardinality(String),
  `run_id` LowCardinality(String) DEFAULT '',
  `rank` UInt32 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (run_id, host, rank, step, event_time);
