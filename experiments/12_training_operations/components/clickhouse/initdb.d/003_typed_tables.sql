CREATE TABLE IF NOT EXISTS training_observability.runs
(
  `run_id` LowCardinality(String),
  `created_time` DateTime64(3) DEFAULT now64(3),
  `updated_time` DateTime64(3) DEFAULT now64(3),
  `status` LowCardinality(String) DEFAULT 'running',
  `model_name` LowCardinality(String) DEFAULT '',
  `model_size` LowCardinality(String) DEFAULT '',
  `source` LowCardinality(String) DEFAULT '',
  `cluster` LowCardinality(String) DEFAULT '',
  `commit` String DEFAULT '',
  `config_json` String DEFAULT '',
  `tags_json` String DEFAULT ''
)
ENGINE = ReplacingMergeTree(updated_time)
ORDER BY run_id;

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

CREATE TABLE IF NOT EXISTS training_observability.metric_arrays
(
  `event_time` DateTime64(3) DEFAULT now64(3),
  `run_id` LowCardinality(String),
  `host` LowCardinality(String) DEFAULT '',
  `rank` UInt32 DEFAULT 0,
  `device` UInt16 DEFAULT 65535,
  `step` UInt64 DEFAULT 0,
  `metric` LowCardinality(String),
  `keys` Array(String) DEFAULT [],
  `values` Array(Float32) DEFAULT [],
  `unit` LowCardinality(String) DEFAULT '',
  `tags_json` String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (run_id, metric, host, rank, device, step, event_time);

CREATE TABLE IF NOT EXISTS training_observability.events
(
  `event_time` DateTime64(3) DEFAULT now64(3),
  `run_id` LowCardinality(String),
  `host` LowCardinality(String) DEFAULT '',
  `rank` UInt32 DEFAULT 0,
  `device` UInt16 DEFAULT 65535,
  `step` UInt64 DEFAULT 0,
  `event_type` LowCardinality(String),
  `severity` LowCardinality(String) DEFAULT 'info',
  `message` String DEFAULT '',
  `payload_json` String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (run_id, event_type, host, rank, device, step, event_time);

CREATE TABLE IF NOT EXISTS training_observability.checkpoints
(
  `event_time`    DateTime64(3) DEFAULT now64(3),
  `run_id`        LowCardinality(String),
  `step`          UInt64,
  `s3_key`        String,
  `loss`          Float64 DEFAULT 0,
  `tag`           LowCardinality(String) DEFAULT 'temporary',
  `is_protected`  UInt8 DEFAULT 0,
  `status`        LowCardinality(String) DEFAULT 'registered',
  `host`          LowCardinality(String) DEFAULT '',
  `duration_s`    Float64 DEFAULT 0,
  `size_bytes`    UInt64 DEFAULT 0,
  `metadata_json` String DEFAULT ''
)
ENGINE = ReplacingMergeTree(event_time)
ORDER BY (run_id, s3_key);
