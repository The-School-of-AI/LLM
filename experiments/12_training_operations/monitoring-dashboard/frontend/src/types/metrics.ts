/** A single data point: [step, value, timestamp] */
export type RawPoint = [number, number, number];

/** Run metadata from the backend */
export interface RunInfo {
  run_id: string;
  start_time: number;
  last_event_time: number;
  latest_step: number;
  is_active: boolean;
  status?: string;
  model_name?: string;
  model_size?: string;
  source?: string;
  cluster?: string;
}

export interface CheckpointRecord {
  step: number;
  s3_key: string;
  loss: number | null;
  tag: string;
  is_protected: boolean;
  status: string;
  duration_s: number | null;
  size_bytes: number | null;
  host: string;
  timestamp: number;
}

export interface TrainingEvent {
  step: number;
  event_type: string;
  severity: string;
  message: string;
  host: string;
  rank: number;
  timestamp: number;
}

export interface MetricArrayLatest {
  step: number;
  keys: string[];
  values: number[];
}

/** Time range filter */
export interface TimeRange {
  from: number;
  to: number;
}

export type TimePreset = '1d' | '3d' | '1w' | 'all';

/** Wire format for SSE payloads */
export interface SSESnapshot {
  version: number;
  runs: Record<string, Record<string, RawPoint[]>>;
  runs_meta?: RunInfo[];
}

export interface SSEDelta {
  version: number;
  runs: Record<string, Record<string, RawPoint[]>>;
}

export interface SSERunsMeta {
  runs_meta: RunInfo[];
}

/** Internal typed-array series for efficient charting */
export interface Series {
  steps: number[];
  values: number[];
  timestamps: number[];
}

export type RunMetrics = Record<string, Series>;
export type AllRunMetrics = Record<string, RunMetrics>;

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export const TAB_KEYS = ['overview', 'architecture', 'milestones', 'infrastructure'] as const;
export type TabKey = (typeof TAB_KEYS)[number];

export const TAB_LABELS: Record<TabKey, string> = {
  overview: 'Overview',
  architecture: 'Architecture Stats',
  milestones: 'Milestones',
  infrastructure: 'Infrastructure & Hardware',
};
