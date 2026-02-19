/** A single data point: [step, value, timestamp] */
export type RawPoint = [number, number, number];

/** Run metadata from the backend */
export interface RunInfo {
  run_id: string;
  start_time: number;
  last_event_time: number;
  latest_step: number;
  is_active: boolean;
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
