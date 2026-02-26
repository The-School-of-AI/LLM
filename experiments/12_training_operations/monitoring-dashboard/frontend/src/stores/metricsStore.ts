import { create } from 'zustand';
import { lttbDecimate } from '../lib/decimation';
import type {
  AllRunMetrics,
  ConnectionStatus,
  RawPoint,
  RunInfo,
  Series,
  SSEDelta,
  SSESnapshot,
  TabKey,
  TimePreset,
  TimeRange,
} from '../types/metrics';

/** Maximum points kept per metric series in the store. */
const MAX_SERIES_POINTS = 4000;

function rawPointsToSeries(points: RawPoint[]): Series {
  return {
    steps: points.map((p) => p[0]),
    values: points.map((p) => p[1]),
    timestamps: points.map((p) => p[2]),
  };
}

function appendSeries(existing: Series, points: RawPoint[]): Series {
  const steps = [...existing.steps, ...points.map((p) => p[0])];
  const values = [...existing.values, ...points.map((p) => p[1])];
  const timestamps = [...existing.timestamps, ...points.map((p) => p[2])];

  // Cap growth with LTTB downsampling so arrays don't grow unbounded
  if (steps.length > MAX_SERIES_POINTS) {
    const { x: dsTimestamps, y: dsValues } = lttbDecimate(timestamps, values, MAX_SERIES_POINTS);
    // Rebuild steps by finding the original step for each kept timestamp
    const tsToStep = new Map<number, number>();
    for (let i = 0; i < timestamps.length; i++) {
      tsToStep.set(timestamps[i], steps[i]);
    }
    const dsSteps = dsTimestamps.map((t) => tsToStep.get(t) ?? 0);
    return { steps: dsSteps, values: dsValues, timestamps: dsTimestamps };
  }

  return { steps, values, timestamps };
}

interface MetricsState {
  version: number;
  runs: AllRunMetrics;
  allRuns: RunInfo[];
  selectedRuns: string[];
  timeRange: TimeRange | null;
  timePreset: TimePreset;
  selectedTab: TabKey;
  connectionStatus: ConnectionStatus;
  setTab: (tab: TabKey) => void;
  setTimePreset: (preset: TimePreset) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  applySnapshot: (data: SSESnapshot) => void;
  applyDelta: (data: SSEDelta) => void;
  applyRunsMeta: (runs: RunInfo[]) => void;
  toggleRun: (runId: string) => void;
  selectAllRuns: () => void;
  deselectAllRuns: () => void;
  setTimeRange: (range: TimeRange | null) => void;
}

export const useMetricsStore = create<MetricsState>((set) => ({
  version: 0,
  runs: {},
  allRuns: [],
  selectedRuns: [],
  timeRange: null,
  timePreset: 'all',
  selectedTab: 'overview',
  connectionStatus: 'connecting',

  setTab: (tab) => set({ selectedTab: tab }),
  setTimePreset: (preset) => {
    const now = Date.now() / 1000;
    const durations: Record<string, number | null> = {
      '1d': 86400,
      '3d': 86400 * 3,
      '1w': 86400 * 7,
      'all': null,
    };
    const dur = durations[preset];
    set({
      timePreset: preset,
      timeRange: dur ? { from: now - dur, to: now } : null,
    });
  },
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  applySnapshot: (data) =>
    set((state) => {
      const runs: AllRunMetrics = {};
      for (const [runId, metrics] of Object.entries(data.runs)) {
        runs[runId] = {};
        for (const [metric, points] of Object.entries(metrics)) {
          runs[runId][metric] = rawPointsToSeries(points);
        }
      }
      const allRunIds = Object.keys(data.runs);
      // Initialize runs_meta from snapshot if present
      const allRuns = data.runs_meta ?? state.allRuns;
      // Auto-select all runs on initial snapshot
      return {
        version: data.version,
        runs,
        allRuns,
        selectedRuns: allRuns.length > 0
          ? allRuns.map((r) => r.run_id)
          : allRunIds,
      };
    }),

  applyDelta: (data) =>
    set((state) => {
      const runs = { ...state.runs };
      const selectedSet = new Set(state.selectedRuns);

      for (const [runId, metrics] of Object.entries(data.runs)) {
        // Auto-select newly appeared runs
        if (!runs[runId]) {
          selectedSet.add(runId);
        }
        if (!runs[runId]) {
          runs[runId] = {};
        } else {
          runs[runId] = { ...runs[runId] };
        }
        for (const [metric, points] of Object.entries(metrics)) {
          if (runs[runId][metric]) {
            runs[runId][metric] = appendSeries(runs[runId][metric], points);
          } else {
            runs[runId][metric] = rawPointsToSeries(points);
          }
        }
      }
      return {
        version: data.version,
        runs,
        selectedRuns: Array.from(selectedSet),
      };
    }),

  applyRunsMeta: (runsMeta) =>
    set((state) => {
      const prevIds = new Set(state.allRuns.map((r) => r.run_id));
      const selectedSet = new Set(state.selectedRuns);
      // Auto-select any brand-new runs
      for (const r of runsMeta) {
        if (!prevIds.has(r.run_id)) {
          selectedSet.add(r.run_id);
        }
      }
      return {
        allRuns: runsMeta,
        selectedRuns: Array.from(selectedSet),
      };
    }),

  toggleRun: (runId) =>
    set((state) => {
      const selected = new Set(state.selectedRuns);
      if (selected.has(runId)) {
        selected.delete(runId);
      } else {
        selected.add(runId);
      }
      return { selectedRuns: Array.from(selected) };
    }),

  selectAllRuns: () =>
    set((state) => ({
      selectedRuns: state.allRuns.map((r) => r.run_id),
    })),

  deselectAllRuns: () => set({ selectedRuns: [] }),

  setTimeRange: (range) => set({ timeRange: range }),
}));
