import { useEffect, useMemo, useState } from 'react'
import { useMetricsStore } from '../stores/metricsStore'
import type { CheckpointRecord, MetricArrayLatest, Series, TrainingEvent } from '../types/metrics'

/**
 * Get the latest value for a metric across all selected runs.
 * Returns { value, step, timestamp, prev } or null if no data.
 */
export function useLatestMetric(metricName: string) {
  const runs = useMetricsStore(s => s.runs)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)

  return useMemo(() => {
    let best: { value: number; step: number; timestamp: number; prev: number | null; runId: string } | null = null
    for (const runId of selectedRuns) {
      const series = runs[runId]?.[metricName]
      if (!series || series.steps.length === 0) continue
      const len = series.steps.length
      const step = series.steps[len - 1]
      const value = series.values[len - 1]
      const timestamp = series.timestamps[len - 1]
      const prev = len >= 2 ? series.values[len - 2] : null
      if (!best || step > best.step) {
        best = { value, step, timestamp, prev, runId }
      }
    }
    return best
  }, [runs, selectedRuns, metricName])
}

/**
 * Get the latest value for a metric matching a prefix across selected runs.
 * Returns all matching metrics with their latest values.
 */
export function useMetricsByPrefix(prefix: string) {
  const runs = useMetricsStore(s => s.runs)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)

  return useMemo(() => {
    const results: Record<string, { value: number; step: number; timestamp: number; runId: string }> = {}
    for (const runId of selectedRuns) {
      const runData = runs[runId]
      if (!runData) continue
      for (const [metricName, series] of Object.entries(runData)) {
        if (!metricName.startsWith(prefix)) continue
        if (series.steps.length === 0) continue
        const len = series.steps.length
        const existing = results[metricName]
        if (!existing || series.steps[len - 1] > existing.step) {
          results[metricName] = {
            value: series.values[len - 1],
            step: series.steps[len - 1],
            timestamp: series.timestamps[len - 1],
            runId,
          }
        }
      }
    }
    return results
  }, [runs, selectedRuns, prefix])
}

/**
 * Get the full time-series data for a metric across selected runs.
 * Returns an array of { runId, series } for all selected runs that have this metric.
 */
export function useMetricSeries(metricName: string): Array<{ runId: string; series: Series }> {
  const runs = useMetricsStore(s => s.runs)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)

  return useMemo(() => {
    const result: Array<{ runId: string; series: Series }> = []
    for (const runId of selectedRuns) {
      const series = runs[runId]?.[metricName]
      if (series && series.steps.length > 0) {
        result.push({ runId, series })
      }
    }
    return result
  }, [runs, selectedRuns, metricName])
}

/**
 * Get checkpoint data from the store (metrics starting with checkpoint_)
 */
export function useCheckpointMetrics() {
  const runs = useMetricsStore(s => s.runs)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)

  return useMemo(() => {
    const checkpoints: Array<{ runId: string; metric: string; step: number; value: number; timestamp: number }> = []
    for (const runId of selectedRuns) {
      const runData = runs[runId]
      if (!runData) continue
      for (const [metricName, series] of Object.entries(runData)) {
        if (!metricName.startsWith('checkpoint_')) continue
        for (let i = 0; i < series.steps.length; i++) {
          checkpoints.push({
            runId,
            metric: metricName,
            step: series.steps[i],
            value: series.values[i],
            timestamp: series.timestamps[i],
          })
        }
      }
    }
    return checkpoints.sort((a, b) => b.step - a.step)
  }, [runs, selectedRuns])
}

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * Fetch checkpoint records for each selected run from the checkpoints table.
 * Polls every 30 s.
 */
export function useCheckpoints(runIds: string[]): CheckpointRecord[] {
  const [records, setRecords] = useState<CheckpointRecord[]>([])
  const key = runIds.join(',')

  useEffect(() => {
    if (runIds.length === 0) { setRecords([]); return }
    const fetch_ = async () => {
      const results = await Promise.allSettled(
        runIds.map(id => fetch(`${API_BASE}/checkpoints/${id}`).then(r => r.json()))
      )
      const all: CheckpointRecord[] = []
      for (const r of results) {
        if (r.status === 'fulfilled') all.push(...(r.value.checkpoints ?? []))
      }
      all.sort((a, b) => b.step - a.step)
      setRecords(all)
    }
    fetch_()
    const id = setInterval(fetch_, 30_000)
    return () => clearInterval(id)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return records
}

/**
 * Fetch training events for each selected run from the events table.
 * Polls every 10 s.
 */
export function useEvents(runIds: string[], limit = 50): TrainingEvent[] {
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const key = runIds.join(',')

  useEffect(() => {
    if (runIds.length === 0) { setEvents([]); return }
    const fetch_ = async () => {
      const results = await Promise.allSettled(
        runIds.map(id =>
          fetch(`${API_BASE}/events/${id}?limit=${limit}`).then(r => r.json())
        )
      )
      const all: TrainingEvent[] = []
      for (const r of results) {
        if (r.status === 'fulfilled') all.push(...(r.value.events ?? []))
      }
      all.sort((a, b) => b.step - a.step)
      setEvents(all.slice(0, limit))
    }
    fetch_()
    const id = setInterval(fetch_, 10_000)
    return () => clearInterval(id)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, limit])

  return events
}

/**
 * Fetch the latest metric_array record for a single run + metric.
 * Polls every 5 s.
 */
export function useMetricArray(runId: string, metric: string): MetricArrayLatest | null {
  const [latest, setLatest] = useState<MetricArrayLatest | null>(null)

  useEffect(() => {
    if (!runId || !metric) { setLatest(null); return }
    const fetch_ = async () => {
      try {
        const data = await fetch(
          `${API_BASE}/metric_arrays/${runId}?metric=${encodeURIComponent(metric)}`
        ).then(r => r.json())
        setLatest(data.latest ?? null)
      } catch {
        // keep previous value on transient error
      }
    }
    fetch_()
    const id = setInterval(fetch_, 5_000)
    return () => clearInterval(id)
  }, [runId, metric])

  return latest
}

/** Format a number for display */
export function formatValue(v: number, decimals = 3): string {
  if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(2) + 'T'
  if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(2) + 'M'
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K'
  return v.toFixed(decimals)
}

/** Format a step number */
export function formatStep(step: number): string {
  if (step >= 1000) return (step / 1000).toFixed(step % 1000 === 0 ? 0 : 1) + 'K'
  return step.toString()
}

/** Calculate delta between current and previous value */
export function getDelta(current: number, prev: number | null | undefined): { text: string; direction: 'dn' | 'up' | 'nu' } {
  if (prev == null) return { text: '—', direction: 'nu' }
  const diff = current - prev
  if (Math.abs(diff) < 1e-6) return { text: '≈ stable', direction: 'nu' }
  const sign = diff < 0 ? '▼' : '▲'
  const dir = diff < 0 ? 'dn' : 'up'
  return { text: `${sign} ${Math.abs(diff).toFixed(3)}`, direction: dir }
}
