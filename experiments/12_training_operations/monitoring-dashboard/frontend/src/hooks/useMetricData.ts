import { useMemo } from 'react'
import { useMetricsStore } from '../stores/metricsStore'
import type { Series } from '../types/metrics'

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
