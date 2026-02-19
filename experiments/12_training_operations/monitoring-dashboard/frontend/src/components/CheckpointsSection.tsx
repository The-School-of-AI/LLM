import { useMemo } from 'react'
import { useCheckpointMetrics, useMetricsByPrefix, useLatestMetric } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

export function CheckpointsSection() {
  const checkpoints = useCheckpointMetrics()

  const ckptTable = useMemo(() => {
    const byStep: Record<number, Record<string, number>> = {}
    for (const cp of checkpoints) {
      if (!byStep[cp.step]) byStep[cp.step] = {}
      byStep[cp.step][cp.metric] = cp.value
    }
    return Object.entries(byStep)
      .map(([step, metrics]) => ({ step: parseInt(step), metrics }))
      .sort((a, b) => b.step - a.step)
  }, [checkpoints])

  const ckptValLoss = useLatestMetric('checkpoint_val_loss')
  const ckptGradNorm = useLatestMetric('checkpoint_grad_norm')
  const ckptLR = useLatestMetric('checkpoint_lr')
  const ckptPerplexity = useLatestMetric('checkpoint_perplexity')
  const ckptSaveDuration = useLatestMetric('checkpoint_save_duration')
  const ckptDiskUsage = useLatestMetric('checkpoint_disk_usage')
  const ckptTokensSeen = useLatestMetric('checkpoint_tokens_seen')

  const benchmarkMetrics = useMetricsByPrefix('checkpoint_benchmark_')

  const benchmarks = useMemo(() => {
    const names = ['mmlu', 'hellaswag', 'arc_c', 'winogrande', 'truthfulqa', 'gsm8k', 'humaneval']
    return names.map(name => {
      const data = benchmarkMetrics[`checkpoint_benchmark_${name}`]
      return {
        name: name.replace('_', '-').toUpperCase().replace('ARC-C', 'ARC-C').replace('HELLASWAG', 'HellaSwag').replace('WINOGRANDE', 'WinoGrande').replace('TRUTHFULQA', 'TruthfulQA').replace('HUMANEVAL', 'HumanEval').replace('GSM8K', 'GSM8K').replace('MMLU', 'MMLU'),
        value: data?.value ?? null,
        step: data?.step ?? null,
      }
    }).filter(b => b.value !== null)
  }, [benchmarkMetrics])

  const latestStep = ckptTable.length > 0 ? ckptTable[0].step : null

  return (
    <>
      <div className="sec-row">
        <h2>Checkpoints</h2>
        <div className="sec-line" /><span className="tag tc">CKPT</span>
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Checkpoint List</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tc">REGISTRY</span>
              <InfoBadge text="All saved checkpoints with their metrics. Shows run, metric name, step, value, and save time." />
            </div>
          </div>
          <div className="pb np">
            <div className="tscr">
              <table className="dt">
                <thead>
                  <tr><th>Run</th><th>Metric</th><th>Step</th><th>Value</th><th>Time</th></tr>
                </thead>
                <tbody>
                  {checkpoints.length > 0 ? checkpoints.slice(0, 50).map((cp, i) => (
                    <tr key={i}>
                      <td style={{ color: 'var(--g)' }}>{cp.runId.slice(0, 12)}</td>
                      <td>{cp.metric.replace('checkpoint_', '')}</td>
                      <td>{cp.step.toLocaleString()}</td>
                      <td>{cp.value.toFixed(4)}</td>
                      <td>{new Date(cp.timestamp * 1000).toLocaleTimeString()}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                        Awaiting checkpoint data
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">
              Checkpoint Stats{latestStep ? ` — step ${latestStep.toLocaleString()}` : ''}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">LATEST</span>
              <InfoBadge text="Key metrics from the most recent checkpoint: save time, disk usage, grad norm, LR, perplexity, and more." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { lbl: 'Save Duration', data: ckptSaveDuration, fmt: (v: number) => `${v.toFixed(1)}s` },
                { lbl: 'Disk Usage', data: ckptDiskUsage, fmt: (v: number) => v >= 1e9 ? `${(v / 1e9).toFixed(1)} GB` : `${(v / 1e6).toFixed(1)} MB` },
                { lbl: 'Grad Norm', data: ckptGradNorm, fmt: (v: number) => v.toFixed(2) },
                { lbl: 'Learning Rate', data: ckptLR, fmt: (v: number) => v.toExponential(1) },
                { lbl: 'Tokens Seen', data: ckptTokensSeen, fmt: (v: number) => v >= 1e12 ? `${(v / 1e12).toFixed(2)}T` : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : `${(v / 1e6).toFixed(1)}M` },
                { lbl: 'Perplexity', data: ckptPerplexity, fmt: (v: number) => v.toFixed(2) },
                { lbl: 'Val Loss', data: ckptValLoss, fmt: (v: number) => v.toFixed(3) },
              ].map(s => (
                <div className="info-box" key={s.lbl}>
                  <div className="lbl">{s.lbl}</div>
                  <div className="val">{s.data ? s.fmt(s.data.value) : '—'}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {benchmarks.length > 0 && (
        <div className="grid g2">
          <div className="pnl">
            <div className="ph">
              <span className="pt">Checkpoint Benchmarks</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="tag to">EVAL</span>
                <InfoBadge text="Benchmark scores (MMLU, HellaSwag, etc.) evaluated at the latest checkpoint step." />
              </div>
            </div>
            <div className="pb np">
              <div className="tscr">
                <table className="dt">
                  <thead>
                    <tr><th>Benchmark</th><th>Score</th><th>Step</th></tr>
                  </thead>
                  <tbody>
                    {benchmarks.map(b => (
                      <tr key={b.name}>
                        <td>{b.name}</td>
                        <td style={{ color: 'var(--g)', fontWeight: 600 }}>{b.value?.toFixed(1)}</td>
                        <td>{b.step?.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div />
        </div>
      )}
    </>
  )
}
