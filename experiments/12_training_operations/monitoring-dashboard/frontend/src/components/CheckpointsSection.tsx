import { useMetricsByPrefix } from '../hooks/useMetricData'
import { useCheckpoints } from '../hooks/useMetricData'
import { useMetricsStore } from '../stores/metricsStore'
import { InfoBadge } from './InfoBadge'
import { useMemo } from 'react'

export function CheckpointsSection() {
  const selectedRuns = useMetricsStore(s => s.selectedRuns)
  const checkpoints = useCheckpoints(selectedRuns)

  const latest = checkpoints[0] ?? null

  const benchmarkMetrics = useMetricsByPrefix('checkpoint_benchmark_')

  const benchmarks = useMemo(() => {
    const names = ['mmlu', 'hellaswag', 'arc_c', 'winogrande', 'truthfulqa', 'gsm8k', 'humaneval']
    return names.map(name => {
      const data = benchmarkMetrics[`checkpoint_benchmark_${name}`]
      return {
        name: name.replace('_', '-').toUpperCase()
          .replace('ARC-C', 'ARC-C').replace('HELLASWAG', 'HellaSwag')
          .replace('WINOGRANDE', 'WinoGrande').replace('TRUTHFULQA', 'TruthfulQA')
          .replace('HUMANEVAL', 'HumanEval').replace('GSM8K', 'GSM8K').replace('MMLU', 'MMLU'),
        value: data?.value ?? null,
        step: data?.step ?? null,
      }
    }).filter(b => b.value !== null)
  }, [benchmarkMetrics])

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
              <InfoBadge text="All saved checkpoints. Shows step, S3 key, loss, tag, status, duration, and size." />
            </div>
          </div>
          <div className="pb np">
            <div className="tscr">
              <table className="dt">
                <thead>
                  <tr>
                    <th>Step</th><th>S3 Key</th><th>Loss</th><th>Tag</th>
                    <th>Status</th><th>Duration</th><th>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.length > 0 ? checkpoints.slice(0, 50).map((cp, i) => (
                    <tr key={i}>
                      <td>{cp.step.toLocaleString()}</td>
                      <td style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)' }}>
                        {cp.s3_key ? '…' + cp.s3_key.slice(-30) : '—'}
                      </td>
                      <td>{cp.loss != null ? cp.loss.toFixed(4) : '—'}</td>
                      <td>{cp.tag || '—'}</td>
                      <td style={{ color: cp.status === 'saved' ? 'var(--g)' : 'var(--o)' }}>{cp.status}</td>
                      <td>{cp.duration_s != null ? `${cp.duration_s.toFixed(1)}s` : '—'}</td>
                      <td>
                        {cp.size_bytes != null
                          ? cp.size_bytes >= 1e9
                            ? `${(cp.size_bytes / 1e9).toFixed(1)} GB`
                            : `${(cp.size_bytes / 1e6).toFixed(1)} MB`
                          : '—'}
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
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
              Checkpoint Stats{latest ? ` — step ${latest.step.toLocaleString()}` : ''}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">LATEST</span>
              <InfoBadge text="Key metrics from the most recent checkpoint: loss, save duration, and size." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                {
                  lbl: 'Loss',
                  val: latest?.loss != null ? latest.loss.toFixed(4) : '—',
                },
                {
                  lbl: 'Duration',
                  val: latest?.duration_s != null ? `${latest.duration_s.toFixed(1)}s` : '—',
                },
                {
                  lbl: 'Size',
                  val: latest?.size_bytes != null
                    ? latest.size_bytes >= 1e9
                      ? `${(latest.size_bytes / 1e9).toFixed(1)} GB`
                      : `${(latest.size_bytes / 1e6).toFixed(1)} MB`
                    : '—',
                },
                {
                  lbl: 'Status',
                  val: latest?.status ?? '—',
                },
                {
                  lbl: 'Tag',
                  val: latest?.tag || '—',
                },
                {
                  lbl: 'Protected',
                  val: latest != null ? (latest.is_protected ? 'Yes' : 'No') : '—',
                },
                {
                  lbl: 'Host',
                  val: latest?.host || '—',
                },
              ].map(s => (
                <div className="info-box" key={s.lbl}>
                  <div className="lbl">{s.lbl}</div>
                  <div className="val">{s.val}</div>
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
