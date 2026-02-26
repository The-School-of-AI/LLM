import { useLatestMetric, useMetricsByPrefix } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

export function GeneratedSamples() {
  const coherence = useLatestMetric('checkpoint_quality_coherence')
  const factuality = useLatestMetric('checkpoint_quality_factuality')
  const codeCorrect = useLatestMetric('checkpoint_quality_code_correct')
  const creativity = useLatestMetric('checkpoint_quality_creativity')
  const instruction = useLatestMetric('checkpoint_quality_instruction')
  const fluency = useLatestMetric('checkpoint_quality_fluency')
  const repetition = useLatestMetric('checkpoint_quality_repetition')
  const toxicity = useLatestMetric('checkpoint_quality_toxicity')

  const qualityMetrics = [
    { label: 'Coherence', data: coherence, bg: 'var(--g)', color: 'var(--g)' },
    { label: 'Factuality', data: factuality, bg: 'var(--b)', color: 'var(--b)' },
    { label: 'Code Correct', data: codeCorrect, bg: 'var(--c)', color: 'var(--c)' },
    { label: 'Creativity', data: creativity, bg: 'var(--p)', color: 'var(--p)' },
    { label: 'Instruction', data: instruction, bg: 'var(--pk)', color: 'var(--pk)' },
    { label: 'Fluency', data: fluency, bg: 'var(--y)', color: 'var(--y)' },
  ]

  const hasQualityData = qualityMetrics.some(m => m.data !== null)

  const sampleMetrics = useMetricsByPrefix('checkpoint_sample_')
  const hasSamples = Object.keys(sampleMetrics).length > 0

  return (
    <>
      <div className="sec-row">
        <h2>Generated Samples</h2>
        <div className="sec-line" /><span className="tag tpk">QUALITATIVE</span>
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Sample Outputs</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">LATEST</span>
              <InfoBadge text="Generated text samples from the latest checkpoint, scored for quality and correctness." />
            </div>
          </div>
          <div className="pb" style={{ overflowY: 'auto', maxHeight: 320 }}>
            {hasSamples ? (
              Object.entries(sampleMetrics).map(([key, data]) => (
                <div className="sblk" key={key}>
                  <div className="slbl">{key.replace('checkpoint_sample_', '')}</div>
                  <div style={{ marginTop: 6 }}>Score: {data.value.toFixed(3)} at step {data.step.toLocaleString()}</div>
                </div>
              ))
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40, fontFamily: 'var(--mono)', fontSize: 11 }}>
                Awaiting generated sample data from evaluation runs...
              </div>
            )}
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Sample Quality Metrics</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tp">AUTO-EVAL</span>
              <InfoBadge text="Automated evaluation scores for coherence, factuality, code correctness, creativity, and more." />
            </div>
          </div>
          <div className="pb">
            {qualityMetrics.map(m => {
              const pct = m.data ? Math.min(m.data.value * 100, 100) : 0
              return (
                <div className="pr" key={m.label}>
                  <span className="pr-l" style={{ minWidth: 100 }}>{m.label}</span>
                  <div className="pr-t">
                    <div className="pr-f" style={{ width: m.data ? `${pct}%` : '0%', background: m.bg }} />
                  </div>
                  <span className="pr-p" style={{ color: m.data ? m.color : 'var(--text-muted)' }}>
                    {m.data ? `${pct.toFixed(0)}%` : '—'}
                  </span>
                </div>
              )
            })}

            {!hasQualityData && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20, fontFamily: 'var(--mono)', fontSize: 11 }}>
                Awaiting quality evaluation data...
              </div>
            )}

            {(repetition || toxicity) && (
              <div style={{
                marginTop: 16, padding: 10, background: 'rgba(255,255,255,0.02)',
                borderRadius: 'var(--rad)', border: '1px solid var(--border)'
              }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                  Safety Metrics
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-secondary)' }}>
                  {repetition && <>Repetition: <span style={{ color: 'var(--g)' }}>{(repetition.value * 100).toFixed(1)}%</span><br /></>}
                  {toxicity && <>Toxicity: <span style={{ color: toxicity.value < 0.05 ? 'var(--g)' : 'var(--r)' }}>{toxicity.value.toFixed(3)}</span> (target &lt;0.05)</>}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
