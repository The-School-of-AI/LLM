import { useMemo } from 'react'
import { MetricChart } from './PlotlyChart'
import { useMetricsByPrefix } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

export function MoEAnalytics() {
  const favouriteTokens = useMetricsByPrefix('moe_favourite_tokens')

  const expertTokenData = useMemo(() => {
    const defaultExperts = [
      { n: 'Expert 0', c: '#a78bfa', t: [] as string[] },
      { n: 'Expert 1', c: '#22d3ee', t: [] as string[] },
      { n: 'Expert 2', c: '#f472b6', t: [] as string[] },
      { n: 'Expert 3', c: '#3ecf8e', t: [] as string[] },
      { n: 'Expert 4', c: '#f5a524', t: [] as string[] },
      { n: 'Expert 5', c: '#4d9cf5', t: [] as string[] },
      { n: 'Expert 6', c: '#fbbf24', t: [] as string[] },
      { n: 'Expert 7', c: '#f55656', t: [] as string[] },
    ]

    for (const [key, data] of Object.entries(favouriteTokens)) {
      const match = key.match(/moe_favourite_tokens[._]expert[._](\d+)/)
      if (match) {
        const idx = parseInt(match[1])
        if (idx < defaultExperts.length) {
          defaultExperts[idx].t = [`val:${data.value.toFixed(2)}`]
        }
      }
    }

    return defaultExperts
  }, [favouriteTokens])

  const bucketMetrics = useMetricsByPrefix('current_bucket')
  const bucketData = useMemo(() => {
    const defaultBuckets = [
      { n: 'B0-general', p: 0, c: '#4d9cf5' },
      { n: 'B1-code', p: 0, c: '#a78bfa' },
      { n: 'B2-math', p: 0, c: '#22d3ee' },
      { n: 'B3-medical', p: 0, c: '#f472b6' },
      { n: 'B4-legal', p: 0, c: '#fbbf24' },
      { n: 'B5-finance', p: 0, c: '#3ecf8e' },
      { n: 'B6-instruct', p: 0, c: '#f5a524' },
      { n: 'B7-science', p: 0, c: '#f55656' },
    ]

    for (const [key, data] of Object.entries(bucketMetrics)) {
      for (let i = 0; i < defaultBuckets.length; i++) {
        const bName = defaultBuckets[i].n.toLowerCase().replace('-', '_')
        const bNum = `b${i}`
        if (key.includes(bNum) || key.includes(bName)) {
          defaultBuckets[i].p = data.value
        }
      }
    }

    if (bucketMetrics['current_bucket']) {
      const activeIdx = Math.round(bucketMetrics['current_bucket'].value)
      if (activeIdx >= 0 && activeIdx < defaultBuckets.length) {
        defaultBuckets[activeIdx].p = Math.max(defaultBuckets[activeIdx].p, 1)
      }
    }

    return defaultBuckets
  }, [bucketMetrics])

  const hasBucketData = bucketData.some(b => b.p > 0)

  return (
    <>
      <div className="sec-row">
        <h2>MoE Analytics</h2>
        <div className="sec-line" /><span className="tag tp">MIXTURE-OF-EXPERTS</span>
      </div>

      <div className="grid g3">
        <div className="pnl">
          <div className="ph">
            <span className="pt">MoE Favourite Tokens</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tp">GRAPH</span>
              <InfoBadge text="Top tokens routed to each expert. Reveals expert specialization and potential routing biases." />
            </div>
          </div>
          <div className="pb" style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>
            <div style={{ marginBottom: 10, fontSize: 10, color: 'var(--text-muted)' }}>
              Expert &rarr; Most Routed Tokens (top-5)
            </div>
            {expertTokenData.map(e => (
              <div key={e.n} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span style={{ minWidth: 65, color: e.c, fontSize: 10, fontWeight: 600 }}>{e.n}</span>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {e.t.length > 0 ? e.t.map(t => (
                    <span key={t} style={{
                      background: e.c + '18', color: e.c, padding: '2px 6px',
                      borderRadius: 3, fontSize: 10, border: `1px solid ${e.c}30`
                    }}>{t}</span>
                  )) : (
                    <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>awaiting data</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">MoE Fourier — Subject/Bucket</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tc">SPECTRAL</span>
              <InfoBadge text="Fourier decomposition of expert routing signals. Reveals periodic patterns in expert selection." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['moe_fourier_expert_0', 'moe_fourier_expert_1', 'moe_fourier_expert_2', 'moe_fourier_expert_3']}
              labels={['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3']}
              colors={['#a78bfa', '#22d3ee', '#f472b6', '#3ecf8e']}
              height={260}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--p)' }} />Expert 0</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--c)' }} />Expert 1</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--pk)' }} />Expert 2</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--g)' }} />Expert 3</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Current Bucket Distribution</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag to">LIVE</span>
              <InfoBadge text="Distribution of tokens across data buckets (domains). Shows which subjects are being trained on now." />
            </div>
          </div>
          <div className="pb">
            {bucketData.map(b => (
              <div className="bar-r" key={b.n}>
                <span className="bar-l">{b.n}</span>
                <div className="bar-t">
                  <div className="bar-f" style={{
                    width: hasBucketData ? `${Math.min(b.p * 3.5, 100)}%` : '0%',
                    background: b.c
                  }}>
                    {hasBucketData && b.p > 0 ? `${b.p}%` : ''}
                  </div>
                </div>
              </div>
            ))}
            {!hasBucketData && (
              <div style={{ textAlign: 'center', fontSize: 10, color: 'var(--text-muted)', padding: 8 }}>
                Awaiting bucket distribution data
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
