import { useLatestMetric } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

function MetricBox({ value, label, color, bgColor, fontSize }: {
  value: string; label: string; color: string; bgColor: string; fontSize?: number
}) {
  return (
    <div className="mbox" style={{ background: bgColor }}>
      <div className="mbox-v" style={{ color, fontSize: fontSize ?? 18 }}>{value}</div>
      <div className="mbox-l" style={fontSize ? { fontSize: 9 } : undefined}>{label}</div>
    </div>
  )
}

export function ArchitectureStats() {
  const gsaGateAvg = useLatestMetric('gsa_gate_avg')
  const gsaEntropy = useLatestMetric('gsa_gate_entropy')
  const gsaLayer0_7 = useLatestMetric('gsa_layer_0_7')
  const gsaLayer8_15 = useLatestMetric('gsa_layer_8_15')
  const gsaLayer16_23 = useLatestMetric('gsa_layer_16_23')
  const gsaLayer24_31 = useLatestMetric('gsa_layer_24_31')

  const recDepth = useLatestMetric('recurrence_depth')
  const recGain = useLatestMetric('recurrence_perf_gain')
  const recPass1 = useLatestMetric('recurrence_pass_1')
  const recPass2 = useLatestMetric('recurrence_pass_2')
  const recPass3 = useLatestMetric('recurrence_pass_3')
  const recConvergence = useLatestMetric('recurrence_convergence_delta')
  const recOverhead = useLatestMetric('recurrence_overhead_ms')

  const mhcMamba = useLatestMetric('mhc_mamba_score')
  const mhcAttn = useLatestMetric('mhc_attn_score')
  const mhcHybrid = useLatestMetric('mhc_hybrid_efficiency')
  const mhcSSM = useLatestMetric('mhc_ssm_path')
  const mhcAttnPath = useLatestMetric('mhc_attn_path')
  const mhcCrossGate = useLatestMetric('mhc_cross_gate')
  const mhcMemory = useLatestMetric('mhc_memory_efficiency')

  const gsaLayers = [
    { label: 'Layer 0-7', data: gsaLayer0_7 },
    { label: 'Layer 8-15', data: gsaLayer8_15 },
    { label: 'Layer 16-23', data: gsaLayer16_23 },
    { label: 'Layer 24-31', data: gsaLayer24_31 },
  ]

  const recPasses = [
    { label: 'Pass 1', data: recPass1, opacity: 1 },
    { label: 'Pass 2', data: recPass2, opacity: 0.7 },
    { label: 'Pass 3', data: recPass3, opacity: 0.5 },
  ]

  const mhcPaths = [
    { label: 'SSM Path', data: mhcSSM, grad: 'linear-gradient(90deg,var(--pk),var(--p))', color: 'var(--pk)' },
    { label: 'Attn Path', data: mhcAttnPath, grad: 'linear-gradient(90deg,var(--c),var(--b))', color: 'var(--c)' },
    { label: 'Cross Gate', data: mhcCrossGate, grad: 'linear-gradient(90deg,var(--g),var(--c))', color: 'var(--g)' },
    { label: 'Memory Eff', data: mhcMemory, grad: 'linear-gradient(90deg,var(--y),var(--o))', color: 'var(--y)' },
  ]

  return (
    <>
      <div className="sec-row">
        <h2>Architecture Stats</h2>
        <div className="sec-line" /><span className="tag tp">GSA &middot; REC &middot; mHC</span>
      </div>

      <div className="grid g3">
        {/* GSA */}
        <div className="pnl">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--c)' }} /> GSA Stats</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tc">ATTENTION</span>
              <InfoBadge text="Gated Sparse Attention metrics. Gate values and entropy indicate attention sparsity per layer group." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
              <MetricBox value={gsaGateAvg ? gsaGateAvg.value.toFixed(3) : '—'} label="Avg Gate Value" color="var(--c)" bgColor="var(--c-dim)" />
              <MetricBox value={gsaEntropy ? gsaEntropy.value.toFixed(3) : '—'} label="Gate Entropy" color="var(--b)" bgColor="var(--b-dim)" />
            </div>
            {gsaLayers.map(r => (
              <div className="pr" key={r.label}>
                <span className="pr-l">{r.label}</span>
                <div className="pr-t">
                  <div className="pr-f" style={{
                    width: r.data ? `${(r.data.value * 100).toFixed(0)}%` : '0%',
                    background: 'linear-gradient(90deg,var(--c),var(--b))'
                  }} />
                </div>
                <span className="pr-p" style={{ color: 'var(--c)' }}>{r.data ? r.data.value.toFixed(2) : '—'}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recurrence */}
        <div className="pnl">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--p)' }} /> Recurrence Performance</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tp">REC</span>
              <InfoBadge text="Recurrence loop stats. Shows depth, per-pass loss, convergence delta, and overhead per step." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
              <MetricBox value={recDepth ? recDepth.value.toFixed(0) : '—'} label="Recurrence Depth" color="var(--p)" bgColor="var(--p-dim)" />
              <MetricBox value={recGain ? `+${recGain.value.toFixed(1)}%` : '—'} label="Perf Gain / Loop" color="var(--pk)" bgColor="var(--pk-dim)" />
            </div>
            {recPasses.map(r => {
              const maxVal = recPass1?.value ?? 1
              const pct = r.data ? (r.data.value / maxVal * 100) : 0
              return (
                <div className="pr" key={r.label}>
                  <span className="pr-l">{r.label}</span>
                  <div className="pr-t">
                    <div className="pr-f" style={{ width: `${pct}%`, background: 'var(--p)', opacity: r.opacity }} />
                  </div>
                  <span className="pr-p" style={{ color: 'var(--p)' }}>{r.data ? r.data.value.toFixed(3) : '—'}</span>
                </div>
              )
            })}
            <div style={{
              marginTop: 8, padding: 8, background: 'rgba(255,255,255,0.02)',
              borderRadius: 'var(--rad)', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)'
            }}>
              Convergence &Delta;: <span style={{ color: 'var(--g)' }}>{recConvergence ? recConvergence.value.toFixed(3) : '—'}</span>
              {' '}&middot; Overhead: <span style={{ color: 'var(--o)' }}>{recOverhead ? `+${recOverhead.value.toFixed(0)}ms/step` : '—'}</span>
            </div>
          </div>
        </div>

        {/* mHC */}
        <div className="pnl">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--pk)' }} /> mHC Stats</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tpk">HYBRID</span>
              <InfoBadge text="Mamba-Hybrid-Controller scores. Tracks SSM vs attention path balance and memory efficiency." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 12 }}>
              <MetricBox value={mhcMamba ? mhcMamba.value.toFixed(2) : '—'} label="Mamba Score" color="var(--pk)" bgColor="var(--pk-dim)" fontSize={16} />
              <MetricBox value={mhcAttn ? mhcAttn.value.toFixed(2) : '—'} label="Attn Score" color="var(--c)" bgColor="var(--c-dim)" fontSize={16} />
              <MetricBox value={mhcHybrid ? mhcHybrid.value.toFixed(2) : '—'} label="Hybrid Eff" color="var(--g)" bgColor="var(--g-dim)" fontSize={16} />
            </div>
            {mhcPaths.map(r => {
              const pct = r.data ? (r.data.value * 100) : 0
              return (
                <div className="pr" key={r.label}>
                  <span className="pr-l">{r.label}</span>
                  <div className="pr-t">
                    <div className="pr-f" style={{ width: `${pct}%`, background: r.grad }} />
                  </div>
                  <span className="pr-p" style={{ color: r.color }}>{r.data ? `${(r.data.value * 100).toFixed(0)}%` : '—'}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </>
  )
}
