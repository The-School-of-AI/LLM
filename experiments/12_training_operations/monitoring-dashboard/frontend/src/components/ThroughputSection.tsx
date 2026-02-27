import { MetricChart } from './PlotlyChart'
import { useLatestMetric, formatValue, getDelta } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

function ThroughputStat({ metricName, label, dotColor, unit, tagClass, tagText, decimals = 1, info }: {
  metricName: string; label: string; dotColor: string; unit: string
  tagClass?: string; tagText?: string; decimals?: number; info: string
}) {
  const data = useLatestMetric(metricName)
  const delta = data ? getDelta(data.value, data.prev) : null

  return (
    <div className="pnl stat">
      <div className="ph">
        <span className="pt"><span className="dot" style={{ background: dotColor }} /> {label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {tagClass && tagText && <span className={`tag ${tagClass}`}>{tagText}</span>}
          <InfoBadge text={info} />
        </div>
      </div>
      <div className="pb">
        <div className="sv" style={{ color: dotColor }}>
          {data ? formatValue(data.value, decimals) : '—'}
        </div>
        <div className="ss">{unit}</div>
        {delta && <div className={`sd ${delta.direction}`}>{delta.text}</div>}
      </div>
    </div>
  )
}

export function ThroughputSection() {
  return (
    <>
      <div className="sec-row">
        <h2>Throughput &amp; Efficiency</h2>
        <div className="sec-line" /><span className="tag tb">PERF</span>
      </div>

      <div className="grid g5">
        <ThroughputStat metricName="throughput/tokens_per_sec" label="Token/sec" dotColor="var(--g)" unit="tokens per second"
          info="Number of tokens processed per second. Higher is better for training speed." />
        <ThroughputStat metricName="throughput/batches_per_sec" label="Batch/sec" dotColor="var(--b)" unit="batches per second"
          info="Batches processed per second. Reflects data pipeline and GPU utilization efficiency." />
        <ThroughputStat metricName="router/null_ratio" label="NULL Ratio" dotColor="var(--o)" unit="null-routed tokens"
          tagClass="to" tagText="CRITICAL"
          info="Fraction of tokens routed to the NULL expert. Should stay below 15% for healthy training." />
        <ThroughputStat metricName="tokens/processed_total" label="Total Tokens" dotColor="var(--c)" unit="processed total" decimals={2}
          info="Cumulative tokens processed since training start. Tracks overall training progress." />
        <ThroughputStat metricName="lr" label="Learning Rate" dotColor="var(--p)" unit="current LR" decimals={6}
          info="Current optimizer learning rate. Follows the configured schedule (warmup, decay, etc.)." />
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Throughput Timeline</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tb">TIMESERIES</span>
              <InfoBadge text="Token and batch throughput over time. Drops may indicate data loading bottlenecks or checkpointing pauses." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['throughput/tokens_per_sec', 'throughput/batches_per_sec']}
              labels={['Token/sec', 'Batch/sec']}
              colors={['#3ecf8e', '#4d9cf5']}
              height={280}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--g)' }} />Token/sec</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--b)' }} />Batch/sec</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">NULL Ratio Over Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag to">TIMESERIES</span>
              <InfoBadge text="Null-routed token ratio over time. Dashed line marks the 15% alert threshold." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['router/null_ratio']}
              labels={['NULL Ratio %']}
              colors={['#f5a524']}
              thresholdY={15}
              height={280}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--o)' }} />NULL Ratio %</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'rgba(245,165,36,0.3)' }} />Threshold (15%)</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Learning Rate Schedule</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tp">TIMESERIES</span>
              <InfoBadge text="Learning rate over training steps. Shows warmup, plateau, and decay phases of the LR schedule." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['lr']}
              labels={['Learning Rate']}
              colors={['#a78bfa']}
              height={240}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--p)' }} />Learning Rate</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Step Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tc">TIMESERIES</span>
              <InfoBadge text="Wall-clock time per training step in milliseconds. Spikes may indicate I/O stalls or GC pauses." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['step_time_ms']}
              labels={['Step Time (ms)']}
              colors={['#22d3ee']}
              height={240}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--c)' }} />Step Time (ms)</div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
