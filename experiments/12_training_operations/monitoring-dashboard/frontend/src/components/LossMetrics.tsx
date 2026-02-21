import { MetricChart } from './PlotlyChart'
import { useLatestMetric, formatValue, getDelta } from '../hooks/useMetricData'
import { InfoBadge } from './InfoBadge'

function StatCard({ metricName, label, dotColor, tagClass, tagText, decimals = 3, info }: {
  metricName: string; label: string; dotColor: string; tagClass: string; tagText: string; decimals?: number; info: string
}) {
  const data = useLatestMetric(metricName)
  const delta = data ? getDelta(data.value, data.prev) : null

  return (
    <div className="pnl stat">
      <div className="ph">
        <span className="pt"><span className="dot" style={{ background: dotColor }} /> {label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className={`tag ${tagClass}`}>{tagText}</span>
          <InfoBadge text={info} />
        </div>
      </div>
      <div className="pb">
        <div className="sv" style={{ color: dotColor }}>
          {data ? formatValue(data.value, decimals) : '—'}
        </div>
        <div className="ss">
          {data ? `step ${data.step.toLocaleString()}` : 'awaiting data'}
        </div>
        {delta && (
          <div className={`sd ${delta.direction}`}>{delta.text}</div>
        )}
      </div>
    </div>
  )
}

export function LossMetrics() {
  return (
    <>
      <div className="sec-row">
        <h2>Loss Metrics</h2>
        <div className="sec-line" /><span className="tag tg">CORE</span>
      </div>

      <div className="grid g5">
        <StatCard metricName="loss/train_t_plus_1" label="Loss t+1" dotColor="var(--g)" tagClass="tg" tagText="LATEST"
          info="Next-token prediction loss. Primary training objective measuring model accuracy." />
        <StatCard metricName="loss/train_t_plus_2" label="Loss t+2" dotColor="var(--b)" tagClass="tb" tagText="LATEST"
          info="Two-step lookahead loss. Tracks how well the model predicts further ahead." />
        <StatCard metricName="loss/router_null" label="NULL Router Loss" dotColor="var(--o)" tagClass="to" tagText="WATCH"
          info="Loss from tokens routed to the NULL expert. High values indicate routing inefficiency." />
        <StatCard metricName="loss/router_moe" label="MoE Router Loss" dotColor="var(--p)" tagClass="tp" tagText="MoE"
          info="Mixture-of-Experts router loss. Measures expert load balancing and routing quality." />
        <StatCard metricName="loss" label="Loss (Overall)" dotColor="var(--c)" tagClass="tc" tagText="EVAL"
          info="Combined overall loss across all objectives. Key convergence indicator." />
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Training Loss Curves</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">TIMESERIES</span>
              <InfoBadge text="Training loss over time for all objectives. Downward trend indicates learning. Use scroll to zoom, drag to pan." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['loss/train', 'loss/train_t_plus_1', 'loss/train_t_plus_2', 'loss']}
              labels={['Train Loss', 'Loss t+1', 'Loss t+2', 'Loss (Overall)']}
              colors={['#f472b6', '#3ecf8e', '#4d9cf5', '#22d3ee']}
              height={280}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--pk)' }} />Train Loss</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--g)' }} />Loss t+1</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--b)' }} />Loss t+2</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--c)' }} />Loss (Overall)</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Router Loss (NULL + MoE)</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tp">TIMESERIES</span>
              <InfoBadge text="Router losses track expert assignment quality. Spikes may indicate load imbalance or routing instability." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['loss/router_null', 'loss/router_moe']}
              labels={['NULL Router', 'MoE Router']}
              colors={['#f5a524', '#a78bfa']}
              height={280}
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--o)' }} />NULL Router</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--p)' }} />MoE Router</div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
