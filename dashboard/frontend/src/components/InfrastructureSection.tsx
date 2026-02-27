import { useMemo } from 'react'
import { useLatestMetric, useMetricsByPrefix, formatValue } from '../hooks/useMetricData'
import { MetricChart } from './PlotlyChart'
import { InfoBadge } from './InfoBadge'

export function InfrastructureSection() {
  const gpuMetrics = useMetricsByPrefix('sys.gpu.')
  const gpuIdleTime = useLatestMetric('gpu_idle_time')
  const cpuIdlePercent = useLatestMetric('cpu/idle_percent')
  const netMetrics = useMetricsByPrefix('sys.net.')

  const gpuData = useMemo(() => {
    const gpus: Array<{ id: number; util: number; temp: number | null; mem: number | null }> = []
    const utilPattern = /sys\.gpu\.(\d+)\.util/
    const tempPattern = /sys\.gpu\.(\d+)\.temp/
    const memPattern = /sys\.gpu\.(\d+)\.(memory|mem)/
    const gpuMap: Record<number, { util: number; temp: number | null; mem: number | null }> = {}

    for (const [key, data] of Object.entries(gpuMetrics)) {
      let match = key.match(utilPattern)
      if (match) {
        const id = parseInt(match[1])
        if (!gpuMap[id]) gpuMap[id] = { util: 0, temp: null, mem: null }
        gpuMap[id].util = data.value
        continue
      }
      match = key.match(tempPattern)
      if (match) {
        const id = parseInt(match[1])
        if (!gpuMap[id]) gpuMap[id] = { util: 0, temp: null, mem: null }
        gpuMap[id].temp = data.value
        continue
      }
      match = key.match(memPattern)
      if (match) {
        const id = parseInt(match[1])
        if (!gpuMap[id]) gpuMap[id] = { util: 0, temp: null, mem: null }
        gpuMap[id].mem = data.value
        continue
      }
    }

    for (const [id, data] of Object.entries(gpuMap)) {
      gpus.push({ id: parseInt(id), ...data })
    }
    gpus.sort((a, b) => a.id - b.id)
    return gpus
  }, [gpuMetrics])

  const avgGpuUtil = useMemo(() => {
    if (gpuData.length === 0) return null
    return gpuData.reduce((sum, g) => sum + g.util, 0) / gpuData.length
  }, [gpuData])

  const circumference = 2 * Math.PI * 42
  const gaugeOffset = avgGpuUtil != null ? circumference * (1 - avgGpuUtil / 100) : circumference

  const netBytesIn = netMetrics['sys.net.bytes_recv']
  const netBytesOut = netMetrics['sys.net.bytes_sent']

  return (
    <>
      <div className="sec-row">
        <h2>Infrastructure &amp; Hardware</h2>
        <div className="sec-line" /><span className="tag tr">INFRA</span>
      </div>

      <div className="grid g4">
        <div className="pnl">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--g)' }} /> GPU Utilization</span>
            <InfoBadge text="Average GPU utilization across all devices. Target is >90% for efficient training." />
          </div>
          <div className="pb">
            <div className="gauge">
              <div className="gauge-r">
                <svg viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                  <circle cx="50" cy="50" r="42" fill="none" stroke="var(--g)" strokeWidth="8"
                    strokeDasharray={circumference.toFixed(1)}
                    strokeDashoffset={gaugeOffset.toFixed(1)}
                    strokeLinecap="round" />
                </svg>
                <div className="gauge-v" style={{ color: 'var(--g)' }}>
                  {avgGpuUtil != null ? `${avgGpuUtil.toFixed(0)}%` : '—'}
                </div>
              </div>
              <div className="gauge-c">
                {gpuData.length > 0 ? `Avg across ${gpuData.length}× GPU` : 'Awaiting GPU data'}
              </div>
            </div>
          </div>
        </div>

        <div className="pnl stat">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--b)' }} /> CPU Idle</span>
            <InfoBadge text="CPU idle percentage. High idle means CPU is not a bottleneck; low idle may indicate data preprocessing load." />
          </div>
          <div className="pb">
            <div className="sv" style={{ color: 'var(--b)' }}>
              {cpuIdlePercent ? `${cpuIdlePercent.value.toFixed(1)}%` : '—'}
            </div>
            <div className="ss">cpu idle percent</div>
            {cpuIdlePercent?.prev != null && (
              <div className={`sd ${cpuIdlePercent.value > cpuIdlePercent.prev ? 'up' : 'dn'}`}>
                {cpuIdlePercent.value > cpuIdlePercent.prev ? '\u25B2' : '\u25BC'} {Math.abs(cpuIdlePercent.value - cpuIdlePercent.prev).toFixed(1)}%
              </div>
            )}
          </div>
        </div>

        <div className="pnl stat">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--o)' }} /> GPU Idle Time</span>
            <InfoBadge text="Average GPU idle time. Lower is better — high values suggest pipeline bubbles or synchronization waits." />
          </div>
          <div className="pb">
            <div className="sv" style={{ color: 'var(--o)' }}>
              {gpuIdleTime ? `${gpuIdleTime.value.toFixed(1)}%` : '—'}
            </div>
            <div className="ss">avg idle</div>
            {gpuIdleTime?.prev != null && (
              <div className={`sd ${gpuIdleTime.value < gpuIdleTime.prev ? 'dn' : 'up'}`}>
                {gpuIdleTime.value < gpuIdleTime.prev ? '\u25BC' : '\u25B2'} {Math.abs(gpuIdleTime.value - gpuIdleTime.prev).toFixed(1)}%
              </div>
            )}
          </div>
        </div>

        <div className="pnl stat">
          <div className="ph">
            <span className="pt"><span className="dot" style={{ background: 'var(--c)' }} /> Data Transfer</span>
            <InfoBadge text="Network throughput for inter-node communication. Includes data loading and gradient sync traffic." />
          </div>
          <div className="pb">
            <div className="sv" style={{ color: 'var(--c)' }}>
              {netBytesOut ? formatValue(netBytesOut.value, 1) : netBytesIn ? formatValue(netBytesIn.value, 1) : '—'}
            </div>
            <div className="ss">bytes/s network</div>
            <div style={{ marginTop: 10, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)' }}>
              {netBytesIn && <>In: <span style={{ color: 'var(--c)' }}>{formatValue(netBytesIn.value, 1)}/s</span><br /></>}
              {netBytesOut && <>Out: <span style={{ color: 'var(--p)' }}>{formatValue(netBytesOut.value, 1)}/s</span></>}
              {!netBytesIn && !netBytesOut && 'Awaiting data'}
            </div>
          </div>
        </div>
      </div>

      {/* GPU Heatmap */}
      {gpuData.length > 0 && (
        <div className="pnl">
          <div className="ph">
            <span className="pt">Per-GPU Utilization Heatmap ({gpuData.length}× GPU)</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">LIVE</span>
              <InfoBadge text="Real-time utilization for each GPU. Color intensity reflects load. Also shows temperature and memory." />
            </div>
          </div>
          <div className="pb">
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(gpuData.length, 8)}, 1fr)`, gap: 6 }}>
              {gpuData.map(g => {
                const int = Math.min(g.util / 100, 1)
                return (
                  <div key={g.id} style={{
                    background: `hsla(145,60%,${30 + int * 30}%,${0.3 + int * 0.7})`,
                    borderRadius: 6, padding: '20px 8px', textAlign: 'center',
                    border: `1px solid rgba(62,207,142,${int * 0.3})`
                  }}>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 22, fontWeight: 700, color: `rgba(255,255,255,${0.6 + int * 0.4})` }}>
                      {g.util.toFixed(0)}%
                    </div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: '#5a6478', marginTop: 4 }}>GPU {g.id}</div>
                    <div style={{ fontSize: 9, color: '#5a6478', marginTop: 2 }}>
                      {g.temp != null ? `${g.temp.toFixed(0)}°C` : '—'}
                      {' \u00B7 '}
                      {g.mem != null ? `${g.mem.toFixed(0)}GB` : '—'}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Time series charts */}
      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">CPU Idle % Over Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tb">TIMESERIES</span>
              <InfoBadge text="CPU idle percentage over time. Sustained low values may indicate a data loading or preprocessing bottleneck." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['cpu/idle_percent']}
              labels={['CPU Idle %']}
              colors={['#4d9cf5']}
              height={280}
              yAxisTitle="Idle %"
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--b)' }} />CPU Idle %</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Step Time Over Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tc">TIMESERIES</span>
              <InfoBadge text="Wall-clock time per training step. Spikes indicate I/O stalls, checkpointing, or garbage collection pauses." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['step_time_ms']}
              labels={['Step Time (ms)']}
              colors={['#22d3ee']}
              height={280}
              yAxisTitle="ms"
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--c)' }} />Step Time (ms)</div>
            </div>
          </div>
        </div>
      </div>

      {/* GPU util/temp charts if data exists */}
      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">GPU Utilization Over Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">TIMESERIES</span>
              <InfoBadge text="Per-GPU utilization history. Helps identify imbalanced workloads or failing devices." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['sys.gpu.0.util', 'sys.gpu.1.util', 'sys.gpu.2.util', 'sys.gpu.3.util']}
              labels={['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']}
              colors={['#3ecf8e', '#4d9cf5', '#a78bfa', '#f472b6']}
              height={280}
              yAxisTitle="Utilization %"
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--g)' }} />GPU 0</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--b)' }} />GPU 1</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--p)' }} />GPU 2</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--pk)' }} />GPU 3</div>
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">GPU Temperature Over Time</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag to">TIMESERIES</span>
              <InfoBadge text="Per-GPU temperature history. Sustained temps above 85°C may trigger thermal throttling." />
            </div>
          </div>
          <div className="pb np">
            <MetricChart
              metrics={['sys.gpu.0.temp', 'sys.gpu.1.temp', 'sys.gpu.2.temp', 'sys.gpu.3.temp']}
              labels={['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']}
              colors={['#f5a524', '#f55656', '#fbbf24', '#22d3ee']}
              height={280}
              yAxisTitle="Temp °C"
            />
            <div className="leg">
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--o)' }} />GPU 0</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--r)' }} />GPU 1</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--y)' }} />GPU 2</div>
              <div className="leg-i"><div className="leg-s" style={{ background: 'var(--c)' }} />GPU 3</div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
