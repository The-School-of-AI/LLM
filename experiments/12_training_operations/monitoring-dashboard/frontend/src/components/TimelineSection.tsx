import { useMemo } from 'react'
import { useLatestMetric } from '../hooks/useMetricData'
import { useMetricsStore } from '../stores/metricsStore'
import { InfoBadge } from './InfoBadge'

export function TimelineSection() {
  const timeToB1 = useLatestMetric('time_to_b1')
  const timeToB2 = useLatestMetric('time_to_b2')
  const timeTo3B = useLatestMetric('time_to_3b')
  const timeTo8B = useLatestMetric('time_to_8b')
  const timeTo70B = useLatestMetric('time_to_70b')
  const timeToSFT = useLatestMetric('time_to_sft')

  const allRuns = useMetricsStore(s => s.allRuns)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)

  const b1Done = timeToB1 && timeToB1.value >= 1
  const b1Pct = timeToB1 ? Math.min(timeToB1.value * 100, 100) : 0
  const b2Pct = timeToB2 ? Math.min(timeToB2.value * 100, 100) : 0

  const milestones = [
    { label: '\u2192 3B', data: timeTo3B, bg: 'var(--g)', color: 'var(--g)' },
    { label: '\u2192 8B', data: timeTo8B, bg: 'var(--b)', color: 'var(--b)' },
    { label: '\u2192 70B', data: timeTo70B, bg: 'var(--p)', color: 'var(--p)' },
    { label: '\u2192 SFT', data: timeToSFT, bg: 'var(--pk)', color: 'var(--pk)' },
  ]

  const events = useMemo(() => {
    const evts: Array<{ color: string; time: string; text: string }> = []
    for (const run of allRuns) {
      if (selectedRuns.includes(run.run_id)) {
        evts.push({
          color: run.is_active ? 'var(--g)' : 'var(--text-muted)',
          time: new Date(run.last_event_time * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC',
          text: `Run ${run.run_id} ${run.is_active ? 'active' : 'inactive'} at step ${run.latest_step.toLocaleString()}`,
        })
      }
    }
    evts.sort((a, b) => b.time.localeCompare(a.time))
    return evts.slice(0, 20)
  }, [allRuns, selectedRuns])

  return (
    <>
      <div className="sec-row">
        <h2>Training Timeline &amp; Milestones</h2>
        <div className="sec-line" /><span className="tag tg">PROGRESS</span>
      </div>

      <div className="grid g2">
        <div className="pnl">
          <div className="ph">
            <span className="pt">Stage Progress — B1 / B2</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tg">ETA</span>
              <InfoBadge text="Training stage completion for B1 and B2 phases, plus model scale milestones (3B, 8B, 70B, SFT)." />
            </div>
          </div>
          <div className="pb">
            <div className="pr">
              <span className="pr-l" style={{ minWidth: 90 }}>B1 Stage</span>
              <div className="pr-t">
                <div className="pr-f" style={{ width: `${b1Pct}%`, background: 'var(--g)' }} />
              </div>
              <span className="pr-p" style={{ color: 'var(--g)' }}>
                {b1Done ? 'DONE' : timeToB1 ? `${b1Pct.toFixed(0)}%` : '—'}
              </span>
            </div>
            {timeToB1 && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)', margin: '-2px 0 10px 100px' }}>
                {b1Done ? `Completed` : `In progress`}
              </div>
            )}

            <div className="pr">
              <span className="pr-l" style={{ minWidth: 90 }}>B2 Stage</span>
              <div className="pr-t">
                <div className="pr-f" style={{ width: `${b2Pct}%`, background: 'linear-gradient(90deg,var(--b),var(--c))' }} />
              </div>
              <span className="pr-p" style={{ color: 'var(--b)' }}>
                {timeToB2 ? `${b2Pct.toFixed(0)}%` : '—'}
              </span>
            </div>

            <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 10 }}>Model Scale Milestones</div>
              {milestones.map(m => {
                const pct = m.data ? Math.min(m.data.value * 100, 100) : 0
                const done = pct >= 100
                return (
                  <div className="pr" key={m.label}>
                    <span className="pr-l" style={{ minWidth: 90, color: m.color }}>{m.label}</span>
                    <div className="pr-t">
                      <div className="pr-f" style={{ width: `${pct}%`, background: m.bg }} />
                    </div>
                    <span className="pr-p" style={{ color: done ? m.color : 'var(--text-muted)' }}>
                      {m.data ? (done ? '\u2713' : `${pct.toFixed(0)}%`) : 'Pending'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        <div className="pnl">
          <div className="ph">
            <span className="pt">Training Event Log</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="tag tb">TIMELINE</span>
              <InfoBadge text="Chronological log of training events, run starts, and status changes from selected runs." />
            </div>
          </div>
          <div className="pb" style={{ overflowY: 'auto', maxHeight: 360 }}>
            {events.length > 0 ? events.map((e, i) => (
              <div className="tl-i" key={i}>
                <div className="tl-d" style={{ background: e.color }} />
                <div>
                  <div className="tl-t">{e.time}</div>
                  <div className="tl-x">{e.text}</div>
                </div>
              </div>
            )) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20, fontFamily: 'var(--mono)', fontSize: 11 }}>
                Awaiting training events...
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
