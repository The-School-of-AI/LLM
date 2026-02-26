import { useMetricsStore } from '../stores/metricsStore'
import type { TabKey, TimePreset } from '../types/metrics'
import { TAB_KEYS, TAB_LABELS } from '../types/metrics'

const TIME_PRESETS: { key: TimePreset; label: string }[] = [
  { key: '1d', label: '1 Day' },
  { key: '3d', label: '3 Days' },
  { key: '1w', label: '1 Week' },
  { key: 'all', label: 'All' },
]

const TAB_ICONS: Record<TabKey, string> = {
  overview: '\u25C8',
  architecture: '\u2726',
  milestones: '\u2691',
  infrastructure: '\u2699',
}

export function Topbar() {
  const status = useMetricsStore(s => s.connectionStatus)
  const timePreset = useMetricsStore(s => s.timePreset)
  const setTimePreset = useMetricsStore(s => s.setTimePreset)
  const selectedTab = useMetricsStore(s => s.selectedTab)
  const setTab = useMetricsStore(s => s.setTab)

  return (
    <div className="topbar-wrap">
      <div className="topbar">
        <div className="tb-left">
          <div className="logo">TrainOps<span> // ERA-V4</span></div>
          <div className="crumb">
            Dashboards <span style={{ opacity: 0.4 }}>&rsaquo;</span>{' '}
            <span className="act">LLM Training Operations</span>
          </div>
        </div>
        <div className="tb-right">
          <div className={`tb-btn ${status === 'connected' ? 'active' : ''}`}>
            <div className="live-dot" style={status !== 'connected' ? { background: 'var(--o)' } : undefined} />
            {status === 'connected' ? 'Live' : status === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </div>
          <div className="time-preset-group">
            {TIME_PRESETS.map(p => (
              <button
                key={p.key}
                className={`time-preset-btn${timePreset === p.key ? ' time-preset-active' : ''}`}
                onClick={() => setTimePreset(p.key)}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="tb-btn" onClick={() => window.location.reload()}>&#x27F3; Refresh</div>
        </div>
      </div>
      <nav className="tab-bar">
        {TAB_KEYS.map(key => (
          <button
            key={key}
            className={`tab-btn${selectedTab === key ? ' tab-active' : ''}`}
            onClick={() => setTab(key)}
          >
            <span className="tab-icon">{TAB_ICONS[key]}</span>
            {TAB_LABELS[key]}
          </button>
        ))}
      </nav>
    </div>
  )
}
