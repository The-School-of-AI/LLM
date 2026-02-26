import { useMemo } from 'react'
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { useMetricsStore } from '../stores/metricsStore'

const Plot = createPlotlyComponent(Plotly)

const COLORS = ['#3ecf8e', '#4d9cf5', '#22d3ee', '#a78bfa', '#f472b6', '#f5a524', '#fbbf24', '#f55656']

const DARK_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {
    family: 'JetBrains Mono, monospace',
    size: 10,
    color: '#5a6478',
  },
  margin: { t: 8, r: 12, b: 36, l: 48 },
  xaxis: {
    type: 'date',
    gridcolor: 'rgba(255,255,255,0.04)',
    linecolor: 'rgba(255,255,255,0.04)',
    zerolinecolor: 'rgba(255,255,255,0.04)',
    tickfont: { size: 9, color: '#5a6478' },
    showgrid: true,
    rangeslider: { visible: false },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.04)',
    linecolor: 'rgba(255,255,255,0.04)',
    zerolinecolor: 'rgba(255,255,255,0.06)',
    tickfont: { size: 9, color: '#5a6478' },
    showgrid: true,
  },
  hovermode: 'x unified',
  hoverlabel: {
    bgcolor: '#1a1f2e',
    bordercolor: '#2a3042',
    font: { family: 'JetBrains Mono, monospace', size: 11, color: '#e0e4ef' },
  },
  legend: {
    orientation: 'h',
    x: 0,
    y: -0.2,
    font: { size: 10, color: '#5a6478' },
    bgcolor: 'transparent',
  },
  showlegend: false,
  autosize: true,
}

const CHART_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  displaylogo: false,
  responsive: true,
  scrollZoom: true,
}

interface MetricChartProps {
  metrics: string[]
  labels?: string[]
  colors?: string[]
  thresholdY?: number
  yAxisTitle?: string
  height?: number
}

/**
 * Plotly-based time-series chart that pulls real data from the Zustand store.
 * X-axis uses timestamps, has built-in zoom/pan/hover.
 */
export function MetricChart({
  metrics,
  labels,
  colors,
  thresholdY,
  yAxisTitle,
  height = 180,
}: MetricChartProps) {
  const runs = useMetricsStore(s => s.runs)
  const selectedRuns = useMetricsStore(s => s.selectedRuns)
  const timeRange = useMetricsStore(s => s.timeRange)

  const { traces, shapes } = useMemo(() => {
    const traces: Plotly.Data[] = []
    const palette = colors ?? COLORS

    for (let idx = 0; idx < metrics.length; idx++) {
      const metricName = metrics[idx]
      for (const runId of selectedRuns) {
        const series = runs[runId]?.[metricName]
        if (!series || series.values.length === 0) continue

        // Build x (Date) and y arrays, applying time filter
        const x: string[] = []
        const y: number[] = []
        for (let i = 0; i < series.timestamps.length; i++) {
          const ts = series.timestamps[i]
          if (timeRange && (ts < timeRange.from || ts > timeRange.to)) continue
          x.push(new Date(ts * 1000).toISOString())
          y.push(series.values[i])
        }
        if (x.length === 0) continue

        const color = palette[idx % palette.length]
        traces.push({
          x,
          y,
          type: 'scattergl',
          mode: 'lines',
          name: labels?.[idx] ?? metricName,
          line: { color, width: 1.5, shape: 'spline' },
          fill: 'tozeroy',
          fillcolor: color + '0F',
          hovertemplate: `%{y:.4f}<extra>${labels?.[idx] ?? metricName}</extra>`,
        })
        break // first matching run per metric
      }
    }

    // Threshold horizontal line
    const shapes: Partial<Plotly.Shape>[] = []
    if (thresholdY !== undefined) {
      shapes.push({
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: thresholdY,
        y1: thresholdY,
        line: { color: 'rgba(245,165,36,0.35)', width: 1, dash: 'dash' },
      })
    }

    return { traces, shapes }
  }, [runs, selectedRuns, metrics, labels, colors, timeRange, thresholdY])

  const layout = useMemo<Partial<Plotly.Layout>>(() => ({
    ...DARK_LAYOUT,
    height,
    shapes,
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: yAxisTitle ? { text: yAxisTitle, font: { size: 10, color: '#5a6478' } } : undefined,
    },
  }), [height, shapes, yAxisTitle])

  if (traces.length === 0) {
    return (
      <div className="chart-wrap" style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: '#5a6478' }}>Awaiting data...</span>
      </div>
    )
  }

  return (
    <div className="chart-wrap" style={{ height }}>
      <Plot
        data={traces}
        layout={layout}
        config={CHART_CONFIG}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
