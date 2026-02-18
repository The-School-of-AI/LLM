/**
 * Shared mutable application state.
 * Imported by all modules that need to read or write global state.
 */

export const state = {
    allRuns: [],
    selectedRuns: [],       // array of run objects
    allMetrics: {},         // { category: [{ name, kind, table }] }
    chartInsts: {},         // canvasId -> Chart instance
    summaryCharts: {},      // canvasId -> Chart instance (summary tab)
    refreshTimer: null,
    dropdownOpen: false,
    expandedMetrics: new Set(),  // track which metrics are expanded across refreshes
    zoomMode: {},           // per-chart: 'box' | 'pan' | null
};
