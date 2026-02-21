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
    expandedMetrics: new Set(),  // metric names currently expanded (persists across refreshes)
    zoomMode: {},           // per-chart: 'box' | null
    // Stable color assignment: run_id -> color index. Never re-assigned after first add.
    runColorMap: new Map(),
    // Source-of-truth for checked metrics — persists across discoverMetrics / refresh cycles.
    selectedMetrics: new Set(),
};
