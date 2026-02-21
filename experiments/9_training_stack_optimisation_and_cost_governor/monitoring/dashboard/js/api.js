/**
 * All server API calls.
 */

import { API } from './constants.js';
import { showToast } from './utils.js';

export async function fetchRuns() {
    const data = await fetch(`${API}/api/runs`).then(r => r.json());
    return data.runs || [];
}

export async function fetchMetricsForRun(runId) {
    const data = await fetch(`${API}/api/runs/${runId}/metrics`).then(r => r.json());
    return data.metrics || {};
}

export async function fetchData(runId, metrics) {
    return fetch(`${API}/api/runs/${runId}/data?metrics=${encodeURIComponent(metrics.join(','))}`).then(r => r.json());
}

export async function fetchEvents(runId) {
    const data = await fetch(`${API}/api/runs/${runId}/events`).then(r => r.json());
    return data.events || [];
}
