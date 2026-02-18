/**
 * Metric discovery, sidebar rendering, search filtering, category selection.
 */

import { state } from './state.js';
import { fetchMetricsForRun } from './api.js';
import { showToast } from './utils.js';
import { generateCharts } from './charts.js';
import { loadEvents } from './events.js';

// ─── Discover metrics for the primary selected run ───
export async function discoverMetrics() {
    const primary = state.selectedRuns[0];
    if (!primary) return;
    document.getElementById('metricScroll').innerHTML =
        '<div class="empty" style="padding:1.5rem"><div class="spinner"></div></div>';
    try {
        state.allMetrics = await fetchMetricsForRun(primary.run_id);
        renderMetricSelector(state.allMetrics);
        loadEvents();
        showToast(`Loaded: ${primary.run_id.substring(0, 24)}`);
    } catch (e) {
        showToast('Failed to load metrics');
    }
}

// ─── Render the metric checkbox list ───
export function renderMetricSelector(metrics, filter = '') {
    const el = document.getElementById('metricScroll');
    const low = filter.toLowerCase();
    let html = '', any = false;

    for (const [cat, items] of Object.entries(metrics)) {
        const vis = items.filter(m => {
            const n = typeof m === 'object' ? m.name : m;
            return !low || n.toLowerCase().includes(low);
        });
        if (!vis.length) continue;
        any = true;
        html += `<div class="metric-category">
            <div class="cat-header">
                <span class="cat-name">${cat}</span>
                <div class="cat-actions">
                    <button class="cat-btn" data-cat="${cat}" data-sel="true">all</button>
                    <button class="cat-btn" data-cat="${cat}" data-sel="false">none</button>
                </div>
            </div>`;
        vis.forEach(m => {
            const name = typeof m === 'object' ? m.name : m;
            const kind = typeof m === 'object' ? m.kind : 'scalar';
            const id = `m-${name.replace(/[^a-zA-Z0-9]/g, '-')}`;
            const prev = document.getElementById(id);
            const chk = prev?.checked ? 'checked' : '';
            html += `<div class="metric-item">
                <input type="checkbox" id="${id}" value="${name}" data-cat="${cat}" ${chk}>
                <label for="${id}">${name}</label>
                <span class="kind-badge kind-${kind}">${kind === 'array' ? 'hist' : 'val'}</span>
            </div>`;
        });
        html += '</div>';
    }
    el.innerHTML = any ? html
        : '<div class="empty" style="padding:1rem"><span style="font-size:0.7rem;color:var(--muted)">No metrics match</span></div>';

    // Attach event handlers
    el.querySelectorAll('.metric-item input').forEach(cb => {
        cb.addEventListener('change', () => generateCharts());
    });
    el.querySelectorAll('.cat-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            selCat(btn.dataset.cat, btn.dataset.sel === 'true');
        });
    });
}

// ─── Filter metrics by search input ───
export function filterMetrics() {
    renderMetricSelector(state.allMetrics, document.getElementById('metricSearch').value);
}

// ─── Select all / none in a category ───
function selCat(cat, on) {
    document.querySelectorAll(`[data-cat="${cat}"]`).forEach(cb => {
        if (cb.type === 'checkbox') cb.checked = on;
    });
    generateCharts();
}
