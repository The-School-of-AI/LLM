/**
 * Run selection: dropdown, chips, toggle logic.
 */

import { state } from './state.js';
import { getRunColor, showToast } from './utils.js';
import { fetchRuns } from './api.js';
import { discoverMetrics } from './metrics.js';
import { generateCharts } from './charts.js';

// ─── Load runs from server ───
export async function loadRuns() {
    try {
        state.allRuns = await fetchRuns();
        renderRunDropdown();
    } catch (e) {
        showToast('Could not reach server');
    }
}

// ─── Dropdown open/close ───
export function toggleRunDropdown() {
    state.dropdownOpen = !state.dropdownOpen;
    document.getElementById('runDropdownBtn').classList.toggle('open', state.dropdownOpen);
    document.getElementById('runDropdownPanel').classList.toggle('open', state.dropdownOpen);
}

export function initDropdownClickOutside() {
    document.addEventListener('click', (e) => {
        if (state.dropdownOpen && !e.target.closest('.run-dropdown-wrap')) {
            state.dropdownOpen = false;
            document.getElementById('runDropdownBtn').classList.remove('open');
            document.getElementById('runDropdownPanel').classList.remove('open');
        }
    });
}

// ─── Render the dropdown list ───
function renderRunDropdown() {
    const panel = document.getElementById('runDropdownPanel');
    const selIds = new Set(state.selectedRuns.map(r => r.run_id));

    if (!state.allRuns.length) {
        panel.innerHTML = '<div style="padding:0.75rem;text-align:center;font-size:0.68rem;color:var(--muted)">No runs found</div>';
        updateDropdownBtnText();
        return;
    }

    panel.innerHTML = state.allRuns.map(r => {
        const checked = selIds.has(r.run_id) ? 'checked' : '';
        const sc = { 'running': 'badge-running', 'done': 'badge-done', 'completed': 'badge-done', 'failed': 'badge-failed' }[r.status] || 'badge-unknown';
        return `<div class="run-dropdown-item" data-run-id="${r.run_id}">
            <input type="checkbox" ${checked} tabindex="-1">
            <div class="run-dropdown-label">
                <div class="run-dropdown-name">${r.run_id}</div>
                <div class="run-dropdown-meta">
                    <span class="badge ${sc}">${r.status || 'unknown'}</span>
                    ${r.model_name ? `<span class="run-model">${r.model_name}${r.model_size ? ' &middot; ' + r.model_size : ''}</span>` : ''}
                </div>
            </div>
        </div>`;
    }).join('');

    // Attach click handlers via delegation
    panel.querySelectorAll('.run-dropdown-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleRunSel(item.dataset.runId);
        });
    });

    updateDropdownBtnText();
}

function updateDropdownBtnText() {
    const txt = document.getElementById('runDropdownText');
    if (!state.selectedRuns.length) {
        txt.textContent = 'Select runs...';
        txt.style.color = 'var(--muted)';
    } else {
        txt.textContent = `${state.selectedRuns.length} run${state.selectedRuns.length > 1 ? 's' : ''} selected`;
        txt.style.color = 'var(--text)';
    }
}

// ─── Toggle a run on/off ───
export function toggleRunSel(runId) {
    const idx = state.selectedRuns.findIndex(r => r.run_id === runId);
    const wasFirst = idx === 0;

    if (idx >= 0) {
        state.selectedRuns.splice(idx, 1);
    } else {
        const run = state.allRuns.find(r => r.run_id === runId);
        if (run) state.selectedRuns.push(run);
    }

    renderRunDropdown();
    renderRunChips();

    const newFirst = state.selectedRuns[0]?.run_id;
    if (state.selectedRuns.length && (wasFirst || idx < 0 && state.selectedRuns.length === 1)) {
        discoverMetrics();
    } else {
        generateCharts();
    }
}

// ─── Remove a run (via chip ×) ───
export function removeRun(runId) {
    const idx = state.selectedRuns.findIndex(r => r.run_id === runId);
    if (idx < 0) return;
    const wasFirst = idx === 0;
    state.selectedRuns.splice(idx, 1);
    renderRunDropdown();
    renderRunChips();

    if (!state.selectedRuns.length) {
        document.getElementById('metricScroll').innerHTML =
            '<div class="empty" style="padding:1rem"><span style="font-size:0.72rem">Select a run first</span></div>';
        document.getElementById('chartsContent').innerHTML =
            '<div class="empty"><div class="empty-icon">&#11041;</div><div class="empty-title">Nothing selected</div><div class="empty-sub">Pick runs from the sidebar, then check metrics to chart</div></div>';
        return;
    }
    if (wasFirst) {
        discoverMetrics();
    } else {
        generateCharts();
    }
}

// ─── Render chip badges for selected runs ───
function renderRunChips() {
    const el = document.getElementById('runChips');
    if (!state.selectedRuns.length) { el.innerHTML = ''; return; }
    el.innerHTML = state.selectedRuns.map((r, i) => {
        const col = getRunColor(i);
        return `<div class="run-chip" style="color:${col};border-color:${col}40;background:${col}10">
            <span class="run-chip-label">${r.run_id.substring(0, 22)}</span>
            <button class="run-chip-x" style="color:${col}" data-run-id="${r.run_id}">&times;</button>
        </div>`;
    }).join('');

    // Attach chip removal handlers
    el.querySelectorAll('.run-chip-x').forEach(btn => {
        btn.addEventListener('click', () => removeRun(btn.dataset.runId));
    });
}
