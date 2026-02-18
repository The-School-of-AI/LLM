/**
 * Chart creation, toolbar actions (zoom, expand, download), base chart options.
 */

import { state } from './state.js';
import { ICON } from './constants.js';
import { fmt, stats, getRunColorById, showToast } from './utils.js';
import { fetchData } from './api.js';

// ─── Main entry: rebuild all charts ───
export async function generateCharts() {
    const sel = Array.from(state.selectedMetrics);
    const el = document.getElementById('chartsContent');

    // Destroy existing charts
    Object.values(state.chartInsts).forEach(c => c.destroy());
    state.chartInsts = {};

    if (!sel.length || !state.selectedRuns.length) {
        el.innerHTML = '<div class="empty"><div class="empty-icon">&#11041;</div><div class="empty-title">No metrics selected</div><div class="empty-sub">Check metrics in the sidebar</div></div>';
        return;
    }

    el.innerHTML = '<div class="empty"><div class="spinner"></div></div>';

    // Fetch data for all selected runs in parallel
    const runDataMap = {};
    const results = await Promise.all(
        state.selectedRuns.map(r => fetchData(r.run_id, sel).then(d => ({ runId: r.run_id, data: d })))
    );
    results.forEach(r => runDataMap[r.runId] = r.data);

    el.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'charts-grid';
    el.appendChild(grid);

    sel.forEach((metric, i) => {
        const hasData = state.selectedRuns.some(r => (runDataMap[r.run_id]?.[metric] || []).length > 0);
        if (hasData) createChart(grid, metric, runDataMap, i);
    });
}

// ─── Create a single chart card ───
function createChart(container, metric, runDataMap, idx) {
    const firstData = runDataMap[state.selectedRuns[0].run_id]?.[metric] || [];
    const isHist = firstData.length > 0 && Array.isArray(firstData[0]?.value);
    const multiRun = state.selectedRuns.length > 1;

    const card = document.createElement('div');
    card.className = 'chart-card' + (state.expandedMetrics.has(metric) ? ' expanded' : '');

    const ps = stats(firstData);
    const cvId = `cv${idx}`;

    // Legend for multi-run
    let legendHtml = '';
    if (multiRun && !isHist) {
        legendHtml = '<div class="chart-legend">' + state.selectedRuns.map((r) => {
            const col = getRunColorById(r.run_id);
            return `<div class="leg"><div class="leg-line" style="background:${col}"></div>${r.run_id.substring(0, 18)}</div>`;
        }).join('') + '</div>';
    }

    // Toolbar
    const isExpanded = state.expandedMetrics.has(metric);
    const toolbarHtml = `<div class="chart-toolbar">
        ${!isHist ? `
            <button class="chart-tool-btn" data-mode="box" data-cv="${cvId}" title="Drag to zoom into area">${ICON.zoomBox}</button>
            <button class="chart-tool-btn" data-action="reset" data-cv="${cvId}" title="Reset zoom (Home)">${ICON.home}</button>
            <div class="chart-tool-sep"></div>
        ` : ''}
        <button class="chart-tool-btn chart-tool-expand" data-action="expand" data-cv="${cvId}" data-metric="${metric}" title="Expand / Collapse">${isExpanded ? ICON.collapse : ICON.expand}</button>
        <button class="chart-tool-btn" data-action="download" data-cv="${cvId}" data-metric="${metric}" title="Download as PNG">${ICON.download}</button>
    </div>`;

    card.innerHTML = `
        <div class="chart-header">
            <div class="chart-name">${metric}</div>
            ${toolbarHtml}
        </div>
        <div class="chart-stats-row">
            <div class="cstat"><span class="cstat-label">current</span><span class="cstat-value" style="color:${getRunColorById(state.selectedRuns[0].run_id)}">${ps ? fmt(ps.latest) : '\u2014'}</span></div>
            <div class="cstat"><span class="cstat-label">min</span><span class="cstat-value">${ps ? fmt(ps.min) : '\u2014'}</span></div>
            <div class="cstat"><span class="cstat-label">max</span><span class="cstat-value">${ps ? fmt(ps.max) : '\u2014'}</span></div>
            <div class="cstat"><span class="cstat-label">mean</span><span class="cstat-value">${ps ? fmt(ps.mean) : '\u2014'}</span></div>
        </div>
        ${legendHtml}
        <div class="chart-wrapper"><canvas id="${cvId}"></canvas></div>`;
    container.appendChild(card);

    // Attach toolbar event handlers
    card.querySelectorAll('.chart-tool-btn').forEach(btn => {
        const action = btn.dataset.action;
        const mode = btn.dataset.mode;
        if (mode) {
            btn.addEventListener('click', () => chartSetMode(cvId, mode));
        } else if (action === 'reset') {
            btn.addEventListener('click', () => chartResetZoom(cvId));
        } else if (action === 'expand') {
            btn.addEventListener('click', () => chartToggleExpand(cvId, metric));
        } else if (action === 'download') {
            btn.addEventListener('click', () => chartDownload(cvId, metric));
        }
    });

    const ctx = document.getElementById(cvId);

    try {
        if (isHist) {
            buildHistogramChart(ctx, cvId, metric, runDataMap, multiRun);
        } else {
            buildLineChart(ctx, cvId, metric, runDataMap);
        }
    } catch (e) {
        console.error('Chart render error for', metric, e);
    }
}

// ─── Histogram (bar) chart ───
function buildHistogramChart(ctx, cvId, metric, runDataMap, multiRun) {
    const datasets = state.selectedRuns.map((r) => {
        const d = runDataMap[r.run_id]?.[metric] || [];
        const last = d[d.length - 1] || {};
        const v = last.value || [];
        const col = getRunColorById(r.run_id);
        return { label: r.run_id.substring(0, 18), data: v, backgroundColor: col + '99', borderColor: col, borderWidth: 1 };
    }).filter(ds => ds.data.length > 0);

    const firstWithData = state.selectedRuns.find(r => {
        const d = runDataMap[r.run_id]?.[metric] || [];
        return d.length > 0;
    });
    const fd = runDataMap[firstWithData.run_id][metric];
    const last = fd[fd.length - 1] || {};
    const k = last.keys && last.keys.length ? last.keys : (last.value || []).map((_, i) => i);

    state.chartInsts[cvId] = new Chart(ctx, {
        type: 'bar',
        data: { labels: k, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: multiRun, labels: { color: '#4a6080', font: { family: 'JetBrains Mono', size: 10 } } },
                tooltip: {
                    backgroundColor: 'rgba(13,17,23,0.95)', titleColor: '#4a6080', bodyColor: '#c9d8e8',
                    borderColor: '#1e2a38', borderWidth: 1, padding: 6,
                    titleFont: { family: 'JetBrains Mono', size: 9 },
                    bodyFont: { family: 'JetBrains Mono', size: 10 },
                    displayColors: false,
                },
            },
            scales: {
                x: { grid: { color: '#1e2a38' }, ticks: { color: '#4a6080', font: { family: 'JetBrains Mono', size: 10 }, maxRotation: 45 } },
                y: { grid: { color: '#1e2a38' }, ticks: { color: '#4a6080', font: { family: 'JetBrains Mono', size: 10 } } },
            },
        },
    });
}

// ─── Line chart ───
function buildLineChart(ctx, cvId, metric, runDataMap) {
    const primaryId = state.selectedRuns[0].run_id;
    const datasets = state.selectedRuns.map((r) => {
        const d = runDataMap[r.run_id]?.[metric] || [];
        const col = getRunColorById(r.run_id);
        return {
            label: r.run_id.substring(0, 18),
            data: d.map(p => ({ x: Number(p.x), y: Number(p.y) })),
            borderColor: col,
            backgroundColor: col + '15',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: state.selectedRuns.length === 1,
            borderDash: r.run_id === primaryId ? [] : [5, 3],
        };
    }).filter(ds => ds.data.length > 0);

    // Compute data range for zoom limits
    let allX = [], allY = [];
    datasets.forEach(ds => ds.data.forEach(p => { allX.push(p.x); allY.push(p.y); }));
    const yPad = (Math.max(...allY) - Math.min(...allY)) * 0.05 || 1;
    const dataRange = allX.length ? {
        xMin: Math.min(...allX), xMax: Math.max(...allX),
        yMin: Math.min(...allY) - yPad, yMax: Math.max(...allY) + yPad,
        yRange: Math.max(...allY) - Math.min(...allY),
    } : null;

    state.chartInsts[cvId] = new Chart(ctx, { type: 'line', data: { datasets }, options: baseOpts(true, dataRange) });
}

// ─── Base Chart.js options for line charts ───
function baseOpts(line, dataRange) {
    return {
        responsive: true, maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'nearest', axis: 'x' },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(13,17,23,0.95)', titleColor: '#4a6080', bodyColor: '#c9d8e8',
                borderColor: '#1e2a38', borderWidth: 1, padding: 6,
                titleFont: { family: 'JetBrains Mono', size: 9 },
                bodyFont: { family: 'JetBrains Mono', size: 10 },
                displayColors: false,
                callbacks: line ? {
                    title: i => `Step ${i[0].parsed.x.toLocaleString()}`,
                    label: i => `${fmt(i.parsed.y)}`,
                } : {},
            },
            zoom: {
                pan: { enabled: true, mode: 'x' },
                zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                limits: dataRange ? {
                    x: { min: dataRange.xMin, max: dataRange.xMax, minRange: 10 },
                    y: { min: dataRange.yMin, max: dataRange.yMax, minRange: dataRange.yRange * 0.01 },
                } : {},
            },
        },
        scales: line ? {
            x: {
                type: 'linear',
                title: { display: true, text: 'Step', color: '#4a6080', font: { family: 'JetBrains Mono', size: 10 } },
                grid: { color: '#1e2a38' },
                ticks: { maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 10 }, callback: v => v.toLocaleString() },
            },
            y: {
                grid: { color: '#1e2a38' },
                ticks: { font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 6 },
            },
        } : {
            x: { grid: { color: '#1e2a38' } },
            y: { grid: { color: '#1e2a38' } },
        },
    };
}

// ─── Toolbar actions ───

function chartDownload(cvId, metric) {
    const chart = state.chartInsts[cvId];
    if (!chart) return;
    const link = document.createElement('a');
    link.download = `${metric.replace(/\//g, '_')}.png`;
    link.href = chart.toBase64Image('image/png', 1);
    link.click();
    showToast('Chart saved as PNG');
}

function chartResetZoom(cvId) {
    const chart = state.chartInsts[cvId];
    if (chart) { chart.resetZoom(); showToast('Zoom reset'); }
}

function chartToggleExpand(cvId, metric) {
    const canvas = document.getElementById(cvId);
    if (!canvas) return;
    const card = canvas.closest('.chart-card');
    if (!card) return;
    const expanding = !card.classList.contains('expanded');
    card.classList.toggle('expanded');
    if (expanding) state.expandedMetrics.add(metric); else state.expandedMetrics.delete(metric);
    const btn = card.querySelector('.chart-tool-expand');
    if (btn) btn.innerHTML = expanding ? ICON.collapse : ICON.expand;
    const chart = state.chartInsts[cvId];
    if (chart) setTimeout(() => chart.resize(), 280);
}

function chartSetMode(cvId, mode) {
    const chart = state.chartInsts[cvId];
    if (!chart) return;
    const cur = state.zoomMode[cvId];
    const newMode = cur === mode ? null : mode;
    state.zoomMode[cvId] = newMode;

    const card = document.getElementById(cvId)?.closest('.chart-card');
    if (card) {
        card.querySelectorAll('.chart-tool-btn[data-mode]').forEach(b => b.classList.remove('active'));
        if (newMode) {
            const btn = card.querySelector(`.chart-tool-btn[data-mode="${newMode}"]`);
            if (btn) btn.classList.add('active');
        }
    }

    const zp = chart.options.plugins.zoom;
    if (newMode === 'box') {
        zp.zoom.drag = { enabled: true, backgroundColor: 'rgba(0,212,255,0.1)', borderColor: 'rgba(0,212,255,0.4)', borderWidth: 1 };
        zp.zoom.wheel = { enabled: false };
        zp.zoom.mode = 'xy';
        zp.pan.enabled = false;
    } else {
        zp.zoom.drag = { enabled: false };
        zp.zoom.wheel = { enabled: true };
        zp.zoom.mode = 'x';
        zp.pan.enabled = true;
        zp.pan.mode = 'x';
        delete zp.pan.modifierKey;
    }
    chart.update('none');
    showToast(newMode === 'box' ? 'Drag to zoom' : 'Default mode');
}
