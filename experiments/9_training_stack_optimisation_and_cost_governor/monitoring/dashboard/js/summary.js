/**
 * Summary tab: KPI cards, pie charts, stats table with sparklines, heatmap.
 */

import { state } from './state.js';
import { RUN_COLORS } from './constants.js';
import { fmt, stats, getRunColor, heatColor } from './utils.js';
import { fetchData } from './api.js';

export async function loadSummary() {
    if (!state.selectedRuns.length) return;
    const el = document.getElementById('summaryContent');
    el.innerHTML = '<div class="empty"><div class="spinner"></div></div>';

    // Collect all metric names
    const allMetricNames = [];
    for (const [cat, items] of Object.entries(state.allMetrics)) {
        items.forEach(m => allMetricNames.push(typeof m === 'object' ? m.name : m));
    }
    if (!allMetricNames.length) {
        el.innerHTML = '<div class="empty"><div class="empty-icon">&#9638;</div><div class="empty-title">No metrics</div><div class="empty-sub">No metrics discovered for this run</div></div>';
        return;
    }

    // Fetch data for all runs
    const runDataMap = {};
    const results = await Promise.all(
        state.selectedRuns.map(r => fetchData(r.run_id, allMetricNames).then(d => ({ runId: r.run_id, data: d })))
    );
    results.forEach(r => runDataMap[r.runId] = r.data);

    // Destroy old summary charts
    Object.values(state.summaryCharts).forEach(c => c.destroy());
    state.summaryCharts = {};

    // Compute stats per metric per run
    const metricStats = {};
    const scalarMetrics = [];
    const arrayMetrics = [];
    allMetricNames.forEach(name => {
        metricStats[name] = {};
        let isArr = false;
        state.selectedRuns.forEach(r => {
            const d = runDataMap[r.run_id]?.[name] || [];
            if (d.length && Array.isArray(d[0]?.value)) isArr = true;
            metricStats[name][r.run_id] = { data: d, stats: stats(d) };
        });
        if (isArr) arrayMetrics.push(name); else scalarMetrics.push(name);
    });

    // Count categories
    const catCounts = {};
    for (const [cat, items] of Object.entries(state.allMetrics)) {
        catCounts[cat] = items.length;
    }

    const primaryRun = state.selectedRuns[0];
    const primaryData = runDataMap[primaryRun.run_id] || {};

    // Max step
    let maxStep = 0;
    scalarMetrics.forEach(m => {
        const d = primaryData[m] || [];
        if (d.length) maxStep = Math.max(maxStep, Number(d[d.length - 1].x || 0));
    });

    el.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'summary-grid';
    el.appendChild(grid);

    // ROW 1: KPI Overview
    buildKpiRow(grid, scalarMetrics, arrayMetrics, catCounts, maxStep);

    // ROW 2: Pie charts + Key metrics
    buildPieCard(grid, catCounts, scalarMetrics, arrayMetrics);
    buildTopMetricsCard(grid, scalarMetrics, metricStats, primaryRun);

    // ROW 3: Full stats table
    const sparklineIds = buildStatsTable(grid, allMetricNames, arrayMetrics, metricStats, primaryRun);

    // ROW 4: Heatmap
    if (scalarMetrics.length > 0 && state.selectedRuns.length > 0) {
        buildHeatmap(grid, scalarMetrics, metricStats);
    }

    // Render pie charts
    renderPieCharts(catCounts, scalarMetrics, arrayMetrics);

    // Render sparklines
    renderSparklines(sparklineIds, primaryData);
}

// ─── KPI row ───
function buildKpiRow(grid, scalarMetrics, arrayMetrics, catCounts, maxStep) {
    grid.insertAdjacentHTML('beforeend', `<div class="summary-full">
        <div class="kpi-row">
            <div class="kpi-box"><div class="kpi-value">${state.selectedRuns.length}</div><div class="kpi-label">Runs Selected</div></div>
            <div class="kpi-box"><div class="kpi-value">${scalarMetrics.length}</div><div class="kpi-label">Scalar Metrics</div></div>
            <div class="kpi-box"><div class="kpi-value">${arrayMetrics.length}</div><div class="kpi-label">Array Metrics</div></div>
            <div class="kpi-box"><div class="kpi-value">${maxStep.toLocaleString()}</div><div class="kpi-label">Max Step</div></div>
            <div class="kpi-box"><div class="kpi-value">${Object.keys(catCounts).length}</div><div class="kpi-label">Categories</div></div>
        </div>
    </div>`);
}

// ─── Pie chart card (categories + types) ───
function buildPieCard(grid, catCounts, scalarMetrics, arrayMetrics) {
    const pieCard = document.createElement('div');
    pieCard.className = 'summary-card';
    pieCard.innerHTML = `<div class="summary-card-title">Metrics by Category</div>
        <div class="pie-row"><div class="pie-wrap"><canvas id="pie-cats"></canvas></div>
        <div class="pie-wrap"><canvas id="pie-types"></canvas></div></div>`;
    grid.appendChild(pieCard);
}

// ─── Top metrics card ───
function buildTopMetricsCard(grid, scalarMetrics, metricStats, primaryRun) {
    const topCard = document.createElement('div');
    topCard.className = 'summary-card';
    const topScalars = scalarMetrics
        .map(m => ({ name: m, s: metricStats[m][primaryRun.run_id]?.stats }))
        .filter(x => x.s)
        .sort((a, b) => Math.abs(b.s.latest) - Math.abs(a.s.latest))
        .slice(0, 8);
    topCard.innerHTML = `<div class="summary-card-title">Key Metrics (${primaryRun.run_id.substring(0, 20)})</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem">${topScalars.map(x => {
            const trend = x.s.latest > x.s.mean ? 'trend-up' : x.s.latest < x.s.mean ? 'trend-down' : 'trend-flat';
            const arrow = x.s.latest > x.s.mean ? '&#9650;' : x.s.latest < x.s.mean ? '&#9660;' : '&#9654;';
            return `<div style="background:var(--bg);border-radius:0.4rem;padding:0.55rem 0.7rem">
                <div style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);margin-bottom:0.2rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${x.name}</div>
                <div style="display:flex;align-items:baseline;gap:0.4rem">
                    <span style="font-family:var(--mono);font-size:0.95rem;font-weight:700;color:var(--text)">${fmt(x.s.latest)}</span>
                    <span class="${trend}" style="font-size:0.65rem">${arrow} ${fmt(x.s.mean)} avg</span>
                </div>
            </div>`;
        }).join('')}</div>`;
    grid.appendChild(topCard);
}

// ─── Full stats table ───
function buildStatsTable(grid, allMetricNames, arrayMetrics, metricStats, primaryRun) {
    const tableCard = document.createElement('div');
    tableCard.className = 'summary-card summary-full';
    let tableHtml = `<div class="summary-card-title">All Metrics \u2014 Detailed Stats</div>
        <div style="overflow-x:auto"><table class="stats-table"><thead><tr>
            <th>Metric</th><th>Category</th><th>Kind</th>
            ${state.selectedRuns.map((r, i) => `<th style="color:${getRunColor(i)}">Latest (${r.run_id.substring(0, 12)})</th>`).join('')}
            <th>Min</th><th>Max</th><th>Mean</th><th>Trend</th><th>Sparkline</th>
        </tr></thead><tbody>`;

    const sparklineIds = [];
    allMetricNames.forEach((name, mi) => {
        let cat = '\u2014';
        for (const [c, items] of Object.entries(state.allMetrics)) {
            if (items.some(m => (typeof m === 'object' ? m.name : m) === name)) { cat = c; break; }
        }
        const isArr = arrayMetrics.includes(name);
        const ps0 = metricStats[name][primaryRun.run_id]?.stats;
        if (!ps0) return;

        const trend = ps0.latest > ps0.mean ? 'trend-up' : ps0.latest < ps0.mean ? 'trend-down' : 'trend-flat';
        const arrow = ps0.latest > ps0.mean ? '&#9650;' : ps0.latest < ps0.mean ? '&#9660;' : '&#8212;';

        tableHtml += `<tr>
            <td class="metric-name-cell" title="${name}">${name}</td>
            <td>${cat}</td>
            <td><span class="kind-badge kind-${isArr ? 'array' : 'scalar'}">${isArr ? 'hist' : 'val'}</span></td>`;
        state.selectedRuns.forEach((r, ri) => {
            const rs = metricStats[name][r.run_id]?.stats;
            tableHtml += `<td style="color:${getRunColor(ri)}">${rs ? fmt(rs.latest) : '\u2014'}</td>`;
        });
        const spkId = `spk${mi}`;
        sparklineIds.push({ id: spkId, name, isArr });
        tableHtml += `<td>${fmt(ps0.min)}</td><td>${fmt(ps0.max)}</td><td>${fmt(ps0.mean)}</td>
            <td><span class="${trend}">${arrow}</span></td>
            <td class="sparkline-cell"><canvas id="${spkId}" width="80" height="24"></canvas></td>
        </tr>`;
    });
    tableHtml += '</tbody></table></div>';
    tableCard.innerHTML = tableHtml;
    grid.appendChild(tableCard);
    return sparklineIds;
}

// ─── Heatmap ───
function buildHeatmap(grid, scalarMetrics, metricStats) {
    const heatCard = document.createElement('div');
    heatCard.className = 'summary-card summary-full';
    const heatMetrics = scalarMetrics.slice(0, 20);

    let heatHtml = `<div class="summary-card-title">Heatmap \u2014 Latest Values (Normalized per Metric)</div>
        <div class="heatmap-container"><div class="heatmap-grid" style="grid-template-columns: 120px repeat(${state.selectedRuns.length}, 60px)">`;

    // Header row
    heatHtml += `<div class="heatmap-cell heatmap-header"></div>`;
    state.selectedRuns.forEach((r, i) => {
        heatHtml += `<div class="heatmap-cell heatmap-header" style="color:${getRunColor(i)}" title="${r.run_id}">${r.run_id.substring(0, 8)}</div>`;
    });

    heatMetrics.forEach(m => {
        const vals = state.selectedRuns.map(r => metricStats[m][r.run_id]?.stats?.latest ?? null);
        const nums = vals.filter(v => v !== null);
        const lo = Math.min(...nums), hi = Math.max(...nums);
        const range = hi - lo || 1;

        heatHtml += `<div class="heatmap-cell heatmap-row-label" title="${m}">${m.length > 16 ? m.substring(0, 15) + '\u2026' : m}</div>`;
        vals.forEach(v => {
            if (v === null) {
                heatHtml += `<div class="heatmap-cell" style="background:var(--border)">\u2014</div>`;
            } else {
                const t = (v - lo) / range;
                const bg = heatColor(t);
                heatHtml += `<div class="heatmap-cell" style="background:${bg}" title="${m}: ${fmt(v)}">${fmt(v)}</div>`;
            }
        });
    });

    heatHtml += '</div></div>';
    heatCard.innerHTML = heatHtml;
    grid.appendChild(heatCard);
}

// ─── Render pie charts (called after DOM is built) ───
function renderPieCharts(catCounts, scalarMetrics, arrayMetrics) {
    const catLabels = Object.keys(catCounts);
    const catVals = Object.values(catCounts);
    const catColors = catLabels.map((_, i) => RUN_COLORS[i % RUN_COLORS.length]);

    const pieOpts = (pos) => ({
        responsive: true, maintainAspectRatio: true, cutout: '55%',
        plugins: {
            legend: { position: pos, labels: { color: '#4a6080', font: { family: 'JetBrains Mono', size: 9 }, padding: 8, boxWidth: 10 } },
            tooltip: { backgroundColor: 'rgba(13,17,23,0.95)', bodyColor: '#c9d8e8', padding: 6, bodyFont: { family: 'JetBrains Mono', size: 10 } },
        },
    });

    state.summaryCharts['pie-cats'] = new Chart(document.getElementById('pie-cats'), {
        type: 'doughnut',
        data: { labels: catLabels, datasets: [{ data: catVals, backgroundColor: catColors.map(c => c + 'cc'), borderColor: catColors, borderWidth: 1.5 }] },
        options: pieOpts('bottom'),
    });

    state.summaryCharts['pie-types'] = new Chart(document.getElementById('pie-types'), {
        type: 'doughnut',
        data: { labels: ['Scalar', 'Array/Histogram'], datasets: [{ data: [scalarMetrics.length, arrayMetrics.length], backgroundColor: ['#00d4ff99', '#a78bfa99'], borderColor: ['#00d4ff', '#a78bfa'], borderWidth: 1.5 }] },
        options: pieOpts('bottom'),
    });
}

// ─── Render sparklines (called after DOM is built) ───
function renderSparklines(sparklineIds, primaryData) {
    sparklineIds.forEach(({ id, name, isArr }) => {
        const cvs = document.getElementById(id);
        if (!cvs || isArr) return;
        const d = primaryData[name] || [];
        if (d.length < 2) return;
        const yVals = d.map(p => Number(p.y));
        const last = yVals[yVals.length - 1];
        const first = yVals[0];
        const col = last >= first ? '#00e5a0' : '#ff4d6a';
        state.summaryCharts[id] = new Chart(cvs, {
            type: 'line',
            data: { labels: d.map((_, i) => i), datasets: [{ data: yVals, borderColor: col, borderWidth: 1.2, pointRadius: 0, fill: false, tension: 0.4 }] },
            options: {
                responsive: false, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } },
                animation: false,
            },
        });
    });
}
