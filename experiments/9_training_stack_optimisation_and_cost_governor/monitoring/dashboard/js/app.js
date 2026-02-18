/**
 * Application entry point: init, auto-refresh orchestration, event wiring.
 */

import { state } from './state.js';
import { initChartDefaults } from './constants.js';
import { showToast } from './utils.js';
import { loadRuns, toggleRunDropdown, initDropdownClickOutside } from './runs.js';
import { filterMetrics } from './metrics.js';
import { generateCharts } from './charts.js';
import { loadEvents } from './events.js';
import { loadSummary } from './summary.js';
import { switchTab } from './tabs.js';

// ─── Auto-refresh ───

function setAutoRefresh() {
    clearInterval(state.refreshTimer);
    const ms = parseInt(document.getElementById('refreshInterval').value);
    const dot = document.getElementById('liveDot');
    const lbl = document.getElementById('liveLabel');
    if (ms > 0) {
        dot.classList.remove('off');
        lbl.textContent = `live \u00b7 ${ms >= 60000 ? ms / 60000 + 'm' : ms / 1000 + 's'}`;
        state.refreshTimer = setInterval(autoRefresh, ms);
    } else {
        dot.classList.add('off');
        lbl.textContent = 'live off';
    }
}

async function autoRefresh() {
    if (!state.selectedRuns.length) return;
    await loadRuns();
    const sel = Array.from(document.querySelectorAll('#metricScroll input:checked')).map(c => c.value);
    if (sel.length) await generateCharts();
    if (document.getElementById('panel-events').classList.contains('active')) await loadEvents();
    if (document.getElementById('panel-summary').classList.contains('active')) await loadSummary();
    showToast('\u21ba refreshed');
}

function manualRefresh() {
    loadRuns();
    if (state.selectedRuns.length) {
        const sel = Array.from(document.querySelectorAll('#metricScroll input:checked')).map(c => c.value);
        if (sel.length) generateCharts();
    }
    showToast('\u21ba refreshed');
}

// ─── Wire up DOM events ───

// ─── Sidebar: desktop collapse + mobile drawer ───

function isMobile() {
    return window.matchMedia('(max-width: 680px)').matches;
}

function openMobileSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    sidebar.classList.add('mobile-open');
    backdrop.style.display = 'block';
    // Force reflow so the opacity transition fires
    backdrop.offsetHeight;
    backdrop.classList.add('visible');
    document.body.style.overflow = 'hidden';
}

function closeMobileSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    sidebar.classList.remove('mobile-open');
    backdrop.classList.remove('visible');
    document.body.style.overflow = '';
    // Hide backdrop after transition
    setTimeout(() => {
        if (!backdrop.classList.contains('visible')) backdrop.style.display = 'none';
    }, 280);
}

function wireEvents() {
    initDropdownClickOutside();

    document.getElementById('runDropdownBtn').addEventListener('click', toggleRunDropdown);
    document.getElementById('refreshInterval').addEventListener('change', setAutoRefresh);
    document.getElementById('refreshBtn').addEventListener('click', manualRefresh);
    document.getElementById('metricSearch').addEventListener('input', filterMetrics);

    // Desktop collapse toggle
    document.getElementById('sidebarToggle').addEventListener('click', () => {
        document.getElementById('layout').classList.toggle('sidebar-collapsed');
    });

    // Mobile hamburger opens drawer
    document.getElementById('mobileMenuBtn').addEventListener('click', openMobileSidebar);

    // Backdrop click closes drawer
    document.getElementById('sidebarBackdrop').addEventListener('click', closeMobileSidebar);

    // Close drawer when a metric is checked or a run selected on mobile
    document.getElementById('metricScroll').addEventListener('change', () => {
        if (isMobile()) closeMobileSidebar();
    });

    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const name = tab.id.replace('tab-', '');
            switchTab(name);
        });
    });
}

// ─── Init ───

async function init() {
    initChartDefaults();
    wireEvents();
    await loadRuns();
    setAutoRefresh();
}

init();
